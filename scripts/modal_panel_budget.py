"""Conservative fleet-time allocation for one frozen research queue.

The bound charges every allowed container continuously, including idle slots,
then reserves a shutdown tail. It is an allocation estimate at pinned list
prices. Provider billing and external deployment termination remain separate.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
import threading
import time

SCHEMA = 'faynt.modal-fleet-budget.v1'
CPU_USD_PER_SECOND = 0.0000131
GIB_USD_PER_SECOND = 0.00000222
# This charged tail allows startup/termination after the execution cutoff.
SHUTDOWN_TAIL_SECONDS = 600
RESUME_SCHEMA = 'faynt.modal-budget-resume.v1'
MAX_RESUME_TRANCHES = 64
# The reviewed inclusive cap. A resume authorization may name a higher total
# only through its explicit cap_increase block, and never above the absolute limit.
TOTAL_RAIL_USD = 150
MAX_AUTHORIZED_TOTAL_USD = 400


def digest(value):
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()


def prepare_resume_authorization(**fields):
    """Build offline authorization; live stopped-app verification is separate."""
    return validate_resume_authorization({'schema':RESUME_SCHEMA,**fields})


def validate_cap_increase(value):
    """One explicit user raise of the inclusive cap, bounded by the absolute limit."""
    if (not isinstance(value,dict) or set(value)!={'previous_total_usd','total_usd','instruction'}
            or any(type(value[key]) not in (int,float) or not math.isfinite(value[key])
                for key in ('previous_total_usd','total_usd'))
            or not value['previous_total_usd']<value['total_usd']<=MAX_AUTHORIZED_TOTAL_USD
            or not isinstance(value['instruction'],str) or not value['instruction'].strip()):
        raise ValueError('explicit cap increase requires the prior total, a higher total within the absolute limit and user text')
    return dict(value)


def authorized_total_usd(authorization):
    """The raised inclusive cap a resume authorization names, or None without a raise."""
    increase=authorization.get('cap_increase') if isinstance(authorization,dict) else None
    return None if increase is None else validate_cap_increase(increase)['total_usd']


def validate_resume_authorization(value):
    expected={'schema','queue_sha256','app_name','environment_name','previous_app_id',
        'previous_budget_sha256','budget_policy','user_authorization','prior_cost','created_unix'}
    if isinstance(value,dict) and 'cap_increase' in value:
        expected.add('cap_increase')
    if not isinstance(value,dict) or set(value)!=expected or value['schema']!=RESUME_SCHEMA:
        raise ValueError('exact explicit budget resume authorization required')
    # A raised total passes the policy rail only through the explicit block, so
    # user_authorization.total_usd, budget_policy.total_usd and cap_increase.total_usd must agree.
    policy=validate_policy(value['budget_policy'],authorized_total_usd=authorized_total_usd(value))
    if policy.get('enforcement') is not None or value['environment_name']!='main':
        raise ValueError('resume uses the reviewed main-environment fleet budget')
    for name in ('queue_sha256','previous_budget_sha256'):
        if not isinstance(value[name],str) or not re.fullmatch(r'[0-9a-f]{64}',value[name]):
            raise ValueError('resume requires exact immutable queue and prior budget hashes')
    if (not isinstance(value['app_name'],str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value['app_name'])
            or not isinstance(value['previous_app_id'],str)
            or not re.fullmatch(r'ap-[A-Za-z0-9]+',value['previous_app_id'])):
        raise ValueError('exact deployment and stopped predecessor identity required')
    user=value['user_authorization']; cost=value['prior_cost']
    if (not isinstance(user,dict) or set(user)!={'total_usd','instruction'}
            or user['total_usd']!=policy['total_usd'] or type(user['total_usd']) not in (int,float)
            or not isinstance(user['instruction'],str) or not user['instruction'].strip()):
        raise ValueError('explicit existing user authorization and inclusive total required')
    if (not isinstance(cost,dict) or set(cost)!={'metered_usd','evidence_sha256'}
            or type(cost['metered_usd']) not in (int,float) or not math.isfinite(cost['metered_usd'])
            or not 0<=cost['metered_usd']<=policy.get('prior_spend_usd',0)
            or not isinstance(cost['evidence_sha256'],str)
            or not re.fullmatch(r'[0-9a-f]{64}',cost['evidence_sha256'])):
        raise ValueError('hash-bound prior cost evidence must fit conservative prior allowance')
    if (type(value['created_unix']) not in (int,float) or not math.isfinite(value['created_unix'])
            or value['created_unix']<=0 or policy.get('prior_spend_usd',0)<=0):
        raise ValueError('finite authorization timestamp and prior allowance required')
    return dict(value)


def validate_stop_observation(authorization, observation, *, now=None):
    authorization=validate_resume_authorization(authorization)
    expected={'previous_app_id','profile','app_state','stopped_at','verified_unix','functions'}
    if (not isinstance(observation,dict) or set(observation)!=expected
            or observation['previous_app_id']!=authorization['previous_app_id']
            or observation['profile']!='frisson' or observation['app_state']!='STOPPED'):
        raise ValueError('stopped predecessor app observation required')
    for key in ('stopped_at','verified_unix'):
        if type(observation[key]) not in (int,float) or not math.isfinite(observation[key]) or observation[key]<=0:
            raise ValueError('finite stopped-app timestamps required')
    if observation['stopped_at']>observation['verified_unix']:
        raise ValueError('predecessor stop must precede verification')
    if now is not None and not now-120<=observation['verified_unix']<=now+1:
        raise ValueError('fresh stopped-app observation required before activation')
    functions=observation['functions']
    if (not isinstance(functions,list) or not functions
            or {row.get('tag') for row in functions}!={'run_cloud_game','dispatch_cloud'}):
        raise ValueError('both reviewed predecessor functions must be observed')
    for row in functions:
        if (set(row)!={'tag','function_id','backlog','num_total_tasks','num_running_inputs'}
                or not isinstance(row['function_id'],str) or not re.fullmatch(r'fu-[A-Za-z0-9]+',row['function_id'])
                or any(type(row[key]) is not int or row[key]!=0
                    for key in ('backlog','num_total_tasks','num_running_inputs'))):
            raise ValueError('predecessor functions must have zero runners, inputs and backlog')
    return dict(observation)


async def observe_stopped_app(authorization):
    """Read only exact predecessor lifecycle and function stats through the SDK."""
    authorization=validate_resume_authorization(authorization)
    if os.environ.get('MODAL_PROFILE')!='frisson':
        raise ValueError('authenticated frisson profile required for resume preflight')
    from modal.client import _Client
    from modal_proto import api_pb2
    client=await _Client.from_env(); app_id=authorization['previous_app_id']
    response=await client.stub.AppGetLifecycle(api_pb2.AppGetLifecycleRequest(app_id=app_id))
    lifecycle=response.lifecycle
    if lifecycle.app_state!=api_pb2.APP_STATE_STOPPED:
        raise ValueError('predecessor app must be stopped before a new budget tranche')
    layout=(await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))).app_layout
    if set(layout.function_ids)!={'run_cloud_game','dispatch_cloud'}:
        raise ValueError('predecessor layout must contain exactly the reviewed benchmark functions')
    objects={item.object_id:item for item in layout.objects}
    functions=[]
    for tag in ('run_cloud_game','dispatch_cloud'):
        function_id=layout.function_ids[tag]
        if (function_id not in objects
                or objects[function_id].function_handle_metadata.app_id!=app_id):
            raise ValueError('predecessor function belongs to a different deployment app')
        stats=await client.stub.FunctionGetCurrentStats(
            api_pb2.FunctionGetCurrentStatsRequest(function_id=function_id))
        functions.append({'tag':tag,'function_id':function_id,
            'backlog':stats.backlog,'num_total_tasks':stats.num_total_tasks,
            'num_running_inputs':stats.num_running_inputs})
    return validate_stop_observation(authorization,{'previous_app_id':app_id,'profile':'frisson',
        'app_state':'STOPPED','stopped_at':lifecycle.stopped_at,'verified_unix':time.time(),
        'functions':sorted(functions,key=lambda row:row['tag'])},now=time.time())


def validate_resume_tranche(tranche, previous_budget):
    expected={'schema','authorization','authorization_sha256','previous_budget_sha256',
        'stop_observation','activation'}
    if not isinstance(tranche,dict) or set(tranche)!=expected or tranche['schema']!=RESUME_SCHEMA+'.tranche':
        raise ValueError('immutable budget tranche required')
    authorization=validate_resume_authorization(tranche['authorization'])
    if (tranche['authorization_sha256']!=digest(authorization)
            or tranche['previous_budget_sha256']!=authorization['previous_budget_sha256']
            or digest(previous_budget)!=tranche['previous_budget_sha256']):
        raise ValueError('budget tranche ancestry or authorization binding differs')
    activation=validate_activation(authorization['budget_policy'],tranche['activation'])
    if activation.get('authorized_total_usd')!=authorized_total_usd(authorization):
        raise ValueError('budget tranche activation must record exactly the authorized cap')
    validate_cap_increase_ancestry(authorization,previous_budget)
    validate_stop_observation(authorization,tranche['stop_observation'],now=activation['started_unix'])
    if (not activation['started_unix']-86400<=authorization['created_unix']<=activation['started_unix']
            or previous_budget.get('deadline_unix',float('inf'))>authorization['created_unix']):
        raise ValueError('resume requires fresh authorization after prior tranche expired')
    return tranche


def validate_cap_increase_ancestry(authorization, previous_budget):
    """Bind an explicit raise to the predecessor receipt's exact policy hash.

    A new raise starts from previous_total_usd; a continuation repeats the raise
    its predecessor already recorded. Either way reserve, fleet resources and
    the predecessor's own prior spend must hash to the predecessor policy.
    """
    increase=authorization.get('cap_increase')
    if increase is None:
        return
    recorded=previous_budget.get('authorized_total_usd')
    total=increase['total_usd'] if recorded==increase['total_usd'] else increase['previous_total_usd']
    predecessor={k:v for k,v in authorization['budget_policy'].items() if k!='prior_spend_usd'}
    predecessor['total_usd']=total
    if 'prior_spend_usd' in previous_budget:
        predecessor['prior_spend_usd']=previous_budget['prior_spend_usd']
    try:
        bound=policy_sha(predecessor,authorized_total_usd=recorded)==previous_budget.get('policy_sha256')
    except ValueError:
        bound=False
    if not bound:
        raise ValueError('cap increase must start from the predecessor cap with unchanged reserve and fleet resources')


def validate_resume_successor(tranche, previous_tranche):
    """Validate a new allocation while retaining every prior budget receipt."""
    previous=previous_tranche['authorization']
    validate_resume_tranche(tranche,previous_tranche['activation'])
    authorization=tranche['authorization']
    if any(authorization[name]!=previous[name]
            for name in ('queue_sha256','app_name','environment_name')):
        raise ValueError('successor must preserve the original queue and deployment')
    policy=authorization['budget_policy']; prior_policy=previous['budget_policy']
    increase=authorization.get('cap_increase'); ignored={'prior_spend_usd'}
    if increase is not None and increase!=previous.get('cap_increase'):
        # A new raise names the predecessor total exactly; a continuation repeats
        # the predecessor's identical block and keeps its total.
        if (increase['previous_total_usd']!=prior_policy['total_usd']
                or policy['total_usd']<=prior_policy['total_usd']):
            raise ValueError('explicit cap increase must start from the predecessor cap and raise it')
        ignored.add('total_usd')
    if ({k:v for k,v in policy.items() if k not in ignored}
            !={k:v for k,v in prior_policy.items() if k not in ignored}):
        raise ValueError('successor must preserve the inclusive cap, reserve and fleet resources')
    if (policy.get('prior_spend_usd',0)<prior_policy.get('prior_spend_usd',0)
            or authorization['prior_cost']['metered_usd']<previous['prior_cost']['metered_usd']):
        raise ValueError('successor cumulative cost evidence and allowance cannot decrease')
    # A STOPPED predecessor cannot spend, and activate_resume_budget separately
    # refuses a successor while the prior clock is live, so a stop at any point
    # during or after the prior tranche is at least as safe as one at its cutoff.
    if tranche['stop_observation']['stopped_at']<previous_tranche['activation']['started_unix']:
        raise ValueError('successor requires a stopped-app observation from the prior tranche or later')
    return tranche


def make_policy(*, total_usd=TOTAL_RAIL_USD, reserve_usd=10, worker_memory_mib=16384, prior_spend_usd=0,
        authorized_total_usd=None):
    if type(prior_spend_usd) not in (int,float) or not math.isfinite(prior_spend_usd):
        raise ValueError('finite prior spend required')
    value={'schema': SCHEMA, 'total_usd': total_usd,
        'reserve_usd': reserve_usd, 'worker_cpu': 4,
        'worker_memory_mib': worker_memory_mib, 'maximum_workers': 64,
        'nonpreemptible': False}
    if prior_spend_usd!=0:
        value['prior_spend_usd']=prior_spend_usd
    return validate_policy(value,authorized_total_usd=authorized_total_usd)


def make_provider_policy(*, environment_name, environment_id, **kwargs):
    return validate_policy({**make_policy(**kwargs), 'enforcement': 'provider-budget',
        'environment_name': environment_name, 'environment_id': environment_id})


def validate_policy(value, *, authorized_total_usd=None):
    """Every caller keeps the reviewed rail unless it names the exact raised total."""
    expected = {'schema', 'total_usd', 'reserve_usd', 'worker_cpu',
                'worker_memory_mib', 'maximum_workers', 'nonpreemptible'}
    provider=isinstance(value,dict) and value.get('enforcement')=='provider-budget'
    if provider:
        expected|={'enforcement','environment_name','environment_id'}
    if isinstance(value,dict) and 'prior_spend_usd' in value:
        expected.add('prior_spend_usd')
    if not isinstance(value, dict) or set(value) != expected or value['schema'] != SCHEMA:
        raise ValueError('exact reviewed fleet budget policy required')
    for key in ('total_usd', 'reserve_usd'):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
            raise ValueError('finite dollar allocations required')
    rail=TOTAL_RAIL_USD
    if authorized_total_usd is not None:
        if (provider or type(authorized_total_usd) not in (int,float) or not math.isfinite(authorized_total_usd)
                or not authorized_total_usd<=MAX_AUTHORIZED_TOTAL_USD
                or value['total_usd']!=authorized_total_usd):
            raise ValueError('explicit cap increase must name this exact fleet-time total within the absolute limit')
        rail=authorized_total_usd
    if not 10 <= value['reserve_usd'] < value['total_usd'] <= rail:
        raise ValueError(f'budget must retain at least $10 overhead within ${rail:g} total')
    prior=value.get('prior_spend_usd',0)
    if (type(prior) not in (int,float) or not math.isfinite(prior)
            or not 0<=prior<value['total_usd']-value['reserve_usd']):
        raise ValueError('finite prior spend must fit inside the inclusive authorization')
    if (type(value['worker_cpu']) is not int or value['worker_cpu'] != 4
            or type(value['maximum_workers']) is not int or value['maximum_workers'] != 64
            or type(value['worker_memory_mib']) is not int
            or value['worker_memory_mib'] not in (8192, 16384)
            or value['nonpreemptible'] is not False):
        raise ValueError('reviewed 64-worker hard resources and standard preemptible rate required')
    if provider and (not isinstance(value['environment_name'],str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value['environment_name'])
            or value['environment_name']=='main' or not isinstance(value['environment_id'],str)
            or not re.fullmatch(r'en-[A-Za-z0-9]+',value['environment_id'])):
        raise ValueError('exact isolated provider environment required')
    return dict(value)


def policy_sha(value, *, authorized_total_usd=None):
    return hashlib.sha256(json.dumps(validate_policy(value,authorized_total_usd=authorized_total_usd),
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rates(value, *, authorized_total_usd=None):
    value = validate_policy(value,authorized_total_usd=authorized_total_usd)
    worker = value['worker_cpu'] * CPU_USD_PER_SECOND + value['worker_memory_mib'] / 1024 * GIB_USD_PER_SECOND
    dispatcher = 0.25 * CPU_USD_PER_SECOND + GIB_USD_PER_SECOND
    return {'worker_usd_per_second': worker, 'dispatcher_usd_per_second': dispatcher,
            'fleet_usd_per_second': value['maximum_workers'] * worker + dispatcher}


def validate_provider_observation(value, observation, *, now=None):
    value=validate_policy(value)
    now=time.time() if now is None else now
    if value.get('enforcement')!='provider-budget' or not isinstance(observation,dict):
        raise ValueError('bound live provider-budget observation required')
    expected={'environment_name','environment_id','profile','cycle_compute_budget_usd',
        'effective_cycle_spend_limit_usd','current_cycle_usage_usd','spend_limit_reached','verified_unix'}
    if set(observation)!=expected:
        raise ValueError('complete provider-budget observation required')
    cap=value['total_usd']-value['reserve_usd']-value.get('prior_spend_usd',0)
    if (observation['environment_name']!=value['environment_name']
            or observation['environment_id']!=value['environment_id']
            or observation['profile']!='frisson' or observation['spend_limit_reached'] is not False):
        raise ValueError('provider environment identity or spending status differs')
    for key in ('cycle_compute_budget_usd','effective_cycle_spend_limit_usd',
                'current_cycle_usage_usd','verified_unix'):
        if type(observation[key]) not in (int,float) or not math.isfinite(observation[key]):
            raise ValueError('finite provider budget values required')
    if (observation['cycle_compute_budget_usd']!=cap
            or not 0<observation['effective_cycle_spend_limit_usd']<=cap
            or not 0<=observation['current_cycle_usage_usd']<cap
            or not now-120<=observation['verified_unix']<=now+1):
        raise ValueError('fresh provider observation with unchanged authorized cap required')
    return dict(observation)


async def observe_provider(value, *, remote=False):
    """Read identity and cap using the pinned SDK. This method never changes a cap."""
    value=validate_policy(value)
    authenticated=(os.environ.get('PANEL_ENVIRONMENT')==value.get('environment_name')
        and re.fullmatch(r'ta-[A-Za-z0-9_-]+',os.environ.get('MODAL_TASK_ID',''))
        if remote else os.environ.get('MODAL_PROFILE')=='frisson')
    if value.get('enforcement')!='provider-budget' or not authenticated:
        raise ValueError('authorized frisson profile and provider-budget mode required')
    from google.protobuf.empty_pb2 import Empty
    from modal.client import _Client
    from modal_proto import api_pb2
    client=await _Client.from_env()
    environments=await client.stub.EnvironmentList(Empty())
    matches=[entry for entry in environments.items if entry.name==value['environment_name']]
    if len(matches)!=1 or matches[0].environment_id!=value['environment_id']:
        raise ValueError('live provider environment identity differs')
    observed=await client.stub.EnvironmentGetBudget(
        api_pb2.EnvironmentGetBudgetRequest(environment_id=value['environment_id']))
    receipt={'environment_name':value['environment_name'],'environment_id':value['environment_id'],
        'profile':'frisson','cycle_compute_budget_usd':observed.cycle_budget_dollars
            if observed.HasField('cycle_budget_dollars') else None,
        'effective_cycle_spend_limit_usd':observed.effective_cycle_spend_limit,
        'current_cycle_usage_usd':observed.current_cycle_usage,
        'spend_limit_reached':observed.spend_limit_reached,'verified_unix':time.time()}
    return validate_provider_observation(value,receipt)


def activate(value, started_unix, provider_observation=None, *, authorized_total_usd=None):
    value = validate_policy(value,authorized_total_usd=authorized_total_usd)
    if type(started_unix) not in (int, float) or not math.isfinite(started_unix) or started_unix <= 0:
        raise ValueError('finite activation timestamp required')
    pricing = rates(value,authorized_total_usd=authorized_total_usd)
    allocated = value['total_usd'] - value['reserve_usd'] - value.get('prior_spend_usd',0)
    prior={'prior_spend_usd':value['prior_spend_usd']} if 'prior_spend_usd' in value else {}
    # The receipt records the explicit raise it was computed under, so every later
    # recomputation reproduces the same rail and the tranche binds it to the authorization.
    raised={'authorized_total_usd':authorized_total_usd} if authorized_total_usd is not None else {}
    if value.get('enforcement')=='provider-budget':
        observation=validate_provider_observation(value,provider_observation,now=started_unix)
        instant=datetime.fromtimestamp(started_unix,timezone.utc)
        next_month=datetime(instant.year+(instant.month==12),instant.month%12+1,1,tzinfo=timezone.utc)
        # The monthly cap must remain in the same cycle for this operation.
        execution_seconds=min(24*3600,math.floor(next_month.timestamp()-started_unix)-SHUTDOWN_TAIL_SECONDS)
        if execution_seconds<=0:
            raise ValueError('provider billing cycle ends before safe execution window')
        return {'schema':SCHEMA+'.activation','policy_sha256':policy_sha(value),
            'started_unix':started_unix,'deadline_unix':started_unix+execution_seconds,
            'execution_seconds':execution_seconds,'shutdown_tail_seconds':SHUTDOWN_TAIL_SECONDS,
            'allocated_fleet_usd':allocated,'reserve_usd':value['reserve_usd'],**pricing,
            'enforcement':'provider-budget','provider_observation':observation,
            'provider_compute_cap_usd':allocated,'provider_invoice_cap':False,
            'external_app_stop_required':True,**prior}
    if provider_observation is not None:
        raise ValueError('provider observation requires provider-budget mode')
    charged_seconds = math.floor(allocated / pricing['fleet_usd_per_second'])
    execution_seconds = charged_seconds - SHUTDOWN_TAIL_SECONDS
    if execution_seconds <= 0:
        raise ValueError('budget is insufficient after shutdown reservation')
    return {'schema': SCHEMA + '.activation',
        'policy_sha256': policy_sha(value,authorized_total_usd=authorized_total_usd),
        'started_unix': started_unix, 'deadline_unix': started_unix + execution_seconds,
        'execution_seconds': execution_seconds, 'shutdown_tail_seconds': SHUTDOWN_TAIL_SECONDS,
        'allocated_fleet_usd': allocated, 'reserve_usd': value['reserve_usd'], **pricing,
        'maximum_modeled_fleet_usd': charged_seconds * pricing['fleet_usd_per_second'],
        'provider_invoice_cap': False, 'external_app_stop_required': True,**prior,**raised}


def validate_activation(value, receipt):
    if not isinstance(receipt, dict) or receipt != activate(value, receipt.get('started_unix'),
            receipt.get('provider_observation'),authorized_total_usd=receipt.get('authorized_total_usd')):
        raise ValueError('immutable budget activation receipt required')
    return receipt


def remaining(value, receipt, now=None):
    receipt = validate_activation(value, receipt)
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError('finite current timestamp required')
    return max(0.0, receipt['deadline_unix'] - now)


def clamp_expiry(value, receipt, expires, now=None):
    if remaining(value, receipt, now) <= 0:
        raise TimeoutError('frozen fleet budget deadline reached')
    if type(expires) not in (int, float) or not math.isfinite(expires):
        raise ValueError('finite original game expiry required')
    return min(expires, receipt['deadline_unix'])


@contextmanager
def watchdog(deadline_unix, *, stop_process=None):
    """Terminate this single-use worker if any blocking operation outlives it.

    Modal may retry the input after exit. Each retry independently checks the
    same immutable deadline before acquiring runtime files or starting a game.
    """
    seconds = deadline_unix - time.time()
    if not math.isfinite(seconds) or seconds <= 0:
        raise TimeoutError('frozen fleet budget deadline reached')
    stop_process = os._exit if stop_process is None else stop_process
    timer = threading.Timer(seconds, stop_process, args=(124,))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
