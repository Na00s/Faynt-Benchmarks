"""Deployable E011 full-queue dispatcher and unchanged durable game workers."""
from __future__ import annotations

import prepared_runtime
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import time

import modal_panel_cloud_queue as cloud
import modal_panel_budget as budget_guard

APP_NAME = 'frisson-slippi-full-panel-v1'
STATE_NAME = 'frisson-slippi-full-panel-state-v1'
META = Path('/cloud-metadata')
BLOBS = Path('/cloud-input/blobs')
ARCHIVE = Path('/opt/cloud-metadata.tar.gz')
DURABLE_MODULES = frozenset({'modal_panel_durable_game', 'faynt_d0_durable_game'})

WORKER_PATHS={
    'modal_panel_durable_game':{'root':'/opt/runtime-root','plan':'/opt/full-game-plan.json',
        'volume':'frisson-benchmark-game-results-v1','index':'frisson-benchmark-game-index-v1'},
    'faynt_d0_durable_game':{'root':'/opt/d0-root','plan':'/opt/faynt-d0-game-plan.json',
        'volume':'frisson-faynt-d0-game-results-v1','index':'frisson-faynt-d0-game-index-v1'},
}
WORKER_TIME_LIMITS={
    'modal_panel_durable_game':{'session_seconds':16290,'function_seconds':15990},
    'faynt_d0_durable_game':{'session_seconds':9090,'function_seconds':8790},
}
# Native D0 supervision reserves 30s, then owner bootstrap reserves 2100s.
# Dispatch adds 300s for scheduling/materialization. Active games retain their
# frozen expiry; saved-result inspection remains available until budget cutoff.
D0_NATIVE_ADMISSION_SECONDS = 2130
D0_DISPATCH_MARGIN_SECONDS = 300
# A recorded call still unresolved this long after its own frozen deadline can
# never answer: its container is gone with the app that spawned it.
DEAD_CALL_GRACE_SECONDS = 600
REDISPATCH_REFUSED = 'redispatch-refused-native-start-exists'


def dead_call_record(record, *, now=None):
    """Only a deadline more than the grace period in the past proves a call dead."""
    expires=(record or {}).get('expires_unix')
    if type(expires) not in (int,float) or not math.isfinite(expires):return False
    now=time.time() if now is None else now
    return now-expires>DEAD_CALL_GRACE_SECONDS


def redispatch_scope(call_id):
    """The scope a dead owner's replacement runs under, named by the dead call."""
    if not isinstance(call_id,str) or not re.fullmatch(r'fc-[A-Za-z0-9_-]{1,128}',call_id):
        raise ValueError('redispatch scope requires the dead call identity')
    return '/redispatch-'+call_id


def attempt_plan_sha(row, retry):
    return retry['plan_sha256'] if retry else cloud.execution_plan_sha(row)


def native_start_bound(index, plan_sha):
    """One call per plan: any call or physical slot binding is a consumed start."""
    return (index.get(plan_sha+'/call') is not None
        or any(index.get(plan_sha+f'/physical/{n}') is not None for n in range(3)))


def durable_index():
    import modal
    module=deployment_worker({'durable_module':os.environ.get('PANEL_DURABLE_MODULE',
                                                              'modal_panel_durable_game')})
    return modal.Dict.from_name(WORKER_PATHS[module]['index'],environment_name=runtime_environment())


def validate_redispatch_scope(state, key, label, suffix, dead_call_id):
    """A scoped run must name a dispatcher-superseded, archived owner of this attempt."""
    marker=state.get(key+'redispatch/'+label+suffix)
    if not isinstance(marker,dict) or marker.get('call_id')!=dead_call_id:
        raise ValueError('redispatch scope requires the dispatcher marker for this dead call')
    superseded=state.get(key+'owner/'+label+suffix)
    if not isinstance(superseded,dict) or superseded.get('call_id')!=dead_call_id:
        raise ValueError('redispatch scope requires the superseded owner record')
    if state.get(key+'owner-history/'+label+suffix+'/'+dead_call_id)!=superseded:
        raise ValueError('superseded owner must be archived before a scoped redispatch')
    return marker


def refuse_started_redispatch(index, label, attempt, dead_call_id, plan_sha):
    """A dead owner that already consumed a native start keeps its plan binding."""
    if not native_start_bound(index,plan_sha):return None
    return {'status':REDISPATCH_REFUSED,'label':label,'attempt_number':attempt,
        'dead_call_id':dead_call_id,'plan_sha256':plan_sha,'call_binding':index.get(plan_sha+'/call'),
        'physical_history':[h for h in (index.get(plan_sha+f'/physical/{n}') for n in range(3)) if h is not None],
        'score_admitted':False}


def late_admission(label, expires, *, dispatch=False, now=None, durable_module=None):
    module=durable_module or os.environ.get('PANEL_DURABLE_MODULE','modal_panel_durable_game')
    if module!='faynt_d0_durable_game':return None
    now=time.time() if now is None else now
    if any(type(value) not in (int,float) or not math.isfinite(value) for value in (now,expires)):
        raise ValueError('finite native admission deadline required')
    minimum=D0_NATIVE_ADMISSION_SECONDS+(D0_DISPATCH_MARGIN_SECONDS if dispatch else 0)
    if expires-now>minimum:return None
    return {'status':'budget-admission-deferred','label':label,'expires_unix':expires,
        'observed_unix':now,'minimum_remaining_seconds':minimum,'score_admitted':False,
        'reason':'insufficient remaining time for frozen native bootstrap admission'}


def retry_late_admission(label, expires, session_seconds, resume_tranche, *, existing=None, now=None):
    """Gate new publication while retaining an existing immutable authorization."""
    if existing is not None:return None
    now=time.time() if now is None else now
    if now+600>=expires and resume_tranche is not None:
        expires=min(now+session_seconds,resume_tranche['activation']['deadline_unix'])
    return late_admission(label,expires,dispatch=True,now=now)


def admitted_native_result(d, volume, index, plan_sha, bound, expires, image_id, label, acquire_owner):
    """Late platform retries can recover a saved result without another start."""
    deferred=late_admission(label,expires)
    if deferred is None:
        if not acquire_owner():return None,{'status':'duplicate-dispatch-suppressed'}
        return d.remote_durable(plan_sha,expires,image_id),None
    manifest=d.recover_saved(volume,index,plan_sha,bound)
    return manifest,(deferred if manifest is None else None)


def deployment_worker(deployment):
    module = deployment.get('durable_module', 'modal_panel_durable_game')
    if not isinstance(module, str) or module not in DURABLE_MODULES:
        raise ValueError('explicit reviewed durable worker module required')
    return module


def worker_paths(deployment):
    return dict(WORKER_PATHS[deployment_worker(deployment)])


def worker_time_limits(deployment):
    return dict(WORKER_TIME_LIMITS[deployment_worker(deployment)])


def frozen_game_session_seconds(row):
    """Dispatch uses the exact session bound already frozen into this plan."""
    body=(META/row['plan_path']).read_bytes()
    if cloud.digest(body)!=row['plan_sha256']:
        raise ValueError('frozen per-game plan changed before dispatch')
    limits=json.loads(body)['limits']
    reviewed=worker_time_limits({'durable_module':os.environ.get('PANEL_DURABLE_MODULE',
                                                                 'modal_panel_durable_game')})
    if any(type(limits.get(name)) is not int or limits[name]!=seconds for name,seconds in reviewed.items()):
        raise ValueError('native worker session/function limits differ from frozen plan')
    return limits['session_seconds']


def deployment_environment(deployment):
    name=deployment.get('environment_name','main')
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):
        raise ValueError('bounded Modal environment name required')
    return name


def runtime_environment():
    return deployment_environment({'environment_name':os.environ.get('PANEL_ENVIRONMENT','main')})


def deployment_names(deployment):
    """Resolve an explicit deployment namespace, preserving the legacy defaults."""
    present = {'app_name', 'state_name'} & deployment.keys()
    if present and present != {'app_name', 'state_name'}:
        raise ValueError('app_name and state_name must be configured together')
    names = {'app_name': deployment.get('app_name', APP_NAME),
             'state_name': deployment.get('state_name', STATE_NAME)}
    if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value)
           for value in names.values()):
        raise ValueError('bounded Modal app and state names required')
    fresh = deployment.get('fresh_run', False)
    if type(fresh) is not bool:
        raise ValueError('fresh_run must be boolean')
    if fresh and (names['app_name'] == APP_NAME or names['state_name'] == STATE_NAME):
        raise ValueError('a fresh run requires separate app and state names')
    return names


def runtime_names():
    values = {key: os.environ[env] for key, env in (
        ('app_name', 'PANEL_APP_NAME'), ('state_name', 'PANEL_STATE_NAME')) if env in os.environ}
    return deployment_names(values)


def validate_fresh_deployment(deployment, directory):
    """A repeat starts with an empty score ledger and no adopted submissions."""
    names = deployment_names(deployment)
    deployment_worker(deployment)
    deployment_environment(deployment)
    if not deployment.get('fresh_run', False):
        return names
    body = (Path(directory) / 'metadata/snapshot.json').read_bytes()
    if cloud.digest(body) != deployment['snapshot_sha256']:
        raise ValueError('fresh deployment snapshot binding differs')
    queue = cloud.validate_snapshot(json.loads(body))
    for row in queue['games']:
        initial = row.get('initial_row', {})
        if (row['initial_status'] != 'pending' or initial.get('status') != 'pending'
                or initial.get('attempts') != []
                or any(initial.get(key) is not None for key in ('result', 'cloud_provenance', 'cloud_exclusion'))
                or row.get('existing_claim') or row.get('external_submission')):
            raise ValueError('fresh run cannot import old results, attempts, claims or submissions')
    return names


def validate_activation_receipt(deployment, receipt):
    names = deployment_names(deployment)
    if (receipt.get('snapshot_sha256') != deployment['snapshot_sha256']
            or deployment_names(receipt) != names
            or deployment_worker(receipt) != deployment_worker(deployment)
            or deployment_environment(receipt) != deployment_environment(deployment)
            or receipt.get('budget') != deployment.get('budget')
            or not isinstance(receipt.get('image_id'), str) or not receipt['image_id']):
        raise ValueError('activation requires this exact deployed queue and namespace')
    return names


def budget_policy(deployment, *, authorized_total_usd=None):
    value = deployment.get('budget')
    policy=(budget_guard.validate_policy(value,authorized_total_usd=authorized_total_usd)
        if value is not None else None)
    if policy is not None and policy.get('enforcement')=='provider-budget':
        if policy['environment_name']!=deployment_environment(deployment):
            raise ValueError('provider budget must bind this exact deployment environment')
    return policy


def remote_budget(state, key):
    raw = os.environ.get('PANEL_BUDGET_POLICY')
    if raw is None:
        return None, None
    tranche=active_resume_tranche(state,key)
    if tranche and (tranche['authorization']['app_name']!=runtime_names()['app_name']
            or tranche['authorization']['environment_name']!=runtime_environment()):
        raise ValueError('active tranche must bind this running deployment')
    # Only the validated hash-bound chain can name a cap above the reviewed rail.
    authorized=budget_guard.authorized_total_usd(tranche['authorization']) if tranche else None
    policy = budget_guard.validate_policy(json.loads(raw),authorized_total_usd=authorized)
    receipt = budget_guard.validate_activation(policy,
        tranche['activation'] if tranche else state.get(key + 'budget'))
    if receipt.get('authorized_total_usd')!=authorized:
        raise ValueError('budget receipt must record exactly the selected authorized cap')
    return policy, receipt


def resume_tranche_chain(state,key):
    """Follow immutable successor selections and validate the complete ancestry."""
    pointer=state.get(key+'budget-current')
    chain=[]; seen=set()
    while pointer is not None:
        if (not isinstance(pointer,dict) or set(pointer)!={'authorization_sha256','tranche_sha256'}
                or any(not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v)
                    for v in pointer.values())):
            raise ValueError('exact current budget tranche pointer required')
        if (pointer['authorization_sha256'] in seen
                or len(chain)>=budget_guard.MAX_RESUME_TRANCHES):
            raise ValueError('budget successor chain must be bounded and acyclic')
        tranche=budget_guard.validate_resume_tranche(
            state.get(key+'budget-tranches/'+pointer['authorization_sha256']),
            chain[-1]['activation'] if chain else state.get(key+'budget'))
        authorization=tranche['authorization']
        if (budget_guard.digest(tranche)!=pointer['tranche_sha256']
                or tranche['authorization_sha256']!=pointer['authorization_sha256']
                or authorization['queue_sha256']+'/'!=key):
            raise ValueError('active budget tranche pointer or queue binding differs')
        if chain:
            budget_guard.validate_resume_successor(tranche,chain[-1])
        chain.append(tranche); seen.add(pointer['authorization_sha256'])
        pointer=state.get(key+'budget-successor/'+budget_guard.digest(tranche))
    return tuple(chain)


def active_resume_tranche(state,key):
    """Resolve the latest selected allocation without rewriting earlier receipts."""
    chain=resume_tranche_chain(state,key)
    return chain[-1] if chain else None


def observe_stopped_app(authorization):
    from modal._utils.async_utils import synchronizer
    async def bounded():
        return await asyncio.wait_for(budget_guard.observe_stopped_app(authorization),timeout=30)
    return synchronizer.create_blocking(bounded)()


def activate_resume_budget(state,key,deployment,authorization,*,now=None,stop_observation=None):
    """Append an allocation within the existing cap after fresh stop preflight."""
    authorization=budget_guard.validate_resume_authorization(authorization)
    authorized=budget_guard.authorized_total_usd(authorization)
    current_time=time.time() if now is None else now
    policy=budget_policy(deployment,authorized_total_usd=authorized)
    if (authorization['budget_policy']!=policy or authorization['queue_sha256']+'/'!=key
            or authorization['queue_sha256']!=deployment['snapshot_sha256']
            or authorization['app_name']!=deployment_names(deployment)['app_name']
            or authorization['environment_name']!=deployment_environment(deployment)):
        raise ValueError('resume authorization must bind this exact deployment and budget')
    auth_sha=budget_guard.digest(authorization)
    chain=resume_tranche_chain(state,key)
    existing=chain[-1] if chain else None
    if any(tranche['authorization_sha256']==auth_sha for tranche in chain):
        if existing['authorization_sha256']!=auth_sha:
            raise TimeoutError('historical remaining-budget tranche expired; reactivation cannot extend it')
        if budget_guard.remaining(policy,existing['activation'],current_time)<=0:
            raise TimeoutError('remaining-budget tranche expired; reactivation cannot extend it')
        return existing['activation']
    if len(chain)>=budget_guard.MAX_RESUME_TRANCHES:
        raise ValueError('budget successor chain limit reached')
    if existing and budget_guard.remaining(existing['authorization']['budget_policy'],
            existing['activation'],current_time)>0:
        raise ValueError('a live resume tranche is already selected')
    previous_budget=existing['activation'] if existing else state.get(key+'budget')
    selection_key=(key+'budget-successor/'+budget_guard.digest(existing)
        if existing else key+'budget-current')
    control=state.get(key+'control',{})
    if control.get('enabled') is not False:
        raise ValueError('old queue control must be explicitly disabled before resume')
    observation=observe_stopped_app(authorization) if stop_observation is None else stop_observation
    if now is None:
        current_time=time.time()
    budget_guard.validate_stop_observation(authorization,observation,now=current_time)
    tranche={'schema':budget_guard.RESUME_SCHEMA+'.tranche','authorization':authorization,
        'authorization_sha256':auth_sha,'previous_budget_sha256':authorization['previous_budget_sha256'],
        'stop_observation':observation,
        'activation':budget_guard.activate(policy,current_time,authorized_total_usd=authorized)}
    budget_guard.validate_resume_tranche(tranche,previous_budget)
    if existing:
        budget_guard.validate_resume_successor(tranche,existing)
    if state.get(key+'control',{}).get('enabled') is not False:
        raise ValueError('queue control changed during resume preflight')
    tranche_key=key+'budget-tranches/'+auth_sha
    state.put(tranche_key,tranche,skip_if_exists=True)
    retained=budget_guard.validate_resume_tranche(state.get(tranche_key),previous_budget)
    if retained['authorization_sha256']!=auth_sha:
        raise ValueError('retained budget tranche authorization differs')
    if existing:
        budget_guard.validate_resume_successor(retained,existing)
    if budget_guard.remaining(policy,retained['activation'],current_time)<=0:
        raise TimeoutError('retained remaining-budget tranche expired')
    pointer={'authorization_sha256':auth_sha,'tranche_sha256':budget_guard.digest(retained)}
    state.put(selection_key,pointer,skip_if_exists=True)
    if state.get(selection_key)!=pointer:
        raise ValueError('another resume tranche was selected concurrently')
    return retained['activation']


def observe_provider_budget(policy, *, remote=False):
    from modal._utils.async_utils import synchronizer
    async def bounded():
        return await asyncio.wait_for(budget_guard.observe_provider(policy,remote=remote),timeout=20)
    # Modal's sync handles and cached _Client belong to its shared background
    # event loop. Bridge onto that loop after configure/hydrate has initialized
    # the client so its RPC task context remains valid.
    return synchronizer.create_blocking(bounded)()


def provider_budget_check(state,key,control,policy,budget):
    """Fail closed on missing provider evidence or a meter that rolls backward."""
    if policy is None or policy.get('enforcement')!='provider-budget':
        return None
    observation=observe_provider_budget(policy,remote=True)
    prior=state.get(key+'provider-budget-check',{})
    high=max(prior.get('high_water_usage_usd',0),
        budget['provider_observation']['current_cycle_usage_usd'])
    usage=observation['current_cycle_usage_usd']
    if usage+1e-8<high:
        raise ValueError('provider compute meter decreased; possible billing-cycle reset')
    checked={'observation':observation,'high_water_usage_usd':max(high,usage),
        'checked_unix':time.time()}
    state.put(key+'provider-budget-check',checked)
    current=state.get(key+'control',control)
    state.put(key+'control',{**current,'provider_budget_high_water_usd':checked['high_water_usage_usd'],
        'provider_budget_verified_unix':checked['checked_unix']})
    return checked


def activate_budget(state, key, deployment, *, now=None, provider_observation=None, started_unix=None):
    """Atomic first activation survives canary expansion and repeated activation."""
    if deployment.get('budget') is None:
        return None
    tranche=active_resume_tranche(state,key)
    authorized=budget_guard.authorized_total_usd(tranche['authorization']) if tranche else None
    policy = budget_policy(deployment,authorized_total_usd=authorized)
    if tranche is not None:
        retained=budget_guard.validate_activation(policy,tranche['activation'])
        if started_unix is not None and started_unix!=retained['started_unix']:
            raise ValueError('resume tranche activation clock cannot be reset')
        if budget_guard.remaining(policy,retained,now)<=0:
            raise TimeoutError('remaining-budget tranche expired; reactivation cannot extend it')
        return retained
    current_time=time.time() if now is None else now
    start=current_time if started_unix is None else started_unix
    if not isinstance(start,(int,float)) or not current_time-86400<=start<=current_time:
        raise ValueError('budget migration start must be a past timestamp within 24 hours')
    if policy.get('enforcement')=='provider-budget' and started_unix is not None:
        raise ValueError('explicit historical start is restricted to fleet-time migration')
    if policy.get('enforcement')=='provider-budget':
        provider_observation=(observe_provider_budget(policy)
            if provider_observation is None else provider_observation)
        budget_guard.validate_provider_observation(policy,provider_observation,now=now)
    receipt = budget_guard.activate(policy,start,provider_observation)
    state.put(key + 'budget', receipt, skip_if_exists=True)
    retained = budget_guard.validate_activation(policy, state.get(key + 'budget'))
    if 'authorized_total_usd' in retained:
        raise ValueError('a cap increase requires a selected resume tranche')
    if started_unix is not None and retained['started_unix']!=started_unix:
        raise ValueError('explicit migration start differs from retained activation')
    if budget_guard.remaining(policy, retained, now) <= 0:
        raise TimeoutError('frozen fleet budget expired; reactivation cannot extend it')
    return retained


def disable_dispatch(state, key, control, report):
    """Stop future launches while retaining evidence for external app cleanup."""
    stopped = {**control, 'enabled': False, 'terminal_status': report['status'],
               'disabled_unix': report['time']}
    state.put(key + 'control', stopped)
    report.update(dispatch_enabled=False, app_cleanup_required=True,
                  app_stopped=False,
                  cleanup_rule='Stop this deployment externally after verifying no active game inputs remain.')
    state.put(key + 'health', report)
    return report


def activation_control(state,key,queue,*,image_id,allowlist=None,
                       auto_expand_after_canary=False,strict_phase_order=False,reinspect_held=False):
    """Freeze queue policy and the exact hold set before any dispatch can run."""
    control={'enabled':True,'image_id':image_id,'allowlist':allowlist,
        'auto_expand_after_canary':auto_expand_after_canary,'strict_phase_order':strict_phase_order}
    if reinspect_held:
        tranche=active_resume_tranche(state,key)
        if tranche is None or allowlist is not None:
            raise ValueError('held reinspection requires an explicit resume tranche and the full queue')
        entries=dict(state.items())
        records={name[len(key+'result/'):]:value for name,value in entries.items()
            if name.startswith(key+'result/')}
        definitions={row['label']:row for row in queue['games']}
        pending={label for label,row in definitions.items() if cloud.status(row,records) not in cloud.TERMINAL}
        # A previous allocation can end after a read-only recovery was deferred,
        # before that recovery produced a hold. Give both kinds of saved work a
        # new bounded inspection namespace while retaining the old records.
        retained_labels={name[len(key+'hold/'):] for name in entries if name.startswith(key+'hold/')}
        retained_labels.update(name[len(key+'recovery/'):].split('/')[0]
            for name in entries if name.startswith(key+'recovery/'))
        retained_labels.update(row.get('label') for row in entries.get(key+'health',{}).get('admission_deferred',[]))
        labels=sorted(pending & retained_labels)
        if labels:
            control['held_reinspection']={'tranche_sha256':budget_guard.digest(tranche),'labels':labels}
            cloud.held_reinspection_labels(control,tranche,definitions,records)
    return control


def validate_canary_allowlist(queue, labels, *, require_both_models=False):
    definitions={row['label']:row for row in queue['games']}
    if (not isinstance(labels,list) or any(not isinstance(label,str) for label in labels)
            or len(set(labels))!=len(labels) or not set(labels)<=definitions.keys()):
        raise ValueError('unique canary allowlist labels must belong to this frozen queue')
    if require_both_models:
        games={game['label']:game for game in queue['manifest']['games']}
        if not 2<=len(labels)<=64 or {games[label]['profile'] for label in labels}!={'75m','10m'}:
            raise ValueError('automatic expansion requires canaries for both 75m and 10m')
        if any(definitions[label]['initial_status']!='pending' for label in labels):
            raise ValueError('automatic expansion requires fresh pending canary definitions')
    return labels


def validated_canary_receipt(row, receipt, queue_sha):
    """Inspect the committed receipt emitted only after full native acceptance."""
    if (receipt.get('schema')!=cloud.SCHEMA+'.acceptance' or receipt.get('status')!='complete'
            or receipt.get('label')!=row['label'] or receipt.get('queue_sha256')!=queue_sha
            or receipt.get('queue_plan_sha256')!=row['plan_sha256']
            or receipt.get('plan_sha256')!=cloud.receipt_plan_sha(row,receipt)):
        raise ValueError('canary acceptance must bind the exact frozen game and queue')
    manifest=receipt.get('manifest',{})
    native=receipt.get('native_acceptance',{})
    if (not isinstance(manifest,dict) or manifest.get('status')!='full-policy-durable-game-passed'
            or manifest.get('plan_sha256')!=receipt['plan_sha256']
            or not isinstance(native,dict) or type(native.get('win')) is not bool
            or native.get('winner') not in {'frisson-ai','slippi-ai'}
            or native['win']!=(native['winner']=='frisson-ai')
            or type(native.get('game_frames')) is not int or native['game_frames']<=0
            or not native.get('end_evidence')):
        raise ValueError('canary requires successful durable native game acceptance')
    files=manifest.get('files',[])
    if not isinstance(files,list) or any(not isinstance(item,dict) for item in files):
        raise ValueError('accepted canary requires a durable artifact manifest')
    for kind,suffix in (('summary','summary.json'),('replay','.slp')):
        digest=native.get(kind+'_sha256')
        if (not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest)
                or not any(item.get('sha256')==digest and item.get('name','').endswith(suffix)
                    for item in files)):
            raise ValueError('accepted canary must retain bound summary and SLP artifacts')
    return {'label':row['label'],'plan_sha256':receipt['plan_sha256'],
        'acceptance_receipt_sha256':cloud.digest(cloud.encoded(receipt)),
        'summary_sha256':native['summary_sha256'],'replay_sha256':native['replay_sha256']}


def expand_accepted_canaries(state,key,control,queue,records):
    if not control.get('auto_expand_after_canary',False):
        return control
    if control['auto_expand_after_canary'] is not True:
        raise ValueError('automatic canary expansion requires explicit boolean opt-in')
    allowed=validate_canary_allowlist(queue,control.get('allowlist'),require_both_models=True)
    if any(records.get(label,{}).get('status')!='complete' for label in allowed):
        return control
    definitions={row['label']:row for row in queue['games']}
    canaries=[validated_canary_receipt(definitions[label],records[label],key.removesuffix('/'))
        for label in allowed]
    binding={'schema':cloud.SCHEMA+'.canary-expansion','queue_sha256':key.removesuffix('/'),
        'image_id':control['image_id'],'canaries':canaries}
    previous=state.get(key+'canary-expansion')
    if previous is not None and {name:previous.get(name) for name in binding}!=binding:
        raise ValueError('retained canary expansion binding changed')
    expansion=previous or {**binding,'expanded_unix':time.time()}
    current=state.get(key+'control',control)
    if (not current.get('enabled') or current.get('allowlist')!=allowed
            or current.get('auto_expand_after_canary') is not True
            or current.get('image_id')!=control['image_id']):
        return current
    state.put(key+'canary-expansion',expansion,skip_if_exists=True)
    if state.get(key+'canary-expansion')!=expansion:
        raise ValueError('canary expansion was already bound differently')
    expanded={**current,'allowlist':None,'auto_expand_after_canary':False,'canary_expansion':expansion}
    state.put(key+'control',expanded)
    return expanded


def snapshot():
    if not META.exists():
        with ARCHIVE.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=os.environ['PANEL_ARCHIVE_SHA']:
                raise ValueError('frozen metadata archive changed')
        META.mkdir()
        with tarfile.open(ARCHIVE,'r:gz') as archive:
            members=archive.getmembers()
            if len(members)>10000 or sum(m.size for m in members)>512*1024**2:
                raise ValueError('metadata archive exceeds bounds')
            seen=set()
            for member in members:
                path=Path(member.name)
                if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name in seen:
                    raise ValueError('unsafe metadata archive member')
                seen.add(member.name); target=META/path;target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as src, target.open('xb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
    body=(META/'snapshot.json').read_bytes()
    if cloud.digest(body)!=os.environ['PANEL_SNAPSHOT_SHA']:
        raise ValueError('frozen cloud queue snapshot changed')
    return cloud.validate_snapshot(json.loads(body))


def handles():
    import modal
    return modal.Dict.from_name(runtime_names()['state_name'],environment_name=runtime_environment())


def prefix(): return os.environ['PANEL_SNAPSHOT_SHA']+'/'


def validate_worker_retry_budget(state, key, retry, active_tranche, *, recover_only=False):
    """Historical retry receipts may be inspected without extending execution."""
    if recover_only and retry.get('budget_resume'):
        for tranche in resume_tranche_chain(state, key):
            if retry['budget_resume'].get('tranche_sha256') == budget_guard.digest(tranche):
                cloud.validate_retry_budget_binding(retry, tranche)
                return
    cloud.validate_retry_budget_binding(retry, active_tranche)


def renew_unstarted_retry(state, key, queue, row, metadata_body, retry, attempt, index,
                          session_seconds, resume_tranche, *, now=None, owner_key=None):
    """Read-only recovery re-times an expired attempt that never took ownership.

    The dispatcher binds each unresolved held label to one bounded reinspection
    under the active tranche. Inside that scope the worker archives the expired
    record by digest, proves the durable index still holds no call binding and
    no physical slot for the attempt plan, and replaces retry/<label>/a<N> only
    if it is unchanged since it was read. retry-latest, the attempt number and
    every started attempt stay untouched; no durable index or volume write occurs.
    """
    label=row['label'];suffix=f'/a{attempt}';retry_key=key+'retry/'+label+suffix
    now=time.time() if now is None else now
    definitions={r['label']:r for r in queue['games']}
    records={label:state.get(key+'result/'+label,{})}
    reinspection=cloud.held_reinspection_labels(state.get(key+'control',{}),resume_tranche,definitions,records)
    if label not in reinspection:
        raise ValueError('expired retry renewal requires reinspection binding under the active tranche')
    if state.get(key+'owner/'+label+suffix if owner_key is None else owner_key) is not None:
        raise ValueError('expired retry attempt already has a logical owner')
    expired=state.get(retry_key)
    if expired!=retry:
        raise ValueError('retry authorization changed during recovery')
    plan_sha=expired['plan_sha256']
    def started():
        return (index.get(plan_sha+'/call') is not None
            or any(index.get(plan_sha+f'/physical/{n}') is not None for n in range(3)))
    renewed=cloud.renew_expired_retry(row,metadata_body,expired,now=now,session_seconds=session_seconds,
        resume_tranche=resume_tranche,index_has_start=started())
    deferred=late_admission(label,renewed['expires_unix'],dispatch=True,now=now)
    if deferred is not None:return {**deferred,'expired_retry_retained':True}
    expired_sha=renewed['renews_authorization_sha256']
    history_key=key+'retry-history/'+label+suffix+'/'+expired_sha
    state.put(history_key,expired,skip_if_exists=True)
    if state.get(history_key)!=expired:
        raise ValueError('retained expired retry record differs')
    # Compare immediately before the replacement: the published record and the
    # durable index must both be unchanged since they were read above.
    if state.get(retry_key)!=expired or started():
        raise ValueError('retry authorization or durable index changed before renewal')
    state.put(retry_key,renewed)
    if state.get(retry_key)!=renewed:
        raise ValueError('renewed retry authorization differs')
    return {'status':'retry-renewed','label':label,'attempt_number':renewed['attempt_number'],
        'reason':renewed['reason'],'renews_authorization_sha256':expired_sha,
        'renewal_count':renewed['renewal_count'],'expires_unix':renewed['expires_unix'],
        'score_admitted':False}


def run_cloud_game(label, queue_sha, expires, image_id, *, recover_only=False, attempt=1, redispatch=None):
    """Every physical input observes one queue-wide absolute spend deadline."""
    if os.environ.get('PANEL_BUDGET_POLICY') is None:
        return _run_cloud_game(label,queue_sha,expires,image_id,recover_only=recover_only,attempt=attempt,
                               redispatch=redispatch)
    policy, receipt = remote_budget(handles(), prefix())
    expires = budget_guard.clamp_expiry(policy, receipt, expires)
    with budget_guard.watchdog(receipt['deadline_unix']):
        return _run_cloud_game(label,queue_sha,expires,image_id,recover_only=recover_only,attempt=attempt,
                               redispatch=redispatch)


def _run_cloud_game(label, queue_sha, expires, image_id, *, recover_only=False, attempt=1, redispatch=None):
    """Single-use container, with atomic logical-call ownership before gameplay."""
    import modal
    queue=snapshot(); key=prefix()
    if queue_sha!=os.environ['PANEL_SNAPSHOT_SHA'] or time.time()>queue['queue_deadline_unix']:
        raise ValueError('active frozen queue and bounded deadline required')
    row=next(r for r in queue['games'] if r['label']==label)
    if row['initial_status']!='pending': return {'status':'already-terminal'}
    state=handles(); record_key=key+'result/'+label
    if state.get(record_key,{}).get('status') in cloud.TERMINAL:
        return {'status':'already-terminal'}
    call_id=modal.current_function_call_id()
    suffix='' if attempt==1 else f'/a{attempt}'
    retry=state.get(key+'retry/'+label+suffix) if attempt>1 else None
    if attempt>1 and (retry is None or retry['attempt_number']!=attempt or expires!=retry['expires_unix']):
        raise ValueError('exact bounded retry authorization required')
    resume_tranche=active_resume_tranche(state,key)
    if retry:
        validate_worker_retry_budget(state,key,retry,resume_tranche,recover_only=recover_only)
    external=row.get('external_submission') if attempt==1 else None
    durable_module=deployment_worker({'durable_module':os.environ.get('PANEL_DURABLE_MODULE',
                                                                    'modal_panel_durable_game')})
    owner_key=key+'owner/'+label+suffix
    if redispatch is not None:
        # A dead owner never releases its key. Its replacement binds a scoped
        # owner key, after proving in the durable index that the attempt plan
        # holds no call binding: the dead call consumed no physical start.
        if external:raise ValueError('adopted submissions are never redispatched')
        owner_key=key+'owner/'+label+suffix+redispatch_scope(redispatch)
        validate_redispatch_scope(state,key,label,suffix,redispatch)
        if not recover_only:
            refusal=refuse_started_redispatch(durable_index(),label,attempt,redispatch,
                                              attempt_plan_sha(row,retry))
            if refusal is not None:return refusal
    if durable_module!='faynt_d0_durable_game' and not external and not recover_only:
        if not cloud.bind_call(state,owner_key,call_id,expires,image_id):
            return {'status':'duplicate-dispatch-suppressed'}
    plan_path=META/row['plan_path'];body=plan_path.read_bytes()
    if cloud.digest(body)!=row['plan_sha256']:raise ValueError('frozen per-game plan changed')
    metadata_body=body
    plan,body,plan_sha=(cloud.retry_execution_plan(row,body,retry) if retry else cloud.execution_plan(row,body))
    if retry and retry.get('renews_authorization_sha256') is not None:
        archived=state.get(key+'retry-history/'+label+suffix+'/'+retry['renews_authorization_sha256'])
        cloud.validate_retry_renewal(archived,retry)
    for upload in plan['uploads']:
        cloud.checked_materialize(upload,[BLOBS,META/'blobs'])
    # The exact original validators read local provenance paths too. Both copies
    # above are byte-identical regular files, preserving all nested input proofs.
    locations=worker_paths({'durable_module':durable_module})
    destination=Path(locations['plan'])
    with destination.open('xb') as stream:stream.write(body)
    sys.path.insert(0,'/opt/runtime-root/scripts')
    if locations['root']!='/opt/runtime-root':
        sys.path.insert(0,str(Path(locations['root'])/'scripts'))
    # Modal imports this module before the hash-bound source tree exists. Python
    # caches missing PYTHONPATH entries, so invalidate that negative cache once
    # the exact original modules have been materialized.
    import importlib
    importlib.invalidate_caches()
    if not Path(locations['root'],'scripts',durable_module+'.py').is_file():
        raise ValueError('materialized native worker entrypoint missing')
    d=importlib.import_module(durable_module)
    if (str(d.batch.base.PLAN) if durable_module=='modal_panel_durable_game' else str(d.PLAN))!=locations['plan']:
        raise ValueError('reviewed native worker plan path differs')
    if d.VOLUME!=locations['volume'] or d.INDEX!=locations['index']:
        raise ValueError('reviewed native worker storage identity differs')
    bound,verified=d.plan_binding(destination,plan_sha,remote=True)
    reviewed=worker_time_limits({'durable_module':durable_module})
    if (bound.SESSION_SECONDS!=reviewed['session_seconds']
            or bound.FUNCTION_SECONDS!=reviewed['function_seconds']):
        raise ValueError('bound worker session/function limits changed')
    volume=modal.Volume.from_name(d.VOLUME,environment_name=runtime_environment())
    index=modal.Dict.from_name(d.INDEX,environment_name=runtime_environment())
    if retry:
        volume.reload()
        for failure_index,failure in enumerate(retry['prior_failures']):
            old=failure['manifest'];old_sha=old['plan_sha256']
            old_root=(d.MOUNT/old['storage_path']).resolve()
            _,old_body,verified_old_sha=cloud.prior_execution_plan(row,metadata_body,retry,failure_index)
            old_path=Path(f'/opt/prior-game-plan-{failure_index}.json')
            with old_path.open('xb') as stream:stream.write(old_body)
            old_bound,_=d.plan_binding(old_path,verified_old_sha,remote=True)
            d.check_manifest(old,old_sha,old_root,old_bound)
            if not cloud.verified_failure_retry_reason(old,old_root):
                raise ValueError('retry requires independently reverified eligible infrastructure failure')
            binding=index.get(old_sha+'/call')
            cloud.validate_prior_call_binding(failure,binding,retry)
            actual=[index.get(old_sha+f'/physical/{n}') for n in range(3)]
            if [h for h in actual if h is not None]!=failure['physical_history']:
                raise ValueError('prior physical history changed')
    if external or recover_only:
        manifest=d.recover_saved(volume,index,plan_sha,bound)
        if manifest is None:
            if recover_only:
                if retry and time.time()>retry['expires_unix']:
                    return renew_unstarted_retry(state,key,queue,row,metadata_body,retry,attempt,index,
                        bound.SESSION_SECONDS,resume_tranche,owner_key=owner_key)
                return {'status':'awaiting-native-result'}
            try:manifest=modal.FunctionCall.from_id(external['call_id']).get(timeout=0)
            except TimeoutError:return {'status':'external-call-still-running'}
        # A scoped recovery reads the scoped owner; before that owner bound, the
        # only call that could have saved a result is the superseded owner.
        expected=(external['call_id'] if external else
            (state.get(owner_key) or state.get(key+'owner/'+label+suffix))['call_id'])
        if manifest['call_id']!=expected:
            raise ValueError('adopted call binding changed')
    else:
        manifest,deferred=admitted_native_result(d,volume,index,plan_sha,bound,expires,image_id,label,
            lambda:durable_module!='faynt_d0_durable_game' or
                cloud.bind_call(state,owner_key,call_id,expires,image_id))
        if deferred is not None:return deferred
        if manifest['call_id']!=state.get(owner_key,{}).get('call_id'):
            raise ValueError('native result logical owner binding changed')
    volume.reload()
    history=[index.get(plan_sha+f'/physical/{n}') for n in range(3)]
    history=[h for h in history if h is not None]
    if not 1<=len(history)<=3 or any(h['call_id']!=manifest['call_id'] for h in history):
        raise ValueError('physical attempt provenance differs')
    root=(d.MOUNT/manifest['storage_path']).resolve()
    d.check_manifest(manifest,plan_sha,root,bound)
    retry_reason=cloud.verified_failure_retry_reason(manifest,root)
    if manifest['status']!='full-policy-durable-game-passed' and retry_reason:
        binding=index.get(plan_sha+'/call')
        prior=(retry['prior_failures'] if retry else [])+[{'manifest':manifest,
            'physical_history':history,'call_binding':binding}]
        original_expiry=(retry['expires_unix'] if retry else index.get(plan_sha+'/call')['expires_unix'])
        number=1+sum(len(item['physical_history']) for item in prior)
        retry_key=key+'retry/'+label+f'/a{number}'
        existing_authorization=state.get(retry_key)
        deferred=retry_late_admission(label,original_expiry,bound.SESSION_SECONDS,
            resume_tranche,existing=existing_authorization)
        if deferred is not None:return {**deferred,'failed_attempt_retained':True}
        authorization=cloud.authorize_failure_retry(row,metadata_body,prior,
            reason=retry_reason,expires_unix=original_expiry,now=time.time(),
            session_seconds=bound.SESSION_SECONDS,resume_tranche=resume_tranche,
            accepted_record=state.get(record_key),existing_authorization=existing_authorization)
        # An already resumed live deadline keeps its original tranche binding.
        if retry and retry.get('budget_resume') and 'budget_resume' not in authorization:
            authorization['budget_resume']=retry['budget_resume']
        cloud.validate_retry_budget_binding(authorization,resume_tranche)
        number=authorization['attempt_number']
        state.put(retry_key,authorization,skip_if_exists=True)
        retained=state.get(retry_key)
        if retained!=authorization:
            # Another read-only recovery can publish first after an ambiguous
            # response. Reverify its exact proofs and retain its original expiry.
            authorization=cloud.authorize_failure_retry(row,metadata_body,prior,
                reason=retry_reason,expires_unix=original_expiry,now=time.time(),
                session_seconds=bound.SESSION_SECONDS,resume_tranche=resume_tranche,
                accepted_record=state.get(record_key),existing_authorization=retained)
            if retained!=authorization:raise ValueError('conflicting retry authorization')
        state.put(key+'retry-latest/'+label,number)
        return {'status':'retry-authorized','label':label,'attempt_number':number,
            'reason':retry_reason,'failed_attempt_retained':True,'score_admitted':False}
    accepted=cloud.native_acceptance(d,bound,verified,manifest)
    receipt={'schema':cloud.SCHEMA+'.acceptance','status':'complete','label':label,
        'queue_sha256':queue_sha,'plan_sha256':plan_sha,'queue_plan_sha256':row['plan_sha256'],
        'accepted_unix':time.time(),'native_acceptance':accepted,'manifest':manifest,
        'physical_history':history,'retry_authorization':retry,'existing_claim':row.get('existing_claim'),
        'adopted_submission':external,'worker_call_id':call_id}
    target=d.MOUNT/'cloud-ledger'/queue_sha/(label+'.json')
    if target.exists():
        old=json.loads(target.read_bytes())
        if old['manifest']!=manifest: raise ValueError('committed acceptance differs')
        receipt=old
    else:d.create_json(target,receipt)
    volume.commit()
    state.put(record_key,receipt,skip_if_exists=True)
    if state.get(record_key)!=receipt:raise ValueError('cloud acceptance already differs')
    return {'status':'complete','label':label,'native_acceptance':accepted}


def dispatch_cloud():
    """Periodic singleton: the entire original queue is present on every tick."""
    import modal
    queue=snapshot(); state=handles(); key=prefix()
    control=state.get(key+'control',{})
    if not control.get('enabled'):
        if control.get('terminal_status'):
            return state.get(key+'health',{'status':control['terminal_status'],'dispatch_enabled':False})
        return {'status':'deployed-awaiting-activation','total':len(queue['games'])}
    if time.time()>queue['queue_deadline_unix']:
        return disable_dispatch(state,key,control,
            {'status':'queue-deadline-reached','time':time.time(),'all_games_in_cloud':len(queue['games'])})
    policy, budget = remote_budget(state,key)
    if policy is not None and budget_guard.remaining(policy,budget)<=0:
        return disable_dispatch(state,key,control,{'status':'fleet-budget-deadline-reached',
            'time':time.time(),'all_games_in_cloud':len(queue['games']),'budget':budget})
    try:
        provider_check=provider_budget_check(state,key,control,policy,budget)
    except Exception as error:
        reason=(str(error) if isinstance(error,ValueError) else type(error).__name__)
        return disable_dispatch(state,key,control,{'status':'provider-budget-verification-failed',
            'time':time.time(),'budget':budget,'reason':reason[:500],
            'all_games_in_cloud':len(queue['games'])})
    control=state.get(key+'control',control)
    if not control.get('enabled'):
        return {'status':'dispatch-disabled-during-provider-check','dispatch_enabled':False}
    # max_containers=1 and one input per container serialize every dispatch tick.
    entries=dict(state.items()); records={k[len(key+'result/'):]:v for k,v in entries.items() if k.startswith(key+'result/')}
    try:
        control=expand_accepted_canaries(state,key,control,queue,records)
    except (KeyError,TypeError,ValueError) as error:
        return disable_dispatch(state,key,control,{'status':'canary-acceptance-invalid',
            'time':time.time(),'reason':str(error)[:500],'all_games_in_cloud':len(queue['games'])})
    if not control.get('enabled'):
        return {'status':'dispatch-disabled-during-canary-check','dispatch_enabled':False}
    definitions={r['label']:r for r in queue['games']}
    holds={k[len(key+'hold/'):]:v for k,v in entries.items() if k.startswith(key+'hold/')
        and k[len(key+'hold/'):] in definitions
        and cloud.status(definitions[k[len(key+'hold/'):]],records) not in cloud.TERMINAL}
    def redispatch_released(label,hold):
        # A recovery-exhausted hold on an attempt whose owner call is dead and
        # whose plan has no native binding is released for exactly one scoped
        # redispatch. The marker written by that redispatch names the released
        # hold record, so a hold raised later under the scope holds the row
        # again. Any lookup failure keeps the row held.
        try:
            if hold.get('recovery_exhausted') is not True:return False
            attempt=state.get(key+'retry-latest/'+label,1)
            if hold.get('attempt')!=attempt:return False
            suffix='' if attempt==1 else f'/a{attempt}'
            marker=state.get(key+'redispatch/'+label+suffix)
            if marker is not None:
                return marker.get('released_hold_sha256')==cloud.digest(hold)
            owner=state.get(key+'owner/'+label+suffix)
            if owner is None or not dead_call_record(owner):return False
            retry=state.get(key+'retry/'+label+suffix) if attempt>1 else None
            return not native_start_bound(durable_index(),attempt_plan_sha(definitions[label],retry))
        except Exception:
            return False
    holds={label:hold for label,hold in holds.items() if not redispatch_released(label,hold)}
    resume_tranche=active_resume_tranche(state,key)
    reinspection=cloud.held_reinspection_labels(control,resume_tranche,definitions,records)
    tranche_sha=cloud.budget_digest(resume_tranche) if resume_tranche else None
    reinspection_tokens={label:cloud.held_reinspection_token(control,label,tranche_sha,holds.get(label))
        for label in reinspection}
    eligibility_holds={label for label,hold in holds.items()
        if label not in reinspection or hold.get('reinspection_token',
            hold.get('reinspection_tranche_sha256'))==reinspection_tokens[label]}
    if control.get('allowlist') is not None:
        allowed=validate_canary_allowlist(queue,control['allowlist'])
        eligibility_holds.update(set(definitions)-set(allowed))
    candidates=cloud.eligible(queue,records,holds=eligibility_holds,
        strict_phase_order=control.get('strict_phase_order',False))
    fn=modal.Function.from_name(runtime_names()['app_name'],'run_cloud_game',environment_name=runtime_environment())
    submitted=[]; waiting=[]; errors=[]; deferred=[]; redispatched=[]
    def attempt_scope(label,attempt):
        # A renewed authorization keeps its attempt, retry and owner keys. Its
        # dispatch and recovery intent restart under the renewal identity, so
        # the expired deadline frozen in the earlier intent is never reused and
        # every earlier record stays retained.
        suffix='' if attempt==1 else f'/a{attempt}'
        retry=state.get(key+'retry/'+label+suffix) if attempt>1 else None
        renews=(retry or {}).get('renews_authorization_sha256')
        scope=suffix+('' if renews is None else '/renewal-'+renews)
        # A dead owner that never started natively is superseded once per
        # attempt. Dispatch, owner and recovery intent then live under a scope
        # named by the dead call, so the frozen worker binds a fresh owner key
        # while the superseded records stay retained.
        marker=state.get(key+'redispatch/'+label+suffix)
        if marker is None:return suffix,retry,scope,suffix,None
        redispatch=redispatch_scope(marker['call_id'])
        return suffix,retry,scope+redispatch,suffix+redispatch,marker
    def redispatch_kwargs(marker):
        return {} if marker is None else {'redispatch':marker['call_id']}
    def archive(path,record):
        state.put(path,record,skip_if_exists=True)
        if state.get(path)!=record:
            raise ValueError('archived record differs: '+path[len(key):])
    def retain_hold(row,error,attempt,**details):
        # Separate scheduling evidence from the score ledger. Any unresolved
        # failure may be deferred, regardless of its scientific classification.
        label=row['label'];suffix,_,scope,owner_suffix,marker=attempt_scope(label,attempt)
        retained={'schema':cloud.SCHEMA+'.scheduling-hold','label':label,
            'queue_sha256':os.environ['PANEL_SNAPSHOT_SHA'],
            'queue_plan_sha256':row.get('plan_sha256'),'attempt':attempt,
            'error':str(error)[:1000],'time':time.time(),'score_admitted':False,
            'dispatch':state.get(key+'dispatch/'+label+scope),
            'owner':state.get(key+'owner/'+label+owner_suffix),
            'recovery':state.get(key+'recovery/'+label+scope),**details}
        if marker is not None:
            retained['redispatch_call_id']=marker['call_id']
        if label in reinspection:
            retained['reinspection_tranche_sha256']=tranche_sha
            retained['reinspection_token']=reinspection_tokens[label]
            previous=state.get(key+'hold/'+label)
            if previous is not None:
                state.put(key+'hold-history/'+reinspection_tokens[label]+'/'+label,previous,skip_if_exists=True)
            state.put(key+'hold/'+label,retained)
        else:
            previous=state.get(key+'hold/'+label)
            if (previous is not None and marker is not None
                    and marker.get('released_hold_sha256')==cloud.digest(previous)):
                # The scoped redispatch released this hold. A hold raised under
                # the scope replaces it; the released record stays archived.
                state.put(key+'hold-history/redispatch-'+marker['call_id']+'/'+label,previous,skip_if_exists=True)
                state.put(key+'hold/'+label,retained)
            else:
                state.put(key+'hold/'+label,retained,skip_if_exists=True)
        holds[label]=retained;errors.append(retained)
    def supersede_dead_owner(row,attempt,dispatch,owner,reason):
        # The frozen worker refuses a second owner under the same key, so the
        # attempt replays under a scope named by the dead call. The superseded
        # owner and dispatch records are archived first and never rewritten;
        # the worker proves the durable index holds no native start before it
        # binds the scoped owner. One scope per attempt: a dead scoped owner
        # holds the row for review instead of chaining.
        label=row['label'];suffix,_,scope,_,marker=attempt_scope(label,attempt)
        if marker is not None:
            retain_hold(row,'redispatch scope already consumed: '+reason,attempt);return
        old=owner['call_id']
        archive(key+'owner-history/'+label+suffix+'/'+old,owner)
        if dispatch:
            archive(key+'dispatch-history/'+label+scope+'/'+dispatch.get('call_id',old),dispatch)
        record={'schema':cloud.SCHEMA+'.redispatch','label':label,'attempt':attempt,'call_id':old,
            'reason':reason,'time':time.time(),'dispatch_scope':scope,'superseded_owner':owner,
            'superseded_dispatch':dispatch or None,'score_admitted':False}
        hold=state.get(key+'hold/'+label)
        if hold is not None:
            record['released_hold_sha256']=cloud.digest(hold)
        marker_key=key+'redispatch/'+label+suffix
        state.put(marker_key,record,skip_if_exists=True)
        if state.get(marker_key)!=record:
            raise ValueError('redispatch marker already differs')
        redispatched.append({'label':label,'attempt':attempt,'dead_call_id':old,'reason':reason,
            'scope':scope+redispatch_scope(old)})
        dispatch_one(row)
    def recover_result(row,dispatch,error,attempt):
        if policy is not None and budget_guard.remaining(policy,budget)<=0:
            raise TimeoutError('frozen fleet budget deadline reached')
        label=row['label'];suffix,retry,scope,owner_suffix,marker=attempt_scope(label,attempt)
        recovery_key=key+'recovery/'+label+scope
        if label in reinspection:
            recovery_key+='/resume-'+resume_tranche['authorization_sha256']
            if reinspection_tokens[label]!=tranche_sha:
                recovery_key+='/revision-'+reinspection_tokens[label]
        recovery=state.get(recovery_key,{})
        if recovery.get('call_id'):
            try:
                answer=modal.FunctionCall.from_id(recovery['call_id']).get(timeout=0)
                if answer.get('status') in {'complete','retry-authorized','retry-renewed'}:return
                if answer.get('status')=='budget-admission-deferred':
                    deferred.append(answer);return
            except TimeoutError:
                waiting.append(label);return
            except Exception as failure:
                error=f'{error}; recovery: {failure}'
        count=recovery.get('submissions',0)
        if count>=3:
            # Exhausted read-only recovery of an attempt whose owner is dead
            # and whose plan has no native binding replays the attempt under
            # a redispatch scope instead of holding it forever.
            owner=state.get(key+'owner/'+label+owner_suffix)
            if (marker is None and owner is not None and dead_call_record(owner)
                    and not native_start_bound(durable_index(),attempt_plan_sha(row,retry))):
                supersede_dead_owner(row,attempt,state.get(key+'dispatch/'+label+scope,{}),owner,
                    f'recovery exhausted behind a dead owner: {error}'[:1000]);return
            retain_hold(row,error,attempt,recovery_exhausted=True);return
        intent={'submissions':count+1,'original_error':str(error)[:1000],'submitted_unix':time.time()}
        state.put(recovery_key,intent)
        call=fn.spawn(label,os.environ['PANEL_SNAPSHOT_SHA'],dispatch['expires_unix'],
            control['image_id'],recover_only=True,attempt=attempt,**redispatch_kwargs(marker))
        state.put(recovery_key,{**intent,'call_id':call.object_id})
        submitted.append({'label':label,'call_id':call.object_id,'read_only_recovery':True})
    def dispatch_one(row):
        label=row['label'];attempt=state.get(key+'retry-latest/'+label,1)
        suffix,retry,scope,owner_suffix,marker=attempt_scope(label,attempt)
        job_key=key+'dispatch/'+label+scope
        dispatch=state.get(job_key,{})
        owner=state.get(key+'owner/'+label+owner_suffix)
        call_id=owner['call_id'] if owner else dispatch.get('call_id')
        if call_id:
            try:
                result=modal.FunctionCall.from_id(call_id).get(timeout=0)
            except TimeoutError:
                # An unresolved call is still running until its own recorded
                # deadline is more than the grace period in the past. Then it
                # is dead: an owned attempt replays under a redispatch scope;
                # a never-owned one archives the dead intent and replays here
                # with a fresh deadline under the ordinary submission bound.
                if not dead_call_record(owner or dispatch):
                    waiting.append(label);return
                if owner:
                    supersede_dead_owner(row,attempt,dispatch,owner,
                        'owned call unresolved past its recorded deadline');return
                archive(key+'dispatch-history/'+label+scope+'/'+call_id,dispatch)
                redispatched.append({'label':label,'attempt':attempt,'dead_call_id':call_id,
                    'reason':'never-owned call unresolved past its recorded deadline','scope':scope})
                dispatch={k:v for k,v in dispatch.items() if k not in ('call_id','expires_unix')}
            except Exception as error:
                # A saved native result survives an interrupted return or a
                # host-side acceptance error. A separate read-only call repeats
                # full acceptance; it cannot invoke a game owner or extend play.
                recover_result(row,dispatch,error,attempt)
                return
            else:
                if result.get('status') in {'complete','retry-authorized','retry-renewed'}:return
                if result.get('status')=='budget-admission-deferred':
                    # A deferral is current only while the tranche it names is live.
                    # Once that deadline has passed the recorded call can never admit
                    # the attempt, so re-reading its answer would defer forever. An
                    # owner that answered before any native start is superseded by
                    # a scoped redispatch; a retained failed attempt keeps its
                    # read-only recovery, which classifies the saved result.
                    stale=result.get('expires_unix')
                    if type(stale) in (int,float) and time.time()>stale:
                        if owner and not result.get('failed_attempt_retained'):
                            supersede_dead_owner(row,attempt,dispatch,owner,
                                'stale admission deferral from an expired tranche');return
                        recover_result(row,dispatch or {'expires_unix':stale},
                            'stale admission deferral from an expired tranche',attempt)
                        return
                    deferred.append(result);return
                if result.get('status')==REDISPATCH_REFUSED:
                    # The worker found a native binding for this plan: the dead
                    # owner did start. Ordinary read-only recovery classifies it.
                    recover_result(row,dispatch,'scoped redispatch refused: native start already bound',attempt)
                    return
                if owner:
                    retain_hold(row,'owned call returned without durable acceptance',attempt,returned=result);return
                # Adoption calls do no new gameplay and may poll an older running call.
        if dispatch and not call_id and time.time()-dispatch['submitted_unix']<300:
            waiting.append(label);return
        count=dispatch.get('submissions',0)
        if marker is not None:
            # The submission bound spans the superseded and the scoped intent.
            count+=(marker.get('superseded_dispatch') or {}).get('submissions',0)
        if count>=3 and not row.get('external_submission'):
            retain_hold(row,'ambiguous dispatch retry bound reached',attempt);return
        # The deadline is frozen before first spawn and reused after lost replies.
        expires=dispatch.get('expires_unix',retry['expires_unix'] if retry else
            time.time()+frozen_game_session_seconds(row))
        if policy is not None:
            expires=budget_guard.clamp_expiry(policy,budget,expires)
        admission=late_admission(label,expires,dispatch=True)
        if (retry and not dispatch and owner is None and label in reinspection
                and time.time()>retry['expires_unix'] and not row.get('external_submission')):
            # An authorization whose deadline passed before any dispatch took
            # ownership can neither be spawned nor renewed by a deferral. A
            # bounded read-only recovery lets the worker renew it under the
            # active tranche; the renewed record dispatches on a later tick
            # under its own renewal scope.
            recover_result(row,{'expires_unix':retry['expires_unix']},
                'expired unstarted retry awaiting renewal',attempt);return
        if admission is not None and not row.get('external_submission'):
            deferred.append(admission);return
        intent={'submitted_unix':time.time(),'expires_unix':expires,'submissions':dispatch.get('submissions',0)+1}
        state.put(job_key,intent)
        call=fn.spawn(label,os.environ['PANEL_SNAPSHOT_SHA'],expires,control['image_id'],attempt=attempt,
                      **redispatch_kwargs(marker))
        state.put(job_key,{**intent,'call_id':call.object_id})
        submitted.append({'label':label,'call_id':call.object_id})
    for row in candidates:
        try:
            dispatch_one(row)
        except Exception as error:
            # A per-game lookup, spawn or bookkeeping error must not prevent
            # independent candidates from being processed. Preserve ownership
            # and dispatch intent, including ambiguous submissions, for review.
            retain_hold(row,f'{type(error).__name__}: {error}',
                state.get(key+'retry-latest/'+row['label'],1))
    counts={s:sum(cloud.status(r,records)==s for r in queue['games']) for s in ('complete','quarantined','pending')}
    runnable=counts['pending']-len(holds)
    report={'status':('complete' if counts['pending']==0 else
        'drained-with-unresolved-holds' if runnable==0 else 'running'),'time':time.time(),
        'counts':counts,'submitted':submitted,'waiting':len(waiting),'errors':errors,
        'admission_deferred':deferred,'redispatched':redispatched,
        'held':len(holds),'runnable_pending':runnable,'held_labels':sorted(holds),
        'all_games_in_cloud':len(queue['games']),'maximum_workers':64}
    if budget is not None:
        report['budget']=budget
        report['budget_seconds_remaining']=budget_guard.remaining(policy,budget)
    if provider_check is not None:
        report['provider_budget_check']=provider_check
    if control.get('canary_expansion'):
        report['canary_expansion']=control['canary_expansion']
    if counts['pending']==0:
        disable_dispatch(state,key,control,report)
    elif budget is not None and budget_guard.remaining(policy,budget)<=0:
        report['status']='fleet-budget-deadline-reached'
        disable_dispatch(state,key,control,report)
    else:
        state.put(key+'health',report)
    print(json.dumps(report),flush=True)
    return report


def configure(deployment_path, *, authorized_total_usd=None):
    import modal
    deployment=json.loads(Path(deployment_path).read_bytes())
    names=validate_fresh_deployment(deployment,Path(deployment_path).parent)
    policy=budget_policy(deployment,authorized_total_usd=authorized_total_usd)
    modal_environment=deployment_environment(deployment)
    script=Path(__file__).absolute();pure=script.with_name('modal_panel_cloud_queue.py')
    environment={
        'PYTHONPATH':'/opt/cloud-code:/opt/runtime-root/scripts',
        'FAYNT_BENCHMARK_LOCAL_ROOT':str(Path(__file__).resolve().parents[1]),
        'PANEL_ARCHIVE_SHA':deployment['archive_sha256'],
        'PANEL_SNAPSHOT_SHA':deployment['snapshot_sha256'],
        'PANEL_APP_NAME':names['app_name'],'PANEL_STATE_NAME':names['state_name'],
        'PANEL_ENVIRONMENT':modal_environment,
        'PANEL_DURABLE_MODULE':deployment_worker(deployment)}
    if policy is not None:
        environment['PANEL_BUDGET_POLICY']=json.dumps(policy,sort_keys=True,separators=(',',':'))
    image=modal.Image.from_registry(prepared_runtime.image_reference()).env({**prepared_runtime.image_environment(), **environment})
    image=image.add_local_file(script,'/opt/cloud-code/'+script.name,copy=False)
    image=image.add_local_file(pure,'/opt/cloud-code/'+pure.name,copy=False)
    image=image.add_local_file(script.with_name('prepared_runtime.py'),'/opt/cloud-code/prepared_runtime.py',copy=False)
    image=image.add_local_file(script.with_name('modal_panel_budget.py'),'/opt/cloud-code/modal_panel_budget.py',copy=False)
    image=image.add_local_file(deployment['archive'],str(ARCHIVE),copy=False)
    for row in deployment['inputs']:
        image=image.add_local_file(row['local'],str(BLOBS/row['sha256']),copy=False)
    app=modal.App(names['app_name'],include_source=False)
    volume=modal.Volume.from_name(worker_paths(deployment)['volume'],create_if_missing=True,environment_name=modal_environment)
    modal.Dict.from_name(names['state_name'],create_if_missing=True,environment_name=modal_environment).hydrate()
    modal.Dict.from_name(worker_paths(deployment)['index'],create_if_missing=True,environment_name=modal_environment).hydrate()
    memory=policy['worker_memory_mib'] if policy is not None else 16384
    worker=app.function(image=image,include_source=False,cpu=(4,4),memory=(memory,memory),
        max_containers=64,min_containers=0,buffer_containers=0,
        timeout=worker_time_limits(deployment)['function_seconds'],startup_timeout=300,
        scaledown_window=2,nonpreemptible=(policy is None),single_use_containers=True,
        volumes={'/persist-benchmark':volume},retries=modal.Retries(max_retries=2,initial_delay=1.0))(run_cloud_game)
    dispatcher=app.function(image=image,include_source=False,cpu=(0.25,0.25),memory=(1024,1024),
        max_containers=1,min_containers=0,buffer_containers=0,timeout=900,startup_timeout=120,
        scaledown_window=2,single_use_containers=True,schedule=modal.Cron('* * * * *'))(dispatch_cloud)
    return app,worker,dispatcher,image,deployment


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('deployment',type=Path)
    p.add_argument('--deploy',action='store_true');p.add_argument('--activate',action='store_true')
    p.add_argument('--allow-label',action='append',default=None,
        help='Bound initial cloud qualification to these original pending labels.')
    p.add_argument('--auto-expand-after-canary',action='store_true',
        help='Expand this queue after every allowlisted 75M and 10M canary has strict native acceptance.')
    p.add_argument('--budget-started-unix',type=float,
        help='Preserve an explicitly recorded original activation start during a fleet-budget migration.')
    p.add_argument('--resume-authorization',type=Path,
        help='Explicit remaining-budget authorization, verified and activated before redeploying a stopped app.')
    p.add_argument('--reinspect-held',action='store_true',
        help='Bind one reinspection of all unresolved held labels to this approved resume tranche.')
    p.add_argument('--strict-phase-order',action='store_true',
        help='Finish runnable earlier blocks before starting later benchmark blocks.')
    a=p.parse_args()
    if a.resume_authorization and (not a.activate or a.budget_started_unix is not None):
        p.error('--resume-authorization requires --activate and its own immutable new activation clock')
    if a.reinspect_held and (not a.activate or a.allow_label is not None):
        p.error('--reinspect-held requires --activate on the full resumed queue')
    if a.auto_expand_after_canary:
        if not a.activate:
            p.error('--auto-expand-after-canary requires --activate and --allow-label for both model profiles')
        validate_canary_allowlist(json.loads((a.deployment.parent/'metadata/snapshot.json').read_bytes()),
            a.allow_label,require_both_models=True)
    import modal
    # A raised cap enters this process only through the explicit resume
    # authorization, which is verified against the stopped predecessor below.
    resume=(budget_guard.validate_resume_authorization(json.loads(a.resume_authorization.read_bytes()))
        if a.resume_authorization else None)
    authorized=budget_guard.authorized_total_usd(resume) if resume is not None else None
    app,worker,dispatcher,image,deployment=configure(a.deployment,authorized_total_usd=authorized)
    if a.resume_authorization:
        # A same-name deployment can reuse its previous app ID. Verify its
        # stopped state and select the new budget before redeployment starts.
        # Control remains disabled throughout this preflight/build interval.
        resume_state=modal.Dict.from_name(deployment_names(deployment)['state_name'],
            environment_name=deployment_environment(deployment))
        activate_resume_budget(resume_state,deployment['snapshot_sha256']+'/',deployment,resume)
    if a.deploy:
        with modal.enable_output():app.deploy(environment_name=deployment_environment(deployment))
        receipt={'app_id':app.app_id,'image_id':image.object_id,'snapshot_sha256':deployment['snapshot_sha256'],
            **deployment_names(deployment),
            'durable_module':deployment_worker(deployment),
            'environment_name':deployment_environment(deployment),
            'budget':budget_policy(deployment,authorized_total_usd=authorized),
            'source_sha256':cloud.digest(Path(__file__).read_bytes()),
            'queue_source_sha256':cloud.digest(Path(__file__).with_name('modal_panel_cloud_queue.py').read_bytes()),
            'budget_source_sha256':cloud.digest(Path(__file__).with_name('modal_panel_budget.py').read_bytes()),
            'deployed_unix':time.time()}
        from modal_panel_queue import new_file
        new_file(a.deployment.parent/f"deployed-{time.time_ns()}.json",receipt)
        temporary=a.deployment.parent/'.deployed.json.tmp'
        temporary.write_bytes(cloud.encoded(receipt));temporary.replace(a.deployment.parent/'deployed.json')
        print(json.dumps(receipt),flush=True)
    if a.activate:
        receipt=json.loads((a.deployment.parent/'deployed.json').read_bytes())
        names=validate_activation_receipt(deployment,receipt)
        state=modal.Dict.from_name(names['state_name'],environment_name=deployment_environment(deployment))
        budget=activate_budget(state,deployment['snapshot_sha256']+'/',deployment,started_unix=a.budget_started_unix)
        control=activation_control(state,deployment['snapshot_sha256']+'/',
            json.loads((a.deployment.parent/'metadata/snapshot.json').read_bytes()),
            image_id=receipt['image_id'],allowlist=a.allow_label,
            auto_expand_after_canary=a.auto_expand_after_canary,
            strict_phase_order=a.strict_phase_order,reinspect_held=a.reinspect_held)
        state.put(deployment['snapshot_sha256']+'/control',control)
        call=modal.Function.from_name(names['app_name'],'dispatch_cloud',environment_name=deployment_environment(deployment)).spawn()
        print(json.dumps({'dispatcher_call_id':call.object_id,
            'all_games':len(json.loads((a.deployment.parent/'metadata/snapshot.json').read_bytes())['games']),
            'budget':budget}),flush=True)
