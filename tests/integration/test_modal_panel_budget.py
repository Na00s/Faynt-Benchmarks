"""No network, cloud mutations, emulator process, or real exit in these tests."""
import copy
import json
from datetime import datetime, timezone
import sys
import types

import pytest

import modal_panel_budget as budget
import modal_panel_cloud_app as app


class Store(dict):
    def put(self,key,value,skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key]=value
        return True


def test_list_price_math_includes_dispatcher_and_shutdown_tail():
    policy=budget.make_policy()
    receipt=budget.activate(policy,1000)
    assert receipt['worker_usd_per_second']==pytest.approx(0.00008792)
    assert receipt['fleet_usd_per_second']==pytest.approx(0.005632375)
    assert receipt['maximum_modeled_fleet_usd']<=140
    assert receipt['maximum_modeled_fleet_usd']+receipt['reserve_usd']<=150
    assert receipt['deadline_unix']==1000+receipt['execution_seconds']
    assert receipt['execution_seconds']+600==int(140/receipt['fleet_usd_per_second'])
    assert receipt['external_app_stop_required'] is True
    assert receipt['provider_invoice_cap'] is False


def test_eight_gib_prices_are_explicit_and_lower():
    policy=budget.make_policy(worker_memory_mib=8192)
    assert budget.rates(policy)['worker_usd_per_second']==pytest.approx(0.00007016)
    assert budget.activate(policy,1000)['execution_seconds']>budget.activate(budget.make_policy(),1000)['execution_seconds']


@pytest.mark.parametrize('key,value',[
    ('total_usd',151),('total_usd',float('inf')),('total_usd',float('nan')),('total_usd',True),
    ('reserve_usd',9),('reserve_usd',150),('worker_cpu',3),('worker_cpu',4.0),
    ('maximum_workers',65),('worker_memory_mib',32768),('nonpreemptible',True),
    ('nonpreemptible',0),('schema','unknown'),('unexpected',1)])
def test_budget_rejects_unreviewed_or_under_reserved_values(key,value):
    policy=budget.make_policy();policy[key]=value
    with pytest.raises(ValueError):budget.validate_policy(policy)


def test_raised_total_requires_the_explicit_matching_keyword():
    assert budget.TOTAL_RAIL_USD==150 and budget.MAX_AUTHORIZED_TOTAL_USD==400
    raised={**budget.make_policy(prior_spend_usd=100),'total_usd':200}
    with pytest.raises(ValueError):budget.validate_policy(raised)
    with pytest.raises(ValueError):budget.make_policy(total_usd=200)
    with pytest.raises(ValueError):budget.activate(raised,1000)
    assert budget.validate_policy(raised,authorized_total_usd=200)==raised
    assert budget.make_policy(total_usd=200,prior_spend_usd=100,authorized_total_usd=200)==raised
    for keyword in (201,150,401,'200',True,float('nan'),float('inf')):
        with pytest.raises(ValueError):budget.validate_policy(raised,authorized_total_usd=keyword)
    assert budget.validate_policy({**raised,'total_usd':400},authorized_total_usd=400)['total_usd']==400
    with pytest.raises(ValueError):budget.validate_policy({**raised,'total_usd':400.5},authorized_total_usd=400.5)
    provider,_=provider_fixture()
    with pytest.raises(ValueError):budget.validate_policy(provider,authorized_total_usd=150)
    receipt=budget.activate(raised,1000,authorized_total_usd=200)
    assert receipt['allocated_fleet_usd']==90 and receipt['authorized_total_usd']==200
    assert receipt['maximum_modeled_fleet_usd']+10+100<=200
    assert budget.validate_activation(raised,receipt)==receipt
    stripped={k:v for k,v in receipt.items() if k!='authorized_total_usd'}
    with pytest.raises(ValueError):budget.validate_activation(raised,stripped)
    plain=budget.activate(budget.make_policy(prior_spend_usd=100),1000)
    assert 'authorized_total_usd' not in plain
    assert {k:v for k,v in plain.items() if k not in ('policy_sha256','allocated_fleet_usd','deadline_unix',
        'execution_seconds','maximum_modeled_fleet_usd')}=={k:v for k,v in stripped.items()
        if k not in ('policy_sha256','allocated_fleet_usd','deadline_unix','execution_seconds','maximum_modeled_fleet_usd')}


def test_root_receipt_cannot_record_a_raise_without_a_tranche(monkeypatch):
    policy=budget.make_policy();receipt=budget.activate(policy,1000)
    store=Store({'q/budget':{**receipt,'authorized_total_usd':150}})
    with pytest.raises(ValueError,match='resume tranche'):
        app.activate_budget(store,'q/',{'budget':policy},now=1100)
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(policy))
    with pytest.raises(ValueError,match='authorized cap'):app.remote_budget(store,'q/')
    store=Store({'q/budget':receipt})
    assert app.activate_budget(store,'q/',{'budget':policy},now=1100)==receipt
    assert app.remote_budget(store,'q/')==(policy,receipt)


def test_budget_receipt_is_fully_bound_and_cannot_extend_deadline():
    policy=budget.make_policy();receipt=budget.activate(policy,1000)
    assert budget.clamp_expiry(policy,receipt,1e9,now=1001)==receipt['deadline_unix']
    assert budget.clamp_expiry(policy,receipt,1100,now=1001)==1100
    changed=copy.deepcopy(receipt);changed['deadline_unix']+=1
    with pytest.raises(ValueError):budget.validate_activation(policy,changed)
    with pytest.raises(TimeoutError):budget.clamp_expiry(policy,receipt,1e9,now=receipt['deadline_unix'])


def test_repeat_activation_keeps_first_clock():
    deployment={'budget':budget.make_policy()};store=Store()
    first=app.activate_budget(store,'queue/',deployment,now=1000)
    assert app.activate_budget(store,'queue/',deployment,now=1100)==first
    with pytest.raises(TimeoutError,match='cannot extend'):
        app.activate_budget(store,'queue/',deployment,now=first['deadline_unix'])
    assert store['queue/budget']==first


def test_changed_budget_policy_cannot_reuse_activation():
    store=Store();app.activate_budget(store,'q/',{'budget':budget.make_policy()},now=1000)
    with pytest.raises(ValueError):
        app.activate_budget(store,'q/',{'budget':budget.make_policy(worker_memory_mib=8192)},now=1100)


def test_worker_rejects_expiry_before_entering_game(monkeypatch):
    policy=budget.make_policy();receipt=budget.activate(policy,1000)
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(policy));monkeypatch.setenv('PANEL_SNAPSHOT_SHA','q')
    monkeypatch.setattr(app,'handles',lambda:Store({'q/budget':receipt}))
    monkeypatch.setattr(budget.time,'time',lambda:receipt['deadline_unix'])
    monkeypatch.setattr(app,'_run_cloud_game',lambda *a,**k:pytest.fail('game entered after expiry'))
    with pytest.raises(TimeoutError):app.run_cloud_game('label','q',1e9,'im-current')


def test_worker_clamps_original_expiry_and_cancels_watchdog(monkeypatch):
    policy=budget.make_policy();receipt=budget.activate(policy,1000)
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(policy));monkeypatch.setenv('PANEL_SNAPSHOT_SHA','q')
    monkeypatch.setattr(app,'handles',lambda:Store({'q/budget':receipt}))
    monkeypatch.setattr(budget.time,'time',lambda:1001)
    timers=[]
    class Timer:
        def __init__(self,seconds,callback,args):self.seconds=seconds;self.callback=callback;self.args=args;timers.append(self)
        def start(self):self.started=True
        def cancel(self):self.cancelled=True
    monkeypatch.setattr(budget.threading,'Timer',Timer)
    monkeypatch.setattr(app,'_run_cloud_game',lambda *args,**kwargs:(args,kwargs))
    args,kwargs=app.run_cloud_game('label','q',1e9,'im-current',attempt=2)
    assert args[2]==receipt['deadline_unix'] and kwargs['attempt']==2
    assert len(timers)==1 and timers[0].started and timers[0].cancelled and timers[0].daemon
    assert timers[0].args==(124,)


def test_watchdog_owns_only_current_worker_exit_callback(monkeypatch):
    callbacks=[]
    class Timer:
        def __init__(self,seconds,callback,args):callbacks.append((callback,args))
        def start(self):pass
        def cancel(self):pass
    monkeypatch.setattr(budget.threading,'Timer',Timer)
    monkeypatch.setattr(budget.time,'time',lambda:1000)
    called=[]
    with budget.watchdog(1001,stop_process=lambda code:called.append(code)):
        callbacks[0][0](*callbacks[0][1])
    assert called==[124]


def test_expired_dispatcher_disables_without_spawning(monkeypatch):
    policy=budget.make_policy();receipt=budget.activate(policy,1000)
    state=Store({'q/budget':receipt,'q/control':{'enabled':True}})
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(policy));monkeypatch.setenv('PANEL_SNAPSHOT_SHA','q')
    monkeypatch.setattr(app,'handles',lambda:state)
    monkeypatch.setattr(app,'snapshot',lambda:{'queue_deadline_unix':1e10,'games':[]})
    monkeypatch.setattr(budget.time,'time',lambda:receipt['deadline_unix'])
    monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace())
    report=app.dispatch_cloud()
    assert report['status']=='fleet-budget-deadline-reached'
    assert state['q/control']['enabled'] is False and report['app_stopped'] is False


@pytest.mark.parametrize('name',['../main','',None,'a'*65])
def test_invalid_environment_is_rejected(name):
    with pytest.raises(ValueError):app.deployment_environment({'environment_name':name})


def test_environment_defaults_and_explicit_remote_routing(monkeypatch):
    assert app.deployment_environment({})=='main'
    monkeypatch.setenv('PANEL_ENVIRONMENT','faynt-d0-632-980')
    names=[]
    monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace(Dict=types.SimpleNamespace(
        from_name=lambda *a,**k:names.append((a,k)))))
    app.handles()
    assert names[0][1]['environment_name']=='faynt-d0-632-980'


def test_activation_rejects_environment_or_budget_drift():
    deployment={'snapshot_sha256':'a','environment_name':'faynt-d0-632-980','budget':budget.make_policy()}
    receipt={**deployment,'image_id':'im-current'}
    app.validate_activation_receipt(deployment,receipt)
    for key,value in [('environment_name','main'),('budget',None)]:
        with pytest.raises(ValueError):app.validate_activation_receipt(deployment,{**receipt,key:value})


def test_configure_binds_hard_resources_budget_worker_and_environment(tmp_path,monkeypatch):
    deployment={'app_name':'new-app','state_name':'new-state','environment_name':'new-env',
        'snapshot_sha256':'a','archive_sha256':'b','archive':'empty.tar.gz','inputs':[],
        'budget':budget.make_policy(),'durable_module':'faynt_d0_durable_game'}
    path=tmp_path/'deployment.json';path.write_text(json.dumps(deployment))
    rows=[];environment={};files=[]
    class Image:
        def env(self,value):environment.update(value);return self
        def add_local_file(self,*args,**kwargs):files.append(args);return self
    class App:
        def __init__(self,name,**kwargs):rows.append(('app',name))
        def function(self,**kwargs):
            return lambda fn:(rows.append((fn.__name__,kwargs)) or fn)
    def handle(name,**kwargs):rows.append(('handle',name,kwargs));return types.SimpleNamespace(hydrate=lambda:None)
    fake=types.SimpleNamespace(Image=types.SimpleNamespace(from_registry=lambda *a,**k:Image()),
        App=App,Volume=types.SimpleNamespace(from_name=handle),Dict=types.SimpleNamespace(from_name=handle),
        Retries=lambda **k:k,Cron=lambda value:value)
    monkeypatch.setitem(sys.modules,'modal',fake)
    app.configure(path)
    worker=dict((name,value) for name,*value in rows if name=='run_cloud_game')['run_cloud_game'][0]
    assert worker['cpu']==(4,4) and worker['memory']==(16384,16384)
    assert worker['max_containers']==64 and worker['nonpreemptible'] is False
    assert worker['timeout']==8790
    assert worker['min_containers']==0 and worker['buffer_containers']==0
    assert environment['PANEL_ENVIRONMENT']=='new-env'
    assert environment['PANEL_DURABLE_MODULE']=='faynt_d0_durable_game'
    assert json.loads(environment['PANEL_BUDGET_POLICY'])==deployment['budget']
    assert all(row[2]['environment_name']=='new-env' for row in rows if row[0]=='handle')
    assert {row[1] for row in rows if row[0]=='handle'}=={
        'new-state','frisson-faynt-d0-game-results-v1','frisson-faynt-d0-game-index-v1'}
    assert any(str(source).endswith('modal_panel_budget.py') for source,destination in files)


def provider_fixture(now=1789488000):
    policy=budget.make_provider_policy(environment_name='faynt-d0-632-980',environment_id='en-fayntOwned')
    observation={'environment_name':policy['environment_name'],'environment_id':policy['environment_id'],
        'profile':'frisson','cycle_compute_budget_usd':140,'effective_cycle_spend_limit_usd':140,
        'current_cycle_usage_usd':0,'spend_limit_reached':False,'verified_unix':now}
    return policy,observation


def test_provider_cap_has_separate_24_hour_operational_deadline():
    policy,observed=provider_fixture()
    receipt=budget.activate(policy,observed['verified_unix'],observed)
    assert receipt['execution_seconds']==86400
    assert receipt['provider_compute_cap_usd']==140
    assert receipt['provider_invoice_cap'] is False
    assert 'maximum_modeled_fleet_usd' not in receipt
    assert budget.validate_activation(policy,receipt)==receipt


@pytest.mark.parametrize('key,value',[
    ('environment_name','main'),('environment_id','en-different'),('profile','other'),
    ('cycle_compute_budget_usd',141),('cycle_compute_budget_usd',None),
    ('effective_cycle_spend_limit_usd',150),('effective_cycle_spend_limit_usd',0),
    ('current_cycle_usage_usd',140),('current_cycle_usage_usd',-1),
    ('spend_limit_reached',True),('verified_unix',0),('verified_unix',float('nan'))])
def test_provider_cap_rejects_unbound_exhausted_or_stale_observations(key,value):
    policy,observed=provider_fixture();now=observed['verified_unix'];observed[key]=value
    with pytest.raises(ValueError):budget.activate(policy,now,observed)


def test_provider_mode_requires_activation_observation_and_isolated_env():
    policy,observed=provider_fixture()
    with pytest.raises(ValueError):budget.activate(policy,observed['verified_unix'])
    with pytest.raises(ValueError):budget.make_provider_policy(environment_name='main',environment_id='en-owned')
    with pytest.raises(ValueError):app.budget_policy({'budget':policy,'environment_name':'another-env'})


def test_provider_cap_cannot_cross_monthly_reset():
    start=datetime(2026,9,30,20,tzinfo=timezone.utc).timestamp()
    policy,observed=provider_fixture(start)
    receipt=budget.activate(policy,start,observed)
    assert receipt['execution_seconds']==4*3600-600
    start=datetime(2026,9,30,23,55,tzinfo=timezone.utc).timestamp()
    policy,observed=provider_fixture(start)
    with pytest.raises(ValueError,match='billing cycle'):budget.activate(policy,start,observed)


def test_reactivation_rechecks_live_provider_cap_while_retaining_first_deadline(monkeypatch):
    pytest.importorskip("modal", reason="optional Modal SDK is required for this SDK contract test")
    policy,observed=provider_fixture();now=observed['verified_unix']
    deployment={'budget':policy,'environment_name':policy['environment_name']};store=Store();calls=[]
    async def observe(value,**kwargs):calls.append(value);return dict(observed)
    monkeypatch.setattr(budget,'observe_provider',observe)
    first=app.activate_budget(store,'q/',deployment,now=now)
    observed.update(verified_unix=now+1000,current_cycle_usage_usd=2)
    assert app.activate_budget(store,'q/',deployment,now=now+1000)==first
    assert len(calls)==2
    observed['cycle_compute_budget_usd']=150
    with pytest.raises(ValueError):app.activate_budget(store,'q/',deployment,now=now+1000)
    assert store['q/budget']==first


def test_worker_paths_preserve_legacy_and_isolate_d0():
    old=app.worker_paths({});new=app.worker_paths({'durable_module':'faynt_d0_durable_game'})
    assert old['plan']=='/opt/full-game-plan.json' and old['root']=='/opt/runtime-root'
    assert new=={'root':'/opt/d0-root','plan':'/opt/faynt-d0-game-plan.json',
        'volume':'frisson-faynt-d0-game-results-v1','index':'frisson-faynt-d0-game-index-v1'}


def test_provider_meter_high_water_survives_reactivation_and_rejects_rollback(monkeypatch):
    policy,observed=provider_fixture();receipt=budget.activate(policy,observed['verified_unix'],observed)
    control={'enabled':True};state=Store({'q/control':control})
    monkeypatch.setattr(app,'observe_provider_budget',lambda *a,**k:dict(observed))
    observed['current_cycle_usage_usd']=2
    assert app.provider_budget_check(state,'q/',control,policy,receipt)['high_water_usage_usd']==2
    state['q/control']={'enabled':True}
    observed['current_cycle_usage_usd']=1
    with pytest.raises(ValueError,match='meter decreased'):
        app.provider_budget_check(state,'q/',state['q/control'],policy,receipt)
    assert state['q/provider-budget-check']['high_water_usage_usd']==2


def test_provider_check_preserves_concurrent_dispatch_disable(monkeypatch):
    policy,observed=provider_fixture();receipt=budget.activate(policy,observed['verified_unix'],observed)
    state=Store({'q/control':{'enabled':True}})
    def observe(*a,**k):state['q/control']={'enabled':False};return observed
    monkeypatch.setattr(app,'observe_provider_budget',observe)
    app.provider_budget_check(state,'q/',{'enabled':True},policy,receipt)
    assert state['q/control']['enabled'] is False


@pytest.mark.parametrize('failure',[ValueError('changed cap'),TimeoutError(),RuntimeError('private server details')])
def test_dispatcher_disables_if_live_provider_check_fails(monkeypatch,failure):
    policy,observed=provider_fixture();receipt=budget.activate(policy,observed['verified_unix'],observed)
    state=Store({'q/budget':receipt,'q/control':{'enabled':True}})
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(policy));monkeypatch.setenv('PANEL_SNAPSHOT_SHA','q')
    monkeypatch.setattr(app,'handles',lambda:state)
    monkeypatch.setattr(app,'snapshot',lambda:{'queue_deadline_unix':1e10,'games':[]})
    monkeypatch.setattr(budget.time,'time',lambda:observed['verified_unix']+1)
    def observe(*a,**k):raise failure
    monkeypatch.setattr(app,'observe_provider_budget',observe)
    monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace())
    report=app.dispatch_cloud()
    assert report['status']=='provider-budget-verification-failed'
    assert state['q/control']['enabled'] is False
    assert 'private server details' not in report['reason']


def test_provider_observation_uses_only_read_only_sdk_methods(monkeypatch):
    pytest.importorskip("modal", reason="optional Modal SDK is required for this SDK contract test")
    import asyncio
    import modal.client
    policy,observed=provider_fixture();calls=[]
    class Stub:
        async def EnvironmentList(self,request):
            calls.append('EnvironmentList')
            return types.SimpleNamespace(items=[types.SimpleNamespace(
                name=policy['environment_name'],environment_id=policy['environment_id'])])
        async def EnvironmentGetBudget(self,request):
            calls.append('EnvironmentGetBudget')
            assert request.environment_id==policy['environment_id']
            return types.SimpleNamespace(cycle_budget_dollars=140,effective_cycle_spend_limit=140,
                current_cycle_usage=3,spend_limit_reached=False,HasField=lambda key:True)
    class Client:
        @staticmethod
        async def from_env():return types.SimpleNamespace(stub=Stub())
    monkeypatch.setattr(modal.client,'_Client',Client)
    monkeypatch.setenv('MODAL_PROFILE','frisson')
    result=asyncio.run(budget.observe_provider(policy))
    assert result['current_cycle_usage_usd']==3
    assert calls==['EnvironmentList','EnvironmentGetBudget']
    monkeypatch.delenv('MODAL_PROFILE')
    monkeypatch.setenv('PANEL_ENVIRONMENT',policy['environment_name'])
    monkeypatch.setenv('MODAL_TASK_ID','ta-owned123')
    assert asyncio.run(budget.observe_provider(policy,remote=True))['environment_id']==policy['environment_id']


def test_provider_bridge_uses_existing_modal_loop_after_sync_initialization(monkeypatch):
    pytest.importorskip("modal", reason="optional Modal SDK is required for this SDK contract test")
    import asyncio
    from modal._utils.async_utils import synchronizer
    async def existing_loop():return asyncio.get_running_loop()
    owner=synchronizer.create_blocking(existing_loop)()
    policy,observed=provider_fixture()
    async def observe(value,**kwargs):
        assert asyncio.get_running_loop() is owner
        assert asyncio.current_task() is not None
        return observed
    monkeypatch.setattr(budget,'observe_provider',observe)
    monkeypatch.setattr(asyncio,'run',lambda *a,**k:pytest.fail('unowned loop created'))
    assert app.observe_provider_budget(policy)==observed


def test_main_fleet_budget_deducts_prior_cost_and_preserves_migration_clock(monkeypatch):
    policy=budget.make_policy(prior_spend_usd=1)
    deployment={'environment_name':'main','budget':policy};state=Store()
    monkeypatch.setattr(app,'observe_provider_budget',lambda *a,**k:pytest.fail('shared environment queried'))
    receipt=app.activate_budget(state,'q/',deployment,now=2000,started_unix=1000)
    assert receipt['started_unix']==1000
    assert receipt['allocated_fleet_usd']==139 and receipt['prior_spend_usd']==1
    assert receipt['maximum_modeled_fleet_usd']+receipt['reserve_usd']+receipt['prior_spend_usd']<=150
    assert app.activate_budget(state,'q/',deployment,now=2500,started_unix=1000)==receipt
    with pytest.raises(ValueError,match='differs from retained'):
        app.activate_budget(state,'q/',deployment,now=2500,started_unix=2000)
    assert state['q/budget']==receipt


@pytest.mark.parametrize('prior',[-1,140,1000,float('inf'),float('nan'),True,False])
def test_main_budget_rejects_invalid_prior_spend(prior):
    with pytest.raises(ValueError):budget.make_policy(prior_spend_usd=prior)


@pytest.mark.parametrize('start',[2001,0,-1,float('inf'),float('nan')])
def test_migration_cannot_start_in_future_or_reset_a_full_day(start):
    with pytest.raises(ValueError):
        app.activate_budget(Store(),'q/',{'budget':budget.make_policy(prior_spend_usd=1)},now=2000,started_unix=start)


def test_frozen_d0_session_uses_native_9090_bound_and_preserves_legacy(tmp_path,monkeypatch):
    monkeypatch.setattr(app,'META',tmp_path)
    monkeypatch.setenv('PANEL_DURABLE_MODULE','faynt_d0_durable_game')
    plan={'limits':{'session_seconds':9090,'function_seconds':8790}}
    body=json.dumps(plan).encode();(tmp_path/'game.json').write_bytes(body)
    row={'plan_path':'game.json','plan_sha256':app.cloud.digest(body)}
    assert app.frozen_game_session_seconds(row)==9090
    assert app.worker_time_limits({})=={'session_seconds':16290,'function_seconds':15990}
    plan['limits']['session_seconds']=16290
    body=json.dumps(plan).encode();(tmp_path/'game.json').write_bytes(body)
    row['plan_sha256']=app.cloud.digest(body)
    with pytest.raises(ValueError,match='limits differ'):app.frozen_game_session_seconds(row)
    row['plan_sha256']='0'*64
    with pytest.raises(ValueError,match='changed before dispatch'):app.frozen_game_session_seconds(row)
