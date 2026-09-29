"""Explicit remaining-budget tranches, with no cloud operations in tests."""
import copy
import asyncio
import json
import inspect
import types
import pytest

import modal_panel_budget as budget
import modal_panel_cloud_app as app


class Store(dict):
    def put(self,key,value,skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key]=copy.deepcopy(value)
        return True


@pytest.fixture
def case():
    queue='a'*64; key=queue+'/'
    old=budget.activate(budget.make_policy(prior_spend_usd=1),1000)
    now=old['deadline_unix']+100
    policy=budget.make_policy(prior_spend_usd=80)
    deployment={'budget':policy,'snapshot_sha256':queue,'app_name':'faynt-d0-632-980-v1',
        'state_name':'faynt-d0-632-980-state-v1','environment_name':'main'}
    authorization=budget.prepare_resume_authorization(queue_sha256=queue,
        app_name=deployment['app_name'],environment_name='main',previous_app_id='ap-previous',
        previous_budget_sha256=budget.digest(old),budget_policy=policy,
        user_authorization={'total_usd':150,'instruction':'Resume the authorized queue within $150.'},
        prior_cost={'metered_usd':78.61239063,'evidence_sha256':'b'*64},created_unix=now-20)
    observation={'previous_app_id':'ap-previous','profile':'frisson','app_state':'STOPPED',
        'stopped_at':now-30,'verified_unix':now-1,'functions':[
            {'tag':tag,'function_id':'fu-'+tag.replace('_',''),'backlog':0,
             'num_total_tasks':0,'num_running_inputs':0} for tag in ('run_cloud_game','dispatch_cloud')]}
    state=Store({key+'budget':old,key+'control':{'enabled':False},
        key+'accept/game':{'status':'complete'}})
    return state,key,deployment,authorization,observation,now


def activate(case):
    state,key,deployment,authorization,observation,now=case
    return app.activate_resume_budget(state,key,deployment,authorization,now=now,
        stop_observation=observation)


def test_resume_preserves_history_and_scores_with_inclusive_60_dollar_tranche(case):
    state,key,deployment,authorization,observation,now=case
    before=copy.deepcopy(state)
    receipt=activate(case)
    assert all(state[k]==v for k,v in before.items())
    assert receipt['allocated_fleet_usd']==60
    assert receipt['prior_spend_usd']==80 and receipt['reserve_usd']==10
    assert receipt['maximum_modeled_fleet_usd']+80+10<=150
    tranche=app.active_resume_tranche(state,key)
    assert tranche['activation']==receipt
    assert tranche['authorization_sha256']==budget.digest(authorization)
    assert state[key+'budget-current']['tranche_sha256']==budget.digest(tranche)


def test_repeat_resume_and_ordinary_activation_keep_new_clock(case):
    state,key,deployment,authorization,observation,now=case
    receipt=activate(case)
    assert app.activate_resume_budget(state,key,deployment,authorization,now=now+100)==receipt
    assert app.activate_budget(state,key,deployment,now=now+200)==receipt
    with pytest.raises(TimeoutError):
        app.activate_resume_budget(state,key,deployment,authorization,now=receipt['deadline_unix'])
    with pytest.raises(TimeoutError):
        app.activate_budget(state,key,deployment,now=receipt['deadline_unix'])


def test_remote_worker_and_retry_share_exact_tranche(case,monkeypatch):
    state,key,deployment,authorization,observation,now=case
    receipt=activate(case)
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(deployment['budget']))
    monkeypatch.setenv('PANEL_APP_NAME',deployment['app_name'])
    monkeypatch.setenv('PANEL_STATE_NAME',deployment['state_name'])
    monkeypatch.setenv('PANEL_ENVIRONMENT','main')
    assert app.remote_budget(state,key)==(deployment['budget'],receipt)
    monkeypatch.setenv('PANEL_APP_NAME','different-app')
    with pytest.raises(ValueError):app.remote_budget(state,key)


@pytest.mark.parametrize('field,value',[
    ('app_name','other'),('environment_name','other'),('queue_sha256','c'*64),
    ('previous_budget_sha256','d'*64),('created_unix',1)])
def test_wrong_binding_fails_before_pointer_write(case,field,value):
    state,key,deployment,authorization,observation,now=case
    authorization[field]=value
    with pytest.raises(ValueError):activate(case)
    assert key+'budget-current' not in state


@pytest.mark.parametrize('stat',['backlog','num_total_tasks','num_running_inputs'])
def test_predecessor_with_work_cannot_resume(case,stat):
    state,key,deployment,authorization,observation,now=case
    observation['functions'][0][stat]=1
    with pytest.raises(ValueError):activate(case)
    assert key+'budget-current' not in state


def test_preflight_must_be_fresh_and_queue_disabled(case):
    state,key,deployment,authorization,observation,now=case
    observation['verified_unix']=now-121
    with pytest.raises(ValueError):activate(case)
    observation['verified_unix']=now-1;state[key+'control']['enabled']=True
    with pytest.raises(ValueError):activate(case)


def test_cost_evidence_must_be_covered_and_cap_cannot_increase(case):
    state,key,deployment,authorization,observation,now=case
    authorization['prior_cost']['metered_usd']=80.01
    with pytest.raises(ValueError):activate(case)
    authorization['prior_cost']['metered_usd']=78.61
    authorization['budget_policy']['total_usd']=151
    with pytest.raises(ValueError):activate(case)
    # A raise that is consistent across the policy, the deployment and the user
    # text is still implicit without the cap_increase block and still fails.
    deployment['budget']['total_usd']=151
    authorization['user_authorization']['total_usd']=151
    with pytest.raises(ValueError):activate(case)
    assert key+'budget-current' not in state


def test_pointer_and_ancestry_tampering_fail_closed(case):
    state,key,deployment,authorization,observation,now=case
    activate(case)
    state[key+'budget']['deadline_unix']+=1
    with pytest.raises(ValueError):app.active_resume_tranche(state,key)


def test_second_authorization_cannot_reset_selected_tranche(case):
    state,key,deployment,authorization,observation,now=case
    activate(case)
    authorization['created_unix']+=1
    with pytest.raises(ValueError):activate(case)


def successor_case(case, *, prior_spend_usd=100, metered_usd=98):
    receipt=activate(case)
    state,key,deployment,authorization,observation,_=case
    now=receipt['deadline_unix']+100
    policy=budget.make_policy(prior_spend_usd=prior_spend_usd)
    authorization={**copy.deepcopy(authorization),'previous_app_id':'ap-current',
        'previous_budget_sha256':budget.digest(receipt),'budget_policy':policy,
        'prior_cost':{'metered_usd':metered_usd,'evidence_sha256':budget.digest({'metered':metered_usd,'time':now})},
        'created_unix':now-20}
    observation={**copy.deepcopy(observation),'previous_app_id':'ap-current',
        'stopped_at':now-30,'verified_unix':now-1}
    return state,key,{**deployment,'budget':policy},authorization,observation,now


@pytest.fixture
def successor(case):
    return successor_case(case)


def test_successors_preserve_all_prior_receipts_scores_and_selected_edges(successor):
    state,key,deployment,authorization,observation,now=successor
    before=copy.deepcopy(state)
    previous=app.active_resume_tranche(state,key)
    receipt=activate(successor)
    assert all(state[k]==v for k,v in before.items())
    assert receipt['allocated_fleet_usd']==40
    assert receipt['maximum_modeled_fleet_usd']+100+10<=150
    chain=app.resume_tranche_chain(state,key)
    assert len(chain)==2 and chain[0]==previous and chain[1]['activation']==receipt
    edge=state[key+'budget-successor/'+budget.digest(previous)]
    assert edge=={'authorization_sha256':budget.digest(authorization),'tranche_sha256':budget.digest(chain[1])}
    third=successor_case(successor,prior_spend_usd=120,metered_usd=118)
    activate(third)
    assert len(app.resume_tranche_chain(state,key))==3
    assert all(state[k]==v for k,v in before.items())


def test_successor_replay_and_ordinary_activation_preserve_selected_clock(successor):
    state,key,deployment,authorization,observation,now=successor
    receipt=activate(successor)
    before=copy.deepcopy(state)
    assert app.activate_resume_budget(state,key,deployment,authorization,now=now+100)==receipt
    assert app.activate_budget(state,key,deployment,now=now+100)==receipt
    with pytest.raises(ValueError,match='cannot be reset'):
        app.activate_budget(state,key,deployment,now=now+100,started_unix=now+100)
    with pytest.raises(TimeoutError):
        app.activate_resume_budget(state,key,deployment,authorization,now=receipt['deadline_unix'])
    historical=app.resume_tranche_chain(state,key)[0]['authorization']
    with pytest.raises(TimeoutError):
        app.activate_resume_budget(state,key,{**deployment,'budget':historical['budget_policy']},
            historical,now=now)
    assert state==before


@pytest.mark.parametrize('violation',[
    'live-predecessor','prior-allowance','metered-cost','total','reserve','resources',
    'app','environment','queue','ancestry','early-authorization','early-stop','stale-stop',
    'backlog','num_total_tasks','num_running_inputs','enabled'])
def test_successor_rejects_changed_bounds_or_unsafe_predecessor(successor,violation):
    state,key,deployment,authorization,observation,now=successor
    previous=app.active_resume_tranche(state,key)
    if violation=='live-predecessor':
        now=previous['activation']['deadline_unix']-1
    elif violation=='prior-allowance':
        deployment['budget']['prior_spend_usd']=79
        authorization['prior_cost']['metered_usd']=78.8
    elif violation=='metered-cost':authorization['prior_cost']['metered_usd']=78
    elif violation=='total':
        deployment['budget']['total_usd']=149
        authorization['user_authorization']['total_usd']=149
    elif violation=='reserve':deployment['budget']['reserve_usd']=11
    elif violation=='resources':deployment['budget']['worker_memory_mib']=8192
    elif violation in ('app','environment','queue'):
        field={'app':'app_name','environment':'environment_name','queue':'queue_sha256'}[violation]
        authorization[field]='c'*64 if violation=='queue' else 'other'
        deployment['snapshot_sha256' if violation=='queue' else field]=authorization[field]
    elif violation=='ancestry':authorization['previous_budget_sha256']=budget.digest(state[key+'budget'])
    elif violation=='early-authorization':authorization['created_unix']=previous['activation']['deadline_unix']-1
    elif violation=='early-stop':observation['stopped_at']=previous['activation']['started_unix']-1
    elif violation=='stale-stop':observation['verified_unix']=now-121
    elif violation=='enabled':state[key+'control']['enabled']=True
    else:observation['functions'][0][violation]=1
    before=copy.deepcopy(state)
    with pytest.raises(ValueError):
        app.activate_resume_budget(state,key,deployment,authorization,now=now,stop_observation=observation)
    assert state==before


def test_successor_accepts_predecessor_stopped_early_during_its_tranche(successor):
    state,key,deployment,authorization,observation,now=successor
    previous=app.active_resume_tranche(state,key)
    observation['stopped_at']=previous['activation']['started_unix']+1
    assert observation['stopped_at']<previous['activation']['deadline_unix']
    activation=app.activate_resume_budget(state,key,deployment,authorization,now=now,stop_observation=observation)
    assert activation['started_unix']==now
    selected=app.active_resume_tranche(state,key)
    assert selected['stop_observation']['stopped_at']==observation['stopped_at']


def test_successor_first_writer_selects_one_immutable_edge(successor,monkeypatch):
    state,key,deployment,authorization,observation,now=successor
    previous=app.active_resume_tranche(state,key)
    selection_key=key+'budget-successor/'+budget.digest(previous)
    winner_auth={**copy.deepcopy(authorization),'created_unix':authorization['created_unix']+1}
    original_put=state.put
    won=[]
    def race_put(name,value,skip_if_exists=False):
        if name==selection_key and not won:
            won.append(True)
            won.append(app.activate_resume_budget(state,key,deployment,winner_auth,
                now=now+1,stop_observation=observation))
        return original_put(name,value,skip_if_exists=skip_if_exists)
    monkeypatch.setattr(state,'put',race_put)
    with pytest.raises(ValueError,match='concurrently'):
        activate(successor)
    chain=app.resume_tranche_chain(state,key)
    assert len(chain)==2 and chain[-1]['authorization']==winner_auth
    assert chain[-1]['activation']==won[1]
    before=copy.deepcopy(dict(state))
    with pytest.raises(ValueError,match='live resume'):
        activate(successor)
    assert state==before


def test_same_authorization_race_retains_first_activation_clock(successor,monkeypatch):
    state,key,deployment,authorization,observation,now=successor
    tranche_key=key+'budget-tranches/'+budget.digest(authorization)
    original_put=state.put
    inserted=[]
    def race_put(name,value,skip_if_exists=False):
        if name==tranche_key and not inserted:
            inserted.append(True)
            inserted.append(app.activate_resume_budget(state,key,deployment,authorization,
                now=now+1,stop_observation=observation))
        return original_put(name,value,skip_if_exists=skip_if_exists)
    monkeypatch.setattr(state,'put',race_put)
    assert activate(successor)==inserted[1]
    assert app.active_resume_tranche(state,key)['activation']['started_unix']==now+1


@pytest.mark.parametrize('tamper',['receipt','edge','cycle','ancestor','cost'])
def test_successor_chain_rejects_tampered_history(successor,tamper):
    state,key,deployment,authorization,observation,now=successor
    activate(successor)
    first,last=app.resume_tranche_chain(state,key)
    edge_key=key+'budget-successor/'+budget.digest(first)
    last_key=key+'budget-tranches/'+last['authorization_sha256']
    if tamper=='receipt':state[last_key]['activation']['deadline_unix']+=1
    elif tamper=='edge':state[edge_key]['tranche_sha256']='0'*64
    elif tamper=='cycle':state[key+'budget-successor/'+budget.digest(last)]=state[key+'budget-current']
    elif tamper=='ancestor':state[key+'budget']['deadline_unix']+=1
    else:
        last=copy.deepcopy(last)
        last['authorization']['prior_cost']['metered_usd']=1
        last['authorization_sha256']=budget.digest(last['authorization'])
        state[key+'budget-tranches/'+last['authorization_sha256']]=last
        state[edge_key]={'authorization_sha256':last['authorization_sha256'],'tranche_sha256':budget.digest(last)}
    with pytest.raises(ValueError):app.resume_tranche_chain(state,key)


def test_successor_chain_limit_rejects_activation_before_writes(successor,monkeypatch):
    state,key,*_=successor
    monkeypatch.setattr(budget,'MAX_RESUME_TRANCHES',1)
    before=copy.deepcopy(state)
    with pytest.raises(ValueError,match='limit'):
        activate(successor)
    assert state==before


RAISE={'previous_total_usd':150,'total_usd':200,'instruction':'Raise the inclusive cap from $150 to $200.'}


def with_raise(case, cap_increase=RAISE, *, user_total_usd=None, prior_spend_usd=None):
    """Restate a case under an explicit raise; the policy and deployment share one dict."""
    state,key,deployment,authorization,observation,now=case
    total=cap_increase['total_usd']
    policy={**authorization['budget_policy'],'total_usd':total}
    if prior_spend_usd is not None:
        policy['prior_spend_usd']=prior_spend_usd
    authorization={**copy.deepcopy(authorization),'budget_policy':policy,
        'cap_increase':copy.deepcopy(cap_increase),
        'user_authorization':{'total_usd':total if user_total_usd is None else user_total_usd,
            'instruction':'Resume the authorized queue within the raised cap.'}}
    return state,key,{**deployment,'budget':policy},authorization,observation,now


@pytest.fixture
def raised(successor):
    return with_raise(successor)


def test_explicit_hash_bound_cap_increase_raises_successor_cap(raised,monkeypatch):
    state,key,deployment,authorization,observation,now=raised
    before=copy.deepcopy(state)
    previous=app.active_resume_tranche(state,key)
    receipt=activate(raised)
    assert all(state[k]==v for k,v in before.items())
    assert receipt['allocated_fleet_usd']==90 and receipt['prior_spend_usd']==100 and receipt['reserve_usd']==10
    assert receipt['authorized_total_usd']==200
    assert receipt['maximum_modeled_fleet_usd']+100+10<=200
    chain=app.resume_tranche_chain(state,key)
    assert len(chain)==2 and chain[0]==previous and chain[1]['activation']==receipt
    assert chain[1]['authorization']['cap_increase']==RAISE
    assert chain[1]['authorization_sha256']==budget.digest(authorization)
    assert app.activate_resume_budget(state,key,deployment,authorization,now=now+100)==receipt
    assert app.activate_budget(state,key,deployment,now=now+100)==receipt
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(deployment['budget']))
    monkeypatch.setenv('PANEL_APP_NAME',deployment['app_name'])
    monkeypatch.setenv('PANEL_STATE_NAME',deployment['state_name'])
    monkeypatch.setenv('PANEL_ENVIRONMENT','main')
    assert app.remote_budget(state,key)==(deployment['budget'],receipt)
    monkeypatch.setenv('PANEL_BUDGET_POLICY',json.dumps(budget.make_policy(prior_spend_usd=100)))
    with pytest.raises(ValueError):app.remote_budget(state,key)


def test_raised_cap_continues_and_raises_again_only_through_explicit_blocks(raised):
    state,key,*_=raised
    activate(raised)
    continued=with_raise(successor_case(raised,metered_usd=148),prior_spend_usd=150)
    receipt=activate(continued)
    assert receipt['allocated_fleet_usd']==40 and receipt['authorized_total_usd']==200
    again=with_raise(successor_case(continued,metered_usd=178),
        {'previous_total_usd':200,'total_usd':250,'instruction':'Raise the inclusive cap from $200 to $250.'},
        prior_spend_usd=180)
    receipt=activate(again)
    assert receipt['allocated_fleet_usd']==60 and receipt['authorized_total_usd']==250
    assert len(app.resume_tranche_chain(state,key))==4
    # Restating the earlier 150 to 200 block after the 250 tranche is a decrease and fails.
    stale=with_raise(successor_case(again,metered_usd=180),prior_spend_usd=185)
    before=copy.deepcopy(state)
    with pytest.raises(ValueError,match='predecessor cap'):activate(stale)
    assert state==before


def test_first_tranche_raise_binds_previous_total_to_root_policy(case):
    state,key,*_=case
    wrong=with_raise(case,{**RAISE,'previous_total_usd':140})
    before=copy.deepcopy(state)
    with pytest.raises(ValueError):activate(wrong)
    assert state==before
    receipt=activate(with_raise(case))
    assert receipt['allocated_fleet_usd']==110 and receipt['authorized_total_usd']==200


@pytest.mark.parametrize('violation',[
    'previous-total','above-limit','missing-instruction','blank-instruction','user-total',
    'decrease','equal','implicit','policy-total','extra-field','reserve','resources','deployment'])
def test_cap_increase_rejects_unbound_or_decreasing_raises(raised,violation):
    state,key,deployment,authorization,observation,now=raised
    increase=authorization['cap_increase']; policy=authorization['budget_policy']
    user=authorization['user_authorization']
    if violation=='previous-total':increase['previous_total_usd']=149
    elif violation=='above-limit':
        increase['total_usd']=budget.MAX_AUTHORIZED_TOTAL_USD+1
        policy['total_usd']=user['total_usd']=increase['total_usd']
    elif violation=='missing-instruction':del increase['instruction']
    elif violation=='blank-instruction':increase['instruction']='   '
    elif violation=='user-total':user['total_usd']=150
    elif violation=='decrease':
        increase['total_usd']=140;policy['total_usd']=user['total_usd']=140
    elif violation=='equal':
        increase['total_usd']=150;policy['total_usd']=user['total_usd']=150
    elif violation=='implicit':del authorization['cap_increase']
    elif violation=='policy-total':policy['total_usd']=user['total_usd']=201
    elif violation=='extra-field':increase['note']='extra'
    elif violation=='reserve':policy['reserve_usd']=11
    elif violation=='resources':policy['worker_memory_mib']=8192
    else:deployment['budget']=budget.make_policy(prior_spend_usd=100)
    before=copy.deepcopy(state)
    with pytest.raises(ValueError):
        app.activate_resume_budget(state,key,deployment,authorization,now=now,stop_observation=observation)
    assert state==before


def test_resume_cli_preflight_precedes_redeploy_and_writes_no_provider_budget():
    source=inspect.getsource(app)
    cli=source[source.index("if __name__=='__main__':"):]
    assert cli.index('activate_resume_budget(')<cli.index('app.deploy(')<cli.index('budget=activate_budget(')
    assert cli.index('validate_resume_authorization(')<cli.index('configure(')
    observer=inspect.getsource(budget.observe_stopped_app)
    assert 'AppGetLifecycle(' in observer and 'FunctionGetCurrentStats(' in observer
    assert 'SetBudget' not in observer and 'AppStop' not in observer


def test_activation_freezes_strict_phase_and_all_unresolved_holds_atomically(case):
    state,key,deployment,authorization,observation,now=case
    activate(case)
    games=[{'label':label,'initial_status':'pending'} for label in ('pending','accepted','quarantined','unheld')]
    for label in ('pending','accepted','quarantined'):
        state[key+'hold/'+label]={'reason':'retained'}
    state[key+'result/accepted']={'status':'complete'}
    state[key+'result/quarantined']={'status':'quarantined'}
    control=app.activation_control(state,key,{'games':games},image_id='im-new',
        strict_phase_order=True,reinspect_held=True)
    assert control['enabled'] is True and control['strict_phase_order'] is True
    assert control['allowlist'] is None
    assert control['held_reinspection']['labels']==['pending']
    assert control['held_reinspection']['tranche_sha256']==budget.digest(app.active_resume_tranche(state,key))
    assert state[key+'hold/pending']=={'reason':'retained'}


def test_live_preflight_timestamp_is_taken_after_network_read(case,monkeypatch):
    state,key,deployment,authorization,observation,now=case
    ticks=iter([now,now+10])
    monkeypatch.setattr(app.time,'time',lambda:next(ticks))
    observation['verified_unix']=now+9
    monkeypatch.setattr(app,'observe_stopped_app',lambda auth:observation)
    receipt=app.activate_resume_budget(state,key,deployment,authorization)
    assert receipt['started_unix']==now+10


def test_live_sdk_stop_probe_uses_stopped_app_layout_and_binds_functions(case,monkeypatch):
    pytest.importorskip("modal", reason="optional Modal SDK is required for this SDK contract test")
    from modal.client import _Client
    from modal_proto import api_pb2
    state,key,deployment,authorization,observation,now=case
    calls=[]
    class Stub:
        async def AppGetLifecycle(self,request):
            calls.append(('lifecycle',request.app_id))
            return api_pb2.AppGetLifecycleResponse(lifecycle=api_pb2.AppLifecycle(
                app_state=api_pb2.APP_STATE_STOPPED,stopped_at=now-30))
        async def AppGetLayout(self,request):
            calls.append(('layout',request.app_id))
            mapping={tag:'fu-'+tag.replace('_','') for tag in ('run_cloud_game','dispatch_cloud')}
            return api_pb2.AppGetLayoutResponse(app_layout=api_pb2.AppLayout(function_ids=mapping,
                objects=[api_pb2.Object(object_id=function_id,
                    function_handle_metadata=api_pb2.FunctionHandleMetadata(app_id='ap-previous'))
                    for function_id in mapping.values()]))
        async def FunctionGetCurrentStats(self,request):
            calls.append(('stats',request.function_id))
            return api_pb2.FunctionStats()
    async def from_env(cls):return types.SimpleNamespace(stub=Stub())
    monkeypatch.setattr(_Client,'from_env',classmethod(from_env))
    monkeypatch.setenv('MODAL_PROFILE','frisson')
    monkeypatch.setattr(budget.time,'time',lambda:now)
    actual=asyncio.run(budget.observe_stopped_app(authorization))
    assert actual['app_state']=='STOPPED' and len(actual['functions'])==2
    assert len(calls)==4
    assert ('layout','ap-previous') in calls
    authorization['previous_app_id']='ap-different'
    with pytest.raises(ValueError,match='different deployment'):
        asyncio.run(budget.observe_stopped_app(authorization))
