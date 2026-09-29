import copy
import hashlib
from pathlib import Path
import sys
import types
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import modal_panel_cloud_queue as c


class Store(dict):
    def put(self,key,value,skip_if_exists=False):
        if skip_if_exists and key in self:return False
        self[key]=copy.deepcopy(value);return True


def row(label,phase=0,slot=0,initial='pending'):
    return {'label':label,'phase':phase,'worker_slot':slot,'initial_status':initial}


def test_original_slot_order_and_phase_barrier():
    snapshot={'games':[row('accepted',initial='complete'),row('a'),row('b'),row('c',slot=1),row('later',phase=1)]}
    assert [r['label'] for r in c.eligible(snapshot,{})]==['a','c']
    records={'a':{'status':'complete'}}
    assert [r['label'] for r in c.eligible(snapshot,records)]==['b','c']
    records.update(b={'status':'complete'},c={'status':'quarantined'})
    assert [r['label'] for r in c.eligible(snapshot,records)]==['later']


def test_running_and_failed_calls_keep_their_original_slot():
    snapshot={'games':[row('first'),row('second'),row('other',slot=1)]}
    for status in ('running','requires-review','external-wait'):
        assert [r['label'] for r in c.eligible(snapshot,{'first':{'status':status}})]==['first','other']


def test_complete_original_inventory_never_replays():
    snapshot={'games':[row('a',initial='complete'),row('b',initial='quarantined'),row('c')]}
    assert c.eligible(snapshot,{'c':{'status':'complete'}})==[]


def test_any_hold_releases_slot_and_phase_without_scoring():
    snapshot={'games':[row('held'),row('next'),row('later',phase=1)]}
    records={}
    assert [r['label'] for r in c.eligible(snapshot,records,holds={'held'})]==['next']
    assert [r['label'] for r in c.eligible(snapshot,records,holds={'held','next'})]==['later']
    assert c.eligible(snapshot,records,holds={'held','next','later'})==[]
    assert records=={}


def test_strict_phase_keeps_held_supported_open_and_runs_available_same_phase():
    snapshot={'games':[row('held-supported'),row('supported-next'),
        row('supported-independent',slot=1),row('extended',phase=1),row('forced',phase=2)]}
    before=copy.deepcopy(snapshot);records={}
    selected=c.eligible(snapshot,records,holds={'held-supported'},strict_phase_order=True)
    assert [r['label'] for r in selected]==['supported-next','supported-independent']
    records.update({'supported-next':{'status':'complete'},'supported-independent':{'status':'complete'}})
    assert c.eligible(snapshot,records,holds={'held-supported'},strict_phase_order=True)==[]
    assert [r['label'] for r in c.eligible(snapshot,records,holds={'held-supported'})]==['extended']
    assert snapshot==before
    assert records=={'supported-next':{'status':'complete'},'supported-independent':{'status':'complete'}}


def test_strict_phase_advances_supported_extended_forced_then_next_model():
    snapshot={'games':[row('supported'),row('extended',phase=1),row('forced',phase=2),
        row('next-model-supported',phase=3),row('next-model-extended',phase=4),
        row('next-model-forced',phase=5)]}
    records={}
    for game in snapshot['games']:
        assert [r['label'] for r in c.eligible(snapshot,records,strict_phase_order=True)]==[game['label']]
        assert c.eligible(snapshot,records,holds={game['label']},strict_phase_order=True)==[]
        records[game['label']]={'status':'complete'}
    assert c.eligible(snapshot,records,strict_phase_order=True)==[]


def test_strict_phase_preserves_already_completed_later_blocks():
    snapshot={'games':[row('supported'),row('extended',phase=1),row('forced',phase=2),
        row('next-model',phase=3)]}
    records={'extended':{'status':'complete','result':{'win':True}},
        'forced':{'status':'complete','result':{'win':False}}}
    before=copy.deepcopy(records)
    assert c.eligible(snapshot,records,holds={'supported'},strict_phase_order=True)==[]
    assert records==before
    records['supported']={'status':'complete'}
    assert [r['label'] for r in c.eligible(snapshot,records,strict_phase_order=True)]==['next-model']
    assert records['extended']==before['extended'] and records['forced']==before['forced']


def test_strict_phase_keeps_sixty_four_slots_and_respects_requested_limit():
    snapshot={'games':[row(f'{n}-p{p}',slot=n) for n in range(64) for p in (1,2)]}
    holds={f'{n}-p1' for n in range(64)}
    selected=c.eligible(snapshot,{},holds=holds,strict_phase_order=True)
    assert len(selected)==64 and len({r['worker_slot'] for r in selected})==64
    assert all(r['label'].endswith('p2') for r in selected)
    assert len(c.eligible(snapshot,{},holds=holds,strict_phase_order=True,limit=3))==3


@pytest.mark.parametrize('value',[None,1,0,'true','false',[],{}])
def test_strict_phase_option_requires_boolean(value):
    with pytest.raises(ValueError,match='strict_phase_order must be boolean'):
        c.eligible({'games':[]},{},strict_phase_order=value)


def test_sixty_four_slot_cap_and_second_port_order():
    snapshot={'games':[row(f'{n}-p{p}',slot=n) for n in range(64) for p in (1,2)]}
    assert len(c.eligible(snapshot,{}))==64
    assert all(r['label'].endswith('p1') for r in c.eligible(snapshot,{}))


def test_all_first_ports_held_preserves_sixty_four_cap():
    snapshot={'games':[row(f'{n}-p{p}',slot=n) for n in range(64) for p in (1,2)]}
    holds={f'{n}-p1' for n in range(64)}
    selected=c.eligible(snapshot,{},holds=holds)
    assert len(selected)==64
    assert len({r['worker_slot'] for r in selected})==64
    assert all(r['label'].endswith('p2') for r in selected)


def test_lost_spawn_reply_can_repeat_without_second_game_owner():
    store=Store()
    assert c.bind_call(store,'logical','fc-original',100,'im-frozen')
    assert not c.bind_call(store,'logical','fc-duplicate',100,'im-frozen')
    assert c.bind_call(store,'logical','fc-original',100,'im-frozen')


def test_original_expiry_and_image_cannot_change_on_retry():
    store=Store();assert c.bind_call(store,'logical','fc-original',100,'im-frozen')
    assert not c.bind_call(store,'logical','fc-original',200,'im-frozen')
    assert not c.bind_call(store,'logical','fc-original',100,'im-other')


def test_blob_rejects_unapproved_path_before_copy(tmp_path):
    body=b'input';sha=hashlib.sha256(body).hexdigest();(tmp_path/sha).write_bytes(body)
    row={'sha256':sha,'bytes':len(body),'remote':'/etc/unapproved','local':'/etc/unapproved'}
    with pytest.raises(ValueError,match='unapproved'):
        c.checked_materialize(row,tmp_path)


def test_blob_content_identity_is_verified(tmp_path):
    sha='a'*64;(tmp_path/sha).write_bytes(b'bad')
    row={'sha256':sha,'bytes':3,'remote':'/opt/a','local':'/opt/a'}
    with pytest.raises(ValueError,match='hash'):
        c.checked_materialize(row,tmp_path)


def test_failed_native_game_never_becomes_score():
    d=types.SimpleNamespace(MOUNT=Path('/mock'),SUCCESS='success',check_manifest=lambda *a:None)
    m={'storage_path':'x','plan_sha256':'a'*64,'files':[],'native_schema':'native','status':'failed'}
    with pytest.raises(ValueError,match='failed native'):
        c.native_acceptance(d,object(),{},m)


def test_native_validator_is_required_before_acceptance():
    d=types.SimpleNamespace(MOUNT=Path('/mock'),SUCCESS='success',check_manifest=lambda *a:None)
    m={'storage_path':'x','plan_sha256':'a'*64,'files':[],'native_schema':'native','status':'success'}
    def reject(*args):raise ValueError('native checks reject')
    with pytest.raises(ValueError,match='native checks reject'):
        c.native_acceptance(d,types.SimpleNamespace(validate_terminal=reject),{},m)


@pytest.fixture
def dispatcher(monkeypatch):
    import modal_panel_cloud_app as app
    store=Store();sha='a'*64
    monkeypatch.setenv('PANEL_SNAPSHOT_SHA',sha)
    queue={'games':[row('a'),row('b'),row('c',slot=1),row('phase2',phase=1)],'queue_deadline_unix':float('inf')}
    monkeypatch.setattr(app,'snapshot',lambda:queue)
    monkeypatch.setattr(app,'handles',lambda:store)
    monkeypatch.setattr(app,'frozen_game_session_seconds',lambda row:16290)
    store[sha+'/control']={'enabled':True,'image_id':'im-test'}
    calls={};launched=[]
    class Call:
        def __init__(self,call_id):self.object_id=call_id
        def get(self,timeout=0):
            result=calls[self.object_id]
            if result is None:raise TimeoutError()
            if isinstance(result,Exception):raise result
            return result
    def spawn(label,*args,**kwargs):
        call=Call('fc-'+str(len(launched)))
        launched.append((label,(args,kwargs),call.object_id));calls[call.object_id]=None
        return call
    modal=types.SimpleNamespace(Function=types.SimpleNamespace(from_name=lambda *a,**k:types.SimpleNamespace(spawn=spawn)),
        FunctionCall=types.SimpleNamespace(from_id=Call))
    monkeypatch.setitem(sys.modules,'modal',modal)
    return app,store,calls,launched,sha,queue


def test_periodic_ticks_refill_without_local_client(dispatcher):
    app,store,calls,launched,sha,_=dispatcher
    app.dispatch_cloud();assert [r[0] for r in launched]==['a','c']
    app.dispatch_cloud();assert len(launched)==2
    store[sha+'/result/a']={'status':'complete'}
    app.dispatch_cloud();assert [r[0] for r in launched]==['a','c','b']
    store[sha+'/result/b']={'status':'complete'};store[sha+'/result/c']={'status':'complete'}
    app.dispatch_cloud();assert launched[-1][0]=='phase2'


def test_disabled_deployment_cannot_launch(dispatcher):
    app,store,_,launched,sha,_=dispatcher
    store[sha+'/control']={'enabled':False}
    assert app.dispatch_cloud()['status']=='deployed-awaiting-activation'
    assert launched==[]


def test_dispatch_control_enforces_strict_phase_without_blocking_same_phase(dispatcher):
    app,store,_,launched,sha,_=dispatcher
    store[sha+'/control']['strict_phase_order']=True
    for label in ('a','b','c'):
        store[sha+'/hold/'+label]={'error':'retained unresolved game'}
    app.dispatch_cloud()
    assert launched==[]
    assert not any(key.startswith(sha+'/result/') for key in store)
    del store[sha+'/hold/b']
    app.dispatch_cloud()
    assert [item[0] for item in launched]==['b']
    for label in ('a','b','c'):
        store[sha+'/result/'+label]={'status':'complete'}
    app.dispatch_cloud()
    assert [item[0] for item in launched]==['b','phase2']


def test_dispatch_rejects_nonboolean_strict_phase_control_before_submission(dispatcher):
    app,store,_,launched,sha,_=dispatcher
    store[sha+'/control']['strict_phase_order']='true'
    with pytest.raises(ValueError,match='strict_phase_order must be boolean'):
        app.dispatch_cloud()
    assert launched==[]


def test_faulted_owned_call_keeps_error_without_duplicate_gameplay(dispatcher):
    app,store,calls,launched,sha,_=dispatcher
    app.dispatch_cloud();label,args,call=launched[0]
    store[sha+'/owner/'+label]={'call_id':call}
    calls[call]=RuntimeError('three physical attempts exhausted')
    health=app.dispatch_cloud()
    assert len(launched)==3 and launched[-1][0]=='a'
    assert launched[-1][1][1]=={'recover_only':True,'attempt':1}
    assert health['submitted'][0]['read_only_recovery'] is True


def test_exhausted_recovery_defers_game_and_continues_through_phase(dispatcher):
    app,store,calls,launched,sha,_=dispatcher
    app.dispatch_cloud();call=launched[0][2]
    store[sha+'/owner/a']={'call_id':call}
    calls[call]=RuntimeError('arbitrary unresolved scientific or infrastructure failure')
    store[sha+'/recovery/a']={'submissions':3}
    health=app.dispatch_cloud()
    assert health['held']==1
    assert store[sha+'/hold/a']['owner']=={'call_id':call}
    assert sha+'/result/a' not in store
    app.dispatch_cloud()
    assert launched[-1][0]=='b'
    store[sha+'/result/b']={'status':'complete'}
    store[sha+'/result/c']={'status':'complete'}
    app.dispatch_cloud()
    assert launched[-1][0]=='phase2'
    store[sha+'/result/phase2']={'status':'complete'}
    assert app.dispatch_cloud()['status']=='drained-with-unresolved-holds'
    assert sum(label=='a' for label,_,_ in launched)==1


@pytest.mark.parametrize('failure',['owned-return','ambiguous-dispatch','lookup-error'])
def test_every_per_game_hold_leaves_other_candidates_running(dispatcher,failure):
    app,store,calls,launched,sha,_=dispatcher
    if failure=='ambiguous-dispatch':
        store[sha+'/dispatch/a']={'submissions':3,'submitted_unix':0}
    else:
        store[sha+'/owner/a']={'call_id':'fc-missing'}
        if failure=='owned-return':calls['fc-missing']={'status':'unexpected'}
        # Lookup raises, followed by absent dispatch expiry in recovery. The
        # outer per-game guard must preserve this failure and continue to c.
    health=app.dispatch_cloud()
    assert health['held']==1 and launched[-1][0]=='c'
    assert store[sha+'/hold/a']['score_admitted'] is False
    app.dispatch_cloud()
    assert launched[-1][0]=='b'


def test_live_owner_timeout_is_still_observed_without_duplicate_launch(dispatcher):
    app,store,calls,launched,sha,_=dispatcher
    app.dispatch_cloud()
    for _ in range(3):app.dispatch_cloud()
    assert len(launched)==2
    assert sha+'/hold/a' not in store


def test_hold_with_later_validated_result_no_longer_counts_as_unresolved(dispatcher):
    app,store,_,_,sha,_=dispatcher
    store[sha+'/hold/a']={'error':'retained history'}
    store[sha+'/result/a']={'status':'complete'}
    assert app.dispatch_cloud()['held']==0


def test_qualification_allowlist_prevents_unapproved_launches(dispatcher):
    app,store,_,launched,sha,_=dispatcher
    store[sha+'/control']['allowlist']=['c']
    app.dispatch_cloud();assert [r[0] for r in launched]==['c']


def test_adoption_of_existing_call_does_not_launch_follower_early(dispatcher):
    app,store,calls,launched,sha,queue=dispatcher
    queue['games'][0]['external_submission']={'call_id':'fc-legacy'}
    app.dispatch_cloud();calls[launched[0][2]]={'status':'external-call-still-running'}
    app.dispatch_cloud();assert [r[0] for r in launched]==['a','c','a']
    assert all(r[0]!='b' for r in launched)


def test_cloud_deadline_stops_spending(dispatcher):
    app,_,_,launched,_,queue=dispatcher
    queue['queue_deadline_unix']=0
    assert app.dispatch_cloud()['status']=='queue-deadline-reached'
    assert launched==[]


def test_dispatcher_ledger_read_budget_keeps_singleton_and_worker_limits():
    import inspect,modal_panel_cloud_app as app
    source=inspect.getsource(app.configure)
    assert 'max_containers=1,min_containers=0,buffer_containers=0,timeout=900' in source
    assert 'max_containers=64' in source
    assert "timeout=worker_time_limits(deployment)['function_seconds']" in source
    assert app.worker_time_limits({})['function_seconds']==15990
    assert app.worker_time_limits({'durable_module':'faynt_d0_durable_game'})['function_seconds']==8790


def test_new_source_directory_needs_import_cache_invalidation(tmp_path):
    import importlib
    directory=tmp_path/'not-yet-materialized';name='e011_new_source_fixture'
    sys.path.insert(0,str(directory))
    try:
        with pytest.raises(ModuleNotFoundError):importlib.import_module(name)
        assert sys.path_importer_cache.get(str(directory)) is None
        directory.mkdir();(directory/(name+'.py')).write_text('VALUE=41\n')
        importlib.invalidate_caches()
        assert importlib.import_module(name).VALUE==41
    finally:
        sys.path.remove(str(directory));sys.modules.pop(name,None)


def test_native_validator_receives_canonical_volume_paths(tmp_path):
    actual=tmp_path/'actual';actual.mkdir();mount=tmp_path/'mount';mount.symlink_to(actual,target_is_directory=True)
    root=actual/'sha'/'execution';(root/'files/game-0').mkdir(parents=True)
    (root/'files/game-0/result.json').write_text('{"native_acceptance":{"win":true}}')
    d=types.SimpleNamespace(MOUNT=mount,SUCCESS='success',check_manifest=lambda *a:None)
    manifest={'storage_path':'sha/execution','plan_sha256':'a'*64,'native_schema':'native','status':'success',
              'files':[{'name':'game-0/result.json'}]}
    def validate(native,files,*args):
        assert all(p==p.resolve(strict=True) for p in files.values())
    bound=types.SimpleNamespace(validate_terminal=validate)
    plan={'scope':{'selected_game_indices':[0]},'artifact_labels':['selected','peer']}
    assert c.native_acceptance(d,bound,plan,manifest)=={'win':True}


def test_adoption_preserves_original_plan_serialization_and_sha():
    import json
    original=c.encoded({'scope':{'seed':123},'games':['a','b']})
    packaged=(json.dumps(json.loads(original),sort_keys=True,indent=2)+'\n').encode()
    row={'plan_sha256':c.digest(packaged),'existing_claim':{'plan_sha256':c.digest(original)},
         'external_submission':{'plan_sha256':c.digest(original)}}
    plan,body,sha=c.execution_plan(row,packaged)
    assert body==original and sha==c.digest(original) and plan['scope']=={'seed':123}


def test_adoption_refuses_different_original_plan():
    original=c.encoded({'seed':123})
    row={'plan_sha256':c.digest(original),'existing_claim':{'plan_sha256':'a'*64},
         'external_submission':{'plan_sha256':'a'*64}}
    with pytest.raises(ValueError,match='cannot be reconstructed'):c.execution_plan(row,original)


@pytest.fixture
def runtime_timeout(tmp_path):
    import json
    root=tmp_path/'failure';(root/'files/runtime').mkdir(parents=True);(root/'files/supervisor').mkdir()
    values={
        'receipt.json':{'status':'failed','games':[],'fresh_runtime_status':'failed','error':'ValueError: fresh runtime failed'},
        'runtime/receipt.json':{'status':'failed','error':'TimeoutError: total build deadline exhausted',
            'scope':{'gameplay_admitted':False,'dolphin_launches':0,'policy_inference':0,'policy_instances':0,'rom_reads':0},
            'commands':[{'error':'TimeoutError: total build deadline exhausted','exit_code':-15,
                'terminated_owned_process_group':True,'unreaped_leader_group_guard':True,
                'command':['python','-m','pip','install']}]},
        'supervisor/receipt.json':{'terminated_owned_process_group':True,'unreaped_leader_group_guard':True}}
    for name,value in values.items():(root/'files'/name).write_text(json.dumps(value))
    return {'status':'failed','files':[{'name':n} for n in values]},root,values


def test_runtime_timeout_proves_zero_gameplay(runtime_timeout):
    manifest,root,_=runtime_timeout
    assert c.zero_runtime_timeout(manifest,root)
    manifest['files'].append({'name':'game-0/result.json'})
    assert not c.zero_runtime_timeout(manifest,root)


@pytest.mark.parametrize('field',['dolphin_launches','policy_inference','policy_instances','rom_reads'])
def test_runtime_timeout_refuses_any_game_or_policy_activity(runtime_timeout,field):
    import json
    manifest,root,values=runtime_timeout
    values['runtime/receipt.json']['scope'][field]=1
    (root/'files/runtime/receipt.json').write_text(json.dumps(values['runtime/receipt.json']))
    assert not c.zero_runtime_timeout(manifest,root)


@pytest.mark.parametrize('reason',['verified-zero-gameplay-runtime-install-timeout',
    'verified-zero-policy-frame-enet-startup-disconnect',
    'reviewed-countdown-enet-disconnect','reviewed-truncated-terminal-replay'])
def test_retry_changes_only_selected_attempt_label(reason):
    original={'artifact_labels':['game-a1','peer-a1'],'scope':{'selected_game_indices':[0]},'seed':31}
    body=c.encoded(original);row={'label':'game','plan_sha256':c.digest(body)}
    replacement=copy.deepcopy(original);replacement['artifact_labels'][0]='game-a2'
    auth={'queue_plan_sha256':row['plan_sha256'],'original_execution_plan_sha256':row['plan_sha256'],
        'queue_plan_body':body.decode(),'reason':reason,
        'attempt_number':2,'plan_sha256':c.digest(c.encoded(replacement)),
        'prior_failures':[{'physical_history':[{'execution':{'execution_id':'first'}}]}]}
    plan,_,_=c.retry_execution_plan(row,body,auth)
    assert plan['artifact_labels']==['game-a2','peer-a1'] and plan['seed']==31
    auth['attempt_number']=4
    with pytest.raises(ValueError,match='three-attempt'):c.retry_execution_plan(row,body,auth)


def test_scheduled_retry_retains_absolute_deadline(dispatcher):
    app,store,calls,launched,sha,_=dispatcher
    store[sha+'/retry-latest/a']=2
    store[sha+'/retry/a/a2']={'expires_unix':12345}
    app.dispatch_cloud()
    call=next(item for item in launched if item[0]=='a')
    assert call[1][0][1]==12345 and call[1][1]['attempt']==2


@pytest.fixture
def menu_disconnect(tmp_path):
    import json
    prefix='project/artifacts/integration/frisson_ai/game-a1/'
    child={'status':'failed','error':'RuntimeError: EnetDisconnected: '}
    diagnostics={'frames_total':0,'frames_in_generation':0,'current_frame_inference_barriers':0,
        'first_frame':None,'last_frame':None}
    values={
        'receipt.json':{'status':'failed','error':'ValueError: native game acceptance failed',
            'games':[{'index':0,'child':child,'cleanup':{'parent_terminal_cleanup_verified':True}}]},
        'game-0/result.json':child,
        prefix+'summary.json':{'error':'EnetDisconnected: ',
            'execution':{'processed_policy_frames':0,'first_game_frame':None,'last_game_frame':None},
            'policies':{p:{'diagnostics':copy.deepcopy(diagnostics)} for p in ('p1','p2')}},
        'supervisor/receipt.json':{'exit_code':0,'terminated_owned_process_group':True,
            'unreaped_leader_group_guard':True}}
    for name,value in values.items():
        path=tmp_path/'files'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    trace=tmp_path/'files'/prefix/'controller_trace.jsonl';trace.write_bytes(b'')
    manifest={'status':'failed','scope':{'selected_game_indices':[0]},
        'files':[{'name':n,'bytes':(tmp_path/'files'/n).stat().st_size} for n in values]+[
            {'name':prefix+'controller_trace.jsonl','bytes':0}]}
    return manifest,tmp_path,values,prefix


def test_menu_disconnect_proves_zero_frames_and_cleanup(menu_disconnect):
    manifest,root,_,_=menu_disconnect
    assert c.zero_menu_disconnect(manifest,root)
    assert c.zero_gameplay_retry_reason(manifest,root)=='verified-zero-policy-frame-enet-startup-disconnect'


@pytest.mark.parametrize('mutation',['frame','policy','trace','replay','cleanup','supervisor','other-error','missing-proof','peer'])
def test_menu_disconnect_rejects_incomplete_or_gameplay_evidence(menu_disconnect,mutation):
    import json
    manifest,root,values,prefix=menu_disconnect
    if mutation=='frame':values[prefix+'summary.json']['execution']['processed_policy_frames']=1
    if mutation=='policy':values[prefix+'summary.json']['policies']['p2']['diagnostics']['frames_total']=1
    if mutation=='trace':(root/'files'/prefix/'controller_trace.jsonl').write_bytes(b'frame')
    if mutation=='replay':manifest['files'].append({'name':prefix+'replays/game.slp','bytes':0})
    if mutation=='cleanup':values['receipt.json']['games'][0]['cleanup']['parent_terminal_cleanup_verified']=False
    if mutation=='supervisor':values['supervisor/receipt.json']['terminated_owned_process_group']=False
    if mutation=='other-error':values[prefix+'summary.json']['error']='Policy inference failure'
    if mutation=='missing-proof':manifest['files']=[r for r in manifest['files'] if r['name']!='supervisor/receipt.json']
    if mutation=='peer':manifest['files'].append({'name':'game-1/result.json','bytes':0})
    for name,value in values.items():(root/'files'/name).write_text(json.dumps(value))
    assert not c.zero_menu_disconnect(manifest,root)


def test_prior_failure_uses_its_own_exact_attempt_labels():
    original={'artifact_labels':['game-a1','peer-a1'],'scope':{'selected_game_indices':[0]},'seed':31}
    body=c.encoded(original);row={'label':'game','plan_sha256':c.digest(body)}
    second=copy.deepcopy(original);second['artifact_labels'][0]='game-a2'
    third=copy.deepcopy(original);third['artifact_labels'][0]='game-a3'
    auth={'queue_plan_sha256':row['plan_sha256'],'original_execution_plan_sha256':row['plan_sha256'],
        'queue_plan_body':body.decode(),'reason':'verified-zero-policy-frame-enet-startup-disconnect',
        'attempt_number':3,'plan_sha256':c.digest(c.encoded(third)),
        'prior_failures':[
            {'manifest':{'plan_sha256':row['plan_sha256']},'physical_history':[{'execution':{'execution_id':'first'}}]},
            {'manifest':{'plan_sha256':c.digest(c.encoded(second))},'physical_history':[{'execution':{'execution_id':'second'}}]}]}
    assert c.prior_execution_plan(row,body,auth,0)==(original,body,c.digest(body))
    assert c.prior_execution_plan(row,body,auth,1)==(second,c.encoded(second),c.digest(c.encoded(second)))
    auth['prior_failures'][0]['manifest']['plan_sha256']=auth['plan_sha256']
    with pytest.raises(ValueError,match='prior failure changed'):
        c.prior_execution_plan(row,body,auth,0)


def test_prior_validator_binding_is_attempt_specific(menu_disconnect):
    import modal_panel_cloud_app as app
    import inspect
    source=inspect.getsource(app._run_cloud_game)
    assert 'd.plan_binding(old_path,verified_old_sha,remote=True)' in source
    assert 'd.check_manifest(old,old_sha,old_root,old_bound)' in source
    assert 'd.check_manifest(old,old_sha,old_root,bound)' not in source


def test_reviewed_failure_requires_exact_retained_manifest(menu_disconnect,monkeypatch):
    manifest,root,_,_=menu_disconnect
    # Move outside the generic zero-frame classifier to exercise explicit review.
    manifest['files'].append({'name':'retained.slp','bytes':8,'sha256':'a'*64})
    reason='reviewed-truncated-terminal-replay'
    monkeypatch.setattr(c,'REVIEWED_FAILURE_RETRIES',{c.digest(manifest):reason})
    assert c.verified_failure_retry_reason(manifest,root)==reason
    manifest['files'][-1]['sha256']='b'*64
    assert c.verified_failure_retry_reason(manifest,root) is None


@pytest.mark.parametrize('field',['call_id','plan_sha256','execution','scope','status'])
def test_reviewed_failure_does_not_generalize_to_other_attempts(menu_disconnect,monkeypatch,field):
    manifest,root,_,_=menu_disconnect
    manifest['files'].append({'name':'retained.slp','bytes':8,'sha256':'a'*64})
    monkeypatch.setattr(c,'REVIEWED_FAILURE_RETRIES',{
        c.digest(manifest):'reviewed-countdown-enet-disconnect'})
    manifest[field]='changed'
    assert c.verified_failure_retry_reason(manifest,root) is None


def test_retry_review_follows_native_validation_and_does_not_score_failed_attempt():
    import inspect
    import modal_panel_cloud_app as app
    source=inspect.getsource(app._run_cloud_game)
    assert source.index('d.check_manifest(old,old_sha,old_root,old_bound)') < source.index(
        'cloud.verified_failure_retry_reason(old,old_root)')
    assert source.index('d.check_manifest(manifest,plan_sha,root,bound)') < source.index(
        'cloud.verified_failure_retry_reason(manifest,root)')
    assert "'score_admitted':False" in source
    assert "'gameplay_started':False" not in source


def exclusion_fixture():
    original={'artifact_labels':['a-a1','peer-a1'],'scope':{'selected_game_indices':[0]}}
    body=c.encoded(original);definition={'label':'a','initial_status':'pending',
        'initial_row':{'status':'pending','result':None},'plan_sha256':c.digest(body)}
    second=copy.deepcopy(original);second['artifact_labels'][0]='a-a2';second_sha=c.digest(c.encoded(second))
    retry={'queue_plan_sha256':c.digest(body),'original_execution_plan_sha256':c.digest(body),
        'queue_plan_body':body.decode(),'reason':'reviewed-countdown-enet-disconnect',
        'attempt_number':2,'plan_sha256':second_sha,'expires_unix':100,
        'prior_failures':[{'physical_history':[{'execution':{'execution_id':'first'}}]}]}
    failures=[]
    for number,identity in enumerate(('first','second','third'),1):
        manifest={'status':'failed','plan_sha256':c.digest(body) if number==1 else second_sha,
            'physical_number':number,'execution':{'execution_id':identity},
            'call_id':'fc-first' if number==1 else 'fc-retry'}
        failures.append({'manifest':manifest,'physical_start':{
            'execution':manifest['execution'],'call_id':manifest['call_id'],'expires_unix':100}})
    retry['prior_failures']=[{'manifest':copy.deepcopy(failures[0]['manifest']),
        'physical_history':[copy.deepcopy(failures[0]['physical_start'])]}]
    snapshot={'games':[definition]};queue_sha=c.digest(snapshot)
    receipt={'schema':c.SCHEMA+'.infrastructure-exclusion','status':'quarantined','label':'a',
        'queue_sha256':queue_sha,'queue_plan_sha256':c.digest(body),
        'user_authorization':'Record the failed infrastructure attempt and continue.',
        'reason':'infrastructure-failure-three-starts-exhausted','retry_authorization':retry,
        'failed_attempts':failures}
    return snapshot,definition,receipt,queue_sha


def test_explicit_exclusion_projects_no_score_and_releases_original_slot():
    import report_modal_cloud_queue as report
    snapshot,definition,receipt,sha=exclusion_fixture()
    assert c.validate_exclusion(definition,receipt,sha)==receipt
    projected=report.project(snapshot,{'a':receipt})
    assert projected['a']['status']=='quarantined' and projected['a']['result'] is None
    assert c.eligible({'games':[row('a'),row('b')]},{'a':receipt})[0]['label']=='b'


@pytest.mark.parametrize('mutation',['scored','protected','missing-attempt','duplicate','deadline','plan','approval'])
def test_exclusion_rejects_score_or_changed_provenance(mutation):
    _,definition,receipt,sha=exclusion_fixture()
    if mutation=='scored':receipt['native_acceptance']={'win':True}
    if mutation=='protected':definition['initial_status']='complete'
    if mutation=='missing-attempt':receipt['failed_attempts'].pop()
    if mutation=='duplicate':receipt['failed_attempts'][2]=copy.deepcopy(receipt['failed_attempts'][1])
    if mutation=='deadline':receipt['failed_attempts'][2]['physical_start']['expires_unix']=101
    if mutation=='plan':receipt['failed_attempts'][2]['manifest']['plan_sha256']='b'*64
    if mutation=='approval':receipt['user_authorization']=''
    with pytest.raises(ValueError):c.validate_exclusion(definition,receipt,sha)


def test_non_strict_order_fills_idle_lanes_from_later_phases():
    def lane(label,phase,slot):
        return {'label':label,'initial_status':'pending','phase':phase,'worker_slot':slot}
    snapshot={'games':[lane('a2',2,1),lane('b2',2,1),lane('c3',3,1),lane('d3',3,2),lane('e3',3,3)]}
    assert [r['label'] for r in c.eligible(snapshot,{},strict_phase_order=False)]==['a2','d3','e3']
    assert [r['label'] for r in c.eligible(snapshot,{},strict_phase_order=True)]==['a2']
    assert [r['label'] for r in c.eligible(snapshot,{},holds={'a2'},strict_phase_order=False)]==['b2','d3','e3']
    assert [r['label'] for r in c.eligible(snapshot,{},holds={'a2','b2'},strict_phase_order=True)]==[]
