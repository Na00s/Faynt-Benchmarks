"""Pure namespace, fresh-ledger and terminal scheduling checks; no cloud calls."""
import json
import copy
import sys
import types

import pytest

import modal_panel_cloud_app as app
import modal_panel_cloud_queue as cloud
import report_modal_cloud_queue as reporting


NAMES = {'app_name': 'faynt-d0-repeat-v1', 'state_name': 'faynt-d0-repeat-state-v1'}


def test_legacy_names_remain_defaults():
    assert app.deployment_names({}) == {'app_name': app.APP_NAME, 'state_name': app.STATE_NAME}
    assert app.deployment_worker({}) == 'modal_panel_durable_game'


@pytest.mark.parametrize('module', ['faynt_d0_durable_game', 'modal_panel_durable_game'])
def test_only_reviewed_durable_workers_are_selectable(module):
    assert app.deployment_worker({'durable_module': module}) == module


@pytest.mark.parametrize('module', ['os', 'package.module', '../worker', '', None, 3])
def test_unreviewed_durable_module_is_rejected(module):
    with pytest.raises(ValueError, match='reviewed durable'):
        app.deployment_worker({'durable_module': module})


@pytest.mark.parametrize('value', [
    {'app_name': 'new'}, {'state_name': 'new'},
    {'app_name': '../old', 'state_name': 'new'},
    {'app_name': 'new', 'state_name': 3},
    {'app_name': 'a' * 65, 'state_name': 'new'},
    {'fresh_run': True}, {**NAMES, 'fresh_run': 'true'},
    {**NAMES, 'app_name': app.APP_NAME, 'fresh_run': True},
    {**NAMES, 'state_name': app.STATE_NAME, 'fresh_run': True},
])
def test_namespace_rejects_incomplete_unsafe_or_legacy_fresh_names(value):
    with pytest.raises(ValueError):
        app.deployment_names(value)


def test_remote_names_select_only_the_requested_app_and_state(monkeypatch):
    monkeypatch.setenv('PANEL_APP_NAME', NAMES['app_name'])
    monkeypatch.setenv('PANEL_STATE_NAME', NAMES['state_name'])
    calls = []
    monkeypatch.setitem(sys.modules, 'modal', types.SimpleNamespace(
        Dict=types.SimpleNamespace(from_name=lambda *a, **k: calls.append((a, k)))))
    app.handles()
    assert app.runtime_names() == NAMES
    assert calls == [((NAMES['state_name'],), {'environment_name': 'main'})]


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, 'validate_snapshot', lambda value: value)
    queue = {'games': [{'label': 'new-a', 'initial_status': 'pending',
                        'initial_row': {'status': 'pending', 'attempts': []}}]}
    directory = tmp_path / 'metadata'
    directory.mkdir()
    def save():
        body = cloud.encoded(queue)
        (directory / 'snapshot.json').write_bytes(body)
        return {**NAMES, 'fresh_run': True, 'snapshot_sha256': cloud.digest(body)}
    return tmp_path, queue, save


def test_fresh_deployment_has_no_inherited_results(fresh):
    directory, _, save = fresh
    assert app.validate_fresh_deployment(save(), directory) == NAMES


@pytest.mark.parametrize('mutation', ['terminal', 'result', 'attempt', 'claim', 'submission', 'provenance'])
def test_fresh_deployment_rejects_every_prior_execution(fresh, mutation):
    directory, queue, save = fresh
    row = queue['games'][0]
    if mutation == 'terminal': row['initial_status'] = 'complete'
    if mutation == 'result': row['initial_row']['result'] = {'win': True}
    if mutation == 'attempt': row['initial_row']['attempts'] = [{'call_id': 'fc-old'}]
    if mutation == 'claim': row['existing_claim'] = {'token': 'old'}
    if mutation == 'submission': row['external_submission'] = {'call_id': 'fc-old'}
    if mutation == 'provenance': row['initial_row']['cloud_provenance'] = {'old': True}
    with pytest.raises(ValueError, match='cannot import'):
        app.validate_fresh_deployment(save(), directory)


def test_fresh_deployment_rejects_changed_snapshot(fresh):
    directory, _, save = fresh
    deployment = save()
    deployment['snapshot_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='snapshot binding'):
        app.validate_fresh_deployment(deployment, directory)


@pytest.mark.parametrize('mutation', ['snapshot', 'app', 'state', 'image', 'worker'])
def test_activation_rejects_another_deployment_receipt(mutation):
    deployment = {**NAMES, 'snapshot_sha256': 'a' * 64}
    receipt = {**deployment, 'image_id': 'im-current'}
    if mutation == 'snapshot': receipt['snapshot_sha256'] = 'b' * 64
    if mutation == 'app': receipt['app_name'] = 'another-app'
    if mutation == 'state': receipt['state_name'] = 'another-state'
    if mutation == 'image': receipt['image_id'] = None
    if mutation == 'worker': receipt['durable_module'] = 'faynt_d0_durable_game'
    with pytest.raises(ValueError, match='exact deployed queue'):
        app.validate_activation_receipt(deployment, receipt)


def test_activation_accepts_legacy_and_isolated_receipts():
    legacy = {'snapshot_sha256': 'a' * 64}
    assert app.validate_activation_receipt(legacy, {**legacy, 'image_id': 'im-old'})['app_name'] == app.APP_NAME
    deployment = {**NAMES, 'snapshot_sha256': 'b' * 64, 'fresh_run': True}
    assert app.validate_activation_receipt(deployment, {**deployment, 'image_id': 'im-new'}) == NAMES


@pytest.fixture
def dispatch(monkeypatch):
    class Store(dict):
        def put(self, key, value, **kwargs): self[key] = value
    sha = 'c' * 64
    monkeypatch.setenv('PANEL_SNAPSHOT_SHA', sha)
    monkeypatch.setenv('PANEL_APP_NAME', NAMES['app_name'])
    monkeypatch.setenv('PANEL_STATE_NAME', NAMES['state_name'])
    queue = {'games': [{'label': 'new-a', 'initial_status': 'pending', 'phase': 0, 'worker_slot': 0}],
             'queue_deadline_unix': float('inf')}
    store = Store({sha + '/control': {'enabled': True, 'image_id': 'im-new', 'allowlist': None}})
    calls = []
    def lookup(*args, **kwargs):
        calls.append(('lookup', args, kwargs))
        return types.SimpleNamespace(spawn=lambda *a, **k: (calls.append(('spawn', a, k))
            or types.SimpleNamespace(object_id='fc-new')))
    monkeypatch.setitem(sys.modules, 'modal', types.SimpleNamespace(Function=types.SimpleNamespace(from_name=lookup)))
    monkeypatch.setattr(app, 'snapshot', lambda: queue)
    monkeypatch.setattr(app, 'handles', lambda: store)
    monkeypatch.setattr(app, 'frozen_game_session_seconds', lambda row: 16290)
    return queue, store, sha + '/', calls


def test_dispatch_targets_new_app(dispatch):
    _, _, _, calls = dispatch
    app.dispatch_cloud()
    assert calls[0][1] == (NAMES['app_name'], 'run_cloud_game')
    assert calls[1][0] == 'spawn'


def test_dispatch_expiry_uses_selected_native_session_limit(dispatch,monkeypatch):
    _,store,key,calls=dispatch
    monkeypatch.setattr(app.time,'time',lambda:1000)
    monkeypatch.setattr(app,'frozen_game_session_seconds',lambda row:9090)
    app.dispatch_cloud()
    launched=next(row for row in calls if row[0]=='spawn')
    assert launched[1][2]==1000+9090
    assert store[key+'dispatch/new-a']['expires_unix']==1000+9090


@pytest.mark.parametrize('terminal', ['complete', 'quarantined'])
def test_terminal_queue_disables_dispatch_and_preserves_health(dispatch, terminal):
    _, store, key, calls = dispatch
    store[key + 'result/new-a'] = {'status': terminal}
    report = app.dispatch_cloud()
    assert report['status'] == 'complete'
    assert report['counts'][terminal] == 1
    assert report['dispatch_enabled'] is False
    assert report['app_cleanup_required'] is True and report['app_stopped'] is False
    assert store[key + 'control']['enabled'] is False
    assert store[key + 'control']['image_id'] == 'im-new'
    assert app.dispatch_cloud() == report
    assert not any(call[0] == 'spawn' for call in calls)


def test_deadline_disables_dispatch_without_scoring_or_claiming_app_stop(dispatch):
    queue, store, key, calls = dispatch
    queue['queue_deadline_unix'] = 0
    report = app.dispatch_cloud()
    assert report['status'] == 'queue-deadline-reached'
    assert store[key + 'control']['enabled'] is False
    assert report['app_stopped'] is False
    assert app.dispatch_cloud() == report
    assert key + 'result/new-a' not in store and calls == []


def test_holds_remain_visible_without_terminal_disable(dispatch):
    _, store, key, calls = dispatch
    store[key + 'hold/new-a'] = {'error': 'retained failure'}
    assert app.dispatch_cloud()['status'] == 'drained-with-unresolved-holds'
    assert store[key + 'control']['enabled'] is True
    assert not any(call[0] == 'spawn' for call in calls)


def test_canary_allowlist_runs_later_model_after_selected_early_model_finishes(dispatch):
    queue,store,key,calls=dispatch
    queue['games']=[{'label':'75m-canary','initial_status':'pending','phase':0,'worker_slot':0},
        {'label':'75m-rest','initial_status':'pending','phase':0,'worker_slot':1},
        {'label':'10m-canary','initial_status':'pending','phase':3,'worker_slot':0}]
    store[key+'control']['allowlist']=['75m-canary','10m-canary']
    first=app.dispatch_cloud()
    assert [item['label'] for item in first['submitted']]==['75m-canary']
    store[key+'result/75m-canary']={'status':'complete'}
    second=app.dispatch_cloud()
    assert [item['label'] for item in second['submitted']]==['10m-canary']
    assert len(queue['games'])==3
    assert all(call[1][0]!='75m-rest' for call in calls if call[0]=='spawn')


@pytest.mark.parametrize('allowlist',[['unknown'],['new-a','new-a'],'new-a',[None]])
def test_canary_allowlist_requires_exact_frozen_unique_labels(dispatch,allowlist):
    _,store,key,calls=dispatch;store[key+'control']['allowlist']=allowlist
    with pytest.raises(ValueError,match='allowlist'):app.dispatch_cloud()
    assert not any(call[0]=='spawn' for call in calls)


@pytest.fixture
def canaries(dispatch):
    queue,store,key,calls=dispatch
    labels=['75m-canary','10m-canary','75m-remaining']
    queue['games']=[{'label':label,'initial_status':'pending','phase':3 if label.startswith('10m') else 0,
        'worker_slot':0 if label.endswith('canary') else 1,'plan_sha256':str(index+1)*64}
        for index,label in enumerate(labels)]
    queue['manifest']={'games':[{'label':label,'profile':'10m' if label.startswith('10m') else '75m'}
        for label in labels]}
    control=store[key+'control']
    control.update(allowlist=labels[:2],auto_expand_after_canary=True,extra_identity='retain-me')
    store[key+'budget']={'immutable':'budget-start-and-deadline'}
    for row in queue['games'][:2]:
        store[key+'result/'+row['label']]={'schema':cloud.SCHEMA+'.acceptance','status':'complete',
            'label':row['label'],'queue_sha256':key[:-1],'queue_plan_sha256':row['plan_sha256'],
            'plan_sha256':row['plan_sha256'],'native_acceptance':{'win':False,'winner':'slippi-ai',
                'game_frames':1000,'end_evidence':'formal-game-end','summary_sha256':'a'*64,'replay_sha256':'b'*64},
            'manifest':{'status':'full-policy-durable-game-passed','plan_sha256':row['plan_sha256'],
                'files':[{'name':'project/summary.json','sha256':'a'*64},
                         {'name':'project/replays/Game.slp','sha256':'b'*64}]}}
    return queue,store,key,calls


def test_both_native_accepted_canaries_expand_without_touching_budget_or_results(canaries):
    queue,store,key,calls=canaries
    before=copy.deepcopy(store);queue_before=copy.deepcopy(queue)
    report=app.dispatch_cloud()
    assert [row['label'] for row in report['submitted']]==['75m-remaining']
    control=store[key+'control']
    assert control['allowlist'] is None and control['auto_expand_after_canary'] is False
    assert control['image_id']==before[key+'control']['image_id']
    assert control['extra_identity']=='retain-me'
    assert store[key+'budget']==before[key+'budget'] and queue==queue_before
    for label in ['75m-canary','10m-canary']:
        assert store[key+'result/'+label]==before[key+'result/'+label]
    assert control['canary_expansion']==store[key+'canary-expansion']==report['canary_expansion']
    expansion=control['canary_expansion']
    assert len(expansion['canaries'])==2 and expansion['queue_sha256']==key[:-1]
    assert app.expand_accepted_canaries(store,key,control,queue,{})==control


@pytest.mark.parametrize('state',['missing','pending','quarantined','failed'])
def test_canary_auto_expansion_waits_for_every_native_acceptance(canaries,state):
    queue,store,key,_=canaries
    record=store.pop(key+'result/10m-canary')
    if state!='missing':store[key+'result/10m-canary']={**record,'status':state}
    records={label:store.get(key+'result/'+label,{}) for label in ['75m-canary','10m-canary']}
    control=store[key+'control'];before=copy.deepcopy(control)
    assert app.expand_accepted_canaries(store,key,control,queue,records)==before
    assert key+'canary-expansion' not in store and store[key+'control']['allowlist'] is not None


@pytest.mark.parametrize('mutation',['schema','queue','plan','native','native-winner','artifacts','manifest'])
def test_canary_expansion_rejects_malformed_or_unbound_complete_receipt(canaries,mutation):
    _,store,key,calls=canaries
    receipt=store[key+'result/10m-canary']
    if mutation=='schema':receipt['schema']='worker-return-only'
    if mutation=='queue':receipt['queue_sha256']='old'
    if mutation=='plan':receipt['plan_sha256']='old'
    if mutation=='native':receipt['native_acceptance']={}
    if mutation=='native-winner':receipt['native_acceptance']['win']=True
    if mutation=='artifacts':receipt['manifest']['files']=[]
    if mutation=='manifest':receipt['manifest']['status']='failed'
    report=app.dispatch_cloud()
    assert report['status']=='canary-acceptance-invalid'
    assert store[key+'control']['enabled'] is False
    assert store[key+'control']['allowlist']==['75m-canary','10m-canary']
    assert not any(call[0]=='spawn' for call in calls)


def test_canary_expansion_requires_explicit_opt_in(canaries):
    _,store,key,calls=canaries
    store[key+'control']['auto_expand_after_canary']=False
    app.dispatch_cloud()
    assert store[key+'control']['allowlist']==['75m-canary','10m-canary']
    assert key+'canary-expansion' not in store
    assert not any(call[0]=='spawn' for call in calls)


def test_canary_expansion_requires_both_model_profiles(canaries):
    queue,store,key,calls=canaries
    queue['manifest']['games'][1]['profile']='75m'
    report=app.dispatch_cloud()
    assert report['status']=='canary-acceptance-invalid'
    assert 'both 75m and 10m' in report['reason']
    assert key+'canary-expansion' not in store


def test_canary_expansion_preserves_concurrent_pause(canaries):
    queue,store,key,_=canaries
    stale=copy.deepcopy(store[key+'control']);store[key+'control']['enabled']=False
    records={label:store[key+'result/'+label] for label in ['75m-canary','10m-canary']}
    assert app.expand_accepted_canaries(store,key,stale,queue,records)['enabled'] is False
    assert key+'canary-expansion' not in store


def test_report_uses_bound_deployment_state(fresh, monkeypatch):
    directory, queue, save = fresh
    deployment = save()
    (directory / 'deployment.json').write_text(json.dumps(deployment))
    monkeypatch.setattr(reporting, 'project', lambda snapshot, receipts: {})
    monkeypatch.setattr(reporting, 'markdown', lambda *args: 'report\n')
    calls = []
    def state(name, **kwargs):
        calls.append((name, kwargs))
        return {deployment['snapshot_sha256'] + '/health': {'status': 'running'}}
    monkeypatch.setitem(sys.modules, 'modal', types.SimpleNamespace(Dict=types.SimpleNamespace(from_name=state)))
    result = reporting.report(directory)
    assert calls == [(NAMES['state_name'], {'environment_name': 'main'})]
    assert result['app_name'] == NAMES['app_name'] and result['state_name'] == NAMES['state_name']
    deployment['snapshot_sha256'] = '0' * 64
    (directory / 'deployment.json').write_text(json.dumps(deployment))
    with pytest.raises(ValueError, match='different snapshot'):
        reporting.report(directory)
    assert len(calls) == 1
