import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import modal_panel_durable_game as d
from test_modal_panel_queue import queue, transport, bind as bind_claim, ingest
import modal_panel_queue as q


class Store(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if key in self and skip_if_exists: return False
        self[key] = copy.deepcopy(value); return True


class Volume:
    def __init__(self): self.commits = self.reloads = 0; self.crash_after_commit = False
    def commit(self):
        self.commits += 1
        if self.crash_after_commit:
            self.crash_after_commit = False
            raise RuntimeError('synthetic crash after durable commit')
    def reload(self): self.reloads += 1


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'MOUNT', tmp_path / 'volume')
    monkeypatch.setattr(d.batch.base.wire, 'remaining', lambda expires: expires)
    bound = types.SimpleNamespace(SESSION_SECONDS=100, FILE_COUNT=10, FILE_CAP=10000, RETURN_CAP=20000,
                                  allowed_output=lambda n: n in {'summary.json', 'game.slp'})
    plan = {'scope': {'selected_game_indices': [0]}, 'artifact_labels': ['game-a1', 'peer-a1']}
    body = b'synthetic replay'
    result = {'schema': 'native', 'status': d.SUCCESS, 'scope': plan['scope'], 'files': [
        {'name': 'game.slp', 'body': body, 'sha256': hashlib.sha256(body).hexdigest(),
         'original_bytes': len(body), 'original_sha256': hashlib.sha256(body).hexdigest(), 'truncated_tail': False}]}
    return bound, plan, Volume(), Store(), result


def run(rig, number, owner, *, call='fc-test', expires=90):
    b, p, v, s, _ = rig
    return d.run_durable(b, p, 'a'*64, expires, 'im-test', v, s, call,
                         {'task_id': f'ta-test{number}', 'execution_id': f'{number:032x}'}, run_owner=owner)


def test_game_survives_crash_after_commit_without_replay(rig):
    rig[2].crash_after_commit = True
    with pytest.raises(RuntimeError, match='after durable commit'):
        run(rig, 1, lambda: copy.deepcopy(rig[4]))
    saved = run(rig, 2, lambda: pytest.fail('completed game must not execute again'))
    assert saved['execution']['execution_id'] == f'{1:032x}'
    assert rig[2].commits == 1


def test_interrupted_game_restarts_with_same_plan_and_preserves_identity(rig):
    def crash(): raise RuntimeError('synthetic container crash during game')
    with pytest.raises(RuntimeError): run(rig, 1, crash)
    result = run(rig, 2, lambda: copy.deepcopy(rig[4]))
    assert result['physical_number'] == 2 and rig[3]['a'*64+'/physical/0']['execution']['task_id'] == 'ta-test1'


def test_restart_bound_is_three_even_when_platform_restarts_again(rig):
    def crash(): raise RuntimeError('synthetic crash')
    for n in range(1,4):
        with pytest.raises(RuntimeError, match='synthetic'): run(rig, n, crash)
    with pytest.raises(RuntimeError, match='three physical'): run(rig, 4, lambda: pytest.fail('budget exhausted'))


def test_prior_native_attempts_reduce_remaining_restart_budget(rig):
    rig[1]['artifact_labels'][0] = 'game-a3'
    with pytest.raises(RuntimeError): run(rig, 1, lambda: (_ for _ in ()).throw(RuntimeError('crash')))
    with pytest.raises(RuntimeError, match='three physical'): run(rig, 2, lambda: pytest.fail('budget exhausted'))


@pytest.mark.parametrize('change', ['call', 'deadline'])
def test_duplicate_calls_and_extended_deadlines_cannot_launch(rig, change):
    run(rig, 1, lambda: copy.deepcopy(rig[4]))
    with pytest.raises(ValueError, match='different call or extended'):
        run(rig, 2, lambda: pytest.fail('duplicate launch'), call='fc-other' if change=='call' else 'fc-test',
            expires=91 if change=='deadline' else 90)


def test_corrupt_committed_replay_is_rejected_without_rerunning(rig):
    saved = run(rig, 1, lambda: copy.deepcopy(rig[4]))
    (d.MOUNT / saved['storage_path'] / 'files/game.slp').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='bytes differ'):
        run(rig, 2, lambda: pytest.fail('corruption requires audit'))


def test_platform_volume_root_symlink_is_supported(rig,tmp_path,monkeypatch):
    actual=tmp_path/'actual-volume';actual.mkdir()
    link=tmp_path/'mounted-volume';link.symlink_to(actual,target_is_directory=True)
    monkeypatch.setattr(d,'MOUNT',link)
    first=run(rig,1,lambda:copy.deepcopy(rig[4]))
    assert run(rig,2,lambda:pytest.fail('completed game replayed'))==first


def test_committed_result_can_be_read_after_gameplay_deadline(rig,monkeypatch):
    first=run(rig,1,lambda:copy.deepcopy(rig[4]))
    monkeypatch.setattr(d.batch.base.wire,'remaining',lambda expires:-1)
    assert run(rig,2,lambda:pytest.fail('expired game must not restart'))==first


def test_expired_interrupted_game_cannot_restart(rig,monkeypatch):
    with pytest.raises(RuntimeError):run(rig,1,lambda:(_ for _ in ()).throw(RuntimeError('crash')))
    monkeypatch.setattr(d.batch.base.wire,'remaining',lambda expires:-1)
    with pytest.raises(ValueError,match='deadline'):run(rig,2,lambda:pytest.fail('expired game launched'))


def test_cloud_saved_result_recovery_requires_no_worker(rig):
    saved=run(rig,1,lambda:copy.deepcopy(rig[4]))
    class Reader:
        def read_file(self,name):yield (d.MOUNT/name).read_bytes()
    assert d.recover_saved(Reader(),rig[3],'a'*64,rig[0])==saved


def test_download_interruption_is_retained_and_resumes(rig,tmp_path):
    saved=run(rig,1,lambda:copy.deepcopy(rig[4]));output=tmp_path/'download';output.mkdir()
    class Reader:
        fail=True
        def read_file(self,name):
            body=(d.MOUNT/name).read_bytes()
            yield body[:3]
            if self.fail:raise ConnectionError('lost download')
            yield body[3:]
    reader=Reader()
    with pytest.raises(ConnectionError):d.download_result(reader,saved,output,'a'*64,rig[0])
    reader.fail=False
    _,files=d.download_result(reader,saved,output,'a'*64,rig[0])
    assert files['game.slp'].read_bytes()==b'synthetic replay'
    assert len(list((output/'durable/files').glob('game.slp.interrupted-*')))==1


def test_manifest_path_traversal_is_rejected(rig,tmp_path):
    saved=run(rig,1,lambda:copy.deepcopy(rig[4]));saved['files'][0]['name']='../escape'
    output=tmp_path/'download';output.mkdir()
    with pytest.raises(ValueError,match='unsafe'):
        d.download_result(None,saved,output,'a'*64,rig[0])


@pytest.mark.parametrize('mutation',['none','frame','inference','winner','end','trace','replay'])
def test_only_proven_empty_startups_are_retried(mutation):
    from test_modal_panel_empty_startup import summary
    s=summary();prefix='project/artifacts/integration/frisson_ai/example-a1/'
    if mutation=='frame':s['execution']['processed_policy_frames']=1
    if mutation=='inference':s['execution']['inference_counts']['frisson-ai']=1
    if mutation=='winner':s['execution']['winner']='frisson-ai'
    if mutation=='end':s['execution']['natural_game_end']=True
    files=[{'name':prefix+'summary.json','body':json.dumps(s).encode()},
           {'name':prefix+'controller_trace.jsonl','body':b'played' if mutation=='trace' else b''}]
    if mutation=='replay':files.append({'name':prefix+'game.slp','body':b'replay'})
    assert d.retryable_startup({'files':files},0,{'artifact_labels':['example-a1','other-a1']}) is (mutation=='none')


def test_durable_sdk_resource_and_retry_configuration(monkeypatch,tmp_path):
    pytest.importorskip("modal", reason="optional Modal SDK is required for this SDK contract test")
    import modal
    captured={}
    original=modal.App.function
    def capture(self,*args,**kwargs):
        captured.update(kwargs)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(modal.App,'function',capture)
    bound=types.SimpleNamespace(FUNCTION_SECONDS=15990)
    app,fn,image,volume,store=d.configure_app(modal,tmp_path/'plan.json',{'uploads':[]},bound)
    assert captured['nonpreemptible'] is True and captured['max_containers']==1
    assert captured['cpu']==(4,4) and captured['memory']==(16384,16384)
    assert captured['single_use_containers'] is True
    assert captured['volumes']=={str(d.MOUNT):volume}
    assert captured['timeout']==15990
    assert 'is_generator' not in captured


def test_completed_pair_frees_only_its_original_slot(queue):
    from modal_panel_durable_cohort import eligible_pairs
    frozen=queue['frozen']['plan'];state=q.read(queue['output']/'state.json')
    before=list(eligible_pairs(frozen,state,queue['directory']))
    assert len(before)==2 and all(p['phase']==0 for p in before)
    first=q.claim_pair(queue['directory'],queue['frozen']['plan_sha256'],before[0]['pair_id'],
        expected_state_sha256=q.digest(q.read_bytes(queue['output']/'state.json')),first_pending_only=True)
    assert len(list(eligible_pairs(frozen,state,queue['directory'])))==1


@pytest.mark.parametrize('index', [0,1])
def test_selected_native_game_and_validation_functions_unchanged(index):
    panel = d.batch.base.policy.bounded_json(d.batch.base.ROOT / d.batch.base.PANEL)
    labels = [g['label'] for g in panel['games'][154:156]]
    before = dict(vars(d.batch.base))
    b = d.bind(labels, [n+'-a1' for n in labels], index)
    assert b.fixed_scope()['selected_game_indices'] == [index]
    assert (index,) in b.owner.__code__.co_consts
    for name in ('game_child', 'native_validation', 'stage_project', 'assemble_admissions'):
        assert getattr(b, name).__code__ is getattr(d.batch.base, name).__code__
    assert set(b.validate_terminal.__code__.co_names) == set(d.batch.base.validate_terminal.__code__.co_names) - {'enumerate'}
    assert all(vars(d.batch.base)[k] is v for k,v in before.items())


def test_single_claim_keeps_peer_pending_until_first_is_durable(queue):
    j=queue['directory']; sha=queue['frozen']['plan_sha256']; pair=queue['frozen']['plan']['pairs'][0]['pair_id']
    def claim():
        return q.claim_pair(j,sha,pair,expected_state_sha256=q.digest(q.read_bytes(queue['output']/'state.json')),
                            first_pending_only=True)
    first=claim(); assert len(first['attempts']) == 1
    with pytest.raises(RuntimeError, match='existing claim'): claim()
    bind_claim(queue,first); assert ingest(queue,transport(queue,first))['status']=='complete'
    second=claim(); assert len(second['attempts']) == 1
    assert first['attempts'][0]['game']['label'] != second['attempts'][0]['game']['label']
    bind_claim(queue,second); assert ingest(queue,transport(queue,second))['status']=='complete'
    with pytest.raises(ValueError,match='terminal'):claim()
