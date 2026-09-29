"""Expired unstarted retries renew under the active tranche without new starts."""
import ast
import copy
import inspect
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import modal_panel_budget as budget
import modal_panel_cloud_queue as q
import modal_panel_cloud_app as app

REASON = 'verified-prebootstrap-budget-exhaustion'
LABEL = 'faynt-d0-632-980-v1-sp21-75m-fox_d21_ditto_v4-forced-mirror-cptfalcon-battlefield-p2'
OLD_DEADLINE = 1789602658.8854022
SESSION = 9090


class Store(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = copy.deepcopy(value)
        return True


def tranche(deadline, tag):
    return {'authorization_sha256': tag * 64, 'activation': {'deadline_unix': deadline}}


def binding(value):
    return {'tranche_sha256': budget.digest(value),
            'authorization_sha256': value['authorization_sha256'],
            'activation_sha256': budget.digest(value['activation'])}


def frozen(record):
    return [q.encoded(record[field]) for field in q.RENEWAL_FROZEN_FIELDS]


@pytest.fixture
def case():
    """Mirror the published cptfalcon a3: two prior failures, the first a legacy
    receipt without call_binding, budget_resume bound to the tranche whose
    deadline equals expires_unix, and zero physical starts for the a3 plan."""
    original = {'artifact_labels': [LABEL + '-a1', 'peer-a1'],
                'scope': {'selected_game_indices': [0]}, 'seed': 31}
    body = q.encoded(original)
    row = {'label': LABEL, 'initial_status': 'pending', 'plan_sha256': q.digest(body),
           'phase': 0, 'worker_slot': 0}
    second = copy.deepcopy(original); second['artifact_labels'][0] = LABEL + '-a2'
    third = copy.deepcopy(original); third['artifact_labels'][0] = LABEL + '-a3'
    old = tranche(OLD_DEADLINE, 'a')
    expired = {
        'reason': REASON, 'attempt_number': 3,
        'queue_plan_sha256': row['plan_sha256'], 'original_execution_plan_sha256': row['plan_sha256'],
        'queue_plan_body': body.decode(), 'plan_sha256': q.digest(q.encoded(third)),
        'expires_unix': OLD_DEADLINE, 'original_expires_unix': 1789500000.0,
        'prior_failures': [
            {'manifest': {'status': 'failed', 'plan_sha256': row['plan_sha256'], 'call_id': 'fc-first'},
             'physical_history': [{'call_id': 'fc-first', 'expires_unix': 1789500000.0,
                                   'execution': {'execution_id': 'first'}}]},
            {'manifest': {'status': 'failed', 'plan_sha256': q.digest(q.encoded(second)),
                          'call_id': 'fc-second'},
             'physical_history': [{'call_id': 'fc-second', 'expires_unix': OLD_DEADLINE,
                                   'execution': {'execution_id': 'second'}}],
             'call_binding': {'call_id': 'fc-second', 'expires_unix': OLD_DEADLINE, 'image_id': 'im-old'}}],
        'budget_resume': binding(old)}
    q.retry_execution_plan(row, body, expired)
    q.validate_retry_budget_binding(expired, old)
    current = tranche(OLD_DEADLINE + 86400, 'b')
    now = OLD_DEADLINE + 3600
    return row, body, expired, old, current, now


def renew(case, **overrides):
    row, body, expired, _, current, now = case
    kwargs = {'now': now, 'session_seconds': SESSION, 'resume_tranche': current, 'index_has_start': False}
    kwargs.update(overrides)
    return q.renew_expired_retry(row, body, expired, **kwargs)


def test_expired_unstarted_retry_renews_under_active_tranche(case):
    row, body, expired, old, current, now = case
    before = copy.deepcopy(expired)
    renewed = renew(case)
    assert expired == before
    assert renewed['expires_unix'] == now + SESSION
    assert renewed['budget_resume'] == binding(current)
    assert renewed['renews_authorization_sha256'] == q.digest(expired)
    assert renewed['renewal_count'] == 1
    assert frozen(renewed) == frozen(expired)
    assert 'call_binding' not in renewed['prior_failures'][0]
    assert set(renewed) == set(expired) | {'renews_authorization_sha256', 'renewal_count'}
    q.validate_retry_budget_binding(renewed, current)
    assert q.retry_execution_plan(row, body, renewed) == q.retry_execution_plan(row, body, expired)
    with pytest.raises(ValueError):
        q.validate_retry_budget_binding(renewed, old)


def test_renewed_deadline_is_clamped_to_the_active_tranche(case):
    _, _, _, _, _, now = case
    short = tranche(now + 4000, 'c')
    renewed = renew(case, resume_tranche=short)
    assert renewed['expires_unix'] == now + 4000
    assert renewed['budget_resume'] == binding(short)


@pytest.mark.parametrize('started', [True, 'yes', None, 1])
def test_started_attempt_keeps_its_frozen_deadline(case, started):
    with pytest.raises(ValueError, match='started retry attempt|explicit durable index'):
        renew(case, index_has_start=started)


@pytest.mark.parametrize('offset', [-1, 0, -3600])
def test_unexpired_retry_is_not_renewed(case, offset):
    with pytest.raises(ValueError, match='unexpired'):
        renew(case, now=OLD_DEADLINE + offset)


def test_renewal_requires_an_active_tranche_with_budget(case):
    _, _, _, _, _, now = case
    with pytest.raises(ValueError, match='approved current tranche'):
        renew(case, resume_tranche=None)
    with pytest.raises(ValueError, match='insufficient execution budget'):
        renew(case, resume_tranche=tranche(now + 600, 'c'))
    with pytest.raises(ValueError, match='finite resumed budget deadline'):
        renew(case, resume_tranche=tranche(float('inf'), 'c'))
    with pytest.raises(ValueError, match='finite bounded retry timing'):
        renew(case, session_seconds=600)


@pytest.mark.parametrize('field', q.RENEWAL_FROZEN_FIELDS)
def test_renewal_refuses_any_frozen_field_change(case, field):
    _, _, expired, _, _, _ = case
    renewed = renew(case)
    q.validate_retry_renewal(expired, renewed)
    changed = copy.deepcopy(renewed)
    changed[field] = 'changed' if field != 'attempt_number' else 2
    with pytest.raises(ValueError, match='frozen field'):
        q.validate_retry_renewal(expired, changed)


@pytest.mark.parametrize('field', ['plan_sha256', 'queue_plan_body', 'reason', 'original_expires_unix'])
def test_expired_record_must_bind_its_frozen_inputs(case, field):
    row, body, expired, _, current, now = case
    broken = copy.deepcopy(expired)
    broken[field] = 'x' * 64 if field != 'original_expires_unix' else 'later'
    with pytest.raises(ValueError):
        q.renew_expired_retry(row, body, broken, now=now, session_seconds=SESSION,
                              resume_tranche=current, index_has_start=False)


def test_renewal_lineage_validator_rejects_shape_digest_count_and_deadline(case):
    _, _, expired, _, _, _ = case
    renewed = renew(case)
    extra = {**renewed, 'note': 'x'}
    with pytest.raises(ValueError, match='record shape'):
        q.validate_retry_renewal(expired, extra)
    with pytest.raises(ValueError, match='bind the retained expired record'):
        q.validate_retry_renewal(expired, {**renewed, 'renews_authorization_sha256': 'f' * 64})
    with pytest.raises(ValueError, match='count must advance'):
        q.validate_retry_renewal(expired, {**renewed, 'renewal_count': 2})
    with pytest.raises(ValueError, match='extend past'):
        q.validate_retry_renewal(expired, {**renewed, 'expires_unix': expired['expires_unix']})
    with pytest.raises(ValueError, match='retained expired and renewed'):
        q.validate_retry_renewal(None, renewed)


def test_renewal_of_a_renewal_chains_digests(case):
    row, body, expired, _, current, now = case
    first = renew(case)
    later = first['expires_unix'] + 10
    third = tranche(later + 86400, 'c')
    second = q.renew_expired_retry(row, body, first, now=later, session_seconds=SESSION,
                                   resume_tranche=third, index_has_start=False)
    assert first['renews_authorization_sha256'] == q.digest(expired)
    assert second['renews_authorization_sha256'] == q.digest(first)
    assert second['renewal_count'] == 2 and first['renewal_count'] == 1
    assert frozen(second) == frozen(expired)
    assert second['budget_resume'] == binding(third)
    q.validate_retry_renewal(first, second)
    with pytest.raises(ValueError):
        q.validate_retry_renewal(expired, second)


def worker_state(case, *, reinspected=True, owner=False, index_start=None):
    row, body, expired, old, current, now = case
    key = 'q' * 64 + '/'
    state = Store({key + 'retry/' + LABEL + '/a3': expired, key + 'retry-latest/' + LABEL: 3,
                   key + 'control': {'enabled': True, 'image_id': 'im', 'strict_phase_order': True}})
    if reinspected:
        state[key + 'control']['held_reinspection'] = {
            'tranche_sha256': budget.digest(current), 'labels': [LABEL]}
    if owner:
        state[key + 'owner/' + LABEL + '/a3'] = {'call_id': 'fc-owner', 'expires_unix': OLD_DEADLINE}
    index = Store()
    if index_start is not None:
        index[expired['plan_sha256'] + index_start] = {'call_id': 'fc-a3', 'expires_unix': OLD_DEADLINE}
    return state, key, index


def renew_in_worker(case, state, key, index, retry=None, **overrides):
    row, body, expired, old, current, now = case
    kwargs = {'now': now}
    kwargs.update(overrides)
    return app.renew_unstarted_retry(state, key, {'games': [row]}, row, body,
                                     expired if retry is None else retry, 3, index, SESSION,
                                     current, **kwargs)


def test_worker_archives_expired_record_then_replaces_only_that_key(case):
    row, body, expired, old, current, now = case
    state, key, index = worker_state(case)
    before = copy.deepcopy(state)
    result = renew_in_worker(case, state, key, index)
    renewed = renew(case)
    archive = key + 'retry-history/' + LABEL + '/a3/' + q.digest(expired)
    assert result['status'] == 'retry-renewed'
    assert result['attempt_number'] == 3 and result['score_admitted'] is False
    assert result['renews_authorization_sha256'] == q.digest(expired)
    assert result['expires_unix'] == renewed['expires_unix']
    assert state[archive] == expired
    assert q.digest(state[archive]) == renewed['renews_authorization_sha256']
    assert state[key + 'retry/' + LABEL + '/a3'] == renewed
    assert state[key + 'retry-latest/' + LABEL] == 3
    assert set(state) - set(before) == {archive}
    assert all(state[k] == v for k, v in before.items() if k != key + 'retry/' + LABEL + '/a3')
    assert index == {}


def test_worker_renewal_chains_and_keeps_every_archived_record(case):
    row, body, expired, old, current, now = case
    state, key, index = worker_state(case)
    renew_in_worker(case, state, key, index)
    first = state[key + 'retry/' + LABEL + '/a3']
    later = first['expires_unix'] + 10
    third = tranche(later + 86400, 'c')
    state[key + 'control']['held_reinspection']['tranche_sha256'] = budget.digest(third)
    app.renew_unstarted_retry(state, key, {'games': [row]}, row, body, first, 3, index, SESSION,
                              third, now=later)
    second = state[key + 'retry/' + LABEL + '/a3']
    assert second['renews_authorization_sha256'] == q.digest(first)
    assert state[key + 'retry-history/' + LABEL + '/a3/' + q.digest(first)] == first
    assert state[key + 'retry-history/' + LABEL + '/a3/' + q.digest(expired)] == expired
    assert state[key + 'retry-latest/' + LABEL] == 3


@pytest.mark.parametrize('slot', ['/call', '/physical/0', '/physical/2'])
def test_worker_refuses_when_index_holds_a_call_or_physical_slot(case, slot):
    state, key, index = worker_state(case, index_start=slot)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match='started retry attempt'):
        renew_in_worker(case, state, key, index)
    assert state == before


def test_worker_rechecks_index_immediately_before_replacing(case):
    row, body, expired, old, current, now = case
    state, key, index = worker_state(case)
    reads = []
    class LateIndex(Store):
        def get(self, name, default=None):
            reads.append(name)
            if name.endswith('/call') and len(reads) > 4:
                return {'call_id': 'fc-late', 'expires_unix': OLD_DEADLINE}
            return super().get(name, default)
    with pytest.raises(ValueError, match='changed before renewal'):
        renew_in_worker(case, state, key, LateIndex())
    assert state[key + 'retry/' + LABEL + '/a3'] == expired
    assert state[key + 'retry-history/' + LABEL + '/a3/' + q.digest(expired)] == expired


def test_worker_refuses_outside_reinspection_owner_or_changed_record(case):
    row, body, expired, old, current, now = case
    state, key, index = worker_state(case, reinspected=False)
    with pytest.raises(ValueError, match='reinspection'):
        renew_in_worker(case, state, key, index)
    state, key, index = worker_state(case, owner=True)
    with pytest.raises(ValueError, match='logical owner'):
        renew_in_worker(case, state, key, index)
    state, key, index = worker_state(case)
    with pytest.raises(ValueError, match='changed during recovery'):
        renew_in_worker(case, state, key, index, retry={**expired, 'expires_unix': OLD_DEADLINE + 1})
    state, key, index = worker_state(case)
    with pytest.raises(ValueError, match='unexpired'):
        renew_in_worker(case, state, key, index, now=OLD_DEADLINE)
    assert key + 'retry-history/' + LABEL + '/a3/' + q.digest(expired) not in state


def test_worker_defers_instead_of_renewing_when_dispatch_could_not_admit(case, monkeypatch):
    row, body, expired, old, current, now = case
    monkeypatch.setenv('PANEL_DURABLE_MODULE', 'faynt_d0_durable_game')
    state, key, index = worker_state(case)
    short = tranche(now + 2400, 'c')
    state[key + 'control']['held_reinspection']['tranche_sha256'] = budget.digest(short)
    before = copy.deepcopy(state)
    result = app.renew_unstarted_retry(state, key, {'games': [row]}, row, body, expired, 3, index,
                                       SESSION, short, now=now)
    assert result['status'] == 'budget-admission-deferred' and result['expired_retry_retained'] is True
    assert state == before


def code(function):
    """Source lines after the docstring, so comments about what never happens do not count."""
    source = inspect.getsource(function)
    first = ast.parse(source).body[0].body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return ''.join(source.splitlines(keepends=True)[first.end_lineno:])
    return source


def test_worker_checks_index_before_writing_and_never_touches_durable_worker():
    source = code(app.renew_unstarted_retry)
    call_read = source.index("index.get(plan_sha+'/call')")
    slot_read = source.index("index.get(plan_sha+f'/physical/{n}')")
    archive = source.index("state.put(history_key,expired,skip_if_exists=True)")
    recheck = source.index("if state.get(retry_key)!=expired or started():")
    replace = source.index("state.put(retry_key,renewed)")
    assert call_read < slot_read < archive < recheck < replace
    assert 'index.put' not in source and 'volume' not in source
    assert 'faynt_d0_durable_game' not in source and 'importlib' not in source
    assert "state.put(key+'retry-latest/'" not in source
    assert 'skip_if_exists=True' in source[archive:recheck]
    assert source.count('state.put(') == 2
    worker = code(app._run_cloud_game)
    validated = worker.index('cloud.retry_execution_plan(row,body,retry)')
    lineage = worker.index('cloud.validate_retry_renewal(archived,retry)')
    saved = worker.index('d.recover_saved(volume,index,plan_sha,bound)')
    renewal = worker.index('return renew_unstarted_retry(')
    assert validated < lineage < saved < renewal
    branch = worker[saved:renewal]
    assert 'if manifest is None:' in branch and 'if recover_only:' in branch
    assert "if retry and time.time()>retry['expires_unix']:" in branch
    assert worker.index("return {'status':'awaiting-native-result'}") > renewal
    assert 'renew' not in worker[:validated]
    queue_source = code(q.renew_expired_retry)
    assert 'index.get' not in queue_source and 'index.put' not in queue_source and 'state' not in queue_source
    assert 'faynt_d0_durable_game' not in queue_source + code(q.validate_retry_renewal)


@pytest.fixture
def dispatcher(monkeypatch, case):
    row, body, expired, old, current, now = case
    store = Store(); sha = 'a' * 64
    monkeypatch.setenv('PANEL_SNAPSHOT_SHA', sha)
    queue = {'games': [{**row, 'phase': 0, 'worker_slot': 0}], 'queue_deadline_unix': float('inf')}
    monkeypatch.setattr(app, 'snapshot', lambda: queue)
    monkeypatch.setattr(app, 'handles', lambda: store)
    monkeypatch.setattr(app.time, 'time', lambda: now)
    store[sha + '/control'] = {'enabled': True, 'image_id': 'im-test'}
    store[sha + '/retry-latest/' + LABEL] = 3
    calls = {}; launched = []
    class Call:
        def __init__(self, call_id): self.object_id = call_id
        def get(self, timeout=0):
            result = calls[self.object_id]
            if result is None: raise TimeoutError()
            if isinstance(result, Exception): raise result
            return result
    def spawn(label, *args, **kwargs):
        call = Call('fc-' + str(len(launched)))
        launched.append((label, (args, kwargs), call.object_id)); calls[call.object_id] = None
        return call
    modal = types.SimpleNamespace(
        Function=types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(spawn=spawn)),
        FunctionCall=types.SimpleNamespace(from_id=Call))
    monkeypatch.setitem(sys.modules, 'modal', modal)
    return store, calls, launched, sha


def test_dispatcher_dispatches_renewed_attempt_with_renewed_expiry(dispatcher, case):
    store, calls, launched, sha = dispatcher
    renewed = renew(case)
    stale = {'submitted_unix': OLD_DEADLINE - 9000, 'expires_unix': OLD_DEADLINE,
             'submissions': 1, 'call_id': 'fc-stale'}
    store[sha + '/retry/' + LABEL + '/a3'] = renewed
    store[sha + '/dispatch/' + LABEL + '/a3'] = stale
    calls['fc-stale'] = ValueError('prior call or original deadline changed')
    report = app.dispatch_cloud()
    assert [item[0] for item in launched] == [LABEL]
    (args, kwargs) = launched[0][1]
    assert args[1] == renewed['expires_unix'] and kwargs == {'attempt': 3}
    assert store[sha + '/dispatch/' + LABEL + '/a3'] == stale
    scoped = sha + '/dispatch/' + LABEL + '/a3/renewal-' + renewed['renews_authorization_sha256']
    assert store[scoped]['expires_unix'] == renewed['expires_unix'] and store[scoped]['call_id'] == 'fc-0'
    assert store[sha + '/retry-latest/' + LABEL] == 3 and report['errors'] == []
    assert sha + '/hold/' + LABEL not in store
    calls['fc-0'] = None
    app.dispatch_cloud()
    assert len(launched) == 1


def test_dispatcher_treats_renewal_answer_as_resolved(dispatcher, case):
    store, calls, launched, sha = dispatcher
    _, _, expired, _, _, _ = case
    store[sha + '/retry/' + LABEL + '/a3'] = expired
    store[sha + '/dispatch/' + LABEL + '/a3'] = {'submitted_unix': 1, 'expires_unix': OLD_DEADLINE,
                                                'submissions': 1, 'call_id': 'fc-stale'}
    recovery = {'submissions': 1, 'original_error': 'x', 'submitted_unix': 2, 'call_id': 'fc-renew'}
    store[sha + '/recovery/' + LABEL + '/a3'] = recovery
    calls['fc-stale'] = ValueError('prior call or original deadline changed')
    calls['fc-renew'] = {'status': 'retry-renewed', 'label': LABEL, 'attempt_number': 3}
    report = app.dispatch_cloud()
    assert launched == [] and report['errors'] == []
    assert store[sha + '/recovery/' + LABEL + '/a3'] == recovery
    assert sha + '/hold/' + LABEL not in store
    store[sha + '/dispatch/' + LABEL + '/a3']['call_id'] = 'fc-direct'
    calls['fc-direct'] = {'status': 'retry-renewed'}
    report = app.dispatch_cloud()
    assert launched == [] and report['errors'] == []


def test_dispatcher_holds_renewed_attempt_under_its_own_scope(dispatcher, case):
    store, calls, launched, sha = dispatcher
    renewed = renew(case)
    scope = '/a3/renewal-' + renewed['renews_authorization_sha256']
    store[sha + '/retry/' + LABEL + '/a3'] = renewed
    store[sha + '/recovery/' + LABEL + '/a3'] = {'submissions': 3, 'call_id': 'fc-old'}
    calls['fc-old'] = {'status': 'retry-renewed'}
    store[sha + '/dispatch/' + LABEL + scope] = {'submitted_unix': 1, 'submissions': 1,
                                                'expires_unix': renewed['expires_unix'], 'call_id': 'fc-new'}
    calls['fc-new'] = RuntimeError('container lost')
    app.dispatch_cloud()
    recovery = store[sha + '/recovery/' + LABEL + scope]
    assert recovery['submissions'] == 1 and launched[-1][1][1] == {'recover_only': True, 'attempt': 3}
    assert launched[-1][1][0][1] == renewed['expires_unix']
    for _ in range(3):
        calls[recovery['call_id']] = RuntimeError('still lost')
        app.dispatch_cloud()
        recovery = store[sha + '/recovery/' + LABEL + scope]
    hold = store[sha + '/hold/' + LABEL]
    assert hold['recovery_exhausted'] is True and hold['attempt'] == 3
    assert hold['dispatch']['call_id'] == 'fc-new' and hold['recovery']['submissions'] == 3
    assert store[sha + '/recovery/' + LABEL + '/a3'] == {'submissions': 3, 'call_id': 'fc-old'}
