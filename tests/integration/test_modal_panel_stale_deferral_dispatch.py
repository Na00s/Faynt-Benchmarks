"""A stale admission deferral from an expired tranche never defers forever.

An owner that answered before any native start is superseded by a scoped
redispatch; a retained failed attempt and a never-owned row keep read-only
recovery, which classifies the saved result."""
import ast
import copy
import inspect
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import modal_panel_cloud_app as app

LABEL = 'faynt-d0-632-980-v1-sp21-75m-fox_d21_ditto_v4-forced-mirror-cptfalcon-battlefield-p1'
PEER = 'faynt-d0-632-980-v1-sp21-10m-fox_d21_ditto_v4-forced-mirror-cptfalcon-battlefield-p1'
OLD_DEADLINE = 1789602658.8854022
NOW = OLD_DEADLINE + 3600
SESSION = 9090
STALE_ERROR = 'stale admission deferral from an expired tranche'
OWNER = {'call_id': 'fc-owner', 'expires_unix': OLD_DEADLINE, 'image_id': 'im-old'}
SCOPE = '/redispatch-fc-owner'


class Store(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = copy.deepcopy(value)
        return True


def deferral(expires):
    """The exact answer a D0 worker gives after taking the owner slot too late."""
    answer = app.late_admission(LABEL, expires, now=expires - 1000, durable_module='faynt_d0_durable_game')
    assert answer['status'] == 'budget-admission-deferred' and answer['expires_unix'] == expires
    return answer


@pytest.fixture
def dispatcher(monkeypatch):
    """A first-attempt owned row in phase 0 and one runnable peer in phase 1."""
    store = Store(); sha = 'a' * 64
    monkeypatch.setenv('PANEL_SNAPSHOT_SHA', sha)
    monkeypatch.delenv('PANEL_DURABLE_MODULE', raising=False)
    rows = [{'label': LABEL, 'initial_status': 'pending', 'plan_sha256': 'b' * 64, 'phase': 0, 'worker_slot': 0},
            {'label': PEER, 'initial_status': 'pending', 'plan_sha256': 'c' * 64, 'phase': 1, 'worker_slot': 0}]
    queue = {'games': rows, 'queue_deadline_unix': float('inf')}
    monkeypatch.setattr(app, 'snapshot', lambda: queue)
    monkeypatch.setattr(app, 'handles', lambda: store)
    monkeypatch.setattr(app, 'frozen_game_session_seconds', lambda row: SESSION)
    monkeypatch.setattr(app.time, 'time', lambda: NOW)
    store[sha + '/control'] = {'enabled': True, 'image_id': 'im-test'}
    store[sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
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
        FunctionCall=types.SimpleNamespace(from_id=Call),
        Dict=types.SimpleNamespace(from_name=lambda *a, **k: Store()))
    monkeypatch.setitem(sys.modules, 'modal', modal)
    return store, calls, launched, sha


def records(store):
    """Every durable record except the per-tick health report."""
    return {k: v for k, v in store.items() if not k.endswith('/health')}


def recovery_spawns(launched):
    return [item for item in launched if item[0] == LABEL and item[1][1].get('recover_only')]


@pytest.mark.parametrize('dispatch_record', [True, False])
def test_stale_owned_deferral_is_superseded_by_one_scoped_redispatch(dispatcher, dispatch_record):
    store, calls, launched, sha = dispatcher
    frozen = OLD_DEADLINE - 5
    dispatch = {'submitted_unix': OLD_DEADLINE - 9000, 'expires_unix': frozen,
                'submissions': 1, 'call_id': 'fc-owner'}
    if dispatch_record:
        store[sha + '/dispatch/' + LABEL] = copy.deepcopy(dispatch)
    before = copy.deepcopy(store)
    calls['fc-owner'] = deferral(OLD_DEADLINE)
    report = app.dispatch_cloud()
    assert report['admission_deferred'] == [] and report['errors'] == [] and report['waiting'] == 0
    assert report['submitted'] == [{'label': LABEL, 'call_id': 'fc-0'}]
    # The dead owner's replacement runs under its own scope with a fresh deadline.
    assert launched == [(LABEL, ((sha, NOW + SESSION, 'im-test'), {'attempt': 1, 'redispatch': 'fc-owner'}), 'fc-0')]
    marker = store[sha + '/redispatch/' + LABEL]
    assert marker['call_id'] == 'fc-owner' and marker['reason'] == STALE_ERROR and marker['attempt'] == 1
    assert marker['superseded_owner'] == OWNER and marker['dispatch_scope'] == ''
    assert marker['superseded_dispatch'] == (dispatch if dispatch_record else None)
    assert 'released_hold_sha256' not in marker and marker['score_admitted'] is False
    assert store[sha + '/owner-history/' + LABEL + '/fc-owner'] == OWNER
    assert store[sha + '/dispatch/' + LABEL + SCOPE] == {'submitted_unix': NOW, 'expires_unix': NOW + SESSION,
                                                        'submissions': 1, 'call_id': 'fc-0'}
    added = {sha + '/redispatch/' + LABEL, sha + '/owner-history/' + LABEL + '/fc-owner',
             sha + '/dispatch/' + LABEL + SCOPE}
    if dispatch_record:
        added.add(sha + '/dispatch-history/' + LABEL + '/fc-owner')
        assert store[sha + '/dispatch-history/' + LABEL + '/fc-owner'] == dispatch
    assert set(records(store)) - set(before) == added
    assert all(store[k] == v for k, v in before.items())
    assert sha + '/hold/' + LABEL not in store and not any(k.startswith(sha + '/recovery/') for k in store)
    assert report['redispatched'] == [{'label': LABEL, 'attempt': 1, 'dead_call_id': 'fc-owner',
                                       'reason': STALE_ERROR, 'scope': SCOPE}]
    # Second tick: the scoped run is still running, so nothing else is spawned.
    store[sha + '/owner/' + LABEL + SCOPE] = {'call_id': 'fc-0', 'expires_unix': NOW + SESSION, 'image_id': 'im-test'}
    report = app.dispatch_cloud()
    assert report['waiting'] == 1 and report['admission_deferred'] == [] and len(launched) == 1
    assert report['redispatched'] == [] and report['errors'] == []
    # The scoped run classified the attempt and authorized attempt 2. Attempt 2
    # dispatches under its own keys; the superseded first attempt stays retained.
    calls['fc-0'] = {'status': 'retry-authorized', 'label': LABEL, 'attempt_number': 2,
                     'reason': 'verified-preowner-interruption', 'failed_attempt_retained': True,
                     'score_admitted': False}
    report = app.dispatch_cloud()
    assert report['admission_deferred'] == [] and report['errors'] == [] and len(launched) == 1
    store[sha + '/retry-latest/' + LABEL] = 2
    store[sha + '/retry/' + LABEL + '/a2'] = {'attempt_number': 2, 'expires_unix': NOW + SESSION}
    report = app.dispatch_cloud()
    assert launched[-1] == (LABEL, ((sha, NOW + SESSION, 'im-test'), {'attempt': 2}), 'fc-1')
    assert store[sha + '/dispatch/' + LABEL + '/a2']['call_id'] == 'fc-1'
    assert store[sha + '/owner/' + LABEL] == OWNER and report['errors'] == []
    assert store[sha + '/redispatch/' + LABEL] == marker


def test_stale_retained_failure_deferral_keeps_read_only_recovery(dispatcher):
    """A deferral answered after a native failure names a consumed start: the
    saved result is classified by recovery, never replaced by a redispatch."""
    store, calls, launched, sha = dispatcher
    store[sha + '/dispatch/' + LABEL] = {'submitted_unix': OLD_DEADLINE - 9000, 'expires_unix': OLD_DEADLINE,
                                         'submissions': 1, 'call_id': 'fc-owner'}
    before = copy.deepcopy(store)
    calls['fc-owner'] = {**deferral(OLD_DEADLINE), 'failed_attempt_retained': True}
    report = app.dispatch_cloud()
    assert report['submitted'] == [{'label': LABEL, 'call_id': 'fc-0', 'read_only_recovery': True}]
    assert launched == [(LABEL, ((sha, OLD_DEADLINE, 'im-test'), {'recover_only': True, 'attempt': 1}), 'fc-0')]
    assert store[sha + '/recovery/' + LABEL] == {'submissions': 1, 'original_error': STALE_ERROR,
                                                 'submitted_unix': NOW, 'call_id': 'fc-0'}
    assert set(records(store)) - set(before) == {sha + '/recovery/' + LABEL}
    assert report['redispatched'] == [] and sha + '/redispatch/' + LABEL not in store


def test_stale_never_owned_deferral_keeps_read_only_recovery(dispatcher):
    store, calls, launched, sha = dispatcher
    del store[sha + '/owner/' + LABEL]
    store[sha + '/dispatch/' + LABEL] = {'submitted_unix': OLD_DEADLINE - 9000, 'expires_unix': OLD_DEADLINE,
                                         'submissions': 1, 'call_id': 'fc-owner'}
    calls['fc-owner'] = deferral(OLD_DEADLINE)
    report = app.dispatch_cloud()
    assert launched == [(LABEL, ((sha, OLD_DEADLINE, 'im-test'), {'recover_only': True, 'attempt': 1}), 'fc-0')]
    assert report['redispatched'] == [] and sha + '/redispatch/' + LABEL not in store
    assert store[sha + '/recovery/' + LABEL]['original_error'] == STALE_ERROR


@pytest.mark.parametrize('offset', [5000, app.D0_NATIVE_ADMISSION_SECONDS, 1, 0])
def test_current_deferral_is_still_listed_and_spawns_nothing(dispatcher, offset):
    store, calls, launched, sha = dispatcher
    store[sha + '/dispatch/' + LABEL] = {'submitted_unix': NOW - 60, 'expires_unix': NOW + offset,
                                         'submissions': 1, 'call_id': 'fc-owner'}
    before = copy.deepcopy(store)
    calls['fc-owner'] = deferral(NOW + offset)
    for _ in range(2):
        report = app.dispatch_cloud()
        assert report['admission_deferred'] == [calls['fc-owner']]
        assert launched == [] and report['errors'] == [] and report['waiting'] == 0
        assert records(store) == before


def test_recovery_bound_still_holds_the_row_and_frees_the_phase(dispatcher):
    store, calls, launched, sha = dispatcher
    store[sha + '/dispatch/' + LABEL] = {'submitted_unix': OLD_DEADLINE - 9000, 'expires_unix': OLD_DEADLINE,
                                         'submissions': 1, 'call_id': 'fc-owner'}
    calls['fc-owner'] = deferral(OLD_DEADLINE)
    report = app.dispatch_cloud()
    assert len(launched) == 1 and report['held_labels'] == []
    # The scoped run found a native binding: the owner did start after all.
    # Ordinary read-only recovery under the scope is bounded as before.
    calls['fc-0'] = {'status': app.REDISPATCH_REFUSED, 'label': LABEL, 'dead_call_id': 'fc-owner'}
    recovery_key = sha + '/recovery/' + LABEL + SCOPE
    for _ in range(3):
        report = app.dispatch_cloud()
        assert report['admission_deferred'] == []
        # An owned attempt with no saved native manifest answers this every time.
        calls[store[recovery_key]['call_id']] = {'status': 'awaiting-native-result'}
    report = app.dispatch_cloud()
    assert len(recovery_spawns(launched)) == 3
    assert [item[1][1] for item in recovery_spawns(launched)] == [
        {'recover_only': True, 'attempt': 1, 'redispatch': 'fc-owner'}] * 3
    hold = store[sha + '/hold/' + LABEL]
    assert hold['recovery_exhausted'] is True and hold['attempt'] == 1 and hold['score_admitted'] is False
    assert hold['error'] == 'scoped redispatch refused: native start already bound'
    assert hold['owner'] is None and hold['redispatch_call_id'] == 'fc-owner'
    assert hold['recovery']['submissions'] == 3 and hold['dispatch']['call_id'] == 'fc-0'
    assert report['held_labels'] == [LABEL] and report['errors'] == [hold]
    assert store[recovery_key]['submissions'] == 3
    assert store[sha + '/owner/' + LABEL] == OWNER
    # The held row no longer pins phase 0: the next tick starts the phase 1 peer.
    report = app.dispatch_cloud()
    assert [item[0] for item in launched if item[0] == PEER] == [PEER]
    assert launched[-1][1] == ((sha, NOW + SESSION, 'im-test'), {'attempt': 1})
    assert len(recovery_spawns(launched)) == 3 and report['held_labels'] == [LABEL]
    assert store[sha + '/hold/' + LABEL] == hold


@pytest.mark.parametrize('answer', [
    {'status': 'complete', 'label': LABEL, 'native_acceptance': {}},
    {'status': 'retry-authorized', 'label': LABEL, 'attempt_number': 2,
     'reason': 'verified-preowner-interruption', 'failed_attempt_retained': True, 'score_admitted': False},
    {'status': 'retry-renewed', 'label': LABEL, 'attempt_number': 3, 'expires_unix': OLD_DEADLINE},
])
def test_resolved_answers_are_unaffected(dispatcher, answer):
    store, calls, launched, sha = dispatcher
    store[sha + '/dispatch/' + LABEL] = {'submitted_unix': OLD_DEADLINE - 9000, 'expires_unix': OLD_DEADLINE,
                                         'submissions': 1, 'call_id': 'fc-owner'}
    before = copy.deepcopy(store)
    calls['fc-owner'] = answer
    report = app.dispatch_cloud()
    assert launched == [] and report['admission_deferred'] == [] and report['errors'] == []
    assert report['waiting'] == 0 and records(store) == before


def test_stale_check_sits_between_resolved_return_and_current_deferral():
    source = inspect.getsource(app.dispatch_cloud)
    body = ast.parse(source).body[0]
    dispatch_one = next(node for node in ast.walk(body)
                        if isinstance(node, ast.FunctionDef) and node.name == 'dispatch_one')
    text = ast.get_source_segment(source, dispatch_one)
    resolved = text.index("if result.get('status') in {'complete','retry-authorized','retry-renewed'}:return")
    deferred = text.index("if result.get('status')=='budget-admission-deferred':")
    stale = text.index("time.time()>stale")
    superseded = text.index("if owner and not result.get('failed_attempt_retained'):")
    recovery = text.index("recover_result(row,dispatch or {'expires_unix':stale},")
    listed = text.index('deferred.append(result);return')
    refused = text.index("if result.get('status')==REDISPATCH_REFUSED:")
    owned = text.index("retain_hold(row,'owned call returned without durable acceptance'")
    assert resolved < deferred < stale < superseded < recovery < listed < refused < owned
    assert STALE_ERROR in text[superseded:recovery] and STALE_ERROR in text[recovery:listed]
    # Four call sites: interrupted return, stale deferral, refused redispatch
    # and the expired-unstarted-retry renewal branch.
    assert text.count('recover_result(') == 4
    for untouched in (app.late_admission, app.retry_late_admission, app._run_cloud_game, app.renew_unstarted_retry):
        assert 'stale' not in inspect.getsource(untouched)
