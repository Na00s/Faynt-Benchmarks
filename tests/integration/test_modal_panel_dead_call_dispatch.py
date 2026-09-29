"""A recorded call that can never answer is replaced on the same attempt.

A call from a stopped app stays unresolved forever. Once its recorded deadline
is more than the grace period in the past the dispatcher treats it as dead: a
never-owned dispatch is archived and repeated with a fresh deadline; an owned
attempt replays under a redispatch scope named by the dead call, and the worker
proves in the durable index that no physical start was consumed before it binds
the scoped owner. Every superseded record stays retained.
"""
import ast
import copy
import inspect
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import modal_panel_cloud_queue as q
import modal_panel_cloud_app as app
from test_modal_panel_retry_renewal import (  # noqa: F401
    case, worker_state, renew_in_worker, LABEL as RENEWAL_LABEL, OLD_DEADLINE)

LABEL = 'faynt-d0-632-980-v1-sp21-75m-fox_d21_ditto_v4-forced-mirror-cptfalcon-battlefield-p1'
PEER = 'faynt-d0-632-980-v1-sp21-10m-fox_d21_ditto_v4-forced-mirror-cptfalcon-battlefield-p1'
PLAN = 'b' * 64
NOW = 1789700000.0
SESSION = 9090
GRACE = app.DEAD_CALL_GRACE_SECONDS
DEAD = NOW - 35 * 3600
OWNER = {'call_id': 'fc-owner', 'expires_unix': DEAD, 'image_id': 'im-old'}
SCOPE = '/redispatch-fc-owner'
DEAD_OWNED = 'owned call unresolved past its recorded deadline'
DEAD_UNOWNED = 'never-owned call unresolved past its recorded deadline'
STALE = 'stale admission deferral from an expired tranche'
REFUSED_ERROR = 'scoped redispatch refused: native start already bound'


class Store(dict):
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = copy.deepcopy(value)
        return True


def dispatch_record(call_id='fc-owner', expires=DEAD, submissions=1):
    return {'submitted_unix': expires - SESSION, 'expires_unix': expires,
            'submissions': submissions, 'call_id': call_id}


def deferral(expires):
    """The answer a D0 worker gives when it could not admit the attempt in time."""
    answer = app.late_admission(LABEL, expires, now=expires - 1000, durable_module='faynt_d0_durable_game')
    assert answer['status'] == 'budget-admission-deferred'
    return answer


@pytest.fixture
def dispatcher(monkeypatch):
    """A first-attempt row in phase 0 and one runnable peer in phase 1, with a
    fake app, fake calls and a fake durable index the dispatcher may read."""
    store = Store(); index = Store(); sha = 'a' * 64
    monkeypatch.setenv('PANEL_SNAPSHOT_SHA', sha)
    monkeypatch.delenv('PANEL_DURABLE_MODULE', raising=False)
    rows = [{'label': LABEL, 'initial_status': 'pending', 'plan_sha256': PLAN, 'phase': 0, 'worker_slot': 0},
            {'label': PEER, 'initial_status': 'pending', 'plan_sha256': 'c' * 64, 'phase': 1, 'worker_slot': 0}]
    queue = {'games': rows, 'queue_deadline_unix': float('inf')}
    monkeypatch.setattr(app, 'snapshot', lambda: queue)
    monkeypatch.setattr(app, 'handles', lambda: store)
    monkeypatch.setattr(app, 'frozen_game_session_seconds', lambda row: SESSION)
    monkeypatch.setattr(app.time, 'time', lambda: NOW)
    store[sha + '/control'] = {'enabled': True, 'image_id': 'im-test'}
    calls = {}; launched = []; index_names = []
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
    def from_name(name, **kwargs):
        index_names.append((name, kwargs.get('environment_name')))
        return index
    modal = types.SimpleNamespace(
        Function=types.SimpleNamespace(from_name=lambda *a, **k: types.SimpleNamespace(spawn=spawn)),
        FunctionCall=types.SimpleNamespace(from_id=Call),
        Dict=types.SimpleNamespace(from_name=from_name))
    monkeypatch.setitem(sys.modules, 'modal', modal)
    return types.SimpleNamespace(store=store, index=index, calls=calls, launched=launched, sha=sha,
                                 index_names=index_names)


def records(store):
    """Every durable record of the row under test; the peer lane may start once
    the row is held, and the per-tick health report is not evidence."""
    return {k: v for k, v in store.items() if LABEL in k}


def spawns(d, label=LABEL):
    return [item for item in d.launched if item[0] == label]


# ---------------------------------------------------------------- never owned


def test_never_owned_dead_call_is_archived_and_redispatched_with_a_fresh_expiry(dispatcher):
    d = dispatcher; dead = dispatch_record('fc-dead')
    d.store[d.sha + '/dispatch/' + LABEL] = copy.deepcopy(dead)
    d.calls['fc-dead'] = None
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    assert report['waiting'] == 0 and report['errors'] == [] and report['admission_deferred'] == []
    # Same attempt, same scope, no redispatch scope: the worker binds the plain owner key.
    assert d.launched == [(LABEL, ((d.sha, NOW + SESSION, 'im-test'), {'attempt': 1}), 'fc-0')]
    assert d.store[d.sha + '/dispatch-history/' + LABEL + '/fc-dead'] == dead
    assert d.store[d.sha + '/dispatch/' + LABEL] == {'submitted_unix': NOW, 'expires_unix': NOW + SESSION,
                                                     'submissions': 2, 'call_id': 'fc-0'}
    assert set(records(d.store)) - set(before) == {d.sha + '/dispatch-history/' + LABEL + '/fc-dead'}
    assert not any(k.startswith(d.sha + kind) for k in d.store
                   for kind in ('/owner', '/redispatch/', '/hold/', '/recovery/'))
    assert report['redispatched'] == [{'label': LABEL, 'attempt': 1, 'dead_call_id': 'fc-dead',
                                       'reason': DEAD_UNOWNED, 'scope': ''}]
    assert report['submitted'] == [{'label': LABEL, 'call_id': 'fc-0'}]
    # The replacement is running: the row waits and nothing else is spawned.
    report = app.dispatch_cloud()
    assert report['waiting'] == 1 and len(d.launched) == 1 and report['redispatched'] == []
    assert d.store[d.sha + '/dispatch-history/' + LABEL + '/fc-dead'] == dead


def test_never_owned_dead_call_at_the_submission_bound_holds_without_a_spawn(dispatcher):
    d = dispatcher; dead = dispatch_record('fc-dead', submissions=3)
    d.store[d.sha + '/dispatch/' + LABEL] = copy.deepcopy(dead)
    d.calls['fc-dead'] = None
    report = app.dispatch_cloud()
    assert spawns(d) == [] and report['held_labels'] == [LABEL]
    hold = d.store[d.sha + '/hold/' + LABEL]
    assert hold['error'] == 'ambiguous dispatch retry bound reached' and hold['dispatch'] == dead
    assert d.store[d.sha + '/dispatch-history/' + LABEL + '/fc-dead'] == dead
    assert d.store[d.sha + '/dispatch/' + LABEL] == dead


def test_a_second_dead_never_owned_call_archives_under_its_own_id(dispatcher):
    d = dispatcher
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-dead')
    d.calls['fc-dead'] = None
    app.dispatch_cloud()
    # The replacement also dies (app stopped again) after its own deadline.
    d.store[d.sha + '/dispatch/' + LABEL]['expires_unix'] = NOW - GRACE - 1
    second = copy.deepcopy(d.store[d.sha + '/dispatch/' + LABEL])
    report = app.dispatch_cloud()
    assert d.store[d.sha + '/dispatch-history/' + LABEL + '/fc-0'] == second
    assert d.store[d.sha + '/dispatch-history/' + LABEL + '/fc-dead'] == dispatch_record('fc-dead')
    assert d.store[d.sha + '/dispatch/' + LABEL]['submissions'] == 3 and report['errors'] == []
    assert [item[2] for item in spawns(d)] == ['fc-0', 'fc-1']
    # Three submissions on this attempt: a third death holds the row.
    d.store[d.sha + '/dispatch/' + LABEL]['expires_unix'] = NOW - GRACE - 1
    report = app.dispatch_cloud()
    assert len(spawns(d)) == 2 and d.store[d.sha + '/hold/' + LABEL]['error'] == 'ambiguous dispatch retry bound reached'


# ------------------------------------------------------------- grace period


@pytest.mark.parametrize('expires', [NOW + 5000, NOW, NOW - 1, NOW - GRACE + 1, NOW - GRACE])
@pytest.mark.parametrize('owned', [False, True])
def test_a_call_inside_its_deadline_or_grace_is_still_waiting(dispatcher, expires, owned):
    d = dispatcher
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner', expires=expires)
    if owned:
        d.store[d.sha + '/owner/' + LABEL] = {**OWNER, 'expires_unix': expires}
    d.calls['fc-owner'] = None
    before = copy.deepcopy(d.store)
    for _ in range(2):
        report = app.dispatch_cloud()
        assert report['waiting'] == 1 and d.launched == [] and report['errors'] == []
        assert report['redispatched'] == [] and records(d.store) == records(before)


def test_a_call_one_second_past_the_grace_is_dead(dispatcher):
    d = dispatcher
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner', expires=NOW - GRACE - 1)
    d.calls['fc-owner'] = None
    report = app.dispatch_cloud()
    assert len(spawns(d)) == 1 and report['redispatched'][0]['dead_call_id'] == 'fc-owner'


def test_an_owner_without_a_recorded_deadline_is_never_dead(dispatcher):
    d = dispatcher
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner', expires=DEAD)
    d.store[d.sha + '/owner/' + LABEL] = {'call_id': 'fc-owner'}
    d.calls['fc-owner'] = None
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    assert report['waiting'] == 1 and d.launched == [] and records(d.store) == records(before)


def test_dead_call_record_requires_a_finite_past_deadline():
    assert app.dead_call_record({'expires_unix': NOW - GRACE - 1}, now=NOW)
    assert not app.dead_call_record({'expires_unix': NOW - GRACE}, now=NOW)
    assert not app.dead_call_record({'expires_unix': float('inf')}, now=NOW)
    assert not app.dead_call_record({'expires_unix': float('-inf')}, now=NOW)
    assert not app.dead_call_record({'expires_unix': '1'}, now=NOW)
    assert not app.dead_call_record({}, now=NOW) and not app.dead_call_record(None, now=NOW)


# ------------------------------------------------------------------- owned


def owned_dead(d, dispatch=True):
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    if dispatch:
        d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner')
    d.calls['fc-owner'] = None


def assert_scoped_redispatch(d, before, reason, dispatch=True):
    assert d.launched == [(LABEL, ((d.sha, NOW + SESSION, 'im-test'), {'attempt': 1, 'redispatch': 'fc-owner'}), 'fc-0')]
    marker = d.store[d.sha + '/redispatch/' + LABEL]
    assert marker['schema'] == q.SCHEMA + '.redispatch' and marker['call_id'] == 'fc-owner'
    assert marker['attempt'] == 1 and marker['reason'] == reason and marker['time'] == NOW
    assert marker['superseded_owner'] == OWNER and marker['dispatch_scope'] == ''
    assert marker['superseded_dispatch'] == (dispatch_record('fc-owner') if dispatch else None)
    assert marker['score_admitted'] is False
    assert d.store[d.sha + '/owner-history/' + LABEL + '/fc-owner'] == OWNER
    assert d.store[d.sha + '/dispatch/' + LABEL + SCOPE] == {'submitted_unix': NOW, 'expires_unix': NOW + SESSION,
                                                            'submissions': 1, 'call_id': 'fc-0'}
    added = {d.sha + '/redispatch/' + LABEL, d.sha + '/owner-history/' + LABEL + '/fc-owner',
             d.sha + '/dispatch/' + LABEL + SCOPE}
    if dispatch:
        added.add(d.sha + '/dispatch-history/' + LABEL + '/fc-owner')
        assert d.store[d.sha + '/dispatch-history/' + LABEL + '/fc-owner'] == dispatch_record('fc-owner')
    assert set(records(d.store)) - set(before) == added
    assert all(d.store[k] == v for k, v in before.items())
    return marker


@pytest.mark.parametrize('dispatch', [True, False])
def test_owned_dead_call_gets_a_scoped_redispatch_with_a_scoped_owner_key(dispatcher, dispatch):
    d = dispatcher; owned_dead(d, dispatch=dispatch)
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    assert report['waiting'] == 0 and report['errors'] == [] and report['admission_deferred'] == []
    marker = assert_scoped_redispatch(d, before, DEAD_OWNED, dispatch=dispatch)
    assert 'released_hold_sha256' not in marker
    assert report['redispatched'] == [{'label': LABEL, 'attempt': 1, 'dead_call_id': 'fc-owner',
                                       'reason': DEAD_OWNED, 'scope': SCOPE}]
    # The dispatcher polls the scoped owner once the worker binds it, never the dead one.
    scoped_owner = d.sha + '/owner/' + LABEL + SCOPE
    d.store[scoped_owner] = {'call_id': 'fc-0', 'expires_unix': NOW + SESSION, 'image_id': 'im-test'}
    report = app.dispatch_cloud()
    assert report['waiting'] == 1 and len(d.launched) == 1 and report['errors'] == []
    d.calls['fc-0'] = {'status': 'complete', 'label': LABEL, 'native_acceptance': {}}
    report = app.dispatch_cloud()
    assert len(d.launched) == 1 and report['errors'] == [] and report['redispatched'] == []
    assert d.store[d.sha + '/owner/' + LABEL] == OWNER and d.store[scoped_owner]['call_id'] == 'fc-0'
    assert d.store[d.sha + '/redispatch/' + LABEL] == marker


def test_stale_deferred_owned_answer_is_redispatched_scoped(dispatcher):
    d = dispatcher
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner')
    d.calls['fc-owner'] = deferral(DEAD)
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    assert report['admission_deferred'] == [] and report['errors'] == []
    assert_scoped_redispatch(d, before, STALE)
    assert not any(k.startswith(d.sha + '/recovery/') for k in d.store)


def test_a_stale_deferral_needs_no_grace_but_a_live_one_stays_listed(dispatcher):
    d = dispatcher
    d.store[d.sha + '/owner/' + LABEL] = {**OWNER, 'expires_unix': NOW - 1}
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner', expires=NOW - 1)
    d.calls['fc-owner'] = deferral(NOW + 1)
    report = app.dispatch_cloud()
    assert report['admission_deferred'] == [d.calls['fc-owner']] and d.launched == []
    d.calls['fc-owner'] = deferral(NOW - 1)
    report = app.dispatch_cloud()
    assert report['admission_deferred'] == [] and len(spawns(d)) == 1
    assert d.launched[0][1][1] == {'attempt': 1, 'redispatch': 'fc-owner'}


def test_owned_dead_retry_attempt_redispatches_under_its_retry_keys(dispatcher):
    d = dispatcher
    retry = {'attempt_number': 2, 'expires_unix': NOW + 4000, 'plan_sha256': 'd' * 64}
    d.store[d.sha + '/retry-latest/' + LABEL] = 2
    d.store[d.sha + '/retry/' + LABEL + '/a2'] = retry
    d.store[d.sha + '/owner/' + LABEL + '/a2'] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL + '/a2'] = dispatch_record('fc-owner')
    d.calls['fc-owner'] = None
    report = app.dispatch_cloud()
    assert d.launched == [(LABEL, ((d.sha, NOW + 4000, 'im-test'), {'attempt': 2, 'redispatch': 'fc-owner'}), 'fc-0')]
    assert d.store[d.sha + '/redispatch/' + LABEL + '/a2']['dispatch_scope'] == '/a2'
    assert d.store[d.sha + '/owner-history/' + LABEL + '/a2/fc-owner'] == OWNER
    assert d.store[d.sha + '/dispatch-history/' + LABEL + '/a2/fc-owner'] == dispatch_record('fc-owner')
    assert d.store[d.sha + '/dispatch/' + LABEL + '/a2' + SCOPE]['expires_unix'] == NOW + 4000
    assert d.sha + '/redispatch/' + LABEL not in d.store and report['errors'] == []


# ---------------------------------------------------------------- refusal


def test_worker_refuses_a_scoped_redispatch_when_the_index_shows_a_native_binding():
    index = Store()
    assert app.refuse_started_redispatch(index, LABEL, 1, 'fc-owner', PLAN) is None
    binding = {'call_id': 'fc-owner', 'expires_unix': DEAD, 'image_id': 'im-old'}
    index[PLAN + '/call'] = binding
    refusal = app.refuse_started_redispatch(index, LABEL, 1, 'fc-owner', PLAN)
    assert refusal == {'status': app.REDISPATCH_REFUSED, 'label': LABEL, 'attempt_number': 1,
                       'dead_call_id': 'fc-owner', 'plan_sha256': PLAN, 'call_binding': binding,
                       'physical_history': [], 'score_admitted': False}
    slot = {'call_id': 'fc-owner', 'started_unix': DEAD - 100, 'expires_unix': DEAD, 'execution': {}}
    index[PLAN + '/physical/1'] = slot
    assert app.refuse_started_redispatch(index, LABEL, 1, 'fc-owner', PLAN)['physical_history'] == [slot]
    del index[PLAN + '/call']
    assert app.refuse_started_redispatch(index, LABEL, 1, 'fc-owner', PLAN)['call_binding'] is None
    assert index == {PLAN + '/physical/1': slot}


def test_worker_requires_the_dispatcher_marker_and_the_archived_owner():
    state = Store(); key = 'a' * 64 + '/'
    with pytest.raises(ValueError, match='dispatcher marker'):
        app.validate_redispatch_scope(state, key, LABEL, '', 'fc-owner')
    state[key + 'redispatch/' + LABEL] = {'call_id': 'fc-other'}
    with pytest.raises(ValueError, match='dispatcher marker'):
        app.validate_redispatch_scope(state, key, LABEL, '', 'fc-owner')
    state[key + 'redispatch/' + LABEL] = {'call_id': 'fc-owner'}
    with pytest.raises(ValueError, match='superseded owner record'):
        app.validate_redispatch_scope(state, key, LABEL, '', 'fc-owner')
    state[key + 'owner/' + LABEL] = OWNER
    with pytest.raises(ValueError, match='archived'):
        app.validate_redispatch_scope(state, key, LABEL, '', 'fc-owner')
    state[key + 'owner-history/' + LABEL + '/fc-owner'] = OWNER
    assert app.validate_redispatch_scope(state, key, LABEL, '', 'fc-owner') == {'call_id': 'fc-owner'}
    with pytest.raises(ValueError, match='call identity'):
        app.redispatch_scope('owner')
    with pytest.raises(ValueError, match='call identity'):
        app.redispatch_scope(None)
    assert app.redispatch_scope('fc-owner') == SCOPE


def test_refused_scoped_redispatch_takes_the_ordinary_recovery_path(dispatcher):
    d = dispatcher; owned_dead(d)
    app.dispatch_cloud()
    d.calls['fc-0'] = {'status': app.REDISPATCH_REFUSED, 'label': LABEL, 'dead_call_id': 'fc-owner',
                       'call_binding': {'call_id': 'fc-owner', 'expires_unix': DEAD, 'image_id': 'im-old'}}
    recovery_key = d.sha + '/recovery/' + LABEL + SCOPE
    report = app.dispatch_cloud()
    assert d.launched[-1] == (LABEL, ((d.sha, NOW + SESSION, 'im-test'),
                                      {'recover_only': True, 'attempt': 1, 'redispatch': 'fc-owner'}), 'fc-1')
    assert d.store[recovery_key] == {'submissions': 1, 'original_error': REFUSED_ERROR,
                                     'submitted_unix': NOW, 'call_id': 'fc-1'}
    assert report['errors'] == [] and report['redispatched'] == []
    assert d.store[d.sha + '/dispatch/' + LABEL + SCOPE]['submissions'] == 1
    # The recovery classifies the saved failure: attempt 2 is authorized.
    d.calls['fc-1'] = {'status': 'retry-authorized', 'label': LABEL, 'attempt_number': 2}
    report = app.dispatch_cloud()
    assert len(d.launched) == 2 and report['errors'] == []


# ----------------------------------------------------------------- bounds


def test_redispatch_scope_is_used_at_most_once_per_dead_owner(dispatcher):
    d = dispatcher; owned_dead(d)
    app.dispatch_cloud()
    marker = copy.deepcopy(d.store[d.sha + '/redispatch/' + LABEL])
    # The scoped owner also dies: no chained scope, the row is held for review.
    d.store[d.sha + '/owner/' + LABEL + SCOPE] = {'call_id': 'fc-0', 'expires_unix': NOW - GRACE - 1,
                                                  'image_id': 'im-test'}
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    assert len(spawns(d)) == 1 and report['held_labels'] == [LABEL]
    hold = d.store[d.sha + '/hold/' + LABEL]
    assert hold['error'] == 'redispatch scope already consumed: ' + DEAD_OWNED
    assert hold['owner']['call_id'] == 'fc-0' and hold['redispatch_call_id'] == 'fc-owner'
    assert d.store[d.sha + '/redispatch/' + LABEL] == marker
    assert set(records(d.store)) - set(before) == {d.sha + '/hold/' + LABEL}
    assert not any(k.startswith(d.sha + '/owner-history/' + LABEL + SCOPE) for k in d.store)
    report = app.dispatch_cloud()
    assert len(spawns(d)) == 1 and report['held_labels'] == [LABEL] and spawns(d, PEER)


def test_normal_submissions_per_attempt_stay_bounded_by_three_across_scopes(dispatcher):
    d = dispatcher
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner', submissions=2)
    d.calls['fc-owner'] = None
    report = app.dispatch_cloud()
    assert len(spawns(d)) == 1 and report['errors'] == []
    assert d.store[d.sha + '/dispatch/' + LABEL + SCOPE]['submissions'] == 1
    # The scoped call dies before binding its owner: two earlier submissions
    # plus this one reach the bound, so the row is held instead of respawned.
    d.store[d.sha + '/dispatch/' + LABEL + SCOPE]['expires_unix'] = NOW - GRACE - 1
    scoped = copy.deepcopy(d.store[d.sha + '/dispatch/' + LABEL + SCOPE])
    report = app.dispatch_cloud()
    assert len(spawns(d)) == 1
    assert d.store[d.sha + '/dispatch-history/' + LABEL + SCOPE + '/fc-0'] == scoped
    assert d.store[d.sha + '/hold/' + LABEL]['error'] == 'ambiguous dispatch retry bound reached'
    assert report['held_labels'] == [LABEL]


def test_three_submissions_before_the_owner_died_leave_no_scoped_spawn(dispatcher):
    d = dispatcher
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner', submissions=3)
    d.calls['fc-owner'] = None
    report = app.dispatch_cloud()
    assert spawns(d) == [] and d.store[d.sha + '/redispatch/' + LABEL]['call_id'] == 'fc-owner'
    assert d.store[d.sha + '/hold/' + LABEL]['error'] == 'ambiguous dispatch retry bound reached'
    assert d.store[d.sha + '/owner-history/' + LABEL + '/fc-owner'] == OWNER


# ------------------------------------------------------------ hold release


def exhausted_hold(d, error=STALE):
    hold = {'schema': q.SCHEMA + '.scheduling-hold', 'label': LABEL, 'queue_sha256': d.sha,
            'queue_plan_sha256': PLAN, 'attempt': 1, 'error': error, 'time': DEAD + 100,
            'score_admitted': False, 'dispatch': dispatch_record('fc-owner'), 'owner': OWNER,
            'recovery': {'submissions': 3, 'original_error': error, 'submitted_unix': DEAD + 50,
                         'call_id': 'fc-rec'}, 'recovery_exhausted': True}
    d.store[d.sha + '/hold/' + LABEL] = copy.deepcopy(hold)
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner')
    d.store[d.sha + '/recovery/' + LABEL] = copy.deepcopy(hold['recovery'])
    d.calls['fc-rec'] = {'status': 'awaiting-native-result'}
    d.calls['fc-owner'] = deferral(DEAD)
    return hold


def test_recovery_exhausted_hold_behind_a_dead_owner_is_released_once(dispatcher):
    d = dispatcher; hold = exhausted_hold(d)
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    assert report['held'] == 0 and report['held_labels'] == [] and report['errors'] == []
    assert d.launched == [(LABEL, ((d.sha, NOW + SESSION, 'im-test'), {'attempt': 1, 'redispatch': 'fc-owner'}), 'fc-0')]
    marker = d.store[d.sha + '/redispatch/' + LABEL]
    assert marker['released_hold_sha256'] == q.digest(hold) and marker['reason'] == STALE
    assert d.store[d.sha + '/hold/' + LABEL] == hold
    assert d.store[d.sha + '/recovery/' + LABEL] == hold['recovery']
    assert d.index_names == [(app.WORKER_PATHS['modal_panel_durable_game']['index'], 'main')]
    assert set(records(d.store)) - set(before) == {
        d.sha + '/redispatch/' + LABEL, d.sha + '/owner-history/' + LABEL + '/fc-owner',
        d.sha + '/dispatch-history/' + LABEL + '/fc-owner', d.sha + '/dispatch/' + LABEL + SCOPE}
    # The marker keeps the row released while the scoped run is running.
    report = app.dispatch_cloud()
    assert report['held'] == 0 and report['waiting'] == 1 and len(d.launched) == 1
    # A hold raised under the scope replaces the released hold and holds again.
    d.store[d.sha + '/owner/' + LABEL + SCOPE] = {'call_id': 'fc-0', 'expires_unix': NOW - GRACE - 1,
                                                  'image_id': 'im-test'}
    report = app.dispatch_cloud()
    assert report['held_labels'] == [LABEL] and len(d.launched) == 1
    assert d.store[d.sha + '/hold-history/redispatch-fc-owner/' + LABEL] == hold
    replaced = d.store[d.sha + '/hold/' + LABEL]
    assert replaced['error'].startswith('redispatch scope already consumed') and replaced != hold
    report = app.dispatch_cloud()
    assert report['held_labels'] == [LABEL] and len(d.launched) == 2 and spawns(d, PEER)
    assert d.store[d.sha + '/hold/' + LABEL] == replaced


def test_hold_stays_when_the_index_shows_a_native_binding(dispatcher):
    d = dispatcher; hold = exhausted_hold(d)
    d.index[PLAN + '/call'] = {'call_id': 'fc-owner', 'expires_unix': DEAD, 'image_id': 'im-old'}
    before = copy.deepcopy(d.store)
    for _ in range(2):
        report = app.dispatch_cloud()
        assert report['held_labels'] == [LABEL] and spawns(d) == [] and report['errors'] == []
        assert records(d.store) == records(before)
    assert spawns(d, PEER) and d.store[d.sha + '/hold/' + LABEL] == hold


@pytest.mark.parametrize('variation', ['not-exhausted', 'other-attempt', 'live-owner', 'no-owner', 'index-error'])
def test_hold_release_requires_exhaustion_current_attempt_and_a_dead_owner(dispatcher, variation):
    d = dispatcher; exhausted_hold(d)
    if variation == 'not-exhausted':
        del d.store[d.sha + '/hold/' + LABEL]['recovery_exhausted']
    elif variation == 'other-attempt':
        d.store[d.sha + '/retry-latest/' + LABEL] = 2
        d.store[d.sha + '/retry/' + LABEL + '/a2'] = {'attempt_number': 2, 'expires_unix': NOW + 4000,
                                                      'plan_sha256': 'd' * 64}
    elif variation == 'live-owner':
        d.store[d.sha + '/owner/' + LABEL]['expires_unix'] = NOW - GRACE
    elif variation == 'no-owner':
        del d.store[d.sha + '/owner/' + LABEL]
    else:
        class Broken(Store):
            def get(self, name, default=None): raise RuntimeError('index unavailable')
        d.index.__class__ = Broken
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    if variation == 'other-attempt':
        # Attempt 2 has its own keys; the old hold on attempt 1 is not released.
        assert report['held_labels'] == [LABEL] and spawns(d) == []
    else:
        assert report['held_labels'] == [LABEL] and spawns(d) == []
    assert records(d.store) == records(before)


def test_exhausted_recovery_behind_a_dead_owner_supersedes_instead_of_holding(dispatcher):
    d = dispatcher
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner')
    d.store[d.sha + '/recovery/' + LABEL] = {'submissions': 3, 'original_error': 'x', 'submitted_unix': DEAD,
                                             'call_id': 'fc-rec'}
    d.calls['fc-owner'] = RuntimeError('result expired')
    d.calls['fc-rec'] = RuntimeError('container lost')
    report = app.dispatch_cloud()
    marker = d.store[d.sha + '/redispatch/' + LABEL]
    assert marker['reason'].startswith('recovery exhausted behind a dead owner: result expired; recovery: container lost')
    assert d.launched[-1][1][1] == {'attempt': 1, 'redispatch': 'fc-owner'} and report['held_labels'] == []
    assert d.store[d.sha + '/recovery/' + LABEL]['submissions'] == 3


def test_exhausted_recovery_with_a_native_binding_still_holds(dispatcher):
    d = dispatcher
    d.store[d.sha + '/owner/' + LABEL] = copy.deepcopy(OWNER)
    d.store[d.sha + '/dispatch/' + LABEL] = dispatch_record('fc-owner')
    d.store[d.sha + '/recovery/' + LABEL] = {'submissions': 3, 'original_error': 'x', 'submitted_unix': DEAD,
                                             'call_id': 'fc-rec'}
    d.index[PLAN + '/call'] = {'call_id': 'fc-owner', 'expires_unix': DEAD, 'image_id': 'im-old'}
    d.calls['fc-owner'] = RuntimeError('result expired')
    d.calls['fc-rec'] = RuntimeError('container lost')
    before = copy.deepcopy(d.store)
    report = app.dispatch_cloud()
    hold = d.store[d.sha + '/hold/' + LABEL]
    assert hold['recovery_exhausted'] is True and spawns(d) == [] and report['held_labels'] == [LABEL]
    assert set(records(d.store)) - set(before) == {d.sha + '/hold/' + LABEL}


# --------------------------------------------------------- worker renewal


def test_worker_renews_an_expired_retry_under_the_scoped_owner_key(case):
    row, body, expired, old, current, now = case
    state, key, index = worker_state(case, owner=True)
    scoped = key + 'owner/' + RENEWAL_LABEL + '/a3' + SCOPE
    with pytest.raises(ValueError, match='logical owner'):
        renew_in_worker(case, state, key, index)
    result = renew_in_worker(case, state, key, index, owner_key=scoped)
    assert result['status'] == 'retry-renewed' and result['attempt_number'] == 3
    assert state[key + 'owner/' + RENEWAL_LABEL + '/a3'] == {'call_id': 'fc-owner', 'expires_unix': OLD_DEADLINE}
    state[scoped] = {'call_id': 'fc-new', 'expires_unix': now + 100}
    with pytest.raises(ValueError, match='logical owner'):
        renew_in_worker(case, state, key, index, retry=state[key + 'retry/' + RENEWAL_LABEL + '/a3'],
                        owner_key=scoped)


# ------------------------------------------------------- source inspection


def code(function):
    source = inspect.getsource(function)
    first = ast.parse(source).body[0].body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        return ''.join(source.splitlines(keepends=True)[first.end_lineno:])
    return source


def test_worker_checks_the_index_before_binding_a_scoped_owner():
    worker = code(app._run_cloud_game)
    scoped_key = worker.index("owner_key=key+'owner/'+label+suffix+redispatch_scope(redispatch)")
    validated = worker.index('validate_redispatch_scope(state,key,label,suffix,redispatch)')
    checked = worker.index('refuse_started_redispatch(durable_index(),label,attempt,redispatch,')
    plain_bind = worker.index('cloud.bind_call(state,owner_key,call_id,expires,image_id)')
    materialized = worker.index("for upload in plan['uploads']:")
    admitted = worker.index('admitted_native_result(d,volume,index,plan_sha,bound,expires,image_id,label,')
    d0_bind = worker.index('cloud.bind_call(state,owner_key,call_id,expires,image_id))')
    assert scoped_key < validated < checked < plain_bind < materialized < admitted < d0_bind
    assert 'if not recover_only:' in worker[validated:checked]
    assert "state.get(owner_key,{}).get('call_id')" in worker[admitted:]
    assert "(state.get(owner_key) or state.get(key+'owner/'+label+suffix))['call_id']" in worker
    assert 'owner_key=owner_key' in worker[worker.index('return renew_unstarted_retry('):]
    # Every owner binding goes through the scoped key; the plain key is only read.
    assert "cloud.bind_call(state,key+'owner/'" not in worker
    assert worker.count('cloud.bind_call(') == 2
    assert 'bind_call' not in worker[:plain_bind]
    # The check reads the index and the state; it never writes either.
    for helper in (app.refuse_started_redispatch, app.validate_redispatch_scope, app.native_start_bound,
                   app.dead_call_record, app.redispatch_scope):
        source = code(helper)
        assert '.put(' not in source and 'volume' not in source and 'importlib' not in source
    assert "index.get(plan_sha+'/call')" in code(app.native_start_bound)
    assert "index.get(plan_sha+f'/physical/{n}')" in code(app.native_start_bound)
    # The frozen native workers do not know about redispatch scopes at all.
    scripts = Path(app.__file__).parent
    for name in ('modal_panel_durable_game.py', 'faynt_d0_durable_game.py'):
        assert 'redispatch' not in (scripts / name).read_text()
    assert 'redispatch' not in code(app.admitted_native_result)
    # The dispatcher never treats a call as dead without the grace period.
    dispatcher = inspect.getsource(app.dispatch_cloud)
    assert dispatcher.count('dead_call_record(') == 3
    assert 'DEAD_CALL_GRACE_SECONDS' in code(app.dead_call_record) and app.DEAD_CALL_GRACE_SECONDS == 600
    text = ast.get_source_segment(dispatcher, next(
        node for node in ast.walk(ast.parse(dispatcher).body[0])
        if isinstance(node, ast.FunctionDef) and node.name == 'dispatch_one'))
    timeout = text.index('except TimeoutError:')
    dead = text.index('if not dead_call_record(owner or dispatch):')
    waiting = text.index('waiting.append(label);return', dead)
    owned = text.index('if owner:', dead)
    archived = text.index("archive(key+'dispatch-history/'+label+scope+'/'+call_id,dispatch)")
    assert timeout < dead < waiting < owned < archived < text.index('except Exception as error:')
