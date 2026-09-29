"""An expired retry that never took ownership is recovered for renewal, not deferred forever."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import modal_panel_budget as budget
import modal_panel_cloud_queue as q
import modal_panel_cloud_app as app
from test_modal_panel_retry_renewal import case, dispatcher, LABEL, OLD_DEADLINE, SESSION  # noqa: F401

REASON = 'expired unstarted retry awaiting renewal'
RECOVERY = {'recover_only': True, 'attempt': 2}


@pytest.fixture
def expired_a2(case):
    """The live shape: an a2 authorization whose deadline was a past tranche
    deadline, one prior failure, and no dispatch, owner or physical start."""
    row, body, expired, old, current, now = case
    second = json.loads(body); second['artifact_labels'][0] = LABEL + '-a2'
    record = {**copy.deepcopy(expired), 'attempt_number': 2,
              'plan_sha256': q.digest(q.encoded(second)),
              'prior_failures': copy.deepcopy(expired['prior_failures'][:1])}
    q.retry_execution_plan(row, body, record)
    q.validate_retry_budget_binding(record, old)
    assert now > record['expires_unix'] == OLD_DEADLINE
    return record


@pytest.fixture
def renewable(monkeypatch, dispatcher, case, expired_a2):
    """Dispatcher tick over one pending row holding the expired a2, under the
    D0 native worker (so late_admission defers) with the label bound for
    reinspection under the active tranche."""
    row, body, expired, old, current, now = case
    store, calls, launched, sha = dispatcher
    monkeypatch.setenv('PANEL_DURABLE_MODULE', 'faynt_d0_durable_game')
    monkeypatch.setattr(app, 'active_resume_tranche', lambda state, key: current)
    store[sha + '/control']['held_reinspection'] = {
        'tranche_sha256': budget.digest(current), 'labels': [LABEL]}
    store[sha + '/retry-latest/' + LABEL] = 2
    store[sha + '/retry/' + LABEL + '/a2'] = expired_a2
    recovery_key = sha + '/recovery/' + LABEL + '/a2/resume-' + current['authorization_sha256']
    return store, calls, launched, sha, recovery_key


def renewed_record(case, expired_a2):
    row, body, _, _, current, now = case
    return q.renew_expired_retry(row, body, expired_a2, now=now, session_seconds=SESSION,
                                 resume_tranche=current, index_has_start=False)


def test_expired_unstarted_retry_spawns_one_read_only_recovery(renewable, case, expired_a2):
    store, calls, launched, sha, recovery_key = renewable
    _, _, _, _, _, now = case
    report = app.dispatch_cloud()
    assert launched == [(LABEL, ((sha, OLD_DEADLINE, 'im-test'), RECOVERY), 'fc-0')]
    assert report['admission_deferred'] == [] and report['errors'] == [] and report['waiting'] == 0
    assert report['submitted'] == [{'label': LABEL, 'call_id': 'fc-0', 'read_only_recovery': True}]
    assert store[recovery_key] == {'submissions': 1, 'original_error': REASON,
                                   'submitted_unix': now, 'call_id': 'fc-0'}
    assert sha + '/dispatch/' + LABEL + '/a2' not in store
    assert sha + '/owner/' + LABEL + '/a2' not in store
    assert sha + '/hold/' + LABEL not in store
    assert store[sha + '/retry/' + LABEL + '/a2'] == expired_a2
    assert store[sha + '/retry-latest/' + LABEL] == 2
    # While the recovery runs the row waits: no second spawn and no deferral.
    report = app.dispatch_cloud()
    assert len(launched) == 1 and report['waiting'] == 1
    assert report['admission_deferred'] == [] and report['errors'] == []
    assert store[recovery_key]['submissions'] == 1


def test_expired_retry_outside_reinspection_is_deferred_without_a_spawn(renewable, expired_a2):
    store, calls, launched, sha, recovery_key = renewable
    del store[sha + '/control']['held_reinspection']
    report = app.dispatch_cloud()
    assert launched == [] and report['errors'] == []
    (deferral,) = report['admission_deferred']
    assert deferral['status'] == 'budget-admission-deferred' and deferral['label'] == LABEL
    assert deferral['expires_unix'] == OLD_DEADLINE and deferral['score_admitted'] is False
    assert not any(key.startswith(sha + '/recovery/') for key in store)
    assert sha + '/dispatch/' + LABEL + '/a2' not in store and sha + '/hold/' + LABEL not in store
    assert store[sha + '/retry/' + LABEL + '/a2'] == expired_a2


def test_unexpired_deferral_stays_deferred_until_the_deadline_passes(monkeypatch, renewable, case):
    """The worker only renews a record whose deadline has passed, so a retry
    that is merely inside the admission window keeps the ordinary deferral
    instead of spending the bounded recovery submissions early."""
    store, calls, launched, sha, recovery_key = renewable
    _, _, _, _, _, now = case
    store[sha + '/retry/' + LABEL + '/a2']['expires_unix'] = now + 2000
    report = app.dispatch_cloud()
    assert launched == [] and recovery_key not in store
    assert [item['expires_unix'] for item in report['admission_deferred']] == [now + 2000]
    monkeypatch.setattr(app.time, 'time', lambda: now + 2001)
    report = app.dispatch_cloud()
    assert launched == [(LABEL, ((sha, now + 2000, 'im-test'), RECOVERY), 'fc-0')]
    assert report['admission_deferred'] == [] and store[recovery_key]['submissions'] == 1


def test_dispatch_record_for_the_current_scope_keeps_the_old_path(renewable, case):
    store, calls, launched, sha, recovery_key = renewable
    _, _, _, _, _, now = case
    stale = {'submitted_unix': now - 9000, 'expires_unix': OLD_DEADLINE, 'submissions': 1}
    store[sha + '/dispatch/' + LABEL + '/a2'] = stale
    report = app.dispatch_cloud()
    assert launched == [] and recovery_key not in store
    assert [item['label'] for item in report['admission_deferred']] == [LABEL]
    assert store[sha + '/dispatch/' + LABEL + '/a2'] == stale
    # A recorded call that fails recovers with the dispatch record's own
    # deadline and error, exactly as before.
    store[sha + '/dispatch/' + LABEL + '/a2'] = {**stale, 'call_id': 'fc-stale'}
    calls['fc-stale'] = ValueError('prior call or original deadline changed')
    report = app.dispatch_cloud()
    assert launched == [(LABEL, ((sha, OLD_DEADLINE, 'im-test'), RECOVERY), 'fc-0')]
    assert report['admission_deferred'] == []
    assert store[recovery_key]['original_error'] == 'prior call or original deadline changed'


def test_renewed_answer_dispatches_under_the_renewal_scope_next_tick(renewable, case, expired_a2):
    store, calls, launched, sha, recovery_key = renewable
    _, _, _, _, _, now = case
    app.dispatch_cloud()
    renewed = renewed_record(case, expired_a2)
    expired_sha = renewed['renews_authorization_sha256']
    # The worker archived the expired record and replaced the published one.
    store[sha + '/retry-history/' + LABEL + '/a2/' + expired_sha] = expired_a2
    store[sha + '/retry/' + LABEL + '/a2'] = renewed
    calls['fc-0'] = {'status': 'retry-renewed', 'label': LABEL, 'attempt_number': 2,
                     'renews_authorization_sha256': expired_sha, 'renewal_count': 1,
                     'expires_unix': renewed['expires_unix'], 'score_admitted': False}
    report = app.dispatch_cloud()
    scope = '/a2/renewal-' + expired_sha
    assert [item[0] for item in launched] == [LABEL, LABEL]
    (args, kwargs) = launched[1][1]
    assert args[1] == renewed['expires_unix'] and kwargs == {'attempt': 2}
    assert store[sha + '/dispatch/' + LABEL + scope] == {
        'submitted_unix': now, 'expires_unix': renewed['expires_unix'], 'submissions': 1, 'call_id': 'fc-1'}
    assert sha + '/dispatch/' + LABEL + '/a2' not in store
    assert store[recovery_key] == {'submissions': 1, 'original_error': REASON,
                                   'submitted_unix': now, 'call_id': 'fc-0'}
    assert report['admission_deferred'] == [] and report['errors'] == []
    assert report['submitted'] == [{'label': LABEL, 'call_id': 'fc-1'}]
    assert store[sha + '/retry-latest/' + LABEL] == 2
    assert store[sha + '/retry-history/' + LABEL + '/a2/' + expired_sha] == expired_a2
    calls['fc-1'] = None
    app.dispatch_cloud()
    assert len(launched) == 2


def test_worker_deferral_answer_is_reported_without_another_recovery(renewable):
    store, calls, launched, sha, recovery_key = renewable
    app.dispatch_cloud()
    answer = {'status': 'budget-admission-deferred', 'label': LABEL, 'expires_unix': OLD_DEADLINE + 5000,
              'score_admitted': False, 'expired_retry_retained': True}
    calls['fc-0'] = answer
    for _ in range(2):
        report = app.dispatch_cloud()
        assert report['admission_deferred'] == [answer] and report['errors'] == []
        assert len(launched) == 1 and store[recovery_key]['submissions'] == 1


def test_recovery_bound_holds_the_row_after_three_recoveries(renewable, case, expired_a2):
    store, calls, launched, sha, recovery_key = renewable
    _, _, _, _, current, _ = case
    for n in range(3):
        report = app.dispatch_cloud()
        assert store[recovery_key]['submissions'] == n + 1
        assert store[recovery_key]['call_id'] == f'fc-{n}' and report['errors'] == []
        calls[f'fc-{n}'] = RuntimeError('container lost')
    report = app.dispatch_cloud()
    assert [item[1][1] for item in launched] == [RECOVERY] * 3
    hold = store[sha + '/hold/' + LABEL]
    assert hold['recovery_exhausted'] is True and hold['attempt'] == 2
    assert hold['dispatch'] is None and hold['owner'] is None
    assert store[recovery_key]['submissions'] == 3 and store[recovery_key]['call_id'] == 'fc-2'
    assert hold['reinspection_token'] == budget.digest(current)
    assert hold['error'].startswith(REASON) and 'container lost' in hold['error']
    assert report['errors'] == [hold] and report['admission_deferred'] == []
    assert store[sha + '/retry/' + LABEL + '/a2'] == expired_a2
    # The held row leaves the candidate set: nothing more is spawned or deferred.
    report = app.dispatch_cloud()
    assert len(launched) == 3 and report['admission_deferred'] == [] and report['errors'] == []
    assert report['held_labels'] == [LABEL] and store[recovery_key]['submissions'] == 3
