"""Read-only recovery retains the exact expired retry budget and deadline."""
import copy

import pytest

import modal_panel_budget as budget
import modal_panel_cloud_app as app


def bound_retry(tranche):
    return {
        'expires_unix': tranche['activation']['deadline_unix'],
        'original_expires_unix': 1,
        'budget_resume': {
            'tranche_sha256': budget.digest(tranche),
            'authorization_sha256': tranche['authorization_sha256'],
            'activation_sha256': budget.digest(tranche['activation']),
        },
    }


@pytest.fixture
def history(monkeypatch):
    old = {'authorization_sha256': 'a' * 64, 'activation': {'deadline_unix': 100}}
    current = {'authorization_sha256': 'b' * 64, 'activation': {'deadline_unix': 200}}
    monkeypatch.setattr(app, 'resume_tranche_chain', lambda state, key: (old, current))
    return old, current


def test_ancestor_retry_can_be_recovered_without_mutation(history):
    old, current = history
    retry = bound_retry(old)
    before = copy.deepcopy(retry)
    app.validate_worker_retry_budget({}, 'queue/', retry, current, recover_only=True)
    assert retry == before


def test_ancestor_retry_cannot_launch_gameplay(history):
    old, current = history
    with pytest.raises(ValueError):
        app.validate_worker_retry_budget({}, 'queue/', bound_retry(old), current)


@pytest.mark.parametrize('recover_only', [False, True])
def test_current_retry_remains_valid(history, recover_only):
    _, current = history
    app.validate_worker_retry_budget({}, 'queue/', bound_retry(current), current,
                                    recover_only=recover_only)


def test_recovery_rejects_unknown_ancestry(history):
    old, current = history
    retry = bound_retry(old)
    retry['budget_resume']['tranche_sha256'] = 'c' * 64
    with pytest.raises(ValueError):
        app.validate_worker_retry_budget({}, 'queue/', retry, current, recover_only=True)


def test_recovery_rejects_extended_old_deadline(history):
    old, current = history
    retry = bound_retry(old)
    retry['expires_unix'] += 1
    with pytest.raises(ValueError):
        app.validate_worker_retry_budget({}, 'queue/', retry, current, recover_only=True)


def test_new_allocation_reinspects_deferred_recovery_without_hold(history, monkeypatch):
    _, current = history
    monkeypatch.setattr(app, 'active_resume_tranche', lambda state, key: current)
    state = {
        'queue/recovery/deferred/a2/resume-old': {'call_id': 'fc-old'},
        'queue/recovery/accepted': {'call_id': 'fc-done'},
        'queue/result/accepted': {'status': 'complete'},
        'queue/health': {'admission_deferred': [{'label': 'admission'}, {'label': 'unknown'}]},
    }
    before = copy.deepcopy(state)
    games = [{'label': label, 'initial_status': 'pending'}
             for label in ('deferred', 'accepted', 'admission', 'fresh')]
    control = app.activation_control(state, 'queue/', {'games': games}, image_id='im-same',
                                     strict_phase_order=True, reinspect_held=True)
    assert control['held_reinspection']['labels'] == ['admission', 'deferred']
    assert control['held_reinspection']['tranche_sha256'] == budget.digest(current)
    assert state == before
    assert control['strict_phase_order'] is True and control['allowlist'] is None
