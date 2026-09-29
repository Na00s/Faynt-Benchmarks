"""Legacy physical history preserves old calls across resumed allocations."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import modal_panel_cloud_queue as q


def fixture():
    binding = {'call_id': 'fc-original', 'expires_unix': 100, 'image_id': 'im-old'}
    failure = {
        'manifest': {'status': 'failed', 'call_id': 'fc-original'},
        'physical_history': [
            {'call_id': 'fc-original', 'expires_unix': 100,
             'execution': {'execution_id': 'old-1'}},
        ],
    }
    authorization = {'expires_unix': 1200, 'original_expires_unix': 500}
    return failure, binding, authorization


def test_legacy_prior_deadline_comes_from_frozen_history():
    failure, binding, authorization = fixture()
    before = copy.deepcopy((failure, binding, authorization))
    q.validate_prior_call_binding(failure, binding, authorization)
    assert (failure, binding, authorization) == before


def test_legacy_two_physical_starts_keep_same_old_call():
    failure, binding, authorization = fixture()
    failure['physical_history'].append({**failure['physical_history'][0],
                                       'execution': {'execution_id': 'old-2'}})
    q.validate_prior_call_binding(failure, binding, authorization)


def test_legacy_unresumed_history_still_validates():
    failure, binding, _ = fixture()
    q.validate_prior_call_binding(failure, binding, {'expires_unix': 100})


@pytest.mark.parametrize('mutation', [
    'missing-binding', 'binding-call', 'binding-expiry', 'manifest-call',
    'empty-history', 'missing-history', 'history-call', 'history-expiry',
    'later-start-call', 'later-start-expiry',
])
def test_legacy_binding_rejects_changed_or_missing_proofs(mutation):
    failure, binding, authorization = fixture()
    if mutation == 'missing-binding':
        binding = None
    elif mutation == 'binding-call':
        binding['call_id'] = 'fc-changed'
    elif mutation == 'binding-expiry':
        binding['expires_unix'] = authorization['expires_unix']
    elif mutation == 'manifest-call':
        failure['manifest']['call_id'] = 'fc-changed'
    elif mutation == 'empty-history':
        failure['physical_history'] = []
    elif mutation == 'missing-history':
        del failure['physical_history']
    elif mutation == 'history-call':
        failure['physical_history'][0]['call_id'] = 'fc-changed'
    elif mutation == 'history-expiry':
        failure['physical_history'][0]['expires_unix'] = authorization['expires_unix']
    else:
        extra = copy.deepcopy(failure['physical_history'][0])
        extra['call_id' if mutation == 'later-start-call' else 'expires_unix'] = (
            'fc-changed' if mutation == 'later-start-call' else 1200)
        failure['physical_history'].append(extra)
    with pytest.raises(ValueError, match='prior call or original deadline changed'):
        q.validate_prior_call_binding(failure, binding, authorization)


def test_explicit_prior_binding_still_requires_exact_proof():
    failure, binding, authorization = fixture()
    failure['call_binding'] = copy.deepcopy(binding)
    q.validate_prior_call_binding(failure, binding, authorization)
    binding['image_id'] = 'im-changed'
    with pytest.raises(ValueError, match='retained prior call or deadline changed'):
        q.validate_prior_call_binding(failure, binding, authorization)
