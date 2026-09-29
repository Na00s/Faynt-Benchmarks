import copy
import pickle

from slippi_panel_audit import digest, json_safe, load_restricted


def test_public_checkpoint_runtime_paths_normalize_without_changing_bytes(tmp_path):
    home = '/' + '/'.join(('home', 'researcher', 'SSBM'))
    nested_home = '/' + '/'.join(('home', 'User', 'researcher', 'SSBM'))
    metadata = {'config': {'policy': {'delay': 21}, 'dataset': {'allowed_characters': 'fox'},
                           'env': {'iso': home + '/SSBM.iso', 'path': nested_home + '/Slippi.AppImage'}},
                'step': 123}
    checkpoint = tmp_path / 'numeric-metadata.pkl'
    checkpoint.write_bytes(pickle.dumps({**metadata, 'state': {}}, protocol=4))
    before = checkpoint.read_bytes()
    expected_hash = digest(checkpoint)
    loaded = load_restricted(checkpoint)
    original = copy.deepcopy(loaded)
    actual = {k: json_safe(v) for k, v in loaded.items() if k != 'state'}
    expected = copy.deepcopy(metadata)
    expected['config']['env'] = {'iso': '${UPSTREAM_RUNTIME}/SSBM/SSBM.iso',
                                'path': '${UPSTREAM_RUNTIME}/SSBM/Slippi.AppImage'}
    assert actual == expected
    assert loaded == original
    assert checkpoint.read_bytes() == before
    assert digest(checkpoint) == expected_hash
    assert json_safe(actual) == actual
    assert json_safe({'name': 'SSBM', 'path': '/opt/runtime/SSBM.iso'}) == {
        'name': 'SSBM', 'path': '/opt/runtime/SSBM.iso'}
