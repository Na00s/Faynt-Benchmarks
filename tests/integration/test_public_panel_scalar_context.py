from types import SimpleNamespace
import json

import numpy as np
import pytest

import slippi_panel_runtime as runtime


@pytest.fixture
def configured_context(monkeypatch):
    """Restore the fresh-process adapter overrides after each scalar fixture."""
    import melee
    from melee_policy.integration import frisson_slippi_match as match
    from melee_policy.integration import frisson_match, game_bundle

    for module, names in (
        (match, ("FRISSON_PORT", "SLIPPI_PORT", "MATCH_STAGE", "DEFAULT_PLAYER_NAME",
                 "SlippiAIPolicySession", "_frisson_request", "_trace_row", "_first_context",
                 "_validate_slippi_release_config_contract", "_selected_replay_identity_audit",
                 "_runtime_reproducibility_record")),
        (frisson_match, ("FRISSON_PORT", "MIMIC_PORT", "MATCH_STAGE")),
        (game_bundle, ("_player_record",)),
        (melee, ("MenuHelper",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(runtime.boundary, "SLIPPI_AI_RELEASE_CONTRACTS",
                        dict(runtime.boundary.SLIPPI_AI_RELEASE_CONTRACTS))

    def configure(port):
        panel = runtime.manifest()
        game = {"frisson_port": port, "slippi_port": 3 - port,
                "stage": "POKEMON_STADIUM", "player_name": "Cody"}
        runtime.configure_game(game, panel["releases"]["fox_d21_ditto_v4"])
        request = SimpleNamespace(stage="POKEMON_STADIUM", player_1_character="YOSHI",
                                  player_2_character="FOX")
        players = {
            port: SimpleNamespace(character=melee.Character.YOSHI, costume=np.uint8(3)),
            3 - port: SimpleNamespace(character=melee.Character.FOX, costume=np.uint8(1)),
        }
        state = SimpleNamespace(frame=np.int32(-123), stage=melee.Stage.POKEMON_STADIUM,
                                players=players)
        return match._first_context, state, request

    return configure


@pytest.mark.parametrize("port", (1, 2))
@pytest.mark.parametrize("frame_type", (int, np.int32, np.int64))
def test_first_context_serializes_native_numpy_scalars(configured_context, port, frame_type):
    first_context, state, request = configured_context(port)
    state.frame = frame_type(-123)
    context = first_context(state, request)

    # json.dumps(sort_keys=True) is the real summary-writing boundary.
    restored = json.loads(json.dumps(context, indent=2, sort_keys=True))
    assert type(context["frame"]) is int
    assert context["frame"] == -123
    assert all(type(value) is bool and value for value in context["checks"].values())
    assert all(type(value) is int for value in context["costumes"].values())
    assert restored["characters"] == {f"p{port}": "YOSHI", f"p{3 - port}": "FOX"}
    assert restored["costumes"] == {f"p{port}": 3, f"p{3 - port}": 1}


@pytest.mark.parametrize("fault", ("frame", "stage", "characters"))
def test_numpy_scalar_conversion_keeps_context_validation(configured_context, fault):
    import melee

    first_context, state, request = configured_context(2)
    if fault == "frame":
        state.frame = np.int32(-122)
    elif fault == "stage":
        state.stage = melee.Stage.FINAL_DESTINATION
    else:
        state.players[2].character = melee.Character.FOX
    with pytest.raises(RuntimeError, match="first physical context mismatch"):
        first_context(state, request)
