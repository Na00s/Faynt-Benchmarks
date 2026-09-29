from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import report_slippi_public_panel as report


def fixture_panel(characters=("FOX",), pairs_per_character=(1,), profiles=("75m",), releases=("test",)):
    games = []
    for profile in profiles:
        for release in releases:
            for character, pair_count in zip(characters, pairs_per_character):
                for index in range(pair_count):
                    pair_id = f"{profile}-{release}-{character}-{index}"
                    for port in (1, 2):
                        games.append({"profile": profile, "release": release, "block": "supported-mirror",
                                      "label": f"{pair_id}-p{port}", "pair_id": pair_id, "seed": index,
                                      "stage": "BATTLEFIELD", "player_name": "Test Player",
                                      "frisson_character": character, "slippi_character": character,
                                      "frisson_port": port, "slippi_port": 3 - port})
    panel = {"experiment_id": "synthetic", "games": games}
    state = {"status": "running", "games": {g["label"]: {"status": "pending"} for g in games}}
    return panel, state


def accept(state, game, win=True, taken=None, conceded=None):
    taken = (4 if win else 2) if taken is None else taken
    conceded = (1 if win else 4) if conceded is None else conceded
    state["games"][game["label"]] = {"status": "complete", "result": {
        "win": win, "winner": "frisson-ai" if win else "slippi-ai",
        "stocks_taken": taken, "stocks_conceded": conceded,
        "stocks_remaining": {"frisson-ai": 4 - conceded, "slippi-ai": 4 - taken},
    }}


def test_empty_results_have_no_estimate_and_full_uncertainty():
    panel, state = fixture_panel()
    row = report.analyze(panel, state)["rows"][0]
    assert row["accepted"] == row["wins"] == row["stocks_taken"] == 0
    assert row["paired_macro_win_fraction"] is None
    assert row["pointwise_interval"]["lower"] == 0
    assert row["pointwise_interval"]["upper"] == 1
    assert row["full_schedule_descriptive_bounds"] == {"lower": 0, "upper": 1, "unknown_weight": 1}


def test_all_wins_retain_nonzero_uncertainty():
    panel, state = fixture_panel(pairs_per_character=(6,))
    for game in panel["games"]:
        accept(state, game)
    row = report.analyze(panel, state)["rows"][0]
    assert row["paired_macro_win_fraction"] == 1
    assert row["wins"] == 12 and row["losses"] == 0
    assert 0 < row["pointwise_interval"]["lower"] < 1
    assert row["pointwise_interval"]["upper"] == 1
    assert row["pointwise_interval"]["effective_pair_count"] == pytest.approx(6)
    assert row["full_schedule_descriptive_bounds"]["lower"] == pytest.approx(1)
    assert row["full_schedule_macro_win_fraction"] == 1


def test_game_counts_and_stocks_include_unpaired_but_primary_excludes_it():
    panel, state = fixture_panel(pairs_per_character=(2,))
    for game, win in zip(panel["games"], (True, False, True)):
        accept(state, game, win)
    row = report.analyze(panel, state)["rows"][0]
    assert (row["accepted"], row["wins"], row["losses"]) == (3, 2, 1)
    assert (row["stocks_taken"], row["stocks_conceded"]) == (10, 6)
    assert row["complete_pairs"] == 1
    assert row["accepted_games_excluded_from_paired_estimate"] == 1
    assert row["paired_macro_win_fraction"] == 0.5
    assert row["full_schedule_macro_win_fraction"] is None
    assert row["full_schedule_descriptive_bounds"] == {"lower": 0.25, "upper": 0.75, "unknown_weight": 0.5}


def test_quarantine_cannot_contribute_retained_winner_or_stocks():
    panel, state = fixture_panel(pairs_per_character=(2,))
    for game in panel["games"]:
        accept(state, game)
    state["games"][panel["games"][1]["label"]]["status"] = "quarantined"
    row = report.analyze(panel, state)["rows"][0]
    assert (row["accepted"], row["wins"], row["quarantined"]) == (3, 3, 1)
    assert row["stocks_taken"] == 12
    assert row["complete_pairs"] == 1
    assert row["accepted_games_excluded_from_paired_estimate"] == 1
    assert row["excluded_cells"][0]["statuses"] == ["complete", "quarantined"]
    assert row["observed_quarantined_games"] == 0


def observe_quarantine(state, game, win=False, taken=1, conceded=4):
    row = state["games"][game["label"]]
    row.update(status="quarantined", observed_outcome={
        "win": win, "winner": "frisson-ai" if win else "slippi-ai",
        "stocks_taken": taken, "stocks_conceded": conceded,
        "source_receipt": "/synthetic/forensic-outcome.json",
        "evidence_class": "forensic-slp-controller-outcome-missing-original-summary",
    })


def test_observed_quarantined_loss_is_separate_and_primary_is_unchanged():
    panel, state = fixture_panel(pairs_per_character=(2,))
    for game in panel["games"]:
        accept(state, game)
    quarantine = panel["games"][1]
    state["games"][quarantine["label"]]["status"] = "quarantined"
    baseline = report.analyze(panel, state)["rows"][0]
    observe_quarantine(state, quarantine)
    before = copy.deepcopy((panel, state))
    data = report.analyze(panel, state)
    row = data["rows"][0]
    assert {k: v for k, v in row.items() if not k.startswith("observed_quarantined_")} == {
        k: v for k, v in baseline.items() if not k.startswith("observed_quarantined_")}
    assert (row["accepted"], row["wins"], row["losses"], row["stocks_taken"], row["stocks_conceded"]) == (3, 3, 0, 12, 3)
    assert (row["observed_quarantined_games"], row["observed_quarantined_wins"],
            row["observed_quarantined_losses"], row["observed_quarantined_stocks_taken"],
            row["observed_quarantined_stocks_conceded"]) == (1, 0, 1, 1, 4)
    assert row["complete_pairs"] == 1
    assert row["paired_macro_win_fraction"] == 1
    outcome = row["observed_quarantined_outcomes"][0]
    assert outcome["label"] == quarantine["label"]
    assert outcome["winner"] == "slippi-ai"
    assert outcome["reason"] == outcome["evidence_class"]
    md = report.render_markdown(data)
    assert "| 1 | 0-1; 1 / 4 | 1 / 2 |" in md
    assert "Survivor-bias warning" in md
    assert "runtime/timing diagnostics" in md
    assert "forensic-slp-controller-outcome-missing-original-summary" in md
    assert "/synthetic/forensic-outcome.json" in md
    assert (panel, state) == before


def test_two_model_observed_quarantine_totals_and_reasons_stay_separate():
    panel, state = fixture_panel(profiles=("75m", "10m"), releases=("test", "other"))
    for game in panel["games"]:
        if game["frisson_port"] == 1:
            win = game["profile"] == "10m"
            observe_quarantine(state, game, win=win, taken=4 if win else 1, conceded=2 if win else 4)
            state["games"][game["label"]]["quarantine_reason"] = "summary missing | timing unavailable\nverified separately"
    data = report.analyze(panel, state)
    totals = data["totals_by_profile"]
    assert (totals["75m"]["observed_quarantined_losses"], totals["75m"]["observed_quarantined_stocks_taken"],
            totals["75m"]["observed_quarantined_stocks_conceded"]) == (2, 2, 8)
    assert (totals["10m"]["observed_quarantined_wins"], totals["10m"]["observed_quarantined_stocks_taken"],
            totals["10m"]["observed_quarantined_stocks_conceded"]) == (2, 8, 4)
    assert all(t["accepted"] == t["wins"] == t["losses"] == 0 for t in totals.values())
    assert all(r["complete_pairs"] == 0 and r["paired_macro_win_fraction"] is None for r in data["rows"])
    md = report.render_markdown(data)
    assert md.count("| Opponent | Block |") == 2
    assert "summary missing \\| timing unavailable verified separately" in md
    assert "Separately observed quarantines: 2 games; 0-2; stocks 2 / 8." in md
    assert "Separately observed quarantines: 2 games; 2-0; stocks 8 / 4." in md


def test_observed_outcome_on_pending_row_is_never_counted():
    panel, state = fixture_panel()
    observe_quarantine(state, panel["games"][0])
    state["games"][panel["games"][0]["label"]]["status"] = "pending"
    row = report.analyze(panel, state)["rows"][0]
    assert row["observed_quarantined_games"] == row["quarantined"] == row["accepted"] == 0


@pytest.mark.parametrize("change,match", [
    ({"win": 1}, "boolean"), ({"stocks_taken": 5}, "invalid stocks"),
    ({"stocks_conceded": True}, "invalid stocks"), ({"winner": "frisson-ai"}, "contradicts"),
    ({"source_receipt": ""}, "source_receipt"), ({"source_receipt": None}, "source_receipt"),
    ({"evidence_class": ""}, "evidence_class"),
])
def test_invalid_explicit_quarantined_outcomes_fail_loudly(change, match):
    panel, state = fixture_panel()
    observe_quarantine(state, panel["games"][0])
    state["games"][panel["games"][0]["label"]]["observed_outcome"].update(change)
    with pytest.raises(ValueError, match=match):
        report.analyze(panel, state)


def test_equal_character_weighting_with_unequal_pair_counts():
    panel, state = fixture_panel(characters=("FOX", "MARTH"), pairs_per_character=(3, 1))
    for game in panel["games"]:
        accept(state, game, game["frisson_character"] == "FOX")
    row = report.analyze(panel, state)["rows"][0]
    assert row["wins"] / row["accepted"] == 0.75
    assert row["paired_macro_win_fraction"] == pytest.approx(0.5)
    expected_sum_sq = 3 * (1 / 6) ** 2 + (1 / 2) ** 2
    assert row["pointwise_interval"]["effective_pair_count"] == pytest.approx(1 / expected_sum_sq)
    assert row["full_schedule_descriptive_bounds"]["lower"] == pytest.approx(0.5)


def test_missing_character_not_silently_treated_as_full_roster_score():
    panel, state = fixture_panel(characters=("FOX", "MARTH"), pairs_per_character=(1, 1))
    for game in panel["games"][:2]:
        accept(state, game)
    row = report.analyze(panel, state)["rows"][0]
    assert row["paired_macro_win_fraction"] == 1
    assert (row["observed_characters"], row["scheduled_characters"]) == (1, 2)
    assert row["full_schedule_macro_win_fraction"] is None
    assert row["full_schedule_descriptive_bounds"] == {"lower": 0.5, "upper": 1, "unknown_weight": 0.5}
    assert row["characters"]["MARTH"]["paired_win_fraction"] is None


def test_two_model_totals_and_tables_remain_separate():
    panel, state = fixture_panel(profiles=("75m", "10m"), releases=("test", "other"))
    for game in panel["games"]:
        accept(state, game, game["profile"] == "75m")
    data = report.analyze(panel, state)
    assert data["totals_by_profile"]["75m"]["wins"] == 4
    assert data["totals_by_profile"]["10m"]["losses"] == 4
    assert sum(t["accepted"] for t in data["totals_by_profile"].values()) == 8
    md = report.render_markdown(data)
    assert md.count("| Opponent | Block |") == 2
    assert md.index("## 75M RL") < md.index("## 10M RL")
    assert "starting CSS assignments" in md
    assert "formal Game End event or separately audited" in md
    assert "anytime-valid" in md and "Universal SOTA" in md


def test_same_pair_id_in_other_model_is_separate_comparison():
    panel, state = fixture_panel(profiles=("75m", "10m"))
    for game in panel["games"]:
        game["pair_id"] = "shared-id"
        accept(state, game)
    assert [r["complete_pairs"] for r in report.analyze(panel, state)["rows"]] == [1, 1]


@pytest.mark.parametrize("mutation,match", [
    (lambda p, s: p["games"].append(copy.deepcopy(p["games"][0])), "duplicated"),
    (lambda p, s: s["games"].pop(p["games"][0]["label"]), "labels differ"),
    (lambda p, s: s["games"][p["games"][0]["label"]].update(status="rejected"), "unrecognized ledger"),
    (lambda p, s: p["games"][1].update(frisson_port=1, slippi_port=2), "opposite physical"),
    (lambda p, s: p["games"][1].update(slippi_port=2), "invalid physical"),
    (lambda p, s: p["games"][1].update(seed=5), "differs in seed"),
    (lambda p, s: p["games"][1].update(stage="DREAMLAND"), "differs in stage"),
    (lambda p, s: p["games"][0].update(block="unknown"), "unrecognized benchmark"),
])
def test_malformed_inputs_fail_loudly(mutation, match):
    panel, state = fixture_panel()
    mutation(panel, state)
    with pytest.raises(ValueError, match=match):
        report.analyze(panel, state)


@pytest.mark.parametrize("change,match", [
    ({"win": 1}, "boolean"), ({"stocks_taken": 5}, "invalid stocks"),
    ({"stocks_conceded": True}, "invalid stocks"), ({"winner": "slippi-ai"}, "contradicts"),
    ({"stocks_conceded": 2}, "contradict remaining"),
])
def test_inconsistent_accepted_results_fail(change, match):
    panel, state = fixture_panel()
    accept(state, panel["games"][0])
    state["games"][panel["games"][0]["label"]]["result"].update(change)
    with pytest.raises(ValueError, match=match):
        report.analyze(panel, state)


def test_hoeffding_formula_family_and_boundaries():
    result = report.weighted_hoeffding([0, 1] * 100, [1 / 200] * 200)
    assert result["estimate"] == 0.5
    assert result["radius"] == pytest.approx(math.sqrt(math.log(2 * 84 / .05) / 400))
    narrower = report.weighted_hoeffding([0, 1] * 100, [1 / 200] * 200, comparisons=1)
    assert narrower["radius"] < result["radius"]
    assert report.weighted_hoeffding([0], [1])["lower"] == 0
    assert report.weighted_hoeffding([1], [1])["upper"] == 1


@pytest.mark.parametrize("values,weights", [([1], []), ([2], [1]), ([math.nan], [1]), ([1], [.5]), ([1], [-1])])
def test_invalid_uncertainty_inputs(values, weights):
    with pytest.raises(ValueError):
        report.weighted_hoeffding(values, weights)


def test_report_does_not_mutate_inputs():
    panel, state = fixture_panel()
    before = copy.deepcopy((panel, state))
    report.analyze(panel, state)
    assert (panel, state) == before


def test_cli_snapshot_hashes_and_source_files_unchanged(tmp_path, capsys):
    panel, state = fixture_panel()
    manifest_bytes = json.dumps(panel).encode()
    state["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    (tmp_path / "state.json").write_text(json.dumps(state))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert report.main(["--panel", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    data = json.loads(Path(output["data"]).read_text())
    assert data["manifest_sha256"] == state["manifest_sha256"]
    assert data["state_sha256"] == hashlib.sha256(before["state.json"]).hexdigest()
    assert Path(output["report"]).is_relative_to(tmp_path / "analysis")
    assert all((tmp_path / name).read_bytes() == content for name, content in before.items())
    assert report.main(["--panel", str(tmp_path)]) == 0
    capsys.readouterr()
    assert len(list((tmp_path / "analysis").glob("*.json"))) == 1


def test_cli_rejects_wrong_binding_and_output_scope(tmp_path):
    panel, state = fixture_panel()
    (tmp_path / "manifest.json").write_text(json.dumps(panel))
    (tmp_path / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="different manifest"):
        report.main(["--panel", str(tmp_path)])
    with pytest.raises(SystemExit):
        report.main(["--panel", str(tmp_path), "--output-dir", str(tmp_path)])


def test_frozen_actual_panel_has_exact_predeclared_comparisons():
    manifest_path = report.DEFAULT_PANEL / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local frozen public-panel manifest unavailable")
    panel = json.loads(manifest_path.read_text())
    state = {"status": "synthetic", "games": {g["label"]: {"status": "pending"} for g in panel["games"]}}
    data = report.analyze(panel, state)
    assert len(data["rows"]) == 84
    assert {k: v["total"] for k, v in data["totals_by_profile"].items()} == {"75m": 1312, "10m": 1312}
    assert sum(r["scheduled_pairs"] for r in data["rows"]) == 1312
