#!/usr/bin/env python3
"""Read-only, paired descriptive analysis of the frozen public Slippi panel.

This script imports no gameplay code and never changes a game ledger. Its only
outputs are analysis JSON/Markdown under the panel's analysis or research folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1"
COMPARISONS = 84  # 2 frozen Frisson models x 14 unique releases x 3 blocks.
ALPHA = 0.05
BLOCKS = ("supported-mirror", "extended-roster", "forced-mirror")
STATUSES = {"pending", "running", "complete", "quarantined"}


def weighted_hoeffding(values, weights, *, alpha=ALPHA, comparisons=COMPARISONS):
    """Two-sided, fixed-look bound for independent bounded pair outcomes.

    For sum(w_i X_i), X_i in [0,1], positive weights summing to one,
    P(|estimate - expectation| >= r) <= 2 exp(-2 r^2 / sum(w_i^2)).
    A union bound spends alpha/comparisons on each predeclared comparison.
    Dependence inside a port pair is unrestricted. Dependence between pairs,
    adaptive sample selection and optional stopping are outside this guarantee.
    """
    if not 0 < alpha < 1 or type(comparisons) is not int or comparisons < 1:
        raise ValueError("invalid confidence family")
    if len(values) != len(weights):
        raise ValueError("values and weights have different lengths")
    if not values:
        return {"estimate": None, "lower": 0.0, "upper": 1.0,
                "radius": None, "effective_pair_count": 0.0}
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("pair outcomes must be finite and bounded by zero and one")
    if any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("pair weights must be positive and sum to one")
    estimate = math.fsum(w * x for w, x in zip(weights, values))
    squared_weights = math.fsum(w * w for w in weights)
    radius = math.sqrt(0.5 * squared_weights * math.log(2 * comparisons / alpha))
    return {"estimate": estimate, "lower": max(0.0, estimate - radius),
            "upper": min(1.0, estimate + radius), "radius": radius,
            "effective_pair_count": 1 / squared_weights}


def _validated_result(result, context):
    if type(result) is not dict:
        raise ValueError(f"{context} result must be a dictionary")
    if type(result.get("win")) is not bool:
        raise ValueError(f"{context} result lacks a boolean win")
    for field in ("stocks_taken", "stocks_conceded"):
        if type(result.get(field)) is not int or not 0 <= result[field] <= 4:
            raise ValueError(f"{context} result has invalid {field}")
    if "winner" in result and result["winner"] != ("frisson-ai" if result["win"] else "slippi-ai"):
        raise ValueError(f"{context} winner contradicts win flag")
    if "stocks_remaining" in result:
        stocks = result["stocks_remaining"]
        if any(type(stocks.get(k)) is not int or not 0 <= stocks[k] <= 4
               for k in ("frisson-ai", "slippi-ai")):
            raise ValueError(f"{context} remaining stocks are invalid")
        if (result["stocks_taken"], result["stocks_conceded"]) != (
                4 - stocks["slippi-ai"], 4 - stocks["frisson-ai"]):
            raise ValueError(f"{context} stock totals contradict remaining stocks")
    return result


def _accepted_result(row):
    return _validated_result(row.get("result", {}), "accepted")


def _observed_quarantine(row):
    """Explicit forensic outcomes stay separate from accepted game evidence."""
    if row["status"] != "quarantined" or "observed_outcome" not in row:
        return None
    result = _validated_result(row["observed_outcome"], "observed quarantined")
    for field in ("winner", "source_receipt", "evidence_class"):
        if type(result.get(field)) is not str or not result[field].strip():
            raise ValueError(f"observed quarantined result lacks {field}")
    reason = next((value for value in (result.get("reason"), row.get("quarantine_reason"),
                                      row.get("reason"), row.get("error"))
                   if type(value) is str and value.strip()), result["evidence_class"])
    return {**{field: result[field] for field in ("win", "winner", "stocks_taken", "stocks_conceded",
                                                "source_receipt", "evidence_class")}, "reason": reason}


def _pair_groups(specs):
    groups = defaultdict(list)
    for game in specs:
        if (game["frisson_port"], game["slippi_port"]) not in ((1, 2), (2, 1)):
            raise ValueError("manifest has invalid physical ports")
        groups[game["pair_id"]].append(game)
    for pair in groups.values():
        if len(pair) != 2 or {g["frisson_port"] for g in pair} != {1, 2}:
            raise ValueError("manifest pair must contain exactly two opposite physical ports")
        for field in ("seed", "stage", "frisson_character", "slippi_character", "player_name"):
            if len({g[field] for g in pair}) != 1:
                raise ValueError(f"manifest pair differs in {field}")
    return groups


def summarize_comparison(specs, ledger):
    """Describe all accepted games; estimate strength from complete pairs only."""
    pairs = _pair_groups(specs)
    accepted = {g["label"]: _accepted_result(ledger[g["label"]])
                for g in specs if ledger[g["label"]]["status"] == "complete"}
    observed_quarantines = []
    for game in specs:
        outcome = _observed_quarantine(ledger[game["label"]])
        if outcome is not None:
            observed_quarantines.append({"label": game["label"], **outcome})
    characters = {}
    complete_pairs = []
    incomplete_pairs = []
    for pair_id, pair in pairs.items():
        char = pair[0]["frisson_character"]
        stats = characters.setdefault(char, {"scheduled_pairs": 0, "complete_pairs": 0,
                                             "pair_outcomes": [], "accepted_games": 0})
        stats["scheduled_pairs"] += 1
        stats["accepted_games"] += sum(g["label"] in accepted for g in pair)
        available = [accepted[g["label"]] for g in pair if g["label"] in accepted]
        detail = {"pair_id": pair_id, "character": char, "stage": pair[0]["stage"],
                  "seed": pair[0]["seed"], "labels": [g["label"] for g in pair]}
        if len(available) == 2:
            outcome = sum(r["win"] for r in available) / 2
            stats["complete_pairs"] += 1
            stats["pair_outcomes"].append(outcome)
            complete_pairs.append({**detail, "win_fraction": outcome})
        else:
            incomplete_pairs.append({**detail, "accepted_games": len(available),
                                     "statuses": [ledger[g["label"]]["status"] for g in pair]})
    observed_chars = sum(bool(v["complete_pairs"]) for v in characters.values())
    values, weights = [], []
    known_schedule_score = 0.0
    known_schedule_weight = 0.0
    for stats in characters.values():
        n = stats["complete_pairs"]
        stats["paired_win_fraction"] = sum(stats["pair_outcomes"]) / n if n else None
        if n:
            values.extend(stats["pair_outcomes"])
            weights.extend([1 / (observed_chars * n)] * n)
            # Every scheduled character keeps equal target weight, including
            # missing characters. Unknown pairs may each attain any value [0,1].
            target_pair_weight = 1 / (len(characters) * stats["scheduled_pairs"])
            known_schedule_score += target_pair_weight * sum(stats["pair_outcomes"])
            known_schedule_weight += target_pair_weight * n
    interval = weighted_hoeffding(values, weights)
    status_counts = Counter(ledger[g["label"]]["status"] for g in specs)
    wins = sum(r["win"] for r in accepted.values())
    return {
        "profile": specs[0]["profile"], "release": specs[0]["release"], "block": specs[0]["block"],
        "accepted": len(accepted), "total": len(specs), "wins": wins, "losses": len(accepted) - wins,
        "stocks_taken": sum(r["stocks_taken"] for r in accepted.values()),
        "stocks_conceded": sum(r["stocks_conceded"] for r in accepted.values()),
        "quarantined": status_counts["quarantined"], "status_counts": dict(status_counts),
        "observed_quarantined_games": len(observed_quarantines),
        "observed_quarantined_wins": sum(r["win"] for r in observed_quarantines),
        "observed_quarantined_losses": sum(not r["win"] for r in observed_quarantines),
        "observed_quarantined_stocks_taken": sum(r["stocks_taken"] for r in observed_quarantines),
        "observed_quarantined_stocks_conceded": sum(r["stocks_conceded"] for r in observed_quarantines),
        "observed_quarantined_outcomes": observed_quarantines,
        "complete_pairs": len(complete_pairs), "scheduled_pairs": len(pairs),
        "incomplete_pairs": len(incomplete_pairs),
        "accepted_games_excluded_from_paired_estimate": len(accepted) - 2 * len(complete_pairs),
        "observed_characters": observed_chars, "scheduled_characters": len(characters),
        "character_metric": "starting-CSS assignment with native in-game transformations allowed",
        "paired_macro_win_fraction": interval["estimate"], "pointwise_interval": interval,
        "all_pairs_complete": not incomplete_pairs,
        "full_schedule_macro_win_fraction": interval["estimate"] if not incomplete_pairs else None,
        "full_schedule_descriptive_bounds": {
            "lower": min(1.0, known_schedule_score),
            "upper": min(1.0, known_schedule_score + max(0.0, 1 - known_schedule_weight)),
            "unknown_weight": max(0.0, 1 - known_schedule_weight),
        },
        "characters": characters, "paired_cells": complete_pairs,
        "excluded_cells": incomplete_pairs,
    }


def analyze(panel, state):
    games = panel["games"]
    labels = [g["label"] for g in games]
    if not games or len(set(labels)) != len(labels):
        raise ValueError("manifest games are empty or labels are duplicated")
    if set(state["games"]) != set(labels):
        raise ValueError("state and manifest game labels differ")
    for row in state["games"].values():
        if row.get("status") not in STATUSES:
            raise ValueError("unrecognized ledger status")
    comparisons = defaultdict(list)
    for game in games:
        if game["block"] not in BLOCKS:
            raise ValueError("unrecognized benchmark block")
        comparisons[(game["profile"], game["release"], game["block"])].append(game)
    if len(comparisons) > COMPARISONS:
        raise ValueError("comparison count exceeds the fixed 84-comparison family")
    results = [summarize_comparison(specs, state["games"]) for specs in comparisons.values()]
    totals = {}
    for row in results:
        total = totals.setdefault(row["profile"], Counter())
        for field in ("accepted", "total", "wins", "losses", "stocks_taken", "stocks_conceded", "quarantined",
                      "observed_quarantined_games", "observed_quarantined_wins", "observed_quarantined_losses",
                      "observed_quarantined_stocks_taken", "observed_quarantined_stocks_conceded"):
            total[field] += row[field]
    return {
        "schema": "e011.slippi_public_panel.paired_report.v1",
        "experiment_id": panel.get("experiment_id"), "state_status": state.get("status"),
        "state_updated_unix": state.get("updated_unix"),
        "exploratory_partial_results": any(not r["all_pairs_complete"] for r in results),
        "uncertainty": {"method": "weighted Hoeffding with Bonferroni", "family_alpha": ALPHA,
                        "fixed_comparisons": COMPARISONS, "unit": "complete physical-port pair",
                        "independent_pairs_assumed": True, "anytime_valid": False},
        "rows": results, "totals_by_profile": dict(totals),
    }


def _percentage(value):
    return "pending" if value is None else f"{100 * value:.1f}%"


def _bounds(lower, upper):
    return f"{100 * lower:.1f}% to {100 * upper:.1f}%"


def _text(value):
    return str(value).replace("\\", "\\\\").replace("`", "\\`").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_markdown(report):
    partial = report["exploratory_partial_results"]
    lines = ["# Public Slippi panel: paired results", "",
             "Exploratory partial results." if partial else "Fixed-schedule descriptive results.", "",
             "Each row is one frozen opponent and block. W-L and stocks include every accepted game. "
             "Stocks are taken / conceded from Frisson's perspective. Explicitly observed quarantined "
             "outcomes appear separately and remain excluded from accepted and paired estimates. "
             "Their column shows W-L; stocks taken / conceded, with a dash when no outcome is recorded.", "",
             "Paired macro averages first average the two port-swapped games, then pairs within each "
             "observed controlled character, then characters equally. Accepted games whose partner is "
             "unfinished or quarantined are excluded from this estimate. An asterisk marks an incomplete "
             "scheduled comparison. Character and pair coverage accompany every estimate.", ""]
    for profile in sorted(report["totals_by_profile"], key=lambda p: (p != "75m", p)):
        lines.extend([f"## {profile.upper()} RL", "",
                      "| Opponent | Block | Accepted / total | W-L | Stocks taken / conceded | Quarantined | Observed quarantine W-L; stocks | Pairs | Characters | Paired macro | Pointwise interval |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
        for r in report["rows"]:
            if r["profile"] != profile:
                continue
            interval = r["pointwise_interval"]
            suffix = "*" if not r["all_pairs_complete"] else ""
            observed = (f"{r['observed_quarantined_wins']}-{r['observed_quarantined_losses']}; "
                        f"{r['observed_quarantined_stocks_taken']} / {r['observed_quarantined_stocks_conceded']}"
                        if r["observed_quarantined_games"] else "-")
            lines.append(f"| {r['release']} | {r['block']} | {r['accepted']} / {r['total']} | "
                         f"{r['wins']}-{r['losses']} | {r['stocks_taken']} / {r['stocks_conceded']} | "
                         f"{r['quarantined']} | {observed} | {r['complete_pairs']} / {r['scheduled_pairs']} | "
                         f"{r['observed_characters']} / {r['scheduled_characters']} | "
                         f"{_percentage(r['paired_macro_win_fraction'])}{suffix} | "
                         f"{_bounds(interval['lower'], interval['upper'])} |")
        t = report["totals_by_profile"][profile]
        lines.extend(["", f"Accepted {t['accepted']} / {t['total']}; {t['wins']}-{t['losses']}; "
                      f"stocks {t['stocks_taken']} / {t['stocks_conceded']}; quarantined {t['quarantined']}.", ""])
        if t["observed_quarantined_games"]:
            lines.extend([f"Separately observed quarantines: {t['observed_quarantined_games']} games; "
                          f"{t['observed_quarantined_wins']}-{t['observed_quarantined_losses']}; stocks "
                          f"{t['observed_quarantined_stocks_taken']} / {t['observed_quarantined_stocks_conceded']}.", ""])
            for r in report["rows"]:
                if r["profile"] == profile:
                    for outcome in r["observed_quarantined_outcomes"]:
                        lines.append(f"- {_text(outcome['label'])}: "
                                     f"{'win' if outcome['win'] else 'loss'}, stocks "
                                     f"{outcome['stocks_taken']} / {outcome['stocks_conceded']}. "
                                     f"Reason: {_text(outcome['reason'])}. "
                                     f"Evidence class: {_text(outcome['evidence_class'])}. "
                                     f"Receipt: {_text(outcome['source_receipt'])}.")
            lines.append("")
    lines.extend([
        "## Uncertainty and missing data", "",
        "Intervals use a two-sided weighted Hoeffding bound with 5% total error split across the "
        "84 predeclared model/release/block comparisons. For pair outcomes X in [0, 1] and weights "
        "w summing to one, the radius is sqrt(log(2 x 84 / 0.05) x sum(w^2) / 2), clipped to [0, 1]. "
        "All-wins samples retain uncertainty. With zero complete pairs the estimate is pending and "
        "the interval spans 0% to 100%.", "",
        "The guarantee assumes independent pair/seed outcomes within each comparison and fixed "
        "outcome-independent inclusion. Pair members may be dependent, and comparisons may share "
        "seeds. These are fixed-look intervals; repeated status views provide exploratory monitoring. "
        "They provide no anytime-valid stopping or selective-claim guarantee. Outcome-related "
        "quarantines can bias the observed subset. Survivor-bias warning: a recoverable quarantined "
        "loss must remain visible when interpreting accepted-only results. The separate observed-outcome "
        "column records available forensic evidence; it does not restore missing original summary or "
        "runtime/timing diagnostics. These outcomes remain outside the primary paired estimate. "
        "Partial estimates describe only currently observed "
        "characters and pairs; the full scheduled-roster estimate remains pending until every pair completes.", "",
        "The JSON companion retains each included pair, excluded pair, character average and a "
        "full-schedule descriptive bound. That bound assigns all missing pairs either zero or one, "
        "preserving the full schedule's equal character weights. It describes possible final finite-schedule "
        "scores and carries no sampling-confidence interpretation. No incomplete-pair game is promoted "
        "into an independent statistical observation.", "",
        "## Scope and interpretation", "",
        "Supported mirrors measure each release on its deployment roster. Extended-roster games "
        "measure Frisson's additional characters against supported opponents. Forced mirrors put both "
        "policies on the same character outside the opponent's deployment roster. For gm-v1 and medium-v1, "
        "the four-character boundary concerns RL deployment; their BC declaration covers all 26 fighters. "
        "Declared rosters do not establish empirical training-example counts.", "",
        "Character rows describe starting CSS assignments. Native Zelda/Sheik transformations remain "
        "allowed during play; a Zelda-start win alone does not establish sustained out-of-roster Zelda "
        "proficiency. The terminal character may differ from the starting assignment.", "",
        "Native Slippi delay is 21 or 24 frames; Frisson policy delay is zero. Frisson uses ring context "
        "128 while its RL actor used prefix context 128. Conditioning follows each frozen upstream "
        "default. These choices characterize the deployed systems and leave latency, context and "
        "conditioning effects entangled with the learned policy. Two games per character and six "
        "pairs per specialist provide limited precision. Training-seed variability is unmeasured.", "",
        "Any favorable finding applies to the named immutable checkpoints, assigned characters, "
        "stages and native-runtime protocol. Universal SOTA and causal architecture-superiority claims "
        "remain unsupported by this panel alone. auto-* routing, private banks and every historical "
        "release remain outside the verified named-checkpoint scope.", "",
        "This report reads coordinator-accepted ledger records and explicitly labeled forensic outcomes. "
        "The coordinator owns replay/controller "
        "acceptance; this reporter checks ledger structure and arithmetic and leaves independent replay "
        "revalidation to the completion audit. A saved replay's natural-end evidence can include a "
        "formal Game End event or separately audited terminal-stock/live-state agreement; these evidence "
        "classes must remain distinguishable in the completion audit. Input hashes identify this exact snapshot.", "",
        f"Manifest SHA-256: `{report.get('manifest_sha256', 'synthetic fixture')}`", "",
        f"State SHA-256: `{report.get('state_sha256', 'synthetic fixture')}`", "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    panel_dir = args.panel.resolve()
    output_dir = (args.output_dir or panel_dir / "analysis").resolve()
    if not any(output_dir.is_relative_to(panel_dir / allowed) for allowed in ("analysis", "research")):
        parser.error("outputs must remain under this panel's analysis or research directory")
    manifest_bytes = (panel_dir / "manifest.json").read_bytes()
    state_bytes = (panel_dir / "state.json").read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    state = json.loads(state_bytes)
    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError("ledger is bound to a different manifest")
    report = analyze(json.loads(manifest_bytes), state)
    report.update(manifest_sha256=manifest_hash, state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # Name snapshots by both inputs: repeated reads preserve prior reports.
    name = f"paired-{manifest_hash[:12]}-{report['state_sha256'][:12]}-{report['source_sha256'][:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / f"{name}.json", output_dir / f"{name}.md"
    if not json_path.exists():
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not md_path.exists():
        md_path.write_text(render_markdown(report))
    print(json.dumps({"report": str(md_path), "data": str(json_path),
                      "totals": report["totals_by_profile"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
