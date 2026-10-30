"""Re-scores one saved live session under each elbow statistic, from its
logs/<session>.traces.json. Read-only: it writes nothing, changes no config, and
reads only that session's own logs and a rules config. It uses no dataset, so no
REHAB24-6 subject is involved.

    python scripts/rescore_session.py                 # newest session with traces
    python scripts/rescore_session.py logs/<session>.csv

Peak, trunk, tempo and the insufficient-data check come from src.rules.score_rep()
unchanged. Only the elbow_straight value is swapped, and it's judged against the
same allowed range score_rep() reports. p10 lives here rather than in src/rules.py
because it's for offline analysis, not a selectable scoring option.

A pass rate here is NOT accuracy: a session has no correct/incorrect labels, and
choosing a statistic from a session you've already seen is post hoc. Any change
to scoring still goes through docs/preregistered_own_clip_eval.md.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules as rules_module
from src.tracking import LOGS_DIR

ELBOW_STATISTICS = ("min", "hold_median", "p10")
EXERCISE_DEFAULT = "shoulder_abduction"


def p10(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=10, method="inclusive")[0]


def elbow_value(rep: dict, statistic: str) -> float:
    if statistic == "p10":
        return p10(rep["elbow_angle_deg"])
    return rules_module.rep_elbow_value(rep, rep["fps"], statistic)


def rescore_rep(rep: dict, config: dict, statistic: str, reference: dict | None = None) -> dict:
    """{"status", "failed_rules", "elbow"} with this elbow statistic."""
    base = rules_module.score_rep(rep, config, rep["fps"], reference=reference)
    value = elbow_value(rep, statistic)
    low, high = base["rules"]["elbow_straight"]["allowed"]
    failed = {rule for rule in base["failed_rules"] if rule != "elbow_straight"}
    if not low <= value <= high:
        failed.add("elbow_straight")
    failed_rules = [rule for rule in rules_module.RULE_IDS if rule in failed]
    if base["status"] == "insufficient_data":
        status = "insufficient_data"
    else:
        status = "fail" if failed_rules else "pass"
    return {"status": status, "failed_rules": failed_rules, "elbow": value}


def summarize(results: list[dict]) -> dict:
    return {
        "n": len(results),
        **{status: sum(r["status"] == status for r in results) for status in ("pass", "fail", "insufficient_data")},
        "rule_failures": {rule: sum(rule in r["failed_rules"] for r in results) for rule in rules_module.RULE_IDS},
    }


def status_mismatches(traces: list[dict], csv_rows: list[dict], config: dict, reference: dict | None = None) -> list[int]:
    """Rep numbers where re-scoring with the config's own statistic disagrees
    with the status logged live, e.g. because the thresholds have since changed."""
    live_statistic = rules_module.elbow_statistic(config)
    return [
        trace["rep"] for trace in traces
        if trace["rep"] <= len(csv_rows)
        and rescore_rep(trace, config, live_statistic, reference)["status"] != csv_rows[trace["rep"] - 1]["status"]
    ]


def find_latest_session(logs_dir: Path) -> Path | None:
    traces = sorted(logs_dir.glob("*.traces.json"), key=lambda p: p.stat().st_mtime)
    return traces[-1].with_name(traces[-1].name.removesuffix(".traces.json") + ".csv") if traces else None


def format_report(traces: list[dict], results: dict[str, list[dict]], allowed: tuple[float, float]) -> list[str]:
    lines = [f"Elbow allowed range: {allowed[0]:g} to {allowed[1]:g} deg", "", "Per statistic:",
             f"{'statistic':12s} {'pass':>5s} {'fail':>5s} {'insuff':>7s} " + " ".join(f"{r:>15s}" for r in rules_module.RULE_IDS)]
    for statistic in ELBOW_STATISTICS:
        s = summarize(results[statistic])
        lines.append(f"{statistic:12s} {s['pass']:5d} {s['fail']:5d} {s['insufficient_data']:7d} "
                     + " ".join(f"{s['rule_failures'][r]:15d}" for r in rules_module.RULE_IDS))
    lines += ["", "Per rep (elbow deg, status, failed rules):"]
    for i, trace in enumerate(traces):
        parts = [f"{statistic} {results[statistic][i]['elbow']:6.1f} {results[statistic][i]['status']:4s} "
                 f"{','.join(results[statistic][i]['failed_rules']) or '-':28s}" for statistic in ELBOW_STATISTICS]
        lines.append(f"rep {trace['rep']:3d} frames {len(trace['elbow_angle_deg']):3d}  " + " | ".join(parts))
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score a saved session under each elbow statistic (read-only).")
    parser.add_argument("session", nargs="?", default=None, help="session CSV in logs/ (default: newest with traces)")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR)
    parser.add_argument("--config", type=Path, default=None, help="rules config (default: the exercise's shipped config)")
    parser.add_argument("--reference", type=Path, default=None, help="reference JSON, for a reference_relative config")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = Path(args.session) if args.session else find_latest_session(args.logs_dir)
    if session is None:
        print(f"No session with traces in {args.logs_dir}. Sessions saved before traces were added can't be re-scored.")
        return 1
    traces_path = session.with_suffix(".traces.json")
    if not traces_path.exists():
        print(f"No traces for {session}. That session was saved without per-frame data, so its only elbow value "
              "is the logged minimum, and it can't be re-scored.")
        return 1

    data = json.loads(traces_path.read_text())
    traces = data["reps"]
    config = json.loads(args.config.read_text()) if args.config else rules_module.load_config(data["exercise"])
    reference = rules_module.load_reference(args.reference) if args.reference else None
    results = {statistic: [rescore_rep(trace, config, statistic, reference) for trace in traces] for statistic in ELBOW_STATISTICS}
    allowed = tuple(rules_module.score_rep(traces[0], config, traces[0]["fps"], reference=reference)["rules"]["elbow_straight"]["allowed"])

    print(f"Session: {session.name}  reps with traces: {len(traces)}  "
          f"mode: {rules_module.active_scoring_mode(config, reference)}")
    print("\n".join(format_report(traces, results, allowed)))

    if session.exists():
        with session.open(newline="") as f:
            mismatches = status_mismatches(traces, list(csv.DictReader(f)), config, reference)
        print(f"\nAgreement with live scoring: {'all reps match' if not mismatches else f'DIFFERS on reps {mismatches}'}")
    print("\nPass rates here aren't accuracy: this session has no labels, and comparing statistics on it is post hoc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
