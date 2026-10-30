"""Exploratory A/B comparison of elbow statistics, from two saved live sessions:
A with locked elbows, B with the elbow deliberately bent on known reps. Read-only:
it writes nothing and changes nothing. It uses no dataset, so no REHAB24-6 subject
is involved.

    python scripts/compare_AB_sessions.py logs/<A>.csv logs/<B>.csv --bent-reps 2,4,6,8,10

For min, hold_median and p10 (computed by scripts/rescore_session.py) it prints
the elbow value's median and range for A, B straight and B bent; how many fall below
135 deg; and the AUC of bent vs all straight reps. It also prints any noise-run
summaries found in results/elbow_noise_*.json.

Labels are by rep NUMBER as the app counted reps. A rep the app missed or
discarded shifts every later label, so check the printed counts against what was
performed. These are a few reps from one person: nothing here is an accuracy figure.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.rescore_session import ELBOW_STATISTICS, elbow_value

THRESHOLD_DEG = 135.0
NOISE_RESULTS_DIR = Path("results")
EXPLORATORY_WARNING = ("EXPLORATORY ONLY: n is tiny, one person, one sitting, and labels come from intended "
                       "movement. These numbers are not accuracy and don't justify changing scoring.")


class MissingTraces(Exception):
    pass


def load_traces(session: Path) -> list[dict]:
    path = session.with_suffix(".traces.json")
    if not path.exists():
        raise MissingTraces(f"No traces at {path}. Record the session with the current app, press Stop session, "
                            "and pass that session's CSV path.")
    reps = json.loads(path.read_text())["reps"]
    if not reps:
        raise MissingTraces(f"{path} holds no reps.")
    return reps


def split_b(traces: list[dict], bent_reps: set[int]) -> tuple[list[dict], list[dict]]:
    """(straight, bent). Raises ValueError if a listed rep isn't in the session."""
    missing = sorted(bent_reps - {trace["rep"] for trace in traces})
    if missing:
        raise ValueError(f"--bent-reps lists {missing}, but session B has reps {[t['rep'] for t in traces]}")
    return [t for t in traces if t["rep"] not in bent_reps], [t for t in traces if t["rep"] in bent_reps]


def auc_lower_is_bent(bent: list[float], straight: list[float]) -> float | None:
    """Chance that a bent rep reads lower than a straight rep, ties counting
    half. 1.0 separates fully, 0.5 is no better than chance."""
    if not bent or not straight:
        return None
    wins = sum(1.0 if b < s else 0.5 if b == s else 0.0 for b in bent for s in straight)
    return wins / (len(bent) * len(straight))


def group_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "median": None, "low": None, "high": None, "below": 0}
    return {"n": len(values), "median": statistics.median(values), "low": min(values), "high": max(values),
            "below": sum(v < THRESHOLD_DEG for v in values)}


def compare(a: list[dict], b_straight: list[dict], b_bent: list[dict]) -> dict:
    report = {}
    for statistic in ELBOW_STATISTICS:
        groups = {name: [elbow_value(rep, statistic) for rep in reps]
                  for name, reps in (("A", a), ("B straight", b_straight), ("B bent", b_bent))}
        report[statistic] = {
            "groups": {name: group_stats(values) for name, values in groups.items()},
            "bent_below": group_stats(groups["B bent"])["below"], "bent_n": len(groups["B bent"]),
            "straight_below": group_stats(groups["A"] + groups["B straight"])["below"],
            "straight_n": len(groups["A"]) + len(groups["B straight"]),
            "auc": auc_lower_is_bent(groups["B bent"], groups["A"] + groups["B straight"]),
        }
    return report


def format_report(report: dict) -> list[str]:
    lines = []
    for statistic, r in report.items():
        lines += ["", f"--- {statistic} ---", f"{'group':12s} {'n':>3s} {'median':>7s} {'range':>15s} {'<135':>5s}"]
        for name, g in r["groups"].items():
            if g["n"]:
                lines.append(f"{name:12s} {g['n']:3d} {g['median']:7.1f} {g['low']:7.1f}-{g['high']:<7.1f} {g['below']:5d}")
            else:
                lines.append(f"{name:12s} {0:3d}  no reps")
        auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
        lines.append(f"Bent below 135: {r['bent_below']}/{r['bent_n']}   Straight below 135: {r['straight_below']}/{r['straight_n']}   "
                     f"AUC bent vs straight: {auc}")
    return lines


def noise_lines(results_dir: Path = NOISE_RESULTS_DIR) -> list[str]:
    paths = sorted(results_dir.glob("elbow_noise_*.json"))
    if not paths:
        return [f"No noise runs found ({results_dir / 'elbow_noise_*.json'})."]
    lines = []
    for path in paths:
        try:
            summary = json.loads(path.read_text())["summary"]
        except (OSError, ValueError, KeyError) as exc:
            lines.append(f"{path.name}: unreadable ({exc})")
            continue
        lines += [f"{path.name}:", f"  {'pose':9s} {'arm':9s} {'gated/frames':>13s} {'mean':>7s} {'std':>6s} {'p5':>7s} {'p95':>7s} {'<135':>7s}"]
        for pose, arms in summary.items():
            for arm, s in arms.items():
                counts = f"{s['n_gated']}/{s['n_frames']}"
                if s["mean"] is None:
                    lines.append(f"  {pose:9s} {arm:9s} {counts:>13s}   no gated frames")
                else:
                    lines.append(f"  {pose:9s} {arm:9s} {counts:>13s} {s['mean']:7.1f} {s['std']:6.1f} {s['p5']:7.1f} "
                                 f"{s['p95']:7.1f} {s['frac_below_threshold']:7.1%}")
    return lines


def discard_count(session: Path) -> int:
    path = session.with_suffix(".discards.csv")
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return sum(1 for _row in csv.DictReader(f))


def parse_rep_list(text: str) -> set[int]:
    try:
        reps = {int(part) for part in text.split(",") if part.strip()}
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} isn't a comma-separated list of rep numbers") from None
    if not reps or min(reps) < 1:
        raise argparse.ArgumentTypeError("--bent-reps needs rep numbers of 1 or more")
    return reps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploratory A/B comparison of elbow statistics (read-only).")
    parser.add_argument("session_a", type=Path, help="session CSV: locked elbows")
    parser.add_argument("session_b", type=Path, help="session CSV: some reps deliberately bent")
    parser.add_argument("--bent-reps", required=True, type=parse_rep_list, help="reps bent in B, e.g. 2,4,6,8,10")
    parser.add_argument("--results-dir", type=Path, default=NOISE_RESULTS_DIR, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        a = load_traces(args.session_a)
        b = load_traces(args.session_b)
        b_straight, b_bent = split_b(b, args.bent_reps)
    except (MissingTraces, ValueError) as exc:
        print(exc)
        return 1

    print(EXPLORATORY_WARNING)
    for label, session, reps in (("A", args.session_a, a), ("B", args.session_b, b)):
        print(f"Session {label}: {session.name}  reps counted: {len(reps)}  discarded segments: {discard_count(session)}")
    print("Check those counts: a missed or discarded rep in B shifts every later bent/straight label.")
    print(f"B bent reps: {sorted(args.bent_reps)}")
    print("\n".join(format_report(compare(a, b_straight, b_bent))))
    print("\nNoise runs:")
    print("\n".join(noise_lines(args.results_dir)))
    print(f"\n{EXPLORATORY_WARNING}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
