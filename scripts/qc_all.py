"""Final QC over the whole manifest, per Amendment 3 of docs/
preregistered_own_clip_eval.md. Read-only. Uses the pose model and rep detector only,
and never scores a rule. Run it once, after every clip passes qc_clip.py, before
evaluate_own_clips.py:

    python scripts/qc_all.py

The evaluator and scripts/qc_clip.py both reset pose tracking at the start of each clip
(Amendment 4), so a clip's rep count shouldn't depend on the clip before it. This
script confirms that by running both ways:
  1. Each clip alone, in its own process, as scripts/qc_clip.py does.
  2. All clips in one process, in the evaluator's order: the first reference row, then
     calibration rows, then eval rows, in manifest order.

It prints detected/planned for each. "OK to score" needs the planned set of clips, all
three checks passing in the combined run, and matching counts in both runs. Otherwise
it prints DO NOT SCORE and exits 1.
"""

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import qc_clip

MANIFEST_PATH = Path("data/own_manifest.csv")
PLANNED_CLIPS = {
    ("reference", "correct"): 3, ("calibration", "correct"): 5, ("eval", "correct"): 6,
    ("eval", "fault_lowarm"): 3, ("eval", "fault_trunk"): 3, ("eval", "fault_elbow"): 3, ("eval", "fault_fast"): 3,
}


def read_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open(newline="") as f:
        return list(csv.DictReader(f))


def evaluator_order(rows: list[dict]) -> list[dict]:
    """The clips evaluate_own_clips.py's main() opens, in the order it opens them."""
    return ([r for r in rows if r["role"] == "reference"][:1] + [r for r in rows if r["role"] == "calibration"]
            + [r for r in rows if r["role"] == "eval"])


def manifest_problems(rows: list[dict]) -> list[str]:
    """Differences from the pre-registered set of clips."""
    problems = []
    counts = Counter((r["role"], r["intended"]) for r in rows)
    for key in PLANNED_CLIPS:
        if counts[key] != 1:
            problems.append(f"{key[0]}/{key[1]}: {counts[key]} manifest rows, expected 1")
    for key in counts:
        if key not in PLANNED_CLIPS:
            problems.append(f"{key[0]}/{key[1]}: not a planned clip")
    for row in rows:
        planned = PLANNED_CLIPS.get((row["role"], row["intended"]))
        if planned is not None and int(row["expected_reps"]) != planned:
            problems.append(f"{Path(row['clip']).name}: expected_reps {row['expected_reps']}, pre-registered {planned}")
    return problems


def detected_alone(clip: Path, planned: int) -> int | None:
    """The clip's rep count from qc_clip.py in its own process. None if unreadable."""
    result = subprocess.run(
        [sys.executable, str(Path(qc_clip.__file__)), str(clip), "--expected-reps", str(planned)],
        capture_output=True, text=True,
    )
    match = re.search(r"rep_count: detected (\d+), planned", result.stdout)
    return int(match.group(1)) if match else None


def compare(results: list[dict]) -> list[str]:
    """Reasons not to score, from [{"name", "planned", "alone", "combined", "checks"}]."""
    problems = []
    for r in results:
        if r["alone"] != r["combined"]:
            problems.append(f"{r['name']}: detected {r['alone']} alone but {r['combined']} combined")
        for check_name, check in r["checks"].items():
            if not check["passed"]:
                problems.append(f"{r['name']}: {check_name} failed in the combined run ({check['detail']})")
    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final detector-only QC over the whole manifest (read-only).")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.manifest.exists():
        print(f"No manifest at {args.manifest}.")
        return 1
    rows = read_rows(args.manifest)
    problems = manifest_problems(rows)
    missing = [r["clip"] for r in rows if not Path(r["clip"]).exists()]
    problems += [f"missing file: {clip}" for clip in missing]
    if problems:
        print("\n".join(problems))
        print("DO NOT SCORE: the manifest doesn't match the plan.")
        return 1

    results = []
    ordered = evaluator_order(rows)
    print("Running each clip alone...")
    alone = {r["clip"]: detected_alone(Path(r["clip"]), int(r["expected_reps"])) for r in ordered}
    print("Running all clips in one process, in the evaluator's order...")
    for row in ordered:
        planned = int(row["expected_reps"])
        trace = qc_clip.collect_detector_trace(Path(row["clip"]))
        results.append({
            "name": Path(row["clip"]).name, "planned": planned, "alone": alone[row["clip"]],
            "combined": trace["rep_count"], "checks": qc_clip.run_checks(trace, planned),
        })

    print(f"\n{'clip':36s} {'planned':>7s} {'alone':>6s} {'combined':>8s}  QC (combined)")
    for r in results:
        passed = all(check["passed"] for check in r["checks"].values())
        print(f"{r['name']:36s} {r['planned']:7d} {str(r['alone']):>6s} {r['combined']:8d}  {'PASS' if passed else 'FAIL'}")
    problems = compare(results)
    if problems:
        print("\n" + "\n".join(problems))
        print("DO NOT SCORE.")
        return 1
    print("\nAll clips pass and the counts match in both runs. OK to score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
