"""Computes Amendment 1's pre-registered primary outcomes (docs/
preregistered_own_clip_eval.md) from scripts/evaluate_own_clips.py's per-rep CSV.
Read-only: it writes nothing and changes nothing.

    python scripts/compute_primary_outcomes.py [results/own_clips_eval.csv]

For each scoring mode the evaluator reports (demo, reference):
  - False-fail on correct reps, counting only peak_angle, trunk_lean and tempo.
  - Detection of each primary fault by its expected rule.
  - The bent-elbow fault, separately, as exploratory with no pass criterion.
Every rate has a Wilson 95% interval. insufficient_data reps are left out of the
denominators and reported.

The verdict applies the amendment's fixed thresholds. It's INCONCLUSIVE under the
pre-registered rules, and NOT VALID if the pinned files at the CSV's commit don't
match the pre-registration or can't be checked. Nothing here can tune anything, and
a pass isn't evidence of accuracy across people.
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_own_clips import rate_with_ci

DEFAULT_CSV = Path("results/own_clips_eval.csv")
MODES = {"demo": "absolute demo config", "reference": "derived reference_relative config"}

PRIMARY_RULES = ("peak_angle", "trunk_lean", "tempo")
PRIMARY_FAULTS = {
    "fault_lowarm": ("arm too low", "peak_angle"),
    "fault_trunk": ("trunk lean", "trunk_lean"),
    "fault_fast": ("too fast", "tempo"),
}
EXPLORATORY_FAULT = ("fault_elbow", "bent elbow", "elbow_straight")
MAX_FALSE_FAIL = 0.20
MIN_DETECTION = 0.70
MAX_INSUFFICIENT = 0.20
PLANNED_SCORABLE = {"correct": 6, "fault_lowarm": 3, "fault_trunk": 3, "fault_elbow": 3, "fault_fast": 3}

PINNED_BLOBS = {
    "config/exercises/shoulder_abduction.json": "79c406f505c73c2ea87671b3773ea81e99e16848",
    "config/pose.json": "c15ab25f8d524734c300a45fcde8285c82ea4411",
    "src/rules.py": "4e85c2b701f86b952e71a4106022739fa41ce8d0",
    "scripts/evaluate_own_clips.py": "09f2715fbb7bb986d654dac3e28042f81cb3bf54",
}
REQUIRED_COLUMNS = {"clip", "intended", "demo_status", "demo_failed_rules", "reference_status", "reference_failed_rules"}


def read_eval_csv(path: Path) -> tuple[dict, list[dict]]:
    """(provenance from the "# key: value" header lines, data rows)."""
    provenance, data_lines = {}, []
    with path.open(newline="") as f:
        for line in f:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                provenance[key.strip()] = value.strip()
            else:
                data_lines.append(line)
    rows = list(csv.DictReader(data_lines))
    missing = REQUIRED_COLUMNS - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    return provenance, rows


def scorable_reps(rows: list[dict], mode: str, intended: str) -> list[dict]:
    return [r for r in rows if r["intended"] == intended and r[f"{mode}_status"] != "insufficient_data"]


def failed_rules(row: dict, mode: str) -> set[str]:
    return {rule for rule in row[f"{mode}_failed_rules"].split(";") if rule}


def mode_outcomes(rows: list[dict], mode: str) -> dict:
    correct = scorable_reps(rows, mode, "correct")
    outcomes = {
        "false_fail": (sum(bool(failed_rules(r, mode) & set(PRIMARY_RULES)) for r in correct), len(correct)),
        "all_rule_false_fail": (sum(r[f"{mode}_status"] == "fail" for r in correct), len(correct)),
        "faults": {},
    }
    for fault, (_label, rule) in PRIMARY_FAULTS.items():
        reps = scorable_reps(rows, mode, fault)
        outcomes["faults"][fault] = (sum(rule in failed_rules(r, mode) for r in reps), len(reps))
    fault, _label, rule = EXPLORATORY_FAULT
    reps = scorable_reps(rows, mode, fault)
    outcomes["exploratory"] = (sum(rule in failed_rules(r, mode) for r in reps), len(reps))
    outcomes["insufficient"] = (sum(r[f"{mode}_status"] == "insufficient_data" for r in rows), len(rows))
    outcomes["scorable_by_group"] = {group: len(scorable_reps(rows, mode, group)) for group in PLANNED_SCORABLE}
    return outcomes


def verdict(outcomes: dict) -> tuple[str, list[str]]:
    """("PASS" | "FAIL" | "INCONCLUSIVE", reasons)."""
    reasons = []
    k, n = outcomes["insufficient"]
    if n == 0 or k / n > MAX_INSUFFICIENT:
        reasons.append(f"insufficient_data on {k}/{n} eval reps (limit {MAX_INSUFFICIENT:.0%})")
    for group, planned in PLANNED_SCORABLE.items():
        if outcomes["scorable_by_group"][group] < planned:
            reasons.append(f"{group}: {outcomes['scorable_by_group'][group]} scorable reps, {planned} planned")
    if reasons:
        return "INCONCLUSIVE", reasons

    k, n = outcomes["false_fail"]
    if k / n > MAX_FALSE_FAIL:
        reasons.append(f"false-fail {k}/{n} is above {MAX_FALSE_FAIL:.0%}")
    for fault, (k, n) in outcomes["faults"].items():
        if k / n < MIN_DETECTION:
            reasons.append(f"{fault} detected {k}/{n}, below {MIN_DETECTION:.0%}")
    return ("FAIL" if reasons else "PASS"), reasons


def pin_check(commit: str | None) -> tuple[bool, list[str]]:
    """(all pinned blobs match at this commit, problems)."""
    if not commit or commit == "unknown":
        return False, ["the CSV doesn't record a usable commit"]
    problems = []
    for path, expected in PINNED_BLOBS.items():
        result = subprocess.run(["git", "rev-parse", f"{commit}:{path}"], capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"{path}: couldn't read it at {commit}")
        elif result.stdout.strip() != expected:
            problems.append(f"{path}: {result.stdout.strip()} at {commit}, pinned {expected}")
    return not problems, problems


def format_mode(mode: str, outcomes: dict) -> list[str]:
    result, reasons = verdict(outcomes)
    lines = [
        "", f"=== {mode}: {MODES[mode]} ===",
        f"False-fail on correct reps (peak, trunk, tempo only; limit {MAX_FALSE_FAIL:.0%}): {rate_with_ci(*outcomes['false_fail'])}",
    ]
    for fault, (label, rule) in PRIMARY_FAULTS.items():
        lines.append(f"Detection, {label} ({fault}) by {rule} (minimum {MIN_DETECTION:.0%}): {rate_with_ci(*outcomes['faults'][fault])}")
    fault, label, rule = EXPLORATORY_FAULT
    lines += [
        f"Exploratory, no pass criterion: {label} ({fault}) by {rule}: {rate_with_ci(*outcomes['exploratory'])}",
        f"Not primary: false-fail counting every rule, including elbow_straight: {rate_with_ci(*outcomes['all_rule_false_fail'])}",
        f"insufficient_data reps, excluded above: {rate_with_ci(*outcomes['insufficient'])}",
        f"Verdict: {result}" + (f" ({'; '.join(reasons)})" if reasons else ""),
    ]
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Amendment 1's primary outcomes (read-only).")
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.csv_path.exists():
        print(f"No evaluator output at {args.csv_path}. Run scripts/evaluate_own_clips.py on the recorded clips first.")
        return 1
    try:
        provenance, rows = read_eval_csv(args.csv_path)
    except ValueError as exc:
        print(exc)
        return 1

    commit = provenance.get("commit")
    print(f"Source: {args.csv_path}  commit: {commit}  dirty: {provenance.get('dirty')}  date: {provenance.get('date')}")
    pins_ok, problems = pin_check(commit)
    if pins_ok:
        print("Pinned files at that commit: match the pre-registration")
    else:
        print("NOT VALID as the pre-registered evaluation. Pinned files: " + "; ".join(problems))
    if provenance.get("dirty") != "False":
        print("Warning: the run's working tree wasn't recorded as clean. Check `git status` output from that run; "
              "any untracked file, such as .vscode/, also sets this flag.")
    for mode in MODES:
        print("\n".join(format_mode(mode, mode_outcomes(rows, mode))))
    print("\nOne person, few reps: a pass here isn't evidence of accuracy across people, and with elbow_straight "
          "advisory it says nothing about detecting a bent elbow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
