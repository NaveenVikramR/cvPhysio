"""Evaluation harness for the project's OWN recorded clips (see docs/
recording_protocol.md), not REHAB24-6. Unlike the REHAB24 scripts,
there is no external rep-boundary annotation for these clips -- rep
detection uses the SAME live src.pipeline.Session/RepDetector pipeline
app.py uses, run once per clip (never twice -- the detected reps are
reused for both scoring modes below).

1. Builds a Reference (src.rules.build_reference()) from the
   `role == "reference"` clip's first 3 detected reps (the
   "therapist" recording).
2. Derives reference_relative tolerances from the `role ==
   "calibration"` clip(s)' detected reps, via the SAME one-sided-
   deviation-from-the-reference-median method scripts/derive_
   reference_tolerances.py uses on REHAB24 data (prints n reps, warns
   if n < 30 -- a single 8-rep calibration clip will always warn).
3. Evaluates every `role == "eval"` clip ONCE (one rep-detection pass
   per clip) against BOTH the demo config (config/exercises/
   shoulder_abduction.json, absolute mode) and the derived
   reference_relative config, reusing the same detected reps for both.

NEVER changes any threshold file itself -- config/exercises/
shoulder_abduction.json is only read, and the derived reference_
relative config lives only in memory / in this run's own output file.

Metrics (Wilson 95% CI on every rate): false-fail rate on correct
eval reps; per fault type, the share flagged by ANY rule and by the
EXPECTED rule (fault_lowarm->peak_angle, fault_trunk->trunk_lean,
fault_elbow->elbow_straight, fault_fast->tempo); which rules fire
wrongly on correct reps; insufficient_data rate; average FPS across
eval clips.

Output: results/own_clips_eval.md and results/own_clips_eval.csv
(one row per scored rep), both starting with commit hash, dirty flag,
landmark_space, config paths, manifest hash, and date.

IMPORTANT: this script's main() requires real recorded video files
listed in data/own_manifest.csv (role=reference/calibration/eval) --
it was built and unit-tested on synthetic/placeholder data only (see
tests/test_evaluate_own_clips.py), and has deliberately NOT been run
against data/own_manifest.csv's own EXAMPLE_ placeholder rows, which
point at video files that do not exist.
"""

import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pose_estimation
from src import rules as rules_module
from src.pipeline import Session, build_smoother

MANIFEST_PATH = Path("data/own_manifest.csv")
RESULTS_DIR = Path("results")
EXERCISE = "shoulder_abduction"
DEMO_CONFIG_PATH = rules_module.CONFIG_DIR / f"{EXERCISE}.json"
N_REFERENCE_REPS = 3
MIN_N_WARNING = 30
FAULT_TO_RULE = {
    "fault_lowarm": "peak_angle",
    "fault_trunk": "trunk_lean",
    "fault_elbow": "elbow_straight",
    "fault_fast": "tempo",
}


def git_provenance() -> tuple[str, bool]:
    """(commit_hash, is_dirty) -- same helper as every other results-
    producing script in this project (duplicated, not imported, so this
    harness has no dependency on scripts/benchmark_rehab24.py's REHAB24-
    specific module state)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
        ).stdout
        return commit, bool(status.strip())
    except Exception:
        return "unknown", True


def manifest_hash(path: Path) -> str:
    if not path.exists():
        return "no-manifest"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_manifest(path: Path) -> list[dict]:
    """Returns [] if the manifest doesn't exist yet -- same "nothing to
    evaluate" convention as scripts/evaluate_rules.py's read_manifest()."""
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Pure, synthetic-data-testable logic (no video, no MediaPipe below this line
# until process_clip_reps())
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n. (nan, nan) if n==0."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def rate_with_ci(k: int, n: int) -> str:
    if n == 0:
        return "N/A (n=0)"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({k / n:.1%}, 95% CI [{lo:.1%}, {hi:.1%}])"


def one_sided_deviation(feature: str, measured: dict, ref_medians: dict) -> float:
    """Same formulas as scripts/derive_reference_tolerances.py: peak_
    angle/elbow_min are penalized only for falling BELOW the reference
    (lower-bound-only rule); trunk_max only for rising ABOVE it (upper-
    bound-only rule); tempo_s is a two-sided fractional deviation."""
    if feature in ("peak_angle", "elbow_min"):
        return max(0.0, ref_medians[feature] - measured[feature])
    if feature == "trunk_max":
        return max(0.0, measured[feature] - ref_medians[feature])
    if feature == "tempo_s":
        return abs(measured[feature] - ref_medians[feature]) / ref_medians[feature]
    raise ValueError(f"Unknown feature {feature!r}")


def derive_tolerances(calibration_reps: list[dict], fps: float, reference_medians: dict) -> dict:
    """calibration_reps: a list of rep dicts, each {"angle_deg": [...],
    "elbow_angle_deg": [...], "trunk_lean_deg": [...]} (one per detected
    calibration rep). Returns {"tol_peak_deg", "tol_elbow_deg",
    "tol_trunk_deg", "tol_tempo_frac", "n"} -- p95 of the one-sided
    deviation from reference_medians, pooled over all calibration_reps.
    """
    deviations = {"peak_angle": [], "elbow_min": [], "trunk_max": [], "tempo_s": []}
    for rep in calibration_reps:
        feat = rules_module.rep_summary_features(rep, fps)
        for key in deviations:
            deviations[key].append(one_sided_deviation(key, feat, reference_medians))
    n = len(calibration_reps)
    return {
        "tol_peak_deg": float(np.percentile(deviations["peak_angle"], 95)) if n else None,
        "tol_elbow_deg": float(np.percentile(deviations["elbow_min"], 95)) if n else None,
        "tol_trunk_deg": float(np.percentile(deviations["trunk_max"], 95)) if n else None,
        "tol_tempo_frac": float(np.percentile(deviations["tempo_s"], 95)) if n else None,
        "n": n,
    }


def rep_count_metrics(detected: int, expected: int) -> dict:
    return {
        "detected": detected, "expected": expected, "exact_match": detected == expected,
        "within_1": abs(detected - expected) <= 1, "signed_error": detected - expected,
    }


def false_fail_rate(correct_results: list[dict]) -> tuple[int, int]:
    """correct_results: score_rep() result dicts for correct-intended
    reps. Returns (k, n) = (reps scored fail, reps actually scorable --
    insufficient_data reps excluded from the denominator, same
    convention as scripts/evaluate_rules.py)."""
    scorable = [r for r in correct_results if r["status"] != "insufficient_data"]
    k = sum(1 for r in scorable if r["status"] == "fail")
    return k, len(scorable)


def fault_detection_rates(results_by_intended: dict[str, list[dict]]) -> dict:
    """results_by_intended: {"fault_lowarm": [score_rep() result, ...], ...}.
    For each fault type: (k, n) flagged by ANY rule (status=="fail"),
    and (k, n) flagged specifically by that fault's EXPECTED rule
    (FAULT_TO_RULE) -- insufficient_data reps excluded from n in both.
    """
    out = {}
    for fault, rule_id in FAULT_TO_RULE.items():
        reps = results_by_intended.get(fault, [])
        scorable = [r for r in reps if r["status"] != "insufficient_data"]
        n = len(scorable)
        k_any = sum(1 for r in scorable if r["status"] == "fail")
        k_expected = sum(1 for r in scorable if rule_id in r["failed_rules"])
        out[fault] = {"rule_id": rule_id, "any": (k_any, n), "expected_rule": (k_expected, n)}
    return out


def wrong_rules_on_correct(correct_results: list[dict]) -> Counter:
    """Which rule(s) fire on a CORRECT rep that got scored fail --
    i.e. false-fail reasons, not failures on fault clips."""
    counts = Counter()
    for r in correct_results:
        if r["status"] == "fail":
            counts.update(r["failed_rules"])
    return counts


def insufficient_data_rate(all_results: list[dict]) -> tuple[int, int]:
    k = sum(1 for r in all_results if r["status"] == "insufficient_data")
    return k, len(all_results)


# ---------------------------------------------------------------------------
# Video-processing glue -- real MediaPipe, real video decode. NOT exercised
# by this script's own tests (see module docstring); only called from
# main(), which this task's instructions say not to run yet.
# ---------------------------------------------------------------------------

def process_clip_reps(video_path: Path, exercise: str) -> tuple[list[dict], float]:
    """Runs the WHOLE clip through Session+RepDetector exactly once
    (Start before frame 0, never Stop -- the same default app.py's
    run_headless() uses with no --span). Returns (detected_reps,
    fps_measured): detected_reps is a list of {"angle_deg",
    "elbow_angle_deg", "trunk_lean_deg", "valid_fraction"} dicts, one
    per rep_complete event, in detection order; fps_measured is frames
    processed / wall-clock seconds (the same live-throughput
    measurement scripts/benchmark_rehab24.py's part_e_fps uses -- a
    real decode+inference speed, deliberately not a cached figure).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    required_roles = rules_module.required_joint_roles()
    session = Session(exercise)
    session.start()
    smoother = build_smoother()
    detected_reps = []
    frame_idx = 0
    n_frames = 0
    start_t = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps_meta
        frame_idx += 1
        landmarks = pose_estimation.get_pose_landmarks(frame)
        event, payload, _visible, _failed = session.process_frame(landmarks, smoother, t, required_roles)
        if event == "rep_complete":
            detected_reps.append({
                "angle_deg": payload["angle_deg"], "elbow_angle_deg": payload["elbow_angle_deg"],
                "trunk_lean_deg": payload["trunk_lean_deg"], "valid_fraction": payload["valid_fraction"],
            })
        n_frames += 1
    elapsed = time.perf_counter() - start_t
    cap.release()
    fps_measured = n_frames / elapsed if elapsed > 0 else float("nan")
    return detected_reps, fps_measured


def main() -> None:
    commit, dirty = git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    today = date.today().isoformat()
    man_hash = manifest_hash(MANIFEST_PATH)
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}  "
          f"manifest_hash={man_hash}  date={today}")

    rows = read_manifest(MANIFEST_PATH)
    if not rows:
        print(f"No manifest found at {MANIFEST_PATH} -- nothing to evaluate.")
        return

    reference_rows = [r for r in rows if r["role"] == "reference"]
    calibration_rows = [r for r in rows if r["role"] == "calibration"]
    eval_rows = [r for r in rows if r["role"] == "eval"]
    if not reference_rows:
        print("No role=reference clip in the manifest -- cannot build a Reference. Nothing to evaluate.")
        return

    print(f"\n[1/3] Building reference from {reference_rows[0]['clip']}...")
    ref_detected, _fps = process_clip_reps(Path(reference_rows[0]["clip"]), EXERCISE)
    if len(ref_detected) < N_REFERENCE_REPS:
        print(f"Only {len(ref_detected)} reps detected in the reference clip -- need >= {N_REFERENCE_REPS}. Aborting.")
        return
    reference_reps = ref_detected[:N_REFERENCE_REPS]
    fps_for_reference = 30.0  # overwritten below once we have a real fps reading
    reference = rules_module.build_reference("own_clips", EXERCISE, reference_reps, fps_for_reference)
    print(f"  Reference medians: {reference['medians']}")

    print(f"\n[2/3] Deriving tolerances from {len(calibration_rows)} calibration clip(s)...")
    calibration_reps = []
    for row in calibration_rows:
        detected, _fps = process_clip_reps(Path(row["clip"]), EXERCISE)
        calibration_reps.extend(detected)
    tolerances = derive_tolerances(calibration_reps, fps_for_reference, reference["medians"])
    print(f"  n calibration reps: {tolerances['n']}  tolerances: {tolerances}")
    if tolerances["n"] < MIN_N_WARNING:
        print(f"  WARNING: n={tolerances['n']} < {MIN_N_WARNING} -- these tolerances are derived from very few reps.")

    demo_config = json.loads(DEMO_CONFIG_PATH.read_text())
    reference_config = {
        "exercise": EXERCISE, "mode": "reference_relative",
        "derived_from": "own recorded calibration clip(s)",
        "n_calibration_reps": tolerances["n"], "date": today, "commit": commit,
        "tol_peak_deg": {"value": tolerances["tol_peak_deg"], "status": "derived"},
        "tol_elbow_deg": {"value": tolerances["tol_elbow_deg"], "status": "derived"},
        "tol_trunk_deg": {"value": tolerances["tol_trunk_deg"], "status": "derived"},
        "tol_tempo_frac": {"value": tolerances["tol_tempo_frac"], "status": "derived"},
        "target_peak_deg": demo_config["target_peak_deg"], "tolerance_deg": demo_config["tolerance_deg"],
        "min_elbow_straight_deg": demo_config["min_elbow_straight_deg"],
        "max_trunk_lean_deg": demo_config["max_trunk_lean_deg"],
        "tempo_min_s": demo_config["tempo_min_s"], "tempo_max_s": demo_config["tempo_max_s"],
        "cue_cooldown_s": demo_config["cue_cooldown_s"], "min_valid_fraction": demo_config["min_valid_fraction"],
        "cues": demo_config["cues"],
    }

    print(f"\n[3/3] Evaluating {len(eval_rows)} eval clip(s) against both configs...")
    csv_rows = []
    per_clip_summary = []
    demo_by_intended, reference_by_intended = {}, {}
    fps_measurements = []
    for row in eval_rows:
        video_path = Path(row["clip"])
        detected_reps, fps_measured = process_clip_reps(video_path, EXERCISE)
        fps_measurements.append(fps_measured)
        expected_reps = int(row["expected_reps"])
        rc = rep_count_metrics(len(detected_reps), expected_reps)
        per_clip_summary.append({"clip": row["clip"], "intended": row["intended"], **rc})
        print(f"  {row['clip']} (intended={row['intended']}): detected={rc['detected']} "
              f"expected={rc['expected']} exact_match={rc['exact_match']} fps={fps_measured:.1f}")

        for rep_idx, rep in enumerate(detected_reps, 1):
            fps_rep = fps_measured if fps_measured and fps_measured > 0 else 30.0
            demo_result = rules_module.score_rep(rep, demo_config, fps_rep)
            reference_result = rules_module.score_rep(rep, reference_config, fps_rep, reference=reference)
            demo_by_intended.setdefault(row["intended"], []).append(demo_result)
            reference_by_intended.setdefault(row["intended"], []).append(reference_result)
            csv_rows.append({
                "clip": row["clip"], "intended": row["intended"], "rep_index": rep_idx,
                "demo_status": demo_result["status"], "demo_failed_rules": ";".join(demo_result["failed_rules"]),
                "reference_status": reference_result["status"],
                "reference_failed_rules": ";".join(reference_result["failed_rules"]),
            })

    avg_fps = float(np.mean(fps_measurements)) if fps_measurements else float("nan")

    def mode_report(by_intended: dict) -> list[str]:
        lines = []
        correct = by_intended.get("correct", [])
        k_ff, n_ff = false_fail_rate(correct)
        lines.append(f"- false-fail rate (correct reps): {rate_with_ci(k_ff, n_ff)}")
        wrong = wrong_rules_on_correct(correct)
        lines.append(f"- which rules fire wrongly on correct reps: {dict(wrong)}")
        fault_rates = fault_detection_rates(by_intended)
        for fault, r in fault_rates.items():
            lines.append(f"- {fault} (expected rule: {r['rule_id']}): any-rule={rate_with_ci(*r['any'])}  "
                          f"expected-rule={rate_with_ci(*r['expected_rule'])}")
        all_results = [r for reps in by_intended.values() for r in reps]
        k_ins, n_ins = insufficient_data_rate(all_results)
        lines.append(f"- insufficient_data rate (all eval reps): {rate_with_ci(k_ins, n_ins)}")
        return lines

    lines = [
        "# Own-clips evaluation\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        f"_landmark_space: `{landmark_space}`_\n",
        f"_configs: demo=`{DEMO_CONFIG_PATH}`  reference_relative=derived in this run_\n",
        f"_manifest_hash: `{man_hash}`  date: {today}_\n",
        "\n## Per-clip rep count\n",
        "| clip | intended | detected | expected | exact_match | within +/-1 | signed_error |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in per_clip_summary:
        lines.append(f"| {s['clip']} | {s['intended']} | {s['detected']} | {s['expected']} | "
                      f"{s['exact_match']} | {s['within_1']} | {s['signed_error']:+d} |")

    lines.append(f"\n## Average FPS across eval clips: {avg_fps:.1f}\n")

    lines.append("\n## Demo config (absolute)\n")
    lines.extend(mode_report(demo_by_intended))
    lines.append("\n## Reference_relative config (derived from calibration clips)\n")
    lines.extend(mode_report(reference_by_intended))

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "own_clips_eval.md").write_text("\n".join(lines))
    if csv_rows:
        csv_path = RESULTS_DIR / "own_clips_eval.csv"
        with csv_path.open("w", newline="") as f:
            f.write(f"# commit: {commit}\n# dirty: {dirty}\n# landmark_space: {landmark_space}\n"
                     f"# demo_config: {DEMO_CONFIG_PATH}\n# reference_config: derived_in_this_run\n"
                     f"# manifest_hash: {man_hash}\n# date: {today}\n")
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"\nWrote {RESULTS_DIR / 'own_clips_eval.md'} and {RESULTS_DIR / 'own_clips_eval.csv'}")


if __name__ == "__main__":
    main()
