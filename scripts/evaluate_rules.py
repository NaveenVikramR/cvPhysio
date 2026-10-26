"""PART C evaluation harness for src/rules.py. Reads data/clips_manifest.
csv (clip, set=calibration|eval, exercise, intended=correct|fault_lowarm|
fault_trunk|fault_elbow|fault_fast, expected_reps) -- a template with a
few EXAMPLE_ rows ships with this script; real eval clips and the real
manifest are gitignored (the user fills them in locally). This script
only reads thresholds from config/exercises/*.json -- it never writes or
changes them.

Per clip that actually exists on disk: detected rep count vs
expected_reps, and per rep, src.rules.score_rep()'s pass/fail and
measured values. Smoothing (src/smoothing.py) is deliberately NOT
applied here -- scripts/measure_smoothing_effect.py already found it can
change rep count on the one real clip tested, so turning it on inside an
accuracy harness by default would conflate two different questions.

Outputs: results/rules_eval.md (human-readable report) and
results/rules_eval.csv (one row per scored rep).
"""

import csv
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules as rules_module
from src.feature_extraction import check_confidence, compute_elbow_angle, compute_trunk_lean_deg, extract_features, select_side
from src.pose_estimation import get_pose_landmarks

MANIFEST_PATH = Path("data/clips_manifest.csv")
RESULTS_DIR = Path("results")
RULE_IDS = ["peak_angle", "elbow_straight", "trunk_lean", "tempo"]
FAULT_TO_RULE = {
    "fault_lowarm": "peak_angle",
    "fault_trunk": "trunk_lean",
    "fault_elbow": "elbow_straight",
    "fault_fast": "tempo",
}


def load_rep_detector_and_process_frame():
    """exec()-slice app.py's RepDetector class + process_frame() out of
    its source text -- same technique as the other scripts/*.py scripts
    in this project, needed because app.py now parses CLI args and
    launches Streamlit/headless logic at import time."""
    source = Path("app.py").read_text()
    start = source.index("CALIBRATION_FRAMES = 15")
    end = source.index("\ndef run_headless(video_path")
    namespace = {
        "statistics": statistics, "Counter": Counter, "deque": deque,
        "check_confidence": check_confidence, "extract_features": extract_features,
        "select_side": select_side, "compute_elbow_angle": compute_elbow_angle,
        "compute_trunk_lean_deg": compute_trunk_lean_deg,
    }
    exec(source[start:end], namespace)
    return namespace["RepDetector"], namespace["process_frame"]


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def score_clip(clip_path: str, exercise: str, config: dict, RepDetectorClass, process_frame) -> tuple[list[dict], float]:
    """Returns (list of score_rep() results, measured fps) for one clip."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": []}
    results = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = get_pose_landmarks(frame)
        event, payload, _visible, _failed = process_frame(detector, parallel, landmarks, exercise, None)
        if event == "rep_complete":
            rep = {
                "angle_deg": payload["angle_deg"],
                "elbow_angle_deg": payload["elbow_angle_deg"],
                "trunk_lean_deg": payload["trunk_lean_deg"],
            }
            results.append(rules_module.score_rep(rep, config, fps))
    cap.release()
    return results, fps


def measure_fps(clip_path: str, exercise: str, config: dict, RepDetectorClass, process_frame) -> float:
    """Real measured frames/sec of the full live loop (pose + smoothing-
    off + confidence + RepDetector + rules scoring) on one clip."""
    cap = cv2.VideoCapture(clip_path)
    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": []}
    n_frames = 0
    start_t = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = get_pose_landmarks(frame)
        event, payload, _visible, _failed = process_frame(detector, parallel, landmarks, exercise, None)
        if event == "rep_complete":
            rep = {
                "angle_deg": payload["angle_deg"],
                "elbow_angle_deg": payload["elbow_angle_deg"],
                "trunk_lean_deg": payload["trunk_lean_deg"],
            }
            rules_module.score_rep(rep, config, cap.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames += 1
    elapsed = time.perf_counter() - start_t
    cap.release()
    return n_frames / elapsed if elapsed > 0 else float("nan")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = read_manifest(MANIFEST_PATH)
    if not rows:
        print(f"No manifest found at {MANIFEST_PATH} -- nothing to evaluate.")
        return

    RepDetectorClass, process_frame = load_rep_detector_and_process_frame()

    missing, usable = [], []
    for row in rows:
        if Path(row["clip"]).exists():
            usable.append(row)
        else:
            missing.append(row)

    print(f"Manifest rows: {len(rows)}  usable (file exists): {len(usable)}  missing: {len(missing)}")
    for row in missing:
        print(f"  MISSING: {row['clip']} (set={row['set']}, intended={row['intended']}) -- not run.")

    csv_rows = []
    per_clip_summary = []
    calibration_measurements = []  # (clip, rep_idx, rule_id, measured)
    eval_rep_rows = []  # (intended, failed_rules, passed)

    config_cache = {}

    for row in usable:
        exercise = row["exercise"]
        if exercise not in config_cache:
            try:
                config_cache[exercise] = rules_module.load_config(exercise)
            except FileNotFoundError as exc:
                config_cache[exercise] = exc
        config = config_cache[exercise]
        if isinstance(config, Exception):
            print(f"  SKIPPED {row['clip']}: {config}")
            continue

        print(f"Scoring {row['clip']} (exercise={exercise}, set={row['set']}, intended={row['intended']})...")
        results, fps = score_clip(row["clip"], exercise, config, RepDetectorClass, process_frame)
        expected = int(row["expected_reps"])
        detected = len(results)
        per_clip_summary.append({
            "clip": row["clip"], "set": row["set"], "intended": row["intended"],
            "expected_reps": expected, "detected_reps": detected,
        })

        for rep_idx, result in enumerate(results, 1):
            csv_row = {
                "clip": row["clip"], "set": row["set"], "exercise": exercise,
                "intended": row["intended"], "rep_index": rep_idx,
                "passed": result["passed"], "failed_rules": ";".join(result["failed_rules"]),
            }
            for rule_id in RULE_IDS:
                r = result["rules"].get(rule_id)
                csv_row[f"{rule_id}_measured"] = r["measured"] if r else ""
                csv_row[f"{rule_id}_allowed_lo"] = r["allowed"][0] if r else ""
                csv_row[f"{rule_id}_allowed_hi"] = r["allowed"][1] if r else ""
                csv_row[f"{rule_id}_passed"] = r["passed"] if r else ""
            csv_rows.append(csv_row)

            if row["set"] == "calibration" and row["intended"] == "correct":
                for rule_id in RULE_IDS:
                    r = result["rules"].get(rule_id)
                    if r:
                        calibration_measurements.append((row["clip"], rep_idx, rule_id, r["measured"]))
            if row["set"] == "eval":
                eval_rep_rows.append((row["intended"], result["failed_rules"], result["passed"]))

    # write per-rep CSV
    csv_path = RESULTS_DIR / "rules_eval.csv"
    if csv_rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote {csv_path} ({len(csv_rows)} rep rows)")
    else:
        print(f"\nNo reps were scored -- {csv_path} not written.")

    # confusion matrix on eval set (ground truth correct/fault vs predicted pass/fail)
    confusion = {"correct": {"pass": 0, "fail": 0}, "fault": {"pass": 0, "fail": 0}}
    for intended, _failed_rules, passed in eval_rep_rows:
        gt = "correct" if intended == "correct" else "fault"
        confusion[gt]["pass" if passed else "fail"] += 1

    detection_rates = {}
    for fault_type, rule_id in FAULT_TO_RULE.items():
        matching = [fr for intended, fr, _p in eval_rep_rows if intended == fault_type]
        if matching:
            hit = sum(1 for fr in matching if rule_id in fr)
            detection_rates[fault_type] = (hit, len(matching))

    correct_eval_reps = [(fr, p) for intended, fr, p in eval_rep_rows if intended == "correct"]
    false_fail_count = sum(1 for _fr, p in correct_eval_reps if not p)
    wrong_rule_counts = Counter()
    for fr, p in correct_eval_reps:
        if not p:
            wrong_rule_counts.update(fr)

    fps_measured = None
    if usable:
        calib_row = next((r for r in usable if r["set"] == "calibration" and Path(r["clip"]).exists()), usable[0])
        exercise = calib_row["exercise"]
        config = config_cache.get(exercise)
        if config and not isinstance(config, Exception):
            print(f"\nMeasuring live-loop FPS on {calib_row['clip']}...")
            fps_measured = measure_fps(calib_row["clip"], exercise, config, RepDetectorClass, process_frame)

    # ---- build markdown report ----
    lines = []
    lines.append("# src/rules.py evaluation\n")
    lines.append(f"Manifest: `{MANIFEST_PATH}` -- {len(rows)} rows, {len(usable)} usable, {len(missing)} missing.\n")

    if missing:
        lines.append("## Missing clips (listed in the manifest, no file on disk)\n")
        for row in missing:
            lines.append(f"- `{row['clip']}` (set={row['set']}, intended={row['intended']}) -- NOT RUN\n")
        lines.append("")

    lines.append("## Rep count: detected vs expected\n")
    lines.append("| clip | set | intended | expected_reps | detected_reps |")
    lines.append("|---|---|---|---|---|")
    for s in per_clip_summary:
        lines.append(f"| {s['clip']} | {s['set']} | {s['intended']} | {s['expected_reps']} | {s['detected_reps']} |")
    lines.append("")

    eval_clips = [s for s in per_clip_summary if s["set"] == "eval"]
    if eval_clips:
        exact = sum(1 for s in eval_clips if s["expected_reps"] == s["detected_reps"])
        lines.append(f"Rep-count accuracy (eval set): {exact}/{len(eval_clips)} clips matched expected_reps exactly.\n")
    else:
        lines.append("Rep-count accuracy (eval set): N/A -- no usable eval-set clips.\n")

    lines.append("## Confusion matrix (eval set): intended vs predicted\n")
    if eval_rep_rows:
        lines.append("| intended \\ predicted | pass | fail |")
        lines.append("|---|---|---|")
        lines.append(f"| correct | {confusion['correct']['pass']} | {confusion['correct']['fail']} |")
        lines.append(f"| fault (any) | {confusion['fault']['pass']} | {confusion['fault']['fail']} |")
        lines.append("")
    else:
        lines.append("N/A -- no usable eval-set reps.\n")

    lines.append("## Per-fault detection rate (matching rule fired)\n")
    if detection_rates:
        lines.append("| fault type | matching rule | detection rate |")
        lines.append("|---|---|---|")
        for fault_type, rule_id in FAULT_TO_RULE.items():
            if fault_type in detection_rates:
                hit, total = detection_rates[fault_type]
                lines.append(f"| {fault_type} | {rule_id} | {hit}/{total} |")
            else:
                lines.append(f"| {fault_type} | {rule_id} | N/A -- no usable clips |")
        lines.append("")
    else:
        lines.append("N/A -- no usable eval-set clips for any fault type.\n")

    lines.append("## False-fail rate on correct reps (eval set)\n")
    if correct_eval_reps:
        lines.append(f"{false_fail_count}/{len(correct_eval_reps)} intended-correct eval reps failed at least one rule.\n")
        if wrong_rule_counts:
            lines.append("Rules that fired wrongly on correct reps:\n")
            for rule_id, count in wrong_rule_counts.most_common():
                lines.append(f"- {rule_id}: {count}")
            lines.append("")
    else:
        lines.append("N/A -- no usable intended-correct eval reps.\n")

    lines.append("## Calibration set: per-rule measured values (correct reps)\n")
    lines.append("For the physiotherapist to choose real thresholds from -- this script does not change config/exercises/*.json.\n")
    if calibration_measurements:
        by_rule = defaultdict(list)
        for _clip, _rep_idx, rule_id, measured in calibration_measurements:
            by_rule[rule_id].append(measured)
        lines.append("| rule | n | values |")
        lines.append("|---|---|---|")
        for rule_id in RULE_IDS:
            values = by_rule.get(rule_id, [])
            if values:
                lines.append(f"| {rule_id} | {len(values)} | {[round(v, 1) for v in values]} |")
        lines.append("")
    else:
        lines.append("N/A -- no usable calibration-set correct reps.\n")

    lines.append("## Live-loop FPS (measured)\n")
    if fps_measured is not None:
        lines.append(f"{fps_measured:.1f} fps, measured on `{calib_row['clip']}` (pose estimation + confidence gate + RepDetector + rules scoring, smoothing off).\n")
    else:
        lines.append("N/A -- no usable clip to measure.\n")

    md_path = RESULTS_DIR / "rules_eval.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
