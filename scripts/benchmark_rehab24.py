"""PART 3 of the REHAB24-6 benchmark task. Reads data/rehab24_manifest.csv
and data/rehab24_subject_split.json (built by scripts/build_rehab24_
manifest.py) and runs, for Exercise 1 (arm abduction) reps:

  a) rep detection: RepDetector's count vs the dataset's annotated rep
     count, per recording -- SPAN-RESTRICTED via src.pipeline.Session
     (Start at the recording's own annotated span's first start_frame,
     Stop right after its last end_frame), the same mechanism app.py's
     --video --span uses. Reported separately per view (front, half-
     profile, profile), per subject, with Wilson 95% CIs on every rate.
  b) angle fidelity: our MediaPipe-derived shoulder-trunk angle vs the
     SAME compute_angle() on the 3D mocap joints, FRONT VIEW ONLY,
     evaluation subjects -- a measurement result, NOT used to set or
     adjust any threshold.
  c) rule scorer vs the dataset's own correctness labels, FRONT VIEW
     ONLY, evaluation subjects -- run TWICE (the derived benchmark
     config and the live demo config), reported side by side: confusion
     matrix, false-fail rate, detection rate, insufficient_data rate,
     which rule fired, majority-class baseline, each with Wilson 95%
     CIs, per subject and overall.
  d) measured FPS on one dataset video (live MediaPipe, never cached --
     this measures real throughput, not cache-lookup speed). Skipped
     entirely under --rescore-only.

The UI-PRMD classifier is NOT run in this script at all (removed from
this run per task instruction -- see scripts/diagnose_misclassification.py
for that separate cross-domain comparison). Never retrains anything,
never changes config/exercises/*.json. Per-frame landmarks are sourced
from scripts/landmark_cache.py (cache hit: no MediaPipe; cache miss:
runs MediaPipe once, then caches) -- --rescore-only refuses to run
MediaPipe at all, erroring out on any cache miss instead.

Output: results/rehab24_benchmark.md (+ per-rep CSVs), every one
starting with commit hash, dirty flag, landmark_space, config path,
and the date.
"""

import json
import math
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

from scripts import landmark_cache
from src import pose_estimation
from src import rules as rules_module
from src.feature_extraction import (
    check_confidence,
    compute_angle,
    compute_elbow_angle,
    compute_trunk_lean_deg,
    extract_features,
    select_side,
)
from src.pipeline import RepDetector, Session, process_frame
from src.pose_estimation import get_pose_landmarks

DATA_DIR = Path("data/rehab24_6")
MANIFEST_PATH = Path("data/rehab24_manifest.csv")
SPLIT_PATH = Path("data/rehab24_subject_split.json")
RESULTS_DIR = Path("results")
EXERCISE = "shoulder_abduction"
BENCHMARK_CONFIG_PATH = Path("config/exercises/benchmarks/rehab24_abduction.json")
DEMO_CONFIG_PATH = rules_module.CONFIG_DIR / f"{EXERCISE}.json"

# Set by main() from --rescore-only; read by every landmark-sourcing
# helper below instead of being threaded through every call site.
RESCORE_ONLY = False

# Mocap joint indices (joints_names.txt) for the SAME (hip, shoulder,
# elbow, wrist) roles feature_extraction.py uses -- right side only,
# since Ex1 is always the right arm (dataset description: "sideway
# raising of the straightened right arm"). "Shoulder"=RightArm, not
# RightShoulder, mirrors UI-PRMD's own bone-start convention (a bone's
# name is the segment it starts, so "Arm" starts at the true
# glenohumeral shoulder joint, same role src/dataset_loader.py already
# uses for UI-PRMD's "UpperArm").
MOCAP_ROLE_TO_JOINT_INDEX = {"hip": 0, "shoulder": 12, "elbow": 13, "wrist": 14}


def git_provenance() -> tuple[str, bool]:
    """(commit_hash, is_dirty) for the current working tree, so every
    results file records exactly what code produced it. Returns
    ("unknown", True) if git isn't available or this isn't a repo --
    missing provenance shouldn't fail the whole run."""
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


def write_csv_with_provenance(df: pd.DataFrame, path: Path, commit: str, dirty: bool,
                               landmark_space: str, config_path: str, date_str: str) -> None:
    with path.open("w", newline="") as f:
        f.write(f"# commit: {commit}\n# dirty: {dirty}\n# landmark_space: {landmark_space}\n"
                f"# config: {config_path}\n# date: {date_str}\n")
        df.to_csv(f, index=False)


def load_manifest_and_split() -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(MANIFEST_PATH)
    split = json.loads(SPLIT_PATH.read_text())
    return manifest, split


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (default z for 95%).
    (nan, nan) if n == 0 -- no rate to report."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def rate_with_ci(k: int, n: int) -> str:
    if n == 0:
        return "N/A (n=0)"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({k / n:.1%}, 95% CI [{lo:.1%}, {hi:.1%}])"


# ---------------------------------------------------------------------------
# landmark source: cache-backed (scripts/landmark_cache.py), respects
# the module-level RESCORE_ONLY flag
# ---------------------------------------------------------------------------

def video_landmark_cache(video_path: Path) -> dict:
    return landmark_cache.get_or_build(video_path, rescore_only=RESCORE_ONLY)


# ---------------------------------------------------------------------------
# (a) rep detection -- span-restricted via src.pipeline.Session
# ---------------------------------------------------------------------------

def recording_span(manifest: pd.DataFrame, video_id: str, camera: str) -> tuple[int, int]:
    reps = manifest[(manifest["video_id"] == video_id) & (manifest["camera"] == camera)]
    return int(reps["start_frame"].min()), int(reps["end_frame"].max())


def run_repdetector_on_recording_span(video_path: Path, start_frame: int, end_frame: int) -> int:
    """Session-gated (Start/Stop-equivalent) rep count over the WHOLE
    recording: Session.start() fires exactly at frame_idx==start_frame,
    Session.stop() right after processing frame_idx==end_frame -- frames
    outside [start_frame, end_frame] are ignored by the rep detector
    entirely, the same mechanism app.py's --video --span uses (both
    drive src.pipeline.Session)."""
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    cache = video_landmark_cache(video_path)
    end_frame = min(end_frame, cache["frame_count"] - 1)
    required_roles = rules_module.required_joint_roles()
    session = Session(EXERCISE)
    rep_count = 0
    fps = cache["fps"]
    for frame_idx in range(cache["frame_count"]):
        t = frame_idx / fps
        if frame_idx == start_frame:
            session.start()
        landmarks = landmark_cache.landmarks_at(cache, frame_idx, landmark_space)
        event, _payload, _visible, _failed = session.process_frame(landmarks, None, t, required_roles)
        if frame_idx == end_frame:
            session.stop()
        if event == "rep_complete":
            rep_count += 1
    return rep_count


def part_a_rep_detection(manifest: pd.DataFrame) -> pd.DataFrame:
    recordings = manifest.drop_duplicates(subset=["video_id", "camera"])[
        ["subject", "video_id", "camera", "view", "video"]
    ]
    rows = []
    for _, rec in recordings.iterrows():
        annotated = len(manifest[(manifest["video_id"] == rec["video_id"]) & (manifest["camera"] == rec["camera"])])
        video_path = DATA_DIR / "videos" / rec["video"]
        if not video_path.exists():
            print(f"  MISSING video: {video_path} -- skipped")
            continue
        start_frame, end_frame = recording_span(manifest, rec["video_id"], rec["camera"])
        try:
            detected = run_repdetector_on_recording_span(video_path, start_frame, end_frame)
        except RuntimeError as exc:
            print(f"  SKIPPED {rec['video_id']} ({rec['camera']}): {exc}")
            continue
        rows.append({
            "subject": int(rec["subject"]), "video_id": rec["video_id"], "camera": rec["camera"],
            "view": rec["view"], "annotated_reps": annotated, "detected_reps": detected,
            "signed_error": detected - annotated, "abs_error": abs(annotated - detected),
            "exact_match": int(annotated == detected),
        })
        print(f"  {rec['video_id']} ({rec['camera']}, {rec['view']}): annotated={annotated} detected={detected} "
              f"span=({start_frame},{end_frame})")
    return pd.DataFrame(rows)


def rep_detection_report_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["N/A -- no recordings processed.\n"]
    lines = []
    for view in sorted(df["view"].unique()):
        sub = df[df["view"] == view]
        k, n = int(sub["exact_match"].sum()), len(sub)
        lines.append(f"\n### view={view}\n")
        lines.append(f"Exact-match rate: {rate_with_ci(k, n)}  "
                      f"mean signed_error={sub['signed_error'].mean():+.2f} (std={sub['signed_error'].std():.2f})  "
                      f"n={n} recordings\n")
        lines.append("| subject | n recordings | exact-match rate (95% CI) | mean signed_error |")
        lines.append("|---|---|---|---|")
        for subject in sorted(sub["subject"].unique()):
            ssub = sub[sub["subject"] == subject]
            k_s, n_s = int(ssub["exact_match"].sum()), len(ssub)
            lines.append(f"| {subject} | {n_s} | {rate_with_ci(k_s, n_s)} | {ssub['signed_error'].mean():+.2f} |")
    return lines


# ---------------------------------------------------------------------------
# (b) angle fidelity -- front view only, evaluation subjects, measurement only
# ---------------------------------------------------------------------------

def load_mocap_joints(video_id: str) -> np.ndarray | None:
    """Returns an (n_frames, 26, 3) array (x, y, z, meters, Y-up) for this
    video_id's 3D joints, synced 1:1 by frame index to the 30 FPS video
    and to Segmentation.csv's first_frame/last_frame -- verified from the
    real file: Ex1/{video_id}-30fps.npy has shape (n, 26, 4), the 4th
    column is a constant 1.0 (homogeneous coordinate, not an occlusion
    flag -- checked across 3 files), and n covers every annotated
    last_frame for that video_id with room to spare. The sibling
    "-120fps.npy" file (the mocap system's native rate) is NOT used here.
    """
    path = DATA_DIR / "3d_joints" / "Ex1" / f"{video_id}-30fps.npy"
    if not path.exists():
        return None
    return np.load(path)[:, :, :3]


def mocap_shoulder_trunk_angle(joints_frame: np.ndarray) -> float:
    hip = joints_frame[MOCAP_ROLE_TO_JOINT_INDEX["hip"]]
    shoulder = joints_frame[MOCAP_ROLE_TO_JOINT_INDEX["shoulder"]]
    elbow = joints_frame[MOCAP_ROLE_TO_JOINT_INDEX["elbow"]]
    return compute_angle(hip, shoulder, elbow)


def mediapipe_angle_series_for_rep(video_path: Path, start_frame: int, end_frame: int) -> list[float | None]:
    """Per-frame compute_shoulder_abduction_angle() (via extract_features)
    over [start_frame, end_frame], None where confidence-gated invalid.
    Uses the exercise's own base required roles (not the rules' wrist-
    inclusive set) since this is a geometry comparison, not a scoring
    run. Landmarks come from the per-video cache (scripts/landmark_
    cache.py) -- cache hit: no MediaPipe; cache miss: runs it once.
    """
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    cache = video_landmark_cache(video_path)
    end_frame = min(end_frame, cache["frame_count"] - 1)
    angles = []
    for frame_idx in range(start_frame, end_frame + 1):
        landmarks = landmark_cache.landmarks_at(cache, frame_idx, landmark_space)
        features = extract_features(landmarks, EXERCISE) if landmarks is not None else None
        angles.append(features["angle_deg"] if features else None)
    return angles


def part_b_angle_fidelity_front(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    eval_front = manifest[manifest["subject"].isin(split["evaluation_subjects"]) & (manifest["view"] == "front")]
    rows = []
    for _, rep in eval_front.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        joints = load_mocap_joints(rep["video_id"])
        if not video_path.exists() or joints is None:
            continue
        try:
            mp_angles = mediapipe_angle_series_for_rep(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        except RuntimeError as exc:
            print(f"  SKIPPED {rep['video_id']}: {exc}")
            continue
        mocap_angles = [
            mocap_shoulder_trunk_angle(joints[f]) for f in range(int(rep["start_frame"]), int(rep["end_frame"]) + 1)
            if f < len(joints)
        ]
        n = min(len(mp_angles), len(mocap_angles))
        for i in range(n):
            if mp_angles[i] is not None:
                rows.append({
                    "subject": int(rep["subject"]), "view": rep["view"],
                    "mediapipe_deg": mp_angles[i], "mocap_deg": mocap_angles[i],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["error_deg"] = df["mediapipe_deg"] - df["mocap_deg"]
    return df


def angle_fidelity_report_lines(df: pd.DataFrame) -> list[str]:
    lines = ["**Measurement result only -- NOT used to set or adjust any threshold.**\n"]
    if df.empty:
        return lines + ["N/A -- no front-view angle-fidelity data for evaluation subjects.\n"]
    mae, bias = df["error_deg"].abs().mean(), df["error_deg"].mean()
    corr = df["mediapipe_deg"].corr(df["mocap_deg"])
    lines.append(f"Overall (front view, evaluation subjects): n={len(df)} frames  MAE={mae:.2f} deg  "
                  f"bias={bias:+.2f} deg  corr={corr:.3f}\n")
    lines.append("| subject | n frames | MAE (deg) | bias (deg) | corr |")
    lines.append("|---|---|---|---|---|")
    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]
        if len(sub) < 2:
            lines.append(f"| {subject} | {len(sub)} | N/A | N/A | N/A |")
            continue
        lines.append(f"| {subject} | {len(sub)} | {sub['error_deg'].abs().mean():.2f} | "
                      f"{sub['error_deg'].mean():+.2f} | {sub['mediapipe_deg'].corr(sub['mocap_deg']):.3f} |")
    return lines


# ---------------------------------------------------------------------------
# (c) rule scorer vs labels -- front view only, evaluation subjects,
# two configs side by side. Classifier NOT run in this script.
# ---------------------------------------------------------------------------

def rep_feature_sequences(video_path: Path, start_frame: int, end_frame: int):
    """Per-rep angle_deg/elbow_angle_deg/trunk_lean_deg + valid_fraction,
    confidence-gated with rules.required_joint_roles() -- same gate as
    live scoring. Landmarks come from the per-video cache (scripts/
    landmark_cache.py) -- cache hit: no MediaPipe; cache miss: runs it
    once, respecting the module-level RESCORE_ONLY flag."""
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    required_roles = rules_module.required_joint_roles()
    cache = video_landmark_cache(video_path)
    fps = cache["fps"]
    end_frame = min(end_frame, cache["frame_count"] - 1)
    angle, elbow, trunk = [], [], []
    total = 0
    for frame_idx in range(start_frame, end_frame + 1):
        total += 1
        landmarks = landmark_cache.landmarks_at(cache, frame_idx, landmark_space)
        if landmarks is None:
            continue
        side = select_side(landmarks)
        visible, _failed = check_confidence(landmarks, side, EXERCISE, required_roles)
        if not visible:
            continue
        features = extract_features(landmarks, EXERCISE)
        angle.append(features["angle_deg"])
        elbow.append(compute_elbow_angle(landmarks))
        trunk.append(compute_trunk_lean_deg(landmarks))
    valid_fraction = len(angle) / total if total else 0.0
    return {"angle_deg": angle, "elbow_angle_deg": elbow, "trunk_lean_deg": trunk, "valid_fraction": valid_fraction}, fps


def part_c_rules_vs_labels_front(manifest: pd.DataFrame, split: dict, config: dict) -> pd.DataFrame:
    eval_front = manifest[manifest["subject"].isin(split["evaluation_subjects"]) & (manifest["view"] == "front")]
    rows = []
    for _, rep in eval_front.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        if not video_path.exists():
            continue
        try:
            seq, fps = rep_feature_sequences(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        except RuntimeError as exc:
            print(f"  SKIPPED {rep['video_id']}: {exc}")
            continue
        if len(seq["angle_deg"]) == 0:
            status, failed_rules = "insufficient_data", []
        else:
            result = rules_module.score_rep(seq, config, fps)
            status, failed_rules = result["status"], result["failed_rules"]
        rows.append({
            "subject": int(rep["subject"]), "view": rep["view"], "label": rep["label"],
            "status": status, "failed_rules": ";".join(failed_rules),
        })
    return pd.DataFrame(rows)


def scorer_rates(df: pd.DataFrame) -> dict:
    scorable = df[df["status"] != "insufficient_data"]
    correct_rows = scorable[scorable["label"] == "correct"]
    incorrect_rows = scorable[scorable["label"] == "incorrect"]
    confusion = {
        "correct": {"pass": int((correct_rows["status"] == "pass").sum()), "fail": int((correct_rows["status"] == "fail").sum())},
        "incorrect": {"pass": int((incorrect_rows["status"] == "pass").sum()), "fail": int((incorrect_rows["status"] == "fail").sum())},
    }
    majority_label = df["label"].mode().iloc[0] if len(df) else None
    maj_k = int((df["label"] == majority_label).sum()) if majority_label is not None else 0
    rule_fire_counts = Counter(
        r for s in scorable[scorable["status"] == "fail"]["failed_rules"] for r in (s.split(";") if s else [])
    )
    return {
        "confusion": confusion,
        "false_fail": (confusion["correct"]["fail"], len(correct_rows)),
        "detection": (confusion["incorrect"]["fail"], len(incorrect_rows)),
        "insufficient": (len(df) - len(scorable), len(df)),
        "majority_baseline": (maj_k, len(df)),
        "rule_fire_counts": dict(rule_fire_counts),
    }


def per_subject_scorer_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["N/A\n"]
    lines = ["| subject | n reps | false-fail (95% CI) | detection (95% CI) | insufficient_data (95% CI) | majority_baseline (95% CI) |",
              "|---|---|---|---|---|---|"]
    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]
        r = scorer_rates(sub)
        lines.append(f"| {subject} | {len(sub)} | {rate_with_ci(*r['false_fail'])} | {rate_with_ci(*r['detection'])} | "
                      f"{rate_with_ci(*r['insufficient'])} | {rate_with_ci(*r['majority_baseline'])} |")
    return lines


def part_c_report_lines(df_benchmark: pd.DataFrame, df_demo: pd.DataFrame,
                         benchmark_config_path: Path, demo_config_path: Path) -> list[str]:
    lines = [f"**Front view, evaluation subjects only. Classifier NOT run.** "
             f"Benchmark config: `{benchmark_config_path}`  Demo config: `{demo_config_path}`\n"]
    if df_benchmark.empty and df_demo.empty:
        return lines + ["N/A -- no front-view evaluation-subject reps scored.\n"]

    r_bench, r_demo = scorer_rates(df_benchmark), scorer_rates(df_demo)
    lines.append("| metric | benchmark config | demo config |")
    lines.append("|---|---|---|")
    lines.append(f"| false-fail rate (correct reps scored fail) | {rate_with_ci(*r_bench['false_fail'])} | {rate_with_ci(*r_demo['false_fail'])} |")
    lines.append(f"| detection rate (incorrect reps scored fail) | {rate_with_ci(*r_bench['detection'])} | {rate_with_ci(*r_demo['detection'])} |")
    lines.append(f"| insufficient_data rate | {rate_with_ci(*r_bench['insufficient'])} | {rate_with_ci(*r_demo['insufficient'])} |")
    lines.append(f"| majority-class baseline | {rate_with_ci(*r_bench['majority_baseline'])} | {rate_with_ci(*r_demo['majority_baseline'])} |")
    lines.append(f"| confusion matrix | {r_bench['confusion']} | {r_demo['confusion']} |")
    lines.append(f"| which rule fired (fail reps) | {r_bench['rule_fire_counts']} | {r_demo['rule_fire_counts']} |")

    lines.append("\n#### per-subject (benchmark config)\n")
    lines.extend(per_subject_scorer_lines(df_benchmark))
    lines.append("\n#### per-subject (demo config)\n")
    lines.extend(per_subject_scorer_lines(df_demo))
    return lines


# ---------------------------------------------------------------------------
# (d) FPS -- live MediaPipe only, never cached; skipped under --rescore-only
# ---------------------------------------------------------------------------

def part_e_fps(video_path: Path) -> float | None:
    if RESCORE_ONLY:
        return None
    required_roles = rules_module.required_joint_roles()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # must read before release() -- a released capture returns -1 (truthy!)
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    n_frames = 0
    frame_idx = 0
    start_t = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        landmarks = get_pose_landmarks(frame)  # LIVE, never cached -- measures real MediaPipe throughput
        process_frame(detector, parallel, landmarks, EXERCISE, None, t, required_roles)
        n_frames += 1
    elapsed = time.perf_counter() - start_t
    cap.release()
    return n_frames / elapsed if elapsed > 0 else float("nan")


def main(config_path: str | None = None, rescore_only: bool = False) -> None:
    """`config_path`, if given, overrides which file is treated as "the
    benchmark config" in part (c)'s side-by-side comparison (default:
    BENCHMARK_CONFIG_PATH, built by scripts/build_benchmark_config.py).
    The demo config (DEMO_CONFIG_PATH) is always the second, fixed side
    of that comparison. `rescore_only`, if True, refuses to run
    MediaPipe at all -- every video must already have a landmark_cache
    entry (see scripts/landmark_cache.py), and (d) FPS is skipped
    entirely (it inherently measures live MediaPipe throughput)."""
    global RESCORE_ONLY
    RESCORE_ONLY = rescore_only

    RESULTS_DIR.mkdir(exist_ok=True)
    commit, dirty = git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    today = date.today().isoformat()
    benchmark_config_path = Path(config_path) if config_path else BENCHMARK_CONFIG_PATH
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}  date={today}  "
          f"rescore_only={RESCORE_ONLY}")
    print(f"Benchmark config: {benchmark_config_path}   Demo config: {DEMO_CONFIG_PATH}")
    if not MANIFEST_PATH.exists() or not SPLIT_PATH.exists():
        print(f"Missing {MANIFEST_PATH} or {SPLIT_PATH} -- run scripts/build_rehab24_manifest.py first.")
        return

    manifest, split = load_manifest_and_split()
    print(f"Manifest: {len(manifest)} rows. Calibration subjects: {split['calibration_subjects']}  "
          f"Evaluation subjects: {split['evaluation_subjects']}")

    config_benchmark = json.loads(benchmark_config_path.read_text())
    config_demo = json.loads(DEMO_CONFIG_PATH.read_text())

    print("\n[a] Rep detection (span-restricted via src.pipeline.Session, all subjects/cameras/views)...")
    part_a = part_a_rep_detection(manifest)

    print("\n[b] Angle fidelity vs mocap (front view, evaluation subjects -- measurement only)...")
    part_b = part_b_angle_fidelity_front(manifest, split)

    print("\n[c] Rule scorer vs labels (front view, evaluation subjects) -- benchmark config...")
    part_c_benchmark = part_c_rules_vs_labels_front(manifest, split, config_benchmark)
    print("[c] ... demo config...")
    part_c_demo = part_c_rules_vs_labels_front(manifest, split, config_demo)

    print(f"\n[d] Measured FPS...{' SKIPPED (--rescore-only)' if RESCORE_ONLY else ''}")
    sample_video = None
    for _, rec in manifest.drop_duplicates(subset=["video"]).iterrows():
        candidate = DATA_DIR / "videos" / rec["video"]
        if candidate.exists():
            sample_video = candidate
            break
    fps_measured = part_e_fps(sample_video) if sample_video else None

    # ---- write outputs ----
    if not part_a.empty:
        write_csv_with_provenance(part_a, RESULTS_DIR / "rehab24_rep_detection.csv", commit, dirty,
                                   landmark_space, str(benchmark_config_path), today)
    if not part_b.empty:
        write_csv_with_provenance(part_b, RESULTS_DIR / "rehab24_angle_fidelity.csv", commit, dirty,
                                   landmark_space, str(benchmark_config_path), today)
    if not part_c_benchmark.empty:
        write_csv_with_provenance(part_c_benchmark, RESULTS_DIR / "rehab24_scorer_vs_labels_benchmark.csv", commit,
                                   dirty, landmark_space, str(benchmark_config_path), today)
    if not part_c_demo.empty:
        write_csv_with_provenance(part_c_demo, RESULTS_DIR / "rehab24_scorer_vs_labels_demo.csv", commit, dirty,
                                   landmark_space, str(DEMO_CONFIG_PATH), today)

    lines = ["# REHAB24-6 benchmark -- shoulder_abduction\n"]
    lines.append(f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n")
    lines.append(f"_landmark_space: `{landmark_space}` (config/pose.json)_\n")
    lines.append(f"_config: benchmark=`{benchmark_config_path}`  demo=`{DEMO_CONFIG_PATH}`_\n")
    lines.append(f"_date: {today}_\n")
    lines.append(f"Calibration subjects: {split['calibration_subjects']}  "
                  f"Evaluation subjects: {split['evaluation_subjects']}\n")

    lines.append("\n## (a) Rep detection (span-restricted)\n")
    lines.extend(rep_detection_report_lines(part_a))

    lines.append("\n## (b) Angle fidelity (front view, evaluation subjects)\n")
    lines.extend(angle_fidelity_report_lines(part_b))

    lines.append("\n## (c) Rule scorer vs labels (front view, evaluation subjects)\n")
    lines.extend(part_c_report_lines(part_c_benchmark, part_c_demo, benchmark_config_path, DEMO_CONFIG_PATH))

    lines.append("\n## (d) Measured FPS\n")
    lines.append(f"{fps_measured:.1f} fps on {sample_video}" if fps_measured else
                 ("N/A (--rescore-only)" if RESCORE_ONLY else "N/A"))

    (RESULTS_DIR / "rehab24_benchmark.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'rehab24_benchmark.md'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                         help="Path to the rules config JSON to use as the 'benchmark config' side of part "
                              "(c)'s comparison, instead of config/exercises/benchmarks/rehab24_abduction.json "
                              "(the demo config, config/exercises/shoulder_abduction.json, is always the other side).")
    parser.add_argument("--rescore-only", action="store_true",
                         help="Never run MediaPipe -- every video must already have a scripts/landmark_cache.py "
                              "entry, or this errors out. Part (d) FPS is skipped entirely in this mode.")
    args = parser.parse_args()
    main(args.config, args.rescore_only)
