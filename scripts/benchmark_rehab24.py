"""PART 3 of the REHAB24-6 benchmark task. Reads data/rehab24_manifest.csv
and data/rehab24_subject_split.json (built by scripts/build_rehab24_
manifest.py) and runs, for Exercise 1 (arm abduction) reps:

  a) rep detection: RepDetector's count vs the dataset's annotated rep
     count, per recording, per view.
  b) angle fidelity: our MediaPipe-derived shoulder-trunk angle vs the
     SAME compute_angle() on the 3D mocap joints, per view, on
     evaluation subjects only.
  c) rule scorer vs the dataset's own correctness labels, on evaluation
     subjects only (config thresholds read, never changed); calibration-
     subject percentiles printed separately for threshold-setting.
  d) the UI-PRMD-trained classifier on the same reps, cross-domain,
     same metrics as (c), clearly labelled.
  e) measured FPS on one dataset video.

Never retrains anything, never changes config/exercises/*.json. Output:
results/rehab24_benchmark.md (+ a per-rep CSV, results/rehab24_benchmark.csv).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules as rules_module
from src.classifier import CorrectnessClassifier
from src.feature_extraction import (
    check_confidence,
    compute_angle,
    compute_elbow_angle,
    compute_trunk_lean_deg,
    extract_features,
    extract_rep_features,
    select_side,
)
from src import pose_estimation
from src.pipeline import RepDetector, process_frame
from src.pose_estimation import get_pose_landmarks

DATA_DIR = Path("data/rehab24_6")
MANIFEST_PATH = Path("data/rehab24_manifest.csv")
SPLIT_PATH = Path("data/rehab24_subject_split.json")
RESULTS_DIR = Path("results")
EXERCISE = "shoulder_abduction"

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


def write_csv_with_provenance(df: pd.DataFrame, path: Path, commit: str, dirty: bool, landmark_space: str) -> None:
    with path.open("w", newline="") as f:
        f.write(f"# commit: {commit}\n# dirty: {dirty}\n# landmark_space: {landmark_space}\n")
        df.to_csv(f, index=False)


def load_manifest_and_split() -> tuple[pd.DataFrame, dict]:
    manifest = pd.read_csv(MANIFEST_PATH)
    split = json.loads(SPLIT_PATH.read_text())
    return manifest, split


# ---------------------------------------------------------------------------
# (a) rep detection
# ---------------------------------------------------------------------------

def run_repdetector_on_recording(video_path: Path, RepDetectorClass, process_frame) -> int:
    """Runs the whole recording through RepDetector (no smoothing, rules'
    required joint roles), returns the number of rep_complete events."""
    required_roles = rules_module.required_joint_roles()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    rep_count = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        landmarks = get_pose_landmarks(frame)
        event, _payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, EXERCISE, None, t, required_roles,
        )
        if event == "rep_complete":
            rep_count += 1
    cap.release()
    return rep_count


def part_a_rep_detection(manifest: pd.DataFrame, RepDetectorClass, process_frame) -> pd.DataFrame:
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
        detected = run_repdetector_on_recording(video_path, RepDetectorClass, process_frame)
        rows.append({
            "subject": rec["subject"], "video_id": rec["video_id"], "camera": rec["camera"],
            "view": rec["view"], "annotated_reps": annotated, "detected_reps": detected,
            "abs_error": abs(annotated - detected),
        })
        print(f"  {rec['video_id']} ({rec['camera']}, {rec['view']}): annotated={annotated} detected={detected}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (b) angle fidelity
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
    inclusive set) since this is a geometry comparison, not a scoring run.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    angles = []
    for _ in range(end_frame - start_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = get_pose_landmarks(frame)
        features = extract_features(landmarks, EXERCISE) if landmarks is not None else None
        angles.append(features["angle_deg"] if features else None)
    cap.release()
    return angles


def part_b_angle_fidelity(manifest: pd.DataFrame, split: dict) -> pd.DataFrame:
    eval_manifest = manifest[manifest["subject"].isin(split["evaluation_subjects"])]
    eval_manifest = eval_manifest[eval_manifest["view"].isin(["front", "profile"])]

    rows = []
    for _, rep in eval_manifest.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        joints = load_mocap_joints(rep["video_id"])
        if not video_path.exists() or joints is None:
            continue
        mp_angles = mediapipe_angle_series_for_rep(video_path, rep["start_frame"], rep["end_frame"])
        mocap_angles = [
            mocap_shoulder_trunk_angle(joints[f]) for f in range(rep["start_frame"], rep["end_frame"] + 1)
            if f < len(joints)
        ]
        n = min(len(mp_angles), len(mocap_angles))
        for i in range(n):
            if mp_angles[i] is not None:
                rows.append({
                    "subject": rep["subject"], "view": rep["view"],
                    "mediapipe_deg": mp_angles[i], "mocap_deg": mocap_angles[i],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["error_deg"] = df["mediapipe_deg"] - df["mocap_deg"]
    return df


def summarize_angle_fidelity(df: pd.DataFrame) -> dict:
    summary = {}
    for view in ("front", "profile"):
        sub = df[df["view"] == view]
        if len(sub) < 2:
            summary[view] = None
            continue
        mae = sub["error_deg"].abs().mean()
        bias = sub["error_deg"].mean()
        corr = sub["mediapipe_deg"].corr(sub["mocap_deg"])
        summary[view] = {"n": len(sub), "mae_deg": mae, "bias_deg": bias, "correlation": corr}
    return summary


# ---------------------------------------------------------------------------
# (c)/(d) scorer / classifier vs labels
# ---------------------------------------------------------------------------

def rep_feature_sequences(video_path: Path, start_frame: int, end_frame: int):
    """Per-rep angle_deg/elbow_angle_deg/trunk_lean_deg + valid_fraction,
    confidence-gated with rules.required_joint_roles() -- same gate as
    live scoring."""
    required_roles = rules_module.required_joint_roles()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # must read before release() -- a released capture returns -1 (truthy!)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    angle, elbow, trunk = [], [], []
    total = 0
    for _ in range(end_frame - start_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        landmarks = get_pose_landmarks(frame)
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
    cap.release()
    valid_fraction = len(angle) / total if total else 0.0
    return {"angle_deg": angle, "elbow_angle_deg": elbow, "trunk_lean_deg": trunk, "valid_fraction": valid_fraction}, fps


def part_c_and_d_rules_and_classifier_vs_labels(
    manifest: pd.DataFrame, split: dict, config: dict, classifier: CorrectnessClassifier | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes rep_feature_sequences() ONCE per evaluation rep and feeds
    it to both the rule scorer and the UI-PRMD classifier, instead of
    running MediaPipe over the same rep segment twice."""
    eval_manifest = manifest[manifest["subject"].isin(split["evaluation_subjects"])]
    rows_c, rows_d = [], []
    for _, rep in eval_manifest.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        if not video_path.exists():
            continue
        seq, fps = rep_feature_sequences(video_path, rep["start_frame"], rep["end_frame"])

        if len(seq["angle_deg"]) == 0:
            status_c, failed_rules = "insufficient_data", []
        else:
            result = rules_module.score_rep(seq, config, fps)
            status_c, failed_rules = result["status"], result["failed_rules"]
        rows_c.append({
            "subject": rep["subject"], "view": rep["view"], "label": rep["label"],
            "status": status_c, "failed_rules": ";".join(failed_rules),
        })

        if classifier is not None:
            if len(seq["angle_deg"]) == 0:
                status_d = "insufficient_data"
            else:
                agg = extract_rep_features(pd.DataFrame({"angle_deg": seq["angle_deg"]}))
                result = classifier.predict(agg)
                status_d = "pass" if result["is_correct"] else "fail"
            rows_d.append({"subject": rep["subject"], "view": rep["view"], "label": rep["label"], "status": status_d})

    return pd.DataFrame(rows_c), pd.DataFrame(rows_d)


def confusion_and_rates(df: pd.DataFrame, view: str) -> dict:
    sub = df[df["view"] == view]
    scorable = sub[sub["status"] != "insufficient_data"]
    insufficient_rate = len(sub) - len(scorable), len(sub)

    confusion = {"correct": {"pass": 0, "fail": 0}, "incorrect": {"pass": 0, "fail": 0}}
    for _, r in scorable.iterrows():
        gt = "correct" if r["label"] == "correct" else "incorrect"
        confusion[gt][r["status"]] += 1

    correct_rows = scorable[scorable["label"] == "correct"]
    incorrect_rows = scorable[scorable["label"] == "incorrect"]
    false_fail_rate = (confusion["correct"]["fail"], len(correct_rows))
    detection_rate = (confusion["incorrect"]["fail"], len(incorrect_rows))

    majority_label = sub["label"].mode().iloc[0] if len(sub) else None
    majority_correct = (sub["label"] == majority_label).sum()
    majority_baseline = (majority_correct, len(sub))

    return {
        "confusion": confusion, "false_fail_rate": false_fail_rate, "detection_rate": detection_rate,
        "insufficient_rate": insufficient_rate, "majority_baseline": majority_baseline,
    }


def calibration_percentiles(manifest: pd.DataFrame, split: dict) -> dict:
    calib_manifest = manifest[
        (manifest["subject"].isin(split["calibration_subjects"])) & (manifest["label"] == "correct")
    ]
    rows = []
    for _, rep in calib_manifest.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        if not video_path.exists():
            continue
        seq, fps = rep_feature_sequences(video_path, rep["start_frame"], rep["end_frame"])
        if len(seq["angle_deg"]) == 0:
            continue
        rows.append({
            "peak_angle": max(seq["angle_deg"]), "elbow_min": min(seq["elbow_angle_deg"]),
            "trunk_max": max(seq["trunk_lean_deg"]), "tempo_s": len(seq["angle_deg"]) / fps,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    return {
        col: {
            "n": len(df), "min": df[col].min(), "p25": df[col].quantile(0.25),
            "median": df[col].median(), "p75": df[col].quantile(0.75), "max": df[col].max(),
        }
        for col in ("peak_angle", "elbow_min", "trunk_max", "tempo_s")
    }


# ---------------------------------------------------------------------------
# (e) FPS
# ---------------------------------------------------------------------------

def part_e_fps(video_path: Path, RepDetectorClass, process_frame) -> float:
    required_roles = rules_module.required_joint_roles()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetectorClass()
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
        landmarks = get_pose_landmarks(frame)
        process_frame(detector, parallel, landmarks, EXERCISE, None, t, required_roles)
        n_frames += 1
    elapsed = time.perf_counter() - start_t
    cap.release()
    return n_frames / elapsed if elapsed > 0 else float("nan")


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    commit, dirty = git_provenance()
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    print(f"Provenance: commit={commit}  dirty={dirty}  landmark_space={landmark_space}")
    if not MANIFEST_PATH.exists() or not SPLIT_PATH.exists():
        print(f"Missing {MANIFEST_PATH} or {SPLIT_PATH} -- run scripts/build_rehab24_manifest.py first.")
        return

    manifest, split = load_manifest_and_split()
    print(f"Manifest: {len(manifest)} rows. Calibration subjects: {split['calibration_subjects']}  "
          f"Evaluation subjects: {split['evaluation_subjects']}")

    RepDetectorClass = RepDetector
    config = rules_module.load_config(EXERCISE)

    print("\n[a] Rep detection...")
    part_a = part_a_rep_detection(manifest, RepDetectorClass, process_frame)

    print("\n[b] Angle fidelity...")
    part_b = part_b_angle_fidelity(manifest, split)

    classifier_path = Path("models/shoulder_abduction.pkl")
    classifier = None
    if classifier_path.exists():
        classifier = CorrectnessClassifier()
        classifier.load(classifier_path)
    else:
        print(f"\n[d] SKIPPED -- no trained classifier at {classifier_path}")

    print("\n[c+d] Rule scorer and UI-PRMD classifier (cross-domain) vs labels...")
    part_c, part_d = part_c_and_d_rules_and_classifier_vs_labels(manifest, split, config, classifier)

    print("\n[e] Measured FPS...")
    sample_video = None
    for _, rec in manifest.drop_duplicates(subset=["video"]).iterrows():
        candidate = DATA_DIR / "videos" / rec["video"]
        if candidate.exists():
            sample_video = candidate
            break
    fps_measured = part_e_fps(sample_video, RepDetectorClass, process_frame) if sample_video else None

    calib_pct = calibration_percentiles(manifest, split)

    # ---- write outputs ----
    if not part_a.empty:
        write_csv_with_provenance(part_a, RESULTS_DIR / "rehab24_rep_detection.csv", commit, dirty, landmark_space)
    if not part_b.empty:
        write_csv_with_provenance(part_b, RESULTS_DIR / "rehab24_angle_fidelity.csv", commit, dirty, landmark_space)
    if not part_c.empty:
        write_csv_with_provenance(part_c, RESULTS_DIR / "rehab24_scorer_vs_labels.csv", commit, dirty, landmark_space)
    if not part_d.empty:
        write_csv_with_provenance(part_d, RESULTS_DIR / "rehab24_classifier_vs_labels.csv", commit, dirty, landmark_space)

    lines = ["# REHAB24-6 benchmark -- shoulder_abduction\n"]
    lines.append(f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n")
    lines.append(f"_landmark_space: `{landmark_space}` (config/pose.json)_\n")
    lines.append(f"Calibration subjects: {split['calibration_subjects']}  "
                  f"Evaluation subjects: {split['evaluation_subjects']}\n")

    lines.append("## (a) Rep detection\n")
    if not part_a.empty:
        exact = (part_a["annotated_reps"] == part_a["detected_reps"]).mean()
        mae = part_a["abs_error"].mean()
        lines.append(f"Exact-match rate: {exact:.1%}  Mean abs error: {mae:.2f} reps  (n={len(part_a)} recordings)\n")
        for view in part_a["view"].unique():
            sub = part_a[part_a["view"] == view]
            lines.append(f"- {view}: exact-match={  (sub['annotated_reps']==sub['detected_reps']).mean():.1%}  "
                          f"MAE={sub['abs_error'].mean():.2f}  n={len(sub)}")
    else:
        lines.append("N/A -- no recordings processed.\n")

    lines.append("\n## (b) Angle fidelity (evaluation subjects, front/profile only)\n")
    if not part_b.empty:
        for view, s in summarize_angle_fidelity(part_b).items():
            lines.append(f"- {view}: {s}" if s else f"- {view}: N/A")
    else:
        lines.append("N/A -- no angle-fidelity data.\n")

    for label, df in (("(c) rule scorer", part_c), ("(d) UI-PRMD classifier (cross-domain)", part_d)):
        lines.append(f"\n## {label} vs labels (evaluation subjects)\n")
        if df.empty:
            lines.append("N/A\n")
            continue
        for view in ("front", "profile"):
            if view not in df["view"].unique():
                continue
            r = confusion_and_rates(df, view)
            lines.append(f"- {view}: confusion={r['confusion']}  "
                          f"false_fail={r['false_fail_rate'][0]}/{r['false_fail_rate'][1]}  "
                          f"detection={r['detection_rate'][0]}/{r['detection_rate'][1]}  "
                          f"insufficient_data={r['insufficient_rate'][0]}/{r['insufficient_rate'][1]}  "
                          f"majority_baseline={r['majority_baseline'][0]}/{r['majority_baseline'][1]}")

    lines.append("\n## Calibration-subject percentiles (correct reps)\n")
    lines.append(json.dumps(calib_pct, indent=2, default=float) if calib_pct else "N/A")

    lines.append(f"\n## (e) Measured FPS\n")
    lines.append(f"{fps_measured:.1f} fps on {sample_video}" if fps_measured else "N/A")

    (RESULTS_DIR / "rehab24_benchmark.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'rehab24_benchmark.md'}")


if __name__ == "__main__":
    main()
