"""Diagnoses WHY the REHAB24-6 benchmark (scripts/benchmark_rehab24.py)
failed as badly as it did. ANALYSIS ONLY -- no changes to src/, config/,
or thresholds; this script only reads and reports. Uses calibration
subjects [1, 2, 6] exclusively (never evaluation subjects [3,4,5,7,8,9]).

PART 1: why rep detection failed.
PART 2: whether the MediaPipe-vs-mocap angle bias is offset/scale/lag/
        mapping/real error (front view only).
PART 3: whether elbow/trunk feature agreement reflects real movement or
        measurement artifacts, on ANNOTATED correct reps.
PART 4: what "incorrect" means in this dataset, and which features can
        discriminate correct vs incorrect at all (AUC).

Output: results/diag_rehab24.md (with a git commit hash/dirty header)
and PNGs under results/diag/.
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_rehab24 import (
    DATA_DIR,
    MOCAP_ROLE_TO_JOINT_INDEX,
    git_provenance,
    load_manifest_and_split,
    load_mocap_joints,
    load_rep_detector_and_process_frame,
    mediapipe_angle_series_for_rep,
    mocap_shoulder_trunk_angle,
    rep_feature_sequences,
)
from src import rules as rules_module
from src.feature_extraction import compute_angle, compute_elbow_angle
from src.pose_estimation import get_pose_landmarks

CALIBRATION_SUBJECTS = [1, 2, 6]
RESULTS_DIR = Path("results")
DIAG_DIR = RESULTS_DIR / "diag"
EXERCISE = "shoulder_abduction"

# Bilateral mocap joints for trunk-lean (Part 3) -- distinct from
# MOCAP_ROLE_TO_JOINT_INDEX's single-side abduction-angle mapping, since
# trunk lean needs a LEFT+RIGHT average the way feature_extraction.
# compute_trunk_lean_deg() does on MediaPipe's landmarks 11/12 (shoulder)
# and 23/24 (hip). "Shoulder"=Arm (7/12), not Shoulder (6/11), for the
# same reason MOCAP_ROLE_TO_JOINT_INDEX uses RightArm: a bone's name is
# the segment it starts, and Arm starts at the true glenohumeral joint.
# "Hip"=UpLeg (16/21), the true per-side hip joint -- NOT the single
# root "Hips" (0) MOCAP_ROLE_TO_JOINT_INDEX uses for the abduction angle
# (a single torso-reference point is fine for that one-sided angle; a
# bilateral average needs two real, distinct side joints).
MOCAP_LEFT_TRUNK = {"shoulder": 7, "hip": 16}
MOCAP_RIGHT_TRUNK = {"shoulder": 12, "hip": 21}


def mocap_elbow_angle(joints_frame: np.ndarray) -> float:
    shoulder = joints_frame[MOCAP_ROLE_TO_JOINT_INDEX["shoulder"]]
    elbow = joints_frame[MOCAP_ROLE_TO_JOINT_INDEX["elbow"]]
    wrist = joints_frame[MOCAP_ROLE_TO_JOINT_INDEX["wrist"]]
    return compute_angle(shoulder, elbow, wrist)


def mocap_trunk_lean_deg(joints_frame: np.ndarray) -> float:
    """Adaptation of feature_extraction.compute_trunk_lean_deg() for
    mocap's Y-UP world (meters), not MediaPipe's Y-DOWN image space --
    compute_trunk_lean_deg() hardcodes up=(0,-1,0), which would be wrong
    (measuring deviation from DOWN, not up) if reused unchanged on Y-up
    mocap data. Not a change to src/feature_extraction.py -- this calls
    the same, unmodified compute_angle() with the correct up-vector for
    this data's own convention."""
    shoulder_mid = (joints_frame[MOCAP_LEFT_TRUNK["shoulder"]] + joints_frame[MOCAP_RIGHT_TRUNK["shoulder"]]) / 2
    hip_mid = (joints_frame[MOCAP_LEFT_TRUNK["hip"]] + joints_frame[MOCAP_RIGHT_TRUNK["hip"]]) / 2
    up_point = hip_mid + np.array([0.0, 1.0, 0.0])  # +Y is up in mocap's world, unlike MediaPipe's -Y
    return compute_angle(shoulder_mid, hip_mid, up_point)


def calibration_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["subject"].isin(CALIBRATION_SUBJECTS)].copy()


# ===========================================================================
# PART 1 -- rep detection failure
# ===========================================================================

def run_repdetector_tracked(video_path: Path, RepDetectorClass, process_frame,
                             frame_start: int = 0, frame_end: int | None = None) -> dict:
    """Like benchmark_rehab24.run_repdetector_on_recording(), but also
    tracks discard reasons and whether calibration ever completed.
    frame_start/frame_end (inclusive), if given, restrict which frames of
    the video are fed in -- used by the span-restricted re-run."""
    required_roles = rules_module.required_joint_roles()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    n_to_read = (frame_end - frame_start + 1) if frame_end is not None else None

    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    rep_count = 0
    discard_reasons: dict[str, int] = {}
    calibration_completed = False
    frame_idx = frame_start
    n_read = 0
    while True:
        if n_to_read is not None and n_read >= n_to_read:
            break
        ok, frame = cap.read()
        if not ok:
            break
        n_read += 1
        t = frame_idx / fps
        frame_idx += 1
        landmarks = get_pose_landmarks(frame)
        event, _payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, EXERCISE, None, t, required_roles,
        )
        if detector.state != "CALIBRATING":
            calibration_completed = True
        if event == "rep_complete":
            rep_count += 1
        elif event == "rep_discarded":
            reason = _payload["reason_code"]
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
    cap.release()
    return {
        "rep_count": rep_count, "discard_reasons": discard_reasons,
        "calibration_completed": calibration_completed, "fps": fps, "total_frames": total_frames,
    }


def part1_rep_detection(manifest: pd.DataFrame) -> dict:
    RepDetectorClass, process_frame = load_rep_detector_and_process_frame()
    calib = calibration_manifest(manifest)
    recordings = calib.drop_duplicates(subset=["video_id", "camera"])

    rows = []
    all_discard_reasons: dict[str, int] = {}
    for _, rec in recordings.iterrows():
        video_path = DATA_DIR / "videos" / rec["video"]
        if not video_path.exists():
            continue
        rep_rows = calib[(calib["video_id"] == rec["video_id"]) & (calib["camera"] == rec["camera"])]
        annotated = len(rep_rows)
        span_start = int(rep_rows["start_frame"].min())
        span_end = int(rep_rows["end_frame"].max())

        full = run_repdetector_tracked(video_path, RepDetectorClass, process_frame)
        duration_s = full["total_frames"] / full["fps"]
        frames_before = span_start
        frames_after = full["total_frames"] - 1 - span_end
        pct_outside_span = 100 * (frames_before + max(frames_after, 0)) / full["total_frames"] if full["total_frames"] else None

        restricted = run_repdetector_tracked(video_path, RepDetectorClass, process_frame, span_start, span_end)

        for reason, count in full["discard_reasons"].items():
            all_discard_reasons[reason] = all_discard_reasons.get(reason, 0) + count

        rows.append({
            "subject": rec["subject"], "video_id": rec["video_id"], "camera": rec["camera"], "view": rec["view"],
            "annotated_reps": annotated, "detected_reps_full": full["rep_count"],
            "signed_error_full": full["rep_count"] - annotated,
            "detected_reps_span_restricted": restricted["rep_count"],
            "signed_error_span_restricted": restricted["rep_count"] - annotated,
            "duration_s": round(duration_s, 2), "total_frames": full["total_frames"],
            "span_start": span_start, "span_end": span_end,
            "frames_before_span": frames_before, "frames_after_span": frames_after,
            "pct_frames_outside_span": round(pct_outside_span, 1) if pct_outside_span is not None else None,
            "calibration_completed_full": full["calibration_completed"],
            "calibration_completed_restricted": restricted["calibration_completed"],
            "discard_reasons_full": dict(full["discard_reasons"]),
        })
        print(f"  {rec['video_id']} ({rec['camera']}, {rec['view']}): annotated={annotated} "
              f"full={full['rep_count']} span_restricted={restricted['rep_count']} "
              f"calib_ok_full={full['calibration_completed']}")

    df = pd.DataFrame(rows)
    return {"per_recording": df, "discard_histogram": all_discard_reasons}


def plot_angle_with_rep_boundaries(video_path: Path, video_id: str, camera: str,
                                    annotated_reps: pd.DataFrame, out_path: Path) -> dict:
    """Full-recording MediaPipe angle_deg trace (raw, no confidence gate
    applied to the plotted line itself -- None gaps shown as breaks),
    with annotated (ground truth) and RepDetector-detected rep
    boundaries both overlaid, for visual diagnosis."""
    from src.feature_extraction import extract_features

    RepDetectorClass, process_frame = load_rep_detector_and_process_frame()
    required_roles = rules_module.required_joint_roles()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetectorClass()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}

    angle_trace = []
    detected_spans = []  # (start_idx, end_idx) in the trace
    cur_rep_start = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        landmarks = get_pose_landmarks(frame)
        features = extract_features(landmarks, EXERCISE) if landmarks is not None else None
        angle_trace.append(features["angle_deg"] if features else None)

        prev_state = detector.state
        event, payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, EXERCISE, None, t, required_roles,
        )
        if prev_state != "MOVING" and detector.state == "MOVING":
            cur_rep_start = frame_idx
        if event == "rep_complete" and cur_rep_start is not None:
            detected_spans.append((cur_rep_start, frame_idx))
            cur_rep_start = None
        frame_idx += 1
    cap.release()

    fig, ax = plt.subplots(figsize=(12, 4))
    t_axis = np.arange(len(angle_trace)) / fps
    ax.plot(t_axis, [a if a is not None else np.nan for a in angle_trace], color="black", linewidth=0.8, label="MediaPipe angle_deg")
    for i, (_, rep) in enumerate(annotated_reps.iterrows()):
        ax.axvspan(rep["start_frame"] / fps, rep["end_frame"] / fps, color="green", alpha=0.15,
                   label="annotated rep" if i == 0 else None)
    for i, (s, e) in enumerate(detected_spans):
        # BUG (fixed): `color="red", alpha=0.0` zeroes the alpha of the
        # WHOLE patch, including the edge -- the red outline was
        # invisible in every previously-generated PNG. facecolor="none"
        # (no fill) + edgecolor="red" with no blanket alpha gives a
        # fully-opaque outline and no fill, which is what was intended.
        ax.axvspan(s / fps, e / fps, facecolor="none", edgecolor="red", linewidth=1.5,
                   label="RepDetector-detected rep" if i == 0 else None)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("abduction angle (deg)")
    ax.set_title(f"{video_id} ({camera}): annotated vs detected rep boundaries")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"n_detected_spans": len(detected_spans), "n_annotated": len(annotated_reps)}


# ===========================================================================
# PART 2 -- angle bias (front view only)
# ===========================================================================

def front_view_recordings(manifest: pd.DataFrame) -> pd.DataFrame:
    calib = calibration_manifest(manifest)
    return calib[calib["view"] == "front"].drop_duplicates(subset=["video_id", "camera"])


def full_recording_angle_series(video_path: Path, video_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Whole-recording (not per-rep) MediaPipe vs mocap angle series,
    aligned by raw frame index (both already verified 1:1-synced to the
    30fps video). Returns (mediapipe_arr, mocap_arr, fps), NaN where
    MediaPipe's confidence gate failed that frame."""
    from src.feature_extraction import extract_features

    joints = load_mocap_joints(video_id)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mp_vals = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks = get_pose_landmarks(frame)
        features = extract_features(landmarks, EXERCISE) if landmarks is not None else None
        mp_vals.append(features["angle_deg"] if features else np.nan)
        frame_idx += 1
    cap.release()

    n = min(len(mp_vals), len(joints))
    mp_arr = np.array(mp_vals[:n])
    mocap_arr = np.array([mocap_shoulder_trunk_angle(joints[f]) for f in range(n)])
    return mp_arr, mocap_arr, fps


def best_lag_cross_correlation(mp_arr: np.ndarray, mocap_arr: np.ndarray, fps: float, max_lag_frames: int = 30) -> dict:
    """Cross-correlates the two (NaN-filled with the series' own mean so
    scipy.signal.correlate doesn't propagate NaN) over +/- max_lag_frames
    and reports the lag (frames, ms) that maximizes correlation, plus the
    MAE at lag=0 vs at the best lag."""
    valid = ~np.isnan(mp_arr) & ~np.isnan(mocap_arr)
    if valid.sum() < 10:
        return {"lag_frames": None, "lag_ms": None, "mae_at_lag0": None, "mae_at_best_lag": None}

    mp_filled = np.where(np.isnan(mp_arr), np.nanmean(mp_arr), mp_arr)
    mocap_filled = mocap_arr
    mp_centered = mp_filled - mp_filled.mean()
    mocap_centered = mocap_filled - mocap_filled.mean()

    corr = scipy_signal.correlate(mp_centered, mocap_centered, mode="full")
    lags = scipy_signal.correlation_lags(len(mp_centered), len(mocap_centered), mode="full")
    mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    best_idx = np.argmax(corr[mask])
    best_lag = lags[mask][best_idx]

    mae_at_lag0 = np.nanmean(np.abs(mp_arr - mocap_arr))
    if best_lag > 0:
        shifted_mp, shifted_mocap = mp_arr[best_lag:], mocap_arr[:-best_lag] if best_lag else mocap_arr
    elif best_lag < 0:
        shifted_mp, shifted_mocap = mp_arr[:best_lag], mocap_arr[-best_lag:]
    else:
        shifted_mp, shifted_mocap = mp_arr, mocap_arr
    mae_at_best_lag = np.nanmean(np.abs(shifted_mp - shifted_mocap)) if len(shifted_mp) else None

    return {
        "lag_frames": int(best_lag), "lag_ms": round(1000 * best_lag / fps, 1),
        "mae_at_lag0": round(float(mae_at_lag0), 2),
        "mae_at_best_lag": round(float(mae_at_best_lag), 2) if mae_at_best_lag is not None else None,
    }


def build_front_series_cache(manifest: pd.DataFrame) -> dict:
    """Computes full_recording_angle_series() ONCE per front-view
    calibration recording -- every other Part 2 function (lag, linear
    fit, rest/peak bias, Bland-Altman) reuses this instead of re-running
    MediaPipe over the same video again."""
    recordings = front_view_recordings(manifest)
    cache = {}
    for _, rec in recordings.iterrows():
        video_path = DATA_DIR / "videos" / rec["video"]
        if not video_path.exists():
            continue
        mp_arr, mocap_arr, fps = full_recording_angle_series(video_path, rec["video_id"])
        cache[rec["video_id"]] = {
            "mp_arr": mp_arr, "mocap_arr": mocap_arr, "fps": fps,
            "subject": rec["subject"], "camera": rec["camera"],
        }
    return cache


def linear_fit_per_subject(cache: dict) -> list[dict]:
    """mediapipe = a*mocap + b, fit per calibration subject, pooled over
    that subject's front-view recording(s)."""
    results = []
    for subject in CALIBRATION_SUBJECTS:
        mp_all = [v["mp_arr"] for v in cache.values() if v["subject"] == subject]
        mocap_all = [v["mocap_arr"] for v in cache.values() if v["subject"] == subject]
        if not mp_all:
            continue
        mp_cat = np.concatenate(mp_all)
        mocap_cat = np.concatenate(mocap_all)
        valid = ~np.isnan(mp_cat) & ~np.isnan(mocap_cat)
        if valid.sum() < 10:
            continue
        fit = scipy_stats.linregress(mocap_cat[valid], mp_cat[valid])
        residuals = mp_cat[valid] - (fit.slope * mocap_cat[valid] + fit.intercept)
        results.append({
            "subject": subject, "n": int(valid.sum()), "slope": round(fit.slope, 3),
            "intercept": round(fit.intercept, 1), "residual_std": round(float(residuals.std()), 2),
            "r_value": round(fit.rvalue, 3),
        })
    return results


def rest_vs_peak_bias(manifest: pd.DataFrame, cache: dict) -> dict:
    """Bias (mediapipe - mocap) at each recording's own rest frames
    (within 5 deg of the recording's global mocap-angle minimum -- a
    proxy for "arm at rest") vs at each annotated rep's peak frame
    (where mocap angle is maximal within [start_frame, end_frame])."""
    calib = calibration_manifest(manifest)
    rest_diffs, peak_diffs = [], []
    for video_id, entry in cache.items():
        mp_arr, mocap_arr, camera = entry["mp_arr"], entry["mocap_arr"], entry["camera"]
        valid = ~np.isnan(mp_arr)

        rest_threshold = np.nanmin(mocap_arr) + 5.0
        rest_mask = valid & (mocap_arr <= rest_threshold)
        rest_diffs.extend((mp_arr[rest_mask] - mocap_arr[rest_mask]).tolist())

        rep_rows = calib[(calib["video_id"] == video_id) & (calib["camera"] == camera)]
        for _, rep in rep_rows.iterrows():
            s, e = int(rep["start_frame"]), min(int(rep["end_frame"]), len(mocap_arr) - 1)
            if e <= s:
                continue
            seg_mocap = mocap_arr[s:e + 1]
            peak_local = s + int(np.nanargmax(seg_mocap))
            if peak_local < len(mp_arr) and not np.isnan(mp_arr[peak_local]):
                peak_diffs.append(mp_arr[peak_local] - mocap_arr[peak_local])

    return {
        "rest_bias_mean": round(float(np.mean(rest_diffs)), 2) if rest_diffs else None,
        "rest_bias_n": len(rest_diffs),
        "peak_bias_mean": round(float(np.mean(peak_diffs)), 2) if peak_diffs else None,
        "peak_bias_n": len(peak_diffs),
    }


def bland_altman_plot(cache: dict, out_path: Path) -> None:
    mp_all = [v["mp_arr"] for v in cache.values()]
    mocap_all = [v["mocap_arr"] for v in cache.values()]
    mp_cat = np.concatenate(mp_all)
    mocap_cat = np.concatenate(mocap_all)
    valid = ~np.isnan(mp_cat) & ~np.isnan(mocap_cat)
    mp_v, mocap_v = mp_cat[valid], mocap_cat[valid]

    mean_vals = (mp_v + mocap_v) / 2
    diffs = mp_v - mocap_v
    bias = diffs.mean()
    sd = diffs.std()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diffs, s=4, alpha=0.3)
    ax.axhline(bias, color="red", linestyle="-", label=f"bias={bias:.1f}")
    ax.axhline(bias + 1.96 * sd, color="red", linestyle="--", label=f"+1.96 SD={bias + 1.96 * sd:.1f}")
    ax.axhline(bias - 1.96 * sd, color="red", linestyle="--", label=f"-1.96 SD={bias - 1.96 * sd:.1f}")
    ax.set_xlabel("mean of MediaPipe and mocap angle (deg)")
    ax.set_ylabel("MediaPipe - mocap (deg)")
    ax.set_title("Bland-Altman: front-view calibration recordings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def bone_length_stability(video_id: str) -> dict:
    joints = load_mocap_joints(video_id)
    if joints is None:
        return {}

    def dist(a_idx, b_idx):
        d = np.linalg.norm(joints[:, a_idx, :] - joints[:, b_idx, :], axis=1)
        return {"mean": float(d.mean()), "std": float(d.std()), "cv_pct": round(100 * d.std() / d.mean(), 2)}

    return {
        "RightShoulder-RightArm (true bone)": dist(11, 12),
        "RightArm-RightForeArm (true bone, upper arm)": dist(12, 13),
        "RightForeArm-RightHand (true bone, forearm)": dist(13, 14),
        "Hips-RightArm (multi-segment, NOT a single bone)": dist(0, 12),
    }


def alternative_mapping_mae(cache: dict) -> dict:
    """MAE if "shoulder" were RightShoulder(11) instead of RightArm(12)
    -- the other joint literally named "shoulder" in joints_names.txt."""
    errors_current, errors_alt = [], []
    for video_id, entry in cache.items():
        joints = load_mocap_joints(video_id)
        if joints is None:
            continue
        mp_arr = entry["mp_arr"]
        n = min(len(mp_arr), len(joints))
        for f in range(n):
            if np.isnan(mp_arr[f]):
                continue
            hip, elbow = joints[f][MOCAP_ROLE_TO_JOINT_INDEX["hip"]], joints[f][MOCAP_ROLE_TO_JOINT_INDEX["elbow"]]
            current_angle = compute_angle(hip, joints[f][12], elbow)  # RightArm
            alt_angle = compute_angle(hip, joints[f][11], elbow)  # RightShoulder
            errors_current.append(abs(mp_arr[f] - current_angle))
            errors_alt.append(abs(mp_arr[f] - alt_angle))
    return {
        "mae_current_mapping_shoulder_eq_RightArm": round(float(np.mean(errors_current)), 2) if errors_current else None,
        "mae_alternative_mapping_shoulder_eq_RightShoulder": round(float(np.mean(errors_alt)), 2) if errors_alt else None,
    }


# ===========================================================================
# PART 3 -- feature agreement on annotated CORRECT reps
# ===========================================================================

def mocap_feature_values_for_rep(video_id: str, start_frame: int, end_frame: int, fps: float) -> dict | None:
    joints = load_mocap_joints(video_id)
    if joints is None:
        return None
    end = min(end_frame, len(joints) - 1)
    if end <= start_frame:
        return None
    angle_series = [mocap_shoulder_trunk_angle(joints[f]) for f in range(start_frame, end + 1)]
    elbow_series = [mocap_elbow_angle(joints[f]) for f in range(start_frame, end + 1)]
    trunk_series = [mocap_trunk_lean_deg(joints[f]) for f in range(start_frame, end + 1)]
    return {
        "peak_angle": max(angle_series), "elbow_min": min(elbow_series), "trunk_max": max(trunk_series),
        "tempo_s": (end - start_frame + 1) / fps,
    }


def mediapipe_feature_values_for_rep(video_path: Path, start_frame: int, end_frame: int) -> dict | None:
    seq, fps = rep_feature_sequences(video_path, start_frame, end_frame)
    if len(seq["angle_deg"]) == 0:
        return None
    return {
        "peak_angle": max(seq["angle_deg"]), "elbow_min": min(seq["elbow_angle_deg"]),
        "trunk_max": max(seq["trunk_lean_deg"]), "tempo_s": (end_frame - start_frame + 1) / fps,
        "valid_fraction": seq["valid_fraction"],
    }


def compute_all_rep_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """mocap- and MediaPipe-derived peak_angle/elbow_min/trunk_max/tempo_s
    for EVERY calibration rep (correct and incorrect), computed exactly
    once -- Part 3 (correct reps only) and Part 4 (AUC, all reps) both
    read from this instead of each re-running MediaPipe over the same
    reps independently."""
    calib = calibration_manifest(manifest)
    rows = []
    for _, rep in calib.iterrows():
        video_path = DATA_DIR / "videos" / rep["video"]
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        mocap_feat = mocap_feature_values_for_rep(rep["video_id"], int(rep["start_frame"]), int(rep["end_frame"]), fps)
        mp_feat = mediapipe_feature_values_for_rep(video_path, int(rep["start_frame"]), int(rep["end_frame"]))
        if mocap_feat is None or mp_feat is None:
            continue
        rows.append({
            "video_id": rep["video_id"], "camera": rep["camera"], "view": rep["view"], "label": rep["label"],
            "label_correct": 1 if rep["label"] == "correct" else 0,
            "mocap_peak_angle": mocap_feat["peak_angle"], "mocap_elbow_min": mocap_feat["elbow_min"],
            "mocap_trunk_max": mocap_feat["trunk_max"], "mocap_tempo_s": mocap_feat["tempo_s"],
            "mp_peak_angle": mp_feat["peak_angle"], "mp_elbow_min": mp_feat["elbow_min"],
            "mp_trunk_max": mp_feat["trunk_max"], "mp_tempo_s": mp_feat["tempo_s"],
            "mp_valid_fraction": mp_feat["valid_fraction"],
        })
    return pd.DataFrame(rows)


def part3_feature_agreement(all_reps: pd.DataFrame) -> dict:
    correct = all_reps[all_reps["label"] == "correct"].copy()
    correct["peak_angle_diff"] = correct["mp_peak_angle"] - correct["mocap_peak_angle"]
    correct["elbow_min_diff"] = correct["mp_elbow_min"] - correct["mocap_elbow_min"]
    correct["trunk_max_diff"] = correct["mp_trunk_max"] - correct["mocap_trunk_max"]
    paired_df = correct[["video_id", "camera", "view", "peak_angle_diff", "elbow_min_diff",
                          "trunk_max_diff", "mp_valid_fraction"]].rename(
        columns={"mp_valid_fraction": "mediapipe_valid_fraction"})

    def percentiles(df, col):
        if df.empty:
            return None
        return {
            "n": len(df), "min": round(df[col].min(), 1), "p25": round(df[col].quantile(0.25), 1),
            "median": round(df[col].median(), 1), "p75": round(df[col].quantile(0.75), 1), "max": round(df[col].max(), 1),
        }

    mocap_cols = {"peak_angle": "mocap_peak_angle", "elbow_min": "mocap_elbow_min",
                  "trunk_max": "mocap_trunk_max", "tempo_s": "mocap_tempo_s"}
    mp_cols = {"peak_angle": "mp_peak_angle", "elbow_min": "mp_elbow_min",
               "trunk_max": "mp_trunk_max", "tempo_s": "mp_tempo_s"}
    return {
        "mocap_percentiles": {k: percentiles(correct, v) for k, v in mocap_cols.items()},
        "mediapipe_percentiles": {k: percentiles(correct, v) for k, v in mp_cols.items()},
        "paired_diffs": paired_df,
    }


# ===========================================================================
# PART 4 -- what the labels mean + AUC
# ===========================================================================

def part4_auc(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for source, cols in (("mocap", ["mocap_peak_angle", "mocap_elbow_min", "mocap_trunk_max", "mocap_tempo_s"]),
                          ("mediapipe", ["mp_peak_angle", "mp_elbow_min", "mp_trunk_max", "mp_tempo_s"])):
        for col in cols:
            if df[col].nunique() < 2 or df["label_correct"].nunique() < 2:
                auc = None
            else:
                auc = roc_auc_score(df["label_correct"], df[col])
            results.append({"source": source, "feature": col, "n": len(df), "auc": round(auc, 3) if auc is not None else None})
    return pd.DataFrame(results)


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_provenance()
    print(f"Provenance: commit={commit}  dirty={dirty}")

    manifest, _split = load_manifest_and_split()
    lines = ["# REHAB24-6 diagnostic -- why the benchmark failed\n",
             f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
             f"Calibration subjects used throughout: {CALIBRATION_SUBJECTS} (evaluation subjects never touched).\n"]

    # ---- PART 1 ----
    print("\n[PART 1] Rep detection failure...")
    part1 = part1_rep_detection(manifest)
    df1 = part1["per_recording"]
    df1.to_csv(DIAG_DIR / "part1_rep_detection.csv", index=False)

    lines.append("\n## PART 1 -- rep detection failure\n")
    lines.append("| video_id | camera | view | annotated | detected(full) | signed_err(full) | "
                  "detected(span) | signed_err(span) | duration_s | %frames_outside_span | calib_ok(full) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in df1.iterrows():
        lines.append(f"| {r['video_id']} | {r['camera']} | {r['view']} | {r['annotated_reps']} | "
                      f"{r['detected_reps_full']} | {r['signed_error_full']:+d} | "
                      f"{r['detected_reps_span_restricted']} | {r['signed_error_span_restricted']:+d} | "
                      f"{r['duration_s']} | {r['pct_frames_outside_span']} | {r['calibration_completed_full']} |")

    exact_full = (df1["signed_error_full"] == 0).mean()
    exact_span = (df1["signed_error_span_restricted"] == 0).mean()
    lines.append(f"\nExact-match rate, full recording: **{exact_full:.1%}**. "
                  f"Span-restricted (RepDetector fed ONLY the annotated exercise span): **{exact_span:.1%}**.\n")
    lines.append(f"Discard-reason histogram (calibration subjects, full recordings): `{part1['discard_histogram']}`\n")
    lines.append(f"Calibration-never-completed count: {(~df1['calibration_completed_full']).sum()}/{len(df1)} recordings.\n")

    print("  Saving 3 angle-vs-time PNGs with rep boundaries...")
    sample_recs = df1.drop_duplicates(subset=["video_id", "camera"]).head(3)
    calib_full = calibration_manifest(manifest)
    for _, rec in sample_recs.iterrows():
        video_path = DATA_DIR / "videos" / calib_full[
            (calib_full["video_id"] == rec["video_id"]) & (calib_full["camera"] == rec["camera"])
        ].iloc[0]["video"]
        rep_rows = calib_full[(calib_full["video_id"] == rec["video_id"]) & (calib_full["camera"] == rec["camera"])]
        out_path = DIAG_DIR / f"part1_{rec['video_id']}_{rec['camera']}.png"
        plot_angle_with_rep_boundaries(video_path, rec["video_id"], rec["camera"], rep_rows, out_path)
        lines.append(f"![{rec['video_id']} {rec['camera']}](diag/{out_path.name})\n")

    # ---- PART 2 ----
    print("\n[PART 2] Angle bias (front view)...")
    front_recs = front_view_recordings(manifest)
    print("  Building front-view MediaPipe/mocap series cache (one MediaPipe pass per recording)...")
    series_cache = build_front_series_cache(manifest)

    lag_results = []
    for video_id, entry in series_cache.items():
        lag = best_lag_cross_correlation(entry["mp_arr"], entry["mocap_arr"], entry["fps"])
        lag["video_id"] = video_id
        lag_results.append(lag)
        print(f"  {video_id}: lag={lag['lag_frames']} frames ({lag['lag_ms']} ms), "
              f"MAE@lag0={lag['mae_at_lag0']} MAE@best_lag={lag['mae_at_best_lag']}")

    fit_results = linear_fit_per_subject(series_cache)
    rest_peak = rest_vs_peak_bias(manifest, series_cache)
    bland_altman_plot(series_cache, DIAG_DIR / "part2_bland_altman.png")
    bone_stability = {video_id: bone_length_stability(video_id) for video_id in series_cache}
    alt_mapping = alternative_mapping_mae(series_cache)

    lines.append("\n## PART 2 -- angle bias (front view, calibration subjects)\n")
    lines.append("### Cross-correlation lag (per recording)\n")
    lines.append("| video_id | lag (frames) | lag (ms) | MAE @ lag=0 | MAE @ best lag |")
    lines.append("|---|---|---|---|---|")
    for lag in lag_results:
        lines.append(f"| {lag['video_id']} | {lag['lag_frames']} | {lag['lag_ms']} | {lag['mae_at_lag0']} | {lag['mae_at_best_lag']} |")

    lines.append("\n### Linear fit mediapipe = a*mocap + b (per subject)\n")
    lines.append("| subject | n | slope a | intercept b | residual std | r |")
    lines.append("|---|---|---|---|---|---|")
    for fit in fit_results:
        lines.append(f"| {fit['subject']} | {fit['n']} | {fit['slope']} | {fit['intercept']} | {fit['residual_std']} | {fit['r_value']} |")

    lines.append(f"\n### Rest vs peak bias\n")
    lines.append(f"Rest frames (within 5 deg of each recording's own mocap-angle minimum): "
                  f"mean(mediapipe-mocap) = {rest_peak['rest_bias_mean']} deg (n={rest_peak['rest_bias_n']}).\n")
    lines.append(f"Peak frames (mocap-angle maximum within each annotated rep): "
                  f"mean(mediapipe-mocap) = {rest_peak['peak_bias_mean']} deg (n={rest_peak['peak_bias_n']}).\n")
    lines.append(f"\n![Bland-Altman](diag/part2_bland_altman.png)\n")

    lines.append("\n### Mocap joint mapping + bone-length stability\n")
    lines.append("joints_names.txt (relevant indices): 0=Hips, 11=RightShoulder, 12=RightArm, 13=RightForeArm, "
                  "14=RightHand. Mapping used for the abduction angle: hip=Hips(0), shoulder=RightArm(12), elbow=RightForeArm(13).\n")
    for video_id, stability in bone_stability.items():
        lines.append(f"\n`{video_id}` bone-length stability (mean, std, CV%):")
        for name, stats in stability.items():
            lines.append(f"- {name}: mean={stats['mean']:.1f}  std={stats['std']:.2f}  CV={stats['cv_pct']}%")
    lines.append(f"\nAlternative mapping (shoulder=RightShoulder(11) instead of RightArm(12)): `{alt_mapping}`\n")

    # ---- PART 3 + 4 shared computation ----
    print("\n[PART 3/4] Computing mocap + MediaPipe features for every calibration rep "
          "(correct and incorrect, annotated boundaries)...")
    all_reps = compute_all_rep_features(manifest)
    all_reps.to_csv(DIAG_DIR / "all_calibration_rep_features.csv", index=False)

    print("\n[PART 3] Feature agreement on annotated correct reps...")
    part3 = part3_feature_agreement(all_reps)
    part3["paired_diffs"].to_csv(DIAG_DIR / "part3_paired_diffs.csv", index=False)

    lines.append("\n## PART 3 -- feature agreement on annotated CORRECT reps\n")
    lines.append("### Mocap-derived percentiles\n")
    lines.append(f"`{part3['mocap_percentiles']}`\n")
    lines.append("### MediaPipe-derived percentiles\n")
    lines.append(f"`{part3['mediapipe_percentiles']}`\n")
    diffs = part3["paired_diffs"]
    if not diffs.empty:
        lines.append("### Per-rep paired differences (mediapipe - mocap)\n")
        for col in ("peak_angle_diff", "elbow_min_diff", "trunk_max_diff"):
            lines.append(f"- {col}: mean={diffs[col].mean():.1f}  std={diffs[col].std():.1f}  "
                          f"median={diffs[col].median():.1f}  n={len(diffs)}")
        lines.append(f"- mediapipe valid_fraction on these reps: mean={diffs['mediapipe_valid_fraction'].mean():.2f}  "
                      f"min={diffs['mediapipe_valid_fraction'].min():.2f}\n")
    lines.append("\nThe earlier calibration-subject percentile table (peak=126 deg, elbow_min=79 deg, "
                  "trunk_max=34 deg, tempo=6.0s) reported in results/rehab24_benchmark.md comes from "
                  "`calibration_percentiles()` in scripts/benchmark_rehab24.py, which calls "
                  "`rep_feature_sequences()` -- confirmed by reading that function: it runs "
                  "`get_pose_landmarks()` on the VIDEO, i.e. it is 100% MediaPipe-derived, not mocap.\n")

    # ---- PART 4 ----
    print("\n[PART 4] Label meaning + AUC...")
    auc_df = part4_auc(all_reps)
    auc_df.to_csv(DIAG_DIR / "part4_auc.csv", index=False)

    lines.append("\n## PART 4 -- what the labels mean + feature AUC\n")
    lines.append("### Dataset's own definition (quoted verbatim)\n")
    lines.append("Segmentation.txt: \"correctness -- indication whether given exercise repetition was "
                  "performed correctly (1) or not (0)\" -- that is the ENTIRE definition in that file; "
                  "no further criteria are given there.\n")
    lines.append("Zenodo record description: \"Ex1 = Arm abduction: sideway raising of the straightened "
                  "right arm\" (the exercise target) and \"A physiotherapist instructed the subjects on "
                  "how to perform the exercises so that at least five repetitions were done in what he "
                  "deemed the correct way and five more incorrectly ... the physiotherapist suggested "
                  "different exercise mistakes for each subject.\"\n")
    lines.append("**No itemized codebook of which mistake each subject performed is released in any "
                  "file inspected (Segmentation.csv/.txt, joints_names.txt, the Zenodo description) -- "
                  "undocumented.** \"Incorrect\" is a holistic physiotherapist judgment, and the dataset "
                  "states outright that the specific mistake type varies per subject.\n")

    lines.append("\n### AUC for separating correct vs incorrect (calibration subjects)\n")
    lines.append("| source | feature | n | AUC |")
    lines.append("|---|---|---|---|")
    for _, r in auc_df.iterrows():
        lines.append(f"| {r['source']} | {r['feature']} | {r['n']} | {r['auc']} |")

    (RESULTS_DIR / "diag_rehab24.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'diag_rehab24.md'}")


if __name__ == "__main__":
    main()
