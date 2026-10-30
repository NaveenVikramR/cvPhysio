"""Compares the lite and heavy pose models against REHAB24-6 mocap, on the
front-view reps of CALIBRATION subjects 1 and 2 ONLY. Evaluation subjects' videos
are never opened: the subject filter below is the only route to a video, and
main() refuses to continue if it ever yields another subject.

For each model: shoulder and elbow angle MAE, slope and r against mocap
(mediapipe = slope*mocap + intercept, per frame, visibility-gated), and inference
fps (time inside the pose model only, on this machine's CPU). Each model runs
fresh over the video up to its last rep, so both get the same tracking history.

Descriptive only: it sets no threshold and changes no config.

Needs data/rehab24_6/ and data/rehab24_manifest.csv (scripts/download_rehab24.py
then scripts/build_rehab24_manifest.py) and both models in models/.

Output: results/pose_model_comparison.md (commit hash at top) and stdout.
"""

import sys
import time
from datetime import date
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts.diagnose_rehab24 import mocap_elbow_angle
from src import pose_estimation
from src.feature_extraction import MIN_VISIBILITY, compute_elbow_angle, extract_features, select_side

CALIBRATION_SUBJECTS = (1, 2)
MODELS = ("lite", "heavy")
RESULTS_PATH = Path("results/pose_model_comparison.md")


def calibration_front_reps(manifest: pd.DataFrame) -> pd.DataFrame:
    return manifest[manifest["subject"].isin(CALIBRATION_SUBJECTS) & (manifest["view"] == "front")]


def fidelity(mediapipe_deg, mocap_deg) -> dict:
    """MAE, slope and r of paired angle series. NaN metrics if under 2 frames."""
    mp_values, mocap_values = np.asarray(mediapipe_deg, float), np.asarray(mocap_deg, float)
    n = len(mp_values)
    if n < 2:
        return {"n_frames": n, "mae": float("nan"), "slope": float("nan"), "r": float("nan")}
    fit = scipy_stats.linregress(mocap_values, mp_values)
    return {"n_frames": n, "mae": float(np.abs(mp_values - mocap_values).mean()),
            "slope": float(fit.slope), "r": float(fit.rvalue)}


def world_landmarks(result) -> list | None:
    if not result.pose_world_landmarks:
        return None
    return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_world_landmarks[0]]


def elbow_gate_passes(landmarks: list) -> bool:
    side = select_side(landmarks)
    return all(landmarks[side[role]][3] >= MIN_VISIBILITY for role in ("elbow", "wrist"))


def run_model_on_video(model: str, video_path: Path, reps: pd.DataFrame, joints: np.ndarray) -> tuple[list[dict], int, float]:
    """Paired per-frame rows for this video's annotated rep frames, plus frames
    run and seconds spent inside the pose model."""
    in_rep = {}
    for _, rep in reps.iterrows():
        for frame_idx in range(int(rep["start_frame"]), int(rep["end_frame"]) + 1):
            in_rep[frame_idx] = int(rep["subject"])
    last_frame = max(in_rep)

    landmarker = pose_estimation.create_landmarker(model)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps > 0:
        raise RuntimeError(f"No usable fps for {video_path}")
    rows, inference_s, n_frames, frame_idx = [], 0.0, 0, 0
    try:
        while frame_idx <= last_frame:
            ok, frame = cap.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            start = time.perf_counter()
            result = landmarker.detect_for_video(image, int(frame_idx * 1000 / fps))
            inference_s += time.perf_counter() - start
            n_frames += 1
            landmarks = world_landmarks(result)
            if frame_idx in in_rep and landmarks is not None and frame_idx < len(joints):
                features = extract_features(landmarks, bm.EXERCISE)
                row = {"subject": in_rep[frame_idx], "shoulder_mp": None, "shoulder_mocap": None,
                       "elbow_mp": None, "elbow_mocap": None}
                if features:
                    row["shoulder_mp"] = features["angle_deg"]
                    row["shoulder_mocap"] = bm.mocap_shoulder_trunk_angle(joints[frame_idx])
                if elbow_gate_passes(landmarks):
                    row["elbow_mp"] = compute_elbow_angle(landmarks)
                    row["elbow_mocap"] = mocap_elbow_angle(joints[frame_idx])
                rows.append(row)
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()
    return rows, n_frames, inference_s


def table_rows(model: str, frames: pd.DataFrame, fps: float) -> list[dict]:
    rows = []
    groups = [("1+2 pooled", frames)] + [(str(s), frames[frames["subject"] == s]) for s in CALIBRATION_SUBJECTS]
    for label, group in groups:
        for joint in ("shoulder", "elbow"):
            pairs = group[[f"{joint}_mp", f"{joint}_mocap"]].dropna().astype(float)
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            rows.append({"model": model, "subjects": label, "angle": joint,
                         **fidelity(pairs[f"{joint}_mp"], pairs[f"{joint}_mocap"]), "inference_fps": fps})
    return rows


def format_table(rows: list[dict]) -> list[str]:
    lines = ["| model | subjects | angle | n frames | MAE (deg) | slope | r | inference fps |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['subjects']} | {r['angle']} | {r['n_frames']} | {r['mae']:.2f} | "
                     f"{r['slope']:.3f} | {r['r']:.3f} | {r['inference_fps']:.1f} |")
    return lines


def main() -> int:
    if not bm.MANIFEST_PATH.exists():
        print(f"Missing {bm.MANIFEST_PATH}. Run scripts/download_rehab24.py then scripts/build_rehab24_manifest.py.")
        return 1
    for model in MODELS:
        if not pose_estimation.model_path(model).exists():
            print(f"Missing {pose_estimation.model_path(model)}. Download it from {pose_estimation.model_url(model)}")
            return 1
    if pose_estimation.load_config().get("landmark_space", "world") != "world":
        print("Aborting: this comparison is specified under world landmarks.")
        return 1

    commit, dirty = bm.git_provenance()
    reps = calibration_front_reps(pd.read_csv(bm.MANIFEST_PATH))
    if not set(reps["subject"].unique()) <= set(CALIBRATION_SUBJECTS):
        print("Aborting: the rep selection includes a non-calibration subject.")
        return 1
    if reps.empty:
        print("No front-view reps for calibration subjects 1, 2 in the manifest.")
        return 1

    print(f"Provenance: commit={commit} dirty={dirty}  subjects={sorted(reps['subject'].unique())}  "
          f"reps={len(reps)}  videos={reps['video'].nunique()}")
    all_rows = []
    for model in MODELS:
        frames, n_frames, inference_s = [], 0, 0.0
        for video, video_reps in reps.groupby("video"):
            video_path = bm.DATA_DIR / "videos" / video
            joints = bm.load_mocap_joints(video_reps["video_id"].iloc[0])
            if not video_path.exists() or joints is None:
                print(f"  MISSING data for {video} -- skipped")
                continue
            rows, n, seconds = run_model_on_video(model, video_path, video_reps, joints)
            frames.extend(rows)
            n_frames += n
            inference_s += seconds
            print(f"  {model}: {video}: {n} frames, {n / seconds:.1f} fps")
        if not frames:
            print(f"No frames collected for {model}.")
            return 1
        all_rows.extend(table_rows(model, pd.DataFrame(frames), n_frames / inference_s))

    table = format_table(all_rows)
    print("\n".join(table))
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text("\n".join([
        "# Pose model comparison: lite vs heavy against mocap\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})  date: {date.today().isoformat()}_\n",
        "REHAB24-6 front view, calibration subjects 1 and 2 only. Descriptive only. "
        "Inference fps is this machine's CPU, pose model time only.\n",
        *table,
    ]) + "\n")
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
