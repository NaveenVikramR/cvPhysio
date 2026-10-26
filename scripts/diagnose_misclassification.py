"""Diagnose why correct-looking live reps get classified incorrect with
high confidence. Pure read-only diagnosis -- does NOT modify the model,
thresholds, or training code. All outputs go to scratch/ (gitignored).

Usage: python scripts/diagnose_misclassification.py
"""

import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import FEATURE_COLUMNS, LABEL_COLUMN, CorrectnessClassifier
from src.dataset_loader import (
    MOVEMENT_TO_EXERCISE,
    discover_rep_files,
    load_dataset_for_exercise,
    load_rep_frames_fk,
)
from src.feature_extraction import (
    check_confidence,
    compute_shoulder_abduction_angle,
    compute_shoulder_rotation_angle,
    extract_features,
    extract_rep_features,
    select_side,
)
from src.pose_estimation import get_pose_landmarks

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = ROOT / "scratch"
SCRATCH.mkdir(exist_ok=True)

CLIPS = [
    ("data/recordings/shoulderAbduction1.mp4", "shoulder_abduction"),
    ("data/recordings/ShoulderInternalExternal1.mp4", "shoulder_rotation"),
    ("data/recordings/ShoulderInteralExternal2.mp4", "shoulder_rotation"),
    ("data/recordings/physitrack_standing_rotation.mp4", "shoulder_rotation"),
]


def load_rep_detector():
    """Pull RepDetector straight out of app.py's source -- same technique
    used throughout this project, avoids importing app.py itself (which
    has Streamlit/webcam calls at module scope)."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.index("CALIBRATION_FRAMES = 15")
    end = src.index("\nst.set_page_config")
    block_src = src[start:end]
    namespace = {}
    exec(
        "import statistics\nfrom collections import Counter, deque\n"
        "LANDMARK_LABELS = {11:'left shoulder',12:'right shoulder',13:'left elbow',14:'right elbow',"
        "15:'left wrist',16:'right wrist',23:'left hip',24:'right hip'}\n" + block_src,
        namespace,
    )
    return namespace["RepDetector"]


RepDetector = load_rep_detector()


def class_probabilities(clf: CorrectnessClassifier, features: dict) -> dict:
    """P(incorrect), P(correct) for one feature dict, reading the
    underlying sklearn model directly (read-only introspection, not a
    change to classifier.py)."""
    row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    proba = clf._model.predict_proba(row)[0]
    classes = list(clf._model.classes_)
    return {cls: float(p) for cls, p in zip(classes, proba)}


def process_clip(video_path: str, exercise: str, clf: CorrectnessClassifier):
    """Runs the real live pipeline over one clip. Returns:
    - rep_rows: list of dicts, one per completed rep (features + proba + timing)
    - frame_trace: list of (frame_idx, time_sec, state, angle_deg or None) for every frame
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = RepDetector()
    rep_rows = []
    frame_trace = []
    frame_idx = 0
    rep_start_frame = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        landmarks = get_pose_landmarks(frame)
        visible, failed, features = False, [], None
        if landmarks is not None:
            visible, failed = check_confidence(landmarks, select_side(landmarks), exercise)
            if visible:
                features = extract_features(landmarks, exercise)

        angle_deg = features["angle_deg"] if features else None
        side = features["side"] if features else None

        state_before = detector.state
        if state_before == "RESTING" and visible and angle_deg is not None:
            # about to know if this frame starts a rep; capture after update
            pass
        event, payload = detector.update(visible, angle_deg, side, failed)

        if state_before == "RESTING" and detector.state == "MOVING":
            rep_start_frame = frame_idx

        frame_trace.append((frame_idx, frame_idx / fps, state_before, angle_deg))

        if event == "rep_complete":
            rep_frames = pd.DataFrame({"angle_deg": payload["angle_deg"]})
            agg = extract_rep_features(rep_frames)
            result = clf.predict(agg)
            proba = class_probabilities(clf, agg)
            end_frame = frame_idx
            start_frame = rep_start_frame if rep_start_frame is not None else end_frame - len(payload["angle_deg"])
            rep_rows.append({
                "clip": Path(video_path).name,
                "exercise": exercise,
                "rep_num": len(rep_rows) + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_s": round(start_frame / fps, 2),
                "end_time_s": round(end_frame / fps, 2),
                "duration_s": round((end_frame - start_frame) / fps, 2),
                "n_frames": len(payload["angle_deg"]),
                "side": max(set(payload["side"]), key=payload["side"].count),
                **{col: agg[col] for col in FEATURE_COLUMNS},
                "peak_angle_deg": agg["angle_max_deg"],
                "P_correct": proba.get(True, float("nan")),
                "P_incorrect": proba.get(False, float("nan")),
                "predicted_is_correct": result["is_correct"],
                "confidence": result["confidence"],
            })
            rep_start_frame = None

    cap.release()
    return rep_rows, frame_trace, fps


# ---------------------------------------------------------------------------
# Step 2: training distribution stats
# ---------------------------------------------------------------------------

def training_distribution_table(exercise: str) -> pd.DataFrame:
    dataset = load_dataset_for_exercise("data/ui_prmd", exercise, modality="kinect")
    rows = []
    for label, sub in [("correct", dataset[dataset[LABEL_COLUMN]]), ("incorrect", dataset[~dataset[LABEL_COLUMN]])]:
        for col in FEATURE_COLUMNS:
            vals = sub[col]
            rows.append({
                "exercise": exercise, "label": label, "feature": col,
                "min": vals.min(), "p5": vals.quantile(0.05), "median": vals.median(),
                "p95": vals.quantile(0.95), "max": vals.max(),
            })
    return pd.DataFrame(rows), dataset


# ---------------------------------------------------------------------------
# Step 3: flag out-of-range features + feature importances + ablation
# ---------------------------------------------------------------------------

def flag_rep_against_training(rep_row: dict, dist_table: pd.DataFrame, exercise: str) -> dict:
    flags = {}
    correct_dist = dist_table[(dist_table.exercise == exercise) & (dist_table.label == "correct")].set_index("feature")
    full_dist = dist_table[dist_table.exercise == exercise].groupby("feature").agg(min=("min", "min"), max=("max", "max"))
    for col in FEATURE_COLUMNS:
        val = rep_row[col]
        c = correct_dist.loc[col]
        outside_correct_p5_p95 = not (c["p5"] <= val <= c["p95"])
        f = full_dist.loc[col]
        outside_full_range = not (f["min"] <= val <= f["max"])
        flags[col] = {
            "value": val,
            "outside_correct_5_95": bool(outside_correct_p5_p95),
            "outside_full_training_range": bool(outside_full_range),
        }
    return flags


def feature_importances(clf: CorrectnessClassifier) -> list:
    importances = clf._model.feature_importances_
    return sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])


def ablation_effect(clf: CorrectnessClassifier, rep_row: dict, dist_table: pd.DataFrame, exercise: str) -> dict:
    """For each feature, replace it with the training-correct median (all
    other features held at this rep's real values) and report how
    P(correct) changes -- isolates each feature's marginal effect on this
    specific rep's prediction."""
    correct_dist = dist_table[(dist_table.exercise == exercise) & (dist_table.label == "correct")].set_index("feature")
    baseline = {col: rep_row[col] for col in FEATURE_COLUMNS}
    baseline_proba = class_probabilities(clf, baseline)
    effects = {}
    for col in FEATURE_COLUMNS:
        modified = dict(baseline)
        modified[col] = correct_dist.loc[col, "median"]
        new_proba = class_probabilities(clf, modified)
        effects[col] = {
            "original_value": baseline[col],
            "median_correct_value": correct_dist.loc[col, "median"],
            "P_correct_before": baseline_proba.get(True, float("nan")),
            "P_correct_after": new_proba.get(True, float("nan")),
            "delta": new_proba.get(True, float("nan")) - baseline_proba.get(True, float("nan")),
        }
    return effects


# ---------------------------------------------------------------------------
# Step 4: segmentation plot
# ---------------------------------------------------------------------------

def plot_segmentation(clip_name: str, frame_trace: list, rep_rows: list, fps: float) -> Path:
    times = [t for _, t, _, a in frame_trace if a is not None]
    angles = [a for _, t, _, a in frame_trace if a is not None]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, angles, color="gray", linewidth=1, label="angle_deg (all frames)")
    for r in rep_rows:
        ax.axvspan(r["start_time_s"], r["end_time_s"], alpha=0.25,
                   color="tab:green" if r["predicted_is_correct"] else "tab:red")
        ax.text((r["start_time_s"] + r["end_time_s"]) / 2, max(angles) if angles else 180,
                f"rep{r['rep_num']}", ha="center", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle_deg")
    ax.set_title(f"{clip_name}: angle vs time, detected rep boundaries (green=correct, red=incorrect)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = SCRATCH / f"segmentation_{Path(clip_name).stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Step 5: geometry check
# ---------------------------------------------------------------------------

def training_side_counts(exercise: str) -> dict:
    """How many training reps used left vs right arm -- load_dataset_for_
    exercise doesn't expose 'side' in its output, so re-derive it directly
    from load_rep_frames_fk() per file (same function training itself
    uses, just also reading its 'side' column here)."""
    movement = {v: k for k, v in MOVEMENT_TO_EXERCISE.items()}[exercise]
    counts = {"left": 0, "right": 0}
    for parsed in discover_rep_files("data/ui_prmd", movement, modality="kinect", source="positions"):
        frames = load_rep_frames_fk(parsed)
        counts[frames["side"].iloc[0]] += 1
    return counts


def example_landmark_frames(exercise: str, live_video_path: str):
    """One example frame's landmarks from the LIVE path (real MediaPipe)
    and one from the TRAINING path (FK-reconstructed), printed side by
    side to confirm identical role-index structure."""
    # live
    cap = cv2.VideoCapture(live_video_path)
    live_landmarks = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lm = get_pose_landmarks(frame)
        if lm is not None:
            visible, failed = check_confidence(lm, select_side(lm), exercise)
            if visible:
                live_landmarks = lm
                break
    cap.release()

    # training: reuse dataset_loader's internals directly (read-only call)
    from src.dataset_loader import _fk_frame_to_landmarks, forward_kinematics_frame, select_side_for_rep
    movement = {v: k for k, v in MOVEMENT_TO_EXERCISE.items()}[exercise]
    parsed = discover_rep_files("data/ui_prmd", movement, modality="kinect", source="positions")[0]
    pos_df = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
    from src.dataset_loader import sibling_path
    ang_path = sibling_path(parsed.path, "positions")
    ang_df = pd.read_csv(ang_path, sep=r"\s+|,", engine="python", header=None)
    ang_named = ang_df.copy()
    ang_named.columns = [f"col_{i}" for i in range(ang_named.shape[1])]
    side = select_side_for_rep(ang_named, exercise, "kinect")
    fk_pos = forward_kinematics_frame(pos_df.to_numpy()[0], ang_df.to_numpy()[0])
    train_landmarks = _fk_frame_to_landmarks(fk_pos, side)

    return live_landmarks, train_landmarks


def main():
    classifiers = {}
    for ex in ["shoulder_abduction", "shoulder_rotation"]:
        clf = CorrectnessClassifier()
        clf.load(f"models/{ex}.pkl")
        classifiers[ex] = clf

    # ---- Step 1: per-rep features + probability, saved to CSV ----
    all_rep_rows = []
    all_frame_traces = {}
    all_fps = {}
    for video_path, exercise in CLIPS:
        print(f"\n{'=' * 70}\nProcessing {video_path} ({exercise})\n{'=' * 70}")
        rep_rows, frame_trace, fps = process_clip(video_path, exercise, classifiers[exercise])
        all_rep_rows.extend(rep_rows)
        all_frame_traces[video_path] = frame_trace
        all_fps[video_path] = fps
        for r in rep_rows:
            print(f"  rep {r['rep_num']}: {r['n_frames']}f [{r['start_time_s']}-{r['end_time_s']}s] "
                  f"side={r['side']} peak={r['peak_angle_deg']:.1f} "
                  f"P(correct)={r['P_correct']:.3f} pred={'correct' if r['predicted_is_correct'] else 'incorrect'}")

    rep_df = pd.DataFrame(all_rep_rows)
    csv_path = SCRATCH / "live_reps.csv"
    rep_df.to_csv(csv_path, index=False)
    print(f"\n[Step 1] Saved {len(rep_df)} rep rows to {csv_path}")

    # ---- Step 2: training distribution table ----
    print(f"\n{'=' * 70}\nStep 2: training distribution (percentiles per feature, per class)\n{'=' * 70}")
    dist_tables = {}
    for ex in ["shoulder_abduction", "shoulder_rotation"]:
        table, dataset = training_distribution_table(ex)
        dist_tables[ex] = table
        print(f"\n-- {ex} --")
        print(table.to_string(index=False, float_format=lambda x: f"{x:7.1f}"))

    # ---- Step 3: flag out-of-range + importances + ablation ----
    print(f"\n{'=' * 70}\nStep 3: out-of-range flags per live rep\n{'=' * 70}")
    for r in all_rep_rows:
        flags = flag_rep_against_training(r, dist_tables[r["exercise"]], r["exercise"])
        out_corr = [c for c, f in flags.items() if f["outside_correct_5_95"]]
        out_full = [c for c, f in flags.items() if f["outside_full_training_range"]]
        print(f"\n{r['clip']} rep {r['rep_num']} (pred={'correct' if r['predicted_is_correct'] else 'incorrect'}, P(correct)={r['P_correct']:.3f}):")
        for col, f in flags.items():
            marker = "**OUTSIDE FULL RANGE**" if f["outside_full_training_range"] else ("outside correct 5-95" if f["outside_correct_5_95"] else "within correct 5-95")
            print(f"    {col:20s} = {f['value']:7.1f}   {marker}")

    print(f"\n{'=' * 70}\nStep 3b: feature importances\n{'=' * 70}")
    for ex in ["shoulder_abduction", "shoulder_rotation"]:
        print(f"\n-- {ex} --")
        for col, imp in feature_importances(classifiers[ex]):
            print(f"    {col:20s} {imp:.3f}")

    print(f"\n{'=' * 70}\nStep 3c: per-feature ablation (3 example reps, physitrack professional clip)\n{'=' * 70}")
    physitrack_reps = [r for r in all_rep_rows if "physitrack" in r["clip"]]
    for r in physitrack_reps[:3]:
        print(f"\n-- {r['clip']} rep {r['rep_num']} (P(correct)={r['P_correct']:.3f}) --")
        effects = ablation_effect(classifiers[r["exercise"]], r, dist_tables[r["exercise"]], r["exercise"])
        for col, e in sorted(effects.items(), key=lambda kv: -abs(kv[1]["delta"])):
            print(f"    {col:20s} {e['original_value']:7.1f} -> median {e['median_correct_value']:7.1f}  "
                  f"P(correct) {e['P_correct_before']:.3f} -> {e['P_correct_after']:.3f}  (delta={e['delta']:+.3f})")

    # ---- Step 4: segmentation plots ----
    print(f"\n{'=' * 70}\nStep 4: segmentation (rep boundaries, clipped/merged check)\n{'=' * 70}")
    for video_path, exercise in CLIPS:
        clip_reps = [r for r in all_rep_rows if r["clip"] == Path(video_path).name]
        out_png = plot_segmentation(Path(video_path).name, all_frame_traces[video_path], clip_reps, all_fps[video_path])
        print(f"  {Path(video_path).name}: {len(clip_reps)} reps, plot saved to {out_png}")
        for r in clip_reps:
            print(f"    rep {r['rep_num']}: {r['start_time_s']}s-{r['end_time_s']}s "
                  f"(duration {r['duration_s']}s, {r['n_frames']} frames), peak={r['peak_angle_deg']:.1f}")

    # ---- Step 5: geometry check ----
    print(f"\n{'=' * 70}\nStep 5: geometry check (arm + landmark roles)\n{'=' * 70}")
    for ex in ["shoulder_abduction", "shoulder_rotation"]:
        counts = training_side_counts(ex)
        print(f"\n-- {ex} -- training side counts: {counts}")
        clip_sides = [(r["clip"], r["side"]) for r in all_rep_rows if r["exercise"] == ex]
        for clip, side in clip_sides:
            print(f"    live {clip}: side={side}")

    print()
    for video_path, exercise in [CLIPS[0], CLIPS[3]]:  # one abduction clip, the physitrack rotation clip
        live_lm, train_lm = example_landmark_frames(exercise, video_path)
        print(f"\n-- {exercise}: example landmark frame comparison ({video_path}) --")
        print(f"  LIVE  (real MediaPipe, {video_path}):")
        for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
            print(f"    idx {idx}: {tuple(round(v, 3) for v in live_lm[idx])}")
        print(f"  TRAINING (FK-reconstructed, from data/ui_prmd):")
        for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
            print(f"    idx {idx}: {tuple(round(v, 3) for v in train_lm[idx])}")


if __name__ == "__main__":
    main()
