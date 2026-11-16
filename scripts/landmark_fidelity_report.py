"""F1 of the landmark-fidelity work: how accurately do MediaPipe's landmarks,
as the live app uses them, reproduce joint ANGLES measured by motion capture?
Measurement only -- does not fit, tune or change anything in src/ or config/.

Data: REHAB24-6 Exercise 1 (right-arm abduction), every annotated rep in
data/rehab24_manifest.csv, both cameras/all views, mocap synced 1:1 by frame
(scripts/benchmark_rehab24.py's load_mocap_joints). Landmarks come from the
per-video cache (scripts/landmark_cache.py) in config/pose.json's
landmark_space.

Why angles, not point positions: MediaPipe's 33 landmarks and the mocap
skeleton's 26 joints sit at different anatomical places in different
coordinate frames, so a per-point position error is not well defined. Angles
are what every downstream module consumes.

Angles compared (MediaPipe always on the RIGHT arm, the arm Ex1 uses -- the
live app's select_side() choice is reported separately):
  shoulder  hip-shoulder-elbow (compute_angle, the abduction angle). Mocap
            computed two ways: with the "Hips" root joint (0) -- the
            convention every earlier REHAB24 script used -- and with the
            anatomical right hip (RightUpLeg, 21), which is what MediaPipe's
            hip landmark corresponds to. The gap between the two shows how
            much of the known offset is definitional rather than tracking.
  elbow     shoulder-elbow-wrist.
  trunk     lean of the hip-mid -> shoulder-mid axis from vertical
            (compute_trunk_lean_deg for MediaPipe, Y-down; same geometry
            with +Y up for mocap).

A frame counts for an angle only if MediaPipe detected a pose and every
landmark that angle uses has visibility >= MIN_VISIBILITY (the live gate).

Metrics per (angle, view, subject group): frames, gate pass rate, bias
(mean MediaPipe - mocap), MAE, MAE after removing that group's bias, Pearson
r, slope of MediaPipe on mocap, and jitter (median |frame-to-frame change of
the error| within a rep). Calibration subjects (1, 2, 6) and evaluation
subjects (3, 4, 5, 7, 8, 9) are always reported separately; any later
correction may be fitted on calibration subjects only.

Usage: python scripts/landmark_fidelity_report.py
Writes results/landmark_fidelity.md and results/landmark_fidelity_frames.csv
(results/ is gitignored).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts import landmark_cache
from src import pose_estimation
from src.feature_extraction import LEFT, MIN_VISIBILITY, RIGHT, compute_angle, compute_trunk_lean_deg, select_side

REPORT_PATH = Path("results/landmark_fidelity.md")
FRAMES_PATH = Path("results/landmark_fidelity_frames.csv")

# REHAB24-6 joints_names.txt indices
MOCAP = {"hips_root": 0, "l_shoulder": 7, "r_shoulder": 12, "r_elbow": 13, "r_wrist": 14,
         "l_hip": 16, "r_hip": 21}

ANGLE_ROLES = {  # landmarks each MediaPipe angle needs to pass the gate
    "shoulder": [RIGHT["hip"], RIGHT["shoulder"], RIGHT["elbow"]],
    "elbow": [RIGHT["shoulder"], RIGHT["elbow"], RIGHT["wrist"]],
    "trunk": [LEFT["shoulder"], RIGHT["shoulder"], LEFT["hip"], RIGHT["hip"]],
}
VIEW_ORDER = ["front", "half-profile", "profile"]


def mocap_angles(j: np.ndarray) -> dict:
    shoulder_mid = (j[MOCAP["l_shoulder"]] + j[MOCAP["r_shoulder"]]) / 2
    hip_mid = (j[MOCAP["l_hip"]] + j[MOCAP["r_hip"]]) / 2
    return {
        "mocap_shoulder_root": compute_angle(j[MOCAP["hips_root"]], j[MOCAP["r_shoulder"]], j[MOCAP["r_elbow"]]),
        "mocap_shoulder": compute_angle(j[MOCAP["r_hip"]], j[MOCAP["r_shoulder"]], j[MOCAP["r_elbow"]]),
        "mocap_elbow": compute_angle(j[MOCAP["r_shoulder"]], j[MOCAP["r_elbow"]], j[MOCAP["r_wrist"]]),
        # Mocap is Y-up, so "up" is +Y here (compute_trunk_lean_deg assumes Y-down).
        "mocap_trunk": compute_angle(shoulder_mid, hip_mid, hip_mid + np.array([0.0, 1.0, 0.0])),
    }


def mediapipe_angles(lm) -> dict:
    out = {f"mp_{name}": np.nan for name in ANGLE_ROLES}
    if lm is None:
        return {**out, "detected": False, "live_side": None}
    for name, idxs in ANGLE_ROLES.items():
        if all(lm[i][3] >= MIN_VISIBILITY for i in idxs):
            if name == "shoulder":
                out["mp_shoulder"] = compute_angle(lm[RIGHT["hip"]], lm[RIGHT["shoulder"]], lm[RIGHT["elbow"]])
            elif name == "elbow":
                out["mp_elbow"] = compute_angle(lm[RIGHT["shoulder"]], lm[RIGHT["elbow"]], lm[RIGHT["wrist"]])
            else:
                out["mp_trunk"] = compute_trunk_lean_deg(lm)
    return {**out, "detected": True, "live_side": "RIGHT" if select_side(lm) is RIGHT else "LEFT"}


def fidelity(mp: np.ndarray, mocap: np.ndarray, rep_ids: np.ndarray | None = None) -> dict:
    """Agreement stats over paired frames; NaN pairs are dropped. Jitter is
    the median |change in error| between consecutive frames of the same rep."""
    mp, mocap = np.asarray(mp, dtype=float), np.asarray(mocap, dtype=float)
    ok = ~(np.isnan(mp) | np.isnan(mocap))
    n = int(ok.sum())
    if n < 3:
        return {"n": n, "bias": np.nan, "mae": np.nan, "mae_debiased": np.nan, "r": np.nan,
                "slope": np.nan, "jitter": np.nan}
    err = mp - mocap
    e = err[ok]
    bias = float(e.mean())
    jitter = np.nan
    if rep_ids is not None:
        rep_ids = np.asarray(rep_ids)
        same_rep = rep_ids[1:] == rep_ids[:-1]
        d = np.abs(np.diff(err))[same_rep & ok[1:] & ok[:-1]]
        jitter = float(np.median(d)) if d.size else np.nan
    # r and slope are undefined when either series is constant.
    varies = np.std(mp[ok]) > 0 and np.std(mocap[ok]) > 0
    return {"n": n, "bias": bias, "mae": float(np.abs(e).mean()), "mae_debiased": float(np.abs(e - bias).mean()),
            "r": float(np.corrcoef(mp[ok], mocap[ok])[0, 1]) if varies else np.nan,
            "slope": float(np.polyfit(mocap[ok], mp[ok], 1)[0]) if varies else np.nan,
            "jitter": jitter}


def collect_frames(manifest: pd.DataFrame, split: dict, landmark_space: str) -> pd.DataFrame:
    rows = []
    calib = set(split["calibration_subjects"])
    for rep_idx, rep in manifest.iterrows():
        if rep["mocap_erroneous"]:
            continue
        joints = bm.load_mocap_joints(rep["video_id"])
        video_path = bm.DATA_DIR / "videos" / rep["video"]
        if joints is None or not video_path.exists():
            continue
        cache = landmark_cache.get_or_build(video_path)
        last = min(int(rep["end_frame"]), cache["frame_count"] - 1, len(joints) - 1)
        for f in range(int(rep["start_frame"]), last + 1):
            lm = landmark_cache.landmarks_at(cache, f, landmark_space)
            rows.append({"rep": rep_idx, "subject": int(rep["subject"]), "view": rep["view"],
                         "camera": rep["camera"], "label": rep["label"],
                         "group": "calibration" if int(rep["subject"]) in calib else "evaluation",
                         "frame": f, **mocap_angles(joints[f]), **mediapipe_angles(lm)})
    return pd.DataFrame(rows)


def fmt(v, digits=1):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def summary_rows(df: pd.DataFrame) -> list[dict]:
    out = []
    pairs = [("shoulder", "mocap_shoulder"), ("shoulder (vs Hips root)", "mocap_shoulder_root"),
             ("elbow", "mocap_elbow"), ("trunk", "mocap_trunk")]
    for view in VIEW_ORDER:
        for group in ("calibration", "evaluation"):
            g = df[(df.view == view) & (df.group == group)]
            if g.empty:
                continue
            for label, mocap_col in pairs:
                mp_col = "mp_" + label.split()[0]
                stats = fidelity(g[mp_col], g[mocap_col], g.rep)
                out.append({"view": view, "group": group, "angle": label,
                            "gate_pass": float(g[mp_col].notna().mean()), **stats})
    return out


def main() -> None:
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()
    df = collect_frames(manifest, split, landmark_space)
    df.to_csv(FRAMES_PATH, index=False)

    lines = ["# Landmark fidelity: MediaPipe joint angles vs motion capture (REHAB24-6 Ex1)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}, landmark_space `{landmark_space}`, "
             f"pose_model `{pose_estimation.configured_pose_model()}`, visibility gate {MIN_VISIBILITY}_", "",
             f"{len(df)} annotated rep frames across {df.rep.nunique()} rep/camera spans, "
             f"subjects {sorted(int(x) for x in df.subject.unique())}. Calibration {split['calibration_subjects']} and "
             f"evaluation {split['evaluation_subjects']} are reported separately.", "",
             "Columns: gate = share of frames where the angle passed the confidence gate; bias = mean(MediaPipe "
             "- mocap); MAE; MAE after removing the group's bias; r; slope of MediaPipe on mocap; jitter = median "
             "|frame-to-frame change of the error|. All in degrees except gate, r and slope.", "",
             "| view | group | angle | frames | gate | bias | MAE | MAE debiased | r | slope | jitter |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary_rows(df):
        lines.append(f"| {r['view']} | {r['group']} | {r['angle']} | {r['n']} | {r['gate_pass']:.0%} | "
                     f"{fmt(r['bias'])} | {fmt(r['mae'])} | {fmt(r['mae_debiased'])} | {fmt(r['r'], 3)} | "
                     f"{fmt(r['slope'], 2)} | {fmt(r['jitter'], 2)} |")

    front = df[df.view == "front"]
    lines += ["", "## Front view, per subject", "",
              "| subject | group | shoulder bias | shoulder MAE debiased | shoulder r | elbow bias | elbow r | "
              "trunk bias | trunk r |", "|---|---|---|---|---|---|---|---|---|"]
    for subj, g in front.groupby("subject"):
        s, e, t = (fidelity(g["mp_" + a], g["mocap_" + a], g.rep) for a in ("shoulder", "elbow", "trunk"))
        lines.append(f"| {subj} | {g.group.iloc[0]} | {fmt(s['bias'])} | {fmt(s['mae_debiased'])} | "
                     f"{fmt(s['r'], 3)} | {fmt(e['bias'])} | {fmt(e['r'], 3)} | {fmt(t['bias'])} | {fmt(t['r'], 3)} |")

    detected = df[df.detected]
    lines += ["", "## Side selection", "",
              "Share of detected frames where the live app's select_side() would pick the LEFT arm "
              "(Ex1 is always the right arm):", ""]
    for view in VIEW_ORDER:
        g = detected[detected.view == view]
        if len(g):
            lines.append(f"- {view}: {(g.live_side == 'LEFT').mean():.1%} of {len(g)} frames")
    lines += ["", f"Pose detected on {df.detected.mean():.1%} of all frames."]

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {REPORT_PATH} and {FRAMES_PATH}")


if __name__ == "__main__":
    main()
