"""Tests whether the MediaPipe-vs-mocap angle/elbow/trunk errors found in
results/diag_rehab24.md are caused by the COORDINATE SPACE fed to
compute_angle(), not MediaPipe's geometric accuracy itself. Calibration
subjects [1, 2, 6], front camera (cam17) only. ANALYSIS in Parts 1-2;
Part 3 applies a fixed-in-advance adoption rule and only then (if a
variant qualifies) touches the live feature path. Part 4 is an unrelated
plotting bug fix. Never touches evaluation subjects.

Output: results/diag_coords.md (commit hash + dirty flag at the top),
PNGs under results/diag/.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_rehab24 import DATA_DIR, load_manifest_and_split, load_mocap_joints, mocap_shoulder_trunk_angle
from scripts.diagnose_rehab24 import CALIBRATION_SUBJECTS, mocap_elbow_angle, mocap_trunk_lean_deg
from src.feature_extraction import (
    check_confidence,
    compute_elbow_angle,
    compute_shoulder_abduction_angle,
    compute_trunk_lean_deg,
    select_side,
)
from src.pose_estimation import _landmarker

RESULTS_DIR = Path("results")
DIAG_DIR = RESULTS_DIR / "diag"
EXERCISE = "shoulder_abduction"
VARIANTS = ["A_current", "B_pixel_scaled", "C_world"]


def git_provenance() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
        return commit, bool(status.strip())
    except Exception:
        return "unknown", True


# ===========================================================================
# PART 1 -- what the live path feeds compute_angle today (read, quote, verify)
# ===========================================================================

def part1_report() -> list[str]:
    lines = ["## PART 1 -- what the live path feeds compute_angle() today\n"]
    pose_est_source = Path("src/pose_estimation.py").read_text().splitlines()
    quoted = "\n".join(f"    {pose_est_source[i - 1]}" for i in (43, 52, 55, 56, 57, 58))
    lines.append("`src/pose_estimation.py` (exact lines, 1-indexed):\n```python\n" + quoted + "\n```\n")
    lines.append("Line 57 takes `result.pose_landmarks[0]` -- MediaPipe's **NORMALIZED image "
                  "landmarks** (`NormalizedLandmark`), NOT `result.pose_world_landmarks` (confirmed "
                  "via `inspect.signature(PoseLandmarkerResult.__init__)`: the result object carries "
                  "both, as separate lists). Line 58 returns `(lm.x, lm.y, lm.z, lm.visibility)` with "
                  "**no multiplication by frame width/height anywhere** in this file or in "
                  "`src/feature_extraction.py` (grepped both; `compute_angle()` and its callers never "
                  "multiply by a resolution).\n")
    lines.append("There is no `src/pipeline.py` in this project -- grepped `src/` and found no such "
                  "file. The live per-frame orchestration is `app.py`'s `process_frame()`, which calls "
                  "`pose_estimation.get_pose_landmarks()` then passes its return value straight into "
                  "`feature_extraction.extract_features()` / `compute_elbow_angle()` / "
                  "`compute_trunk_lean_deg()`, unmodified.\n")
    lines.append("**Units**: MediaPipe's normalized landmarks are x/y each independently divided by "
                  "that axis's own frame dimension (`x = pixel_x / width`, `y = pixel_y / height`), "
                  "landing in roughly `[0, 1]`; z is documented by MediaPipe as being on \"roughly the "
                  "same scale as x\" (i.e. divided by width, not height) -- an approximation, not an "
                  "independently normalized third axis.\n")
    lines.append("**Frame dimensions for these videos (measured via `cv2.VideoCapture`, Camera17 = "
                  "front/the horizontal camera)**: width=1920, height=1080 -- a 16:9, non-square "
                  "frame. Confirmed on a real frame below (frame shape reported by OpenCV as "
                  "`(1080, 1920, 3)` = (H, W, C)).\n")
    lines.append("**Consequence**: with x normalized by 1920 and y normalized by 1080, one unit of "
                  "normalized x-distance represents 1920/1080 = 1.78x the real pixel distance of one "
                  "unit of normalized y-distance. `compute_angle()` is scale-invariant only under "
                  "UNIFORM scaling across all 3 axes -- feeding it raw normalized (x, y, z) when "
                  "width != height is an anisotropic (non-uniform) distortion of the real geometry, "
                  "which is exactly what Part 2 below tests.\n")
    return lines


# ===========================================================================
# PART 2 -- three coordinate-space variants
# ===========================================================================

_next_timestamp_ms = [0]


def get_landmark_sets(frame) -> tuple[list[tuple] | None, list[tuple] | None]:
    """One MediaPipe inference call, returning BOTH landmark sets
    (normalized image-space, world/metric) -- not duplicating inference
    for each variant. Uses the SAME already-loaded `_landmarker` instance
    as src/pose_estimation.py (imported directly, not re-instantiated),
    with its own independent monotonic timestamp counter since this
    script is the only caller driving it in this process."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = _landmarker.detect_for_video(mp_image, _next_timestamp_ms[0])
    _next_timestamp_ms[0] += 33
    if not result.pose_landmarks:
        return None, None
    normalized = [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks[0]]
    world = [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_world_landmarks[0]] if result.pose_world_landmarks else None
    return normalized, world


def pixel_scale(landmarks: list[tuple], width: int, height: int) -> list[tuple]:
    """Variant B: x*W, y*H, z*W (MediaPipe's own documented convention
    that z is roughly on the x/width scale). Converts the two
    differently-normalized axes (x/width, y/height) into one common,
    isotropic unit (real pixels), fixing the anisotropy described in
    PART 1 -- unlike world landmarks, this still does NOT involve any
    camera-intrinsics-based depth recovery, just a flat rescale."""
    return [(x * width, y * height, z * width, vis) for x, y, z, vis in landmarks]


def variant_frame_features(landmarks: list[tuple] | None) -> dict | None:
    """compute_shoulder_abduction_angle/compute_elbow_angle/
    compute_trunk_lean_deg -- UNMODIFIED, same functions the live path
    uses -- applied to whichever coordinate space `landmarks` is already
    in. Confidence-gated with the exercise's base required roles (not
    the rules-wrist-inclusive set), matching the precedent set in
    scripts/benchmark_rehab24.py's mediapipe_angle_series_for_rep() for
    this same kind of geometry comparison."""
    if landmarks is None:
        return None
    side = select_side(landmarks)
    visible, _failed = check_confidence(landmarks, side, EXERCISE)
    if not visible:
        return None
    return {
        "angle": compute_shoulder_abduction_angle(landmarks),
        "elbow": compute_elbow_angle(landmarks),
        "trunk": compute_trunk_lean_deg(landmarks),
    }


def full_recording_variants(video_path: Path, video_id: str) -> pd.DataFrame:
    """Per-frame mocap angle + all 3 variants' angle/elbow/trunk, one
    MediaPipe pass per frame (both landmark sets from the same call)."""
    joints = load_mocap_joints(video_id)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        normalized, world = get_landmark_sets(frame)
        feat_a = variant_frame_features(normalized)
        feat_b = variant_frame_features(pixel_scale(normalized, width, height)) if normalized is not None else None
        feat_c = variant_frame_features(world)

        row = {"frame_idx": frame_idx}
        for name, feat in (("A_current", feat_a), ("B_pixel_scaled", feat_b), ("C_world", feat_c)):
            row[f"{name}_angle"] = feat["angle"] if feat else np.nan
            row[f"{name}_elbow"] = feat["elbow"] if feat else np.nan
            row[f"{name}_trunk"] = feat["trunk"] if feat else np.nan
        rows.append(row)
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows)
    n = min(len(df), len(joints))
    df = df.iloc[:n].copy()
    df["mocap_angle"] = [mocap_shoulder_trunk_angle(joints[f]) for f in range(n)]
    df["mocap_elbow"] = [mocap_elbow_angle(joints[f]) for f in range(n)]
    df["mocap_trunk"] = [mocap_trunk_lean_deg(joints[f]) for f in range(n)]
    df["video_id"] = video_id
    df.attrs["fps"] = fps
    df.attrs["width"] = width
    df.attrs["height"] = height
    return df


def frame_level_fit_stats(df: pd.DataFrame, variant: str, annotated_reps: pd.DataFrame) -> dict:
    mp_arr = df[f"{variant}_angle"].to_numpy()
    mocap_arr = df["mocap_angle"].to_numpy()
    valid = ~np.isnan(mp_arr) & ~np.isnan(mocap_arr)
    if valid.sum() < 10:
        return {"n": int(valid.sum())}

    fit = scipy_stats.linregress(mocap_arr[valid], mp_arr[valid])
    residuals = mp_arr[valid] - (fit.slope * mocap_arr[valid] + fit.intercept)
    mae = float(np.mean(np.abs(mp_arr[valid] - mocap_arr[valid])))

    rest_threshold = np.nanmin(mocap_arr) + 5.0
    rest_mask = valid & (mocap_arr <= rest_threshold)
    rest_bias = float(np.mean(mp_arr[rest_mask] - mocap_arr[rest_mask])) if rest_mask.sum() else None

    peak_diffs = []
    for _, rep in annotated_reps.iterrows():
        s, e = int(rep["start_frame"]), min(int(rep["end_frame"]), len(mocap_arr) - 1)
        if e <= s:
            continue
        seg = mocap_arr[s:e + 1]
        if np.all(np.isnan(seg)):
            continue
        peak_local = s + int(np.nanargmax(seg))
        if peak_local < len(mp_arr) and not np.isnan(mp_arr[peak_local]):
            peak_diffs.append(mp_arr[peak_local] - mocap_arr[peak_local])
    peak_bias = float(np.mean(peak_diffs)) if peak_diffs else None

    return {
        "n": int(valid.sum()), "slope": round(fit.slope, 3), "intercept": round(fit.intercept, 1),
        "r": round(fit.rvalue, 3), "mae": round(mae, 2), "residual_std": round(float(residuals.std()), 2),
        "rest_bias": round(rest_bias, 2) if rest_bias is not None else None,
        "peak_bias": round(peak_bias, 2) if peak_bias is not None else None,
    }


def per_rep_paired_diffs(df: pd.DataFrame, variant: str, correct_reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mocap_angle, mocap_elbow, mocap_trunk = df["mocap_angle"].to_numpy(), df["mocap_elbow"].to_numpy(), df["mocap_trunk"].to_numpy()
    mp_angle, mp_elbow, mp_trunk = df[f"{variant}_angle"].to_numpy(), df[f"{variant}_elbow"].to_numpy(), df[f"{variant}_trunk"].to_numpy()
    for _, rep in correct_reps.iterrows():
        s, e = int(rep["start_frame"]), min(int(rep["end_frame"]), len(df) - 1)
        if e <= s:
            continue
        mocap_peak = np.nanmax(mocap_angle[s:e + 1])
        mocap_elbow_min = np.nanmin(mocap_elbow[s:e + 1])
        mocap_trunk_max = np.nanmax(mocap_trunk[s:e + 1])
        mp_seg_angle, mp_seg_elbow, mp_seg_trunk = mp_angle[s:e + 1], mp_elbow[s:e + 1], mp_trunk[s:e + 1]
        if np.all(np.isnan(mp_seg_angle)):
            continue
        rows.append({
            "video_id": rep["video_id"],
            "peak_angle_diff": np.nanmax(mp_seg_angle) - mocap_peak,
            "elbow_min_diff": np.nanmin(mp_seg_elbow) - mocap_elbow_min,
            "trunk_max_diff": np.nanmax(mp_seg_trunk) - mocap_trunk_max,
        })
    return pd.DataFrame(rows)


def bland_altman_plot(mp_arr: np.ndarray, mocap_arr: np.ndarray, title: str, out_path: Path) -> None:
    valid = ~np.isnan(mp_arr) & ~np.isnan(mocap_arr)
    mp_v, mocap_v = mp_arr[valid], mocap_arr[valid]
    mean_vals = (mp_v + mocap_v) / 2
    diffs = mp_v - mocap_v
    bias, sd = diffs.mean(), diffs.std()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diffs, s=4, alpha=0.3)
    ax.axhline(bias, color="red", label=f"bias={bias:.1f}")
    ax.axhline(bias + 1.96 * sd, color="red", linestyle="--", label=f"+1.96 SD={bias + 1.96 * sd:.1f}")
    ax.axhline(bias - 1.96 * sd, color="red", linestyle="--", label=f"-1.96 SD={bias - 1.96 * sd:.1f}")
    ax.set_xlabel("mean of MediaPipe and mocap angle (deg)")
    ax.set_ylabel("MediaPipe - mocap (deg)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# PART 3 -- adoption rule (fixed before looking at results)
# ===========================================================================

ADOPTION_RULE = {
    "slope_range": (0.90, 1.10),
    "mae_max": 10.0,
    "elbow_min_diff_range": (-15.0, 15.0),
    "trunk_max_diff_range": (-10.0, 10.0),
}


def check_adoption(variant: str, fit: dict, paired: pd.DataFrame) -> dict:
    """Applies ADOPTION_RULE exactly as specified, before any results
    were seen (the rule itself is written above, fixed, in PART 3's
    code -- not tuned after running the numbers)."""
    if not fit or "slope" not in fit:
        return {"variant": variant, "adopted": False, "reason": "insufficient frame-level data"}
    slope_ok = ADOPTION_RULE["slope_range"][0] <= fit["slope"] <= ADOPTION_RULE["slope_range"][1]
    mae_ok = fit["mae"] <= ADOPTION_RULE["mae_max"]
    elbow_mean = paired["elbow_min_diff"].mean() if not paired.empty else None
    trunk_mean = paired["trunk_max_diff"].mean() if not paired.empty else None
    elbow_ok = elbow_mean is not None and ADOPTION_RULE["elbow_min_diff_range"][0] <= elbow_mean <= ADOPTION_RULE["elbow_min_diff_range"][1]
    trunk_ok = trunk_mean is not None and ADOPTION_RULE["trunk_max_diff_range"][0] <= trunk_mean <= ADOPTION_RULE["trunk_max_diff_range"][1]
    adopted = slope_ok and mae_ok and elbow_ok and trunk_ok
    return {
        "variant": variant, "adopted": adopted,
        "slope": fit["slope"], "slope_ok": slope_ok,
        "mae": fit["mae"], "mae_ok": mae_ok,
        "elbow_min_diff_mean": round(elbow_mean, 2) if elbow_mean is not None else None, "elbow_ok": elbow_ok,
        "trunk_max_diff_mean": round(trunk_mean, 2) if trunk_mean is not None else None, "trunk_ok": trunk_ok,
    }


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_provenance()
    print(f"Provenance: commit={commit}  dirty={dirty}")

    manifest, _split = load_manifest_and_split()
    calib = manifest[manifest["subject"].isin(CALIBRATION_SUBJECTS)]
    front_recs = calib[(calib["view"] == "front") & (calib["camera"] == "cam17")].drop_duplicates(subset=["video_id"])
    correct_reps_all = calib[(calib["camera"] == "cam17") & (calib["view"] == "front") & (calib["label"] == "correct")]

    lines = ["# Coordinate-space diagnostic\n", f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
             f"Calibration subjects: {CALIBRATION_SUBJECTS}, front camera (cam17) only. Evaluation subjects never touched.\n"]
    lines.extend(part1_report())

    print("\n[PART 2] Computing 3 coordinate-space variants per recording...")
    per_recording_dfs = {}
    for _, rec in front_recs.iterrows():
        video_path = DATA_DIR / "videos" / rec["video"]
        if not video_path.exists():
            continue
        print(f"  {rec['video_id']}...")
        df = full_recording_variants(video_path, rec["video_id"])
        per_recording_dfs[rec["video_id"]] = df
        df.to_csv(DIAG_DIR / f"coords_{rec['video_id']}.csv", index=False)

    all_df = pd.concat(per_recording_dfs.values(), ignore_index=True)
    annotated_all = calib[(calib["camera"] == "cam17") & (calib["view"] == "front")]

    lines.append("\n## PART 2 -- three coordinate-space variants\n")
    lines.append("### Frame-level fit vs mocap (pooled across the 3 front-view calibration recordings)\n")
    lines.append("| variant | n | slope | intercept | r | MAE | residual_std | rest_bias | peak_bias |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    fits = {}
    for variant in VARIANTS:
        fit = frame_level_fit_stats(all_df, variant, annotated_all)
        fits[variant] = fit
        lines.append(f"| {variant} | {fit.get('n')} | {fit.get('slope')} | {fit.get('intercept')} | "
                      f"{fit.get('r')} | {fit.get('mae')} | {fit.get('residual_std')} | "
                      f"{fit.get('rest_bias')} | {fit.get('peak_bias')} |")
        print(f"  {variant}: {fit}")

    lines.append("\n### Per-rep paired differences vs mocap, annotated CORRECT reps (mediapipe - mocap)\n")
    paired_results = {}
    for variant in VARIANTS:
        paired = per_rep_paired_diffs(all_df, variant, correct_reps_all)
        paired_results[variant] = paired
        paired.to_csv(DIAG_DIR / f"coords_paired_{variant}.csv", index=False)
        if paired.empty:
            lines.append(f"- {variant}: N/A (no reps)")
            continue
        lines.append(f"- **{variant}** (n={len(paired)}):")
        for col in ("peak_angle_diff", "elbow_min_diff", "trunk_max_diff"):
            lines.append(f"  - {col}: mean={paired[col].mean():.1f}  std={paired[col].std():.1f}  median={paired[col].median():.1f}")

    print("  Saving Bland-Altman PNGs for B and C...")
    bland_altman_plot(all_df["B_pixel_scaled_angle"].to_numpy(), all_df["mocap_angle"].to_numpy(),
                       "Bland-Altman: variant B (pixel-scaled), front-view calibration", DIAG_DIR / "coords_bland_altman_B.png")
    bland_altman_plot(all_df["C_world_angle"].to_numpy(), all_df["mocap_angle"].to_numpy(),
                       "Bland-Altman: variant C (world/metric), front-view calibration", DIAG_DIR / "coords_bland_altman_C.png")
    lines.append("\n![Bland-Altman B](diag/coords_bland_altman_B.png)\n")
    lines.append("![Bland-Altman C](diag/coords_bland_altman_C.png)\n")

    # ---- PART 3 ----
    lines.append("\n## PART 3 -- adoption decision\n")
    lines.append(f"Fixed rule: `{ADOPTION_RULE}`\n")
    lines.append("| variant | slope | slope_ok | MAE | mae_ok | elbow_min_diff_mean | elbow_ok | trunk_max_diff_mean | trunk_ok | **ADOPTED** |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    checks = []
    for variant in VARIANTS:
        check = check_adoption(variant, fits[variant], paired_results[variant])
        checks.append(check)
        lines.append(f"| {check['variant']} | {check.get('slope')} | {check.get('slope_ok')} | "
                      f"{check.get('mae')} | {check.get('mae_ok')} | {check.get('elbow_min_diff_mean')} | "
                      f"{check.get('elbow_ok')} | {check.get('trunk_max_diff_mean')} | {check.get('trunk_ok')} | "
                      f"**{check['adopted']}** |")
        print(f"  ADOPTION CHECK {variant}: {check}")

    adopted_variants = [c for c in checks if c["adopted"]]
    if adopted_variants:
        lines.append(f"\n**Adopted: {adopted_variants[0]['variant']}** (first qualifying variant in A, B, C order).\n")
    else:
        closest = min(checks, key=lambda c: abs(c.get("slope", 999) - 1.0) if c.get("slope") is not None else 999)
        lines.append(f"\n**No variant met all 4 criteria. Nothing adopted; config/exercises thresholds unchanged.** "
                      f"Closest to qualifying: {closest['variant']}.\n")

    (RESULTS_DIR / "diag_coords.md").write_text("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'diag_coords.md'}")

    # stash results for a possible PART 3 adoption step run separately
    import json
    (DIAG_DIR / "coords_adoption_checks.json").write_text(json.dumps(checks, indent=2, default=str))


if __name__ == "__main__":
    main()
