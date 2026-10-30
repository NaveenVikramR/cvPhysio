"""ANALYSIS ONLY -- read-only. Does not change src/, config/, any
pre-existing results/ file, the dataset split, or any benchmark. The
only things this script writes are results/elbow_variants.md (new)
and, if a landmark-cache entry is missing, a new entry under
results/cache/ (gitignored MediaPipe-inference cache, the same
mechanism scripts/benchmark_rehab24.py already uses -- not a change to
a committed file).

Calibration subjects **1 and 2 only** -- NOT scripts.diagnose_rehab24.
CALIBRATION_SUBJECTS (which is [1, 2, 6]; subject 6 has no cam17/front
recording in this dataset anyway, see scripts/diagnose_coords_C.py).
Front camera (cam17), view=="front" only. Evaluation subjects
[3, 4, 5, 7, 8, 9] are never read -- SUBJECTS below is hardcoded to
[1, 2], independent of any split file, so there's no path by which an
evaluation subject could leak in even if the split changed later.

Question: does the right elbow angle (shoulder-elbow-wrist) agree
better with mocap when MediaPipe's depth (z) coordinate is dropped?
Three variants of the SAME angle, same frames, same (shoulder, elbow,
wrist) joint triple, through the SAME src.feature_extraction.
compute_angle() (never modified, never reimplemented):
  (a) 3d_world    -- MediaPipe world/metric landmarks, full (x, y, z)
  (b) world_xy    -- the same world landmarks, z forced to 0
  (c) image_px_xy -- MediaPipe NORMALIZED (image-space) landmarks,
                     x*width and y*height (z dropped entirely, not
                     just zeroed -- image-space z has no metric
                     meaning to begin with)
Mocap reference: the same compute_angle() on mocap's own (shoulder,
elbow, wrist) 3D joints (scripts.diagnose_rehab24.mocap_elbow_angle).

Landmarks come from scripts/landmark_cache.py (cache hit: no MediaPipe
run; cache miss: runs the pose model once per video, exactly the
fallback the task asked for). This script proposes no change to any
live threshold, config, or feature -- evidence only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import landmark_cache
from scripts.benchmark_rehab24 import DATA_DIR, git_provenance, load_manifest_and_split, load_mocap_joints
from scripts.diagnose_rehab24 import mocap_elbow_angle
from src.feature_extraction import RIGHT, compute_angle

RESULTS_DIR = Path("results")
SUBJECTS = [1, 2]  # hardcoded -- deliberately NOT CALIBRATION_SUBJECTS ([1, 2, 6])
CAMERA = "cam17"
VIEW = "front"
VARIANTS = ["3d_world", "world_xy", "image_px_xy"]
STRAIGHT_ARM_MOCAP_DEG = 160.0
VISIBILITY_MIN = 0.5  # same confidence-gate threshold used live (CLAUDE.md hard rule)


def elbow_angle_variants(world_frame, normalized_frame, width: int, height: int) -> dict:
    """One frame's full 33-landmark world/normalized lists (or None if
    not detected) -> {"3d_world": .., "world_xy": .., "image_px_xy": ..},
    all three on the SAME right-side (shoulder, elbow, wrist) triple,
    NaN for any variant whose source set is missing or whose
    shoulder/elbow/wrist visibility is below VISIBILITY_MIN."""
    out = {v: np.nan for v in VARIANTS}
    if world_frame is None:
        return out
    vis_ok = all(world_frame[RIGHT[role]][3] >= VISIBILITY_MIN for role in ("shoulder", "elbow", "wrist"))
    if not vis_ok:
        return out

    s, e, w = world_frame[RIGHT["shoulder"]], world_frame[RIGHT["elbow"]], world_frame[RIGHT["wrist"]]
    out["3d_world"] = compute_angle(s, e, w)
    s_xy = (s[0], s[1], 0.0, s[3])
    e_xy = (e[0], e[1], 0.0, e[3])
    w_xy = (w[0], w[1], 0.0, w[3])
    out["world_xy"] = compute_angle(s_xy, e_xy, w_xy)

    if normalized_frame is not None:
        ns = normalized_frame[RIGHT["shoulder"]]
        ne = normalized_frame[RIGHT["elbow"]]
        nw = normalized_frame[RIGHT["wrist"]]
        s_px = (ns[0] * width, ns[1] * height, 0.0, ns[3])
        e_px = (ne[0] * width, ne[1] * height, 0.0, ne[3])
        w_px = (nw[0] * width, nw[1] * height, 0.0, nw[3])
        out["image_px_xy"] = compute_angle(s_px, e_px, w_px)
    return out


def per_recording_frame(video_path: Path, video_id: str, subject: int) -> pd.DataFrame:
    """Per-frame mocap elbow angle + all 3 MediaPipe variants, for the
    whole recording (not restricted to annotated rep spans -- same
    pooling convention as scripts/diagnose_coord_space.py's
    frame_level_fit_stats, which pools over the full per-frame series
    and only restricts to rep spans for the separate rest/peak
    sub-analysis)."""
    cache = landmark_cache.get_or_build(video_path, rescore_only=False)
    joints = load_mocap_joints(video_id)
    if joints is None:
        return pd.DataFrame()
    n = min(cache["frame_count"], len(joints))
    width, height = cache["width"], cache["height"]

    rows = []
    for frame_idx in range(n):
        world_frame = landmark_cache.landmarks_at(cache, frame_idx, "world")
        normalized_frame = landmark_cache.landmarks_at(cache, frame_idx, "normalized")
        variants = elbow_angle_variants(world_frame, normalized_frame, width, height)
        rows.append({
            "subject": subject, "video_id": video_id, "frame_idx": frame_idx,
            "mocap_elbow": mocap_elbow_angle(joints[frame_idx]),
            **{f"{v}_elbow": variants[v] for v in VARIANTS},
        })
    return pd.DataFrame(rows)


def jitter_std(series: np.ndarray) -> float:
    """std of the frame-to-frame first difference -- np.diff naturally
    propagates NaN across any gap touching a missing/invalid frame, so
    a short detection dropout is never counted as a real jump."""
    diffs = np.diff(series)
    valid = ~np.isnan(diffs)
    return float(np.nanstd(diffs[valid])) if valid.any() else float("nan")


def variant_stats(mocap: np.ndarray, variant_arr: np.ndarray) -> dict:
    valid = ~np.isnan(mocap) & ~np.isnan(variant_arr)
    n = int(valid.sum())
    if n < 10:
        return {"n": n}
    mo, va = mocap[valid], variant_arr[valid]
    diff = va - mo
    fit = scipy_stats.linregress(mo, va)
    return {
        "n": n,
        "mae": round(float(np.mean(np.abs(diff))), 2),
        "bias": round(float(np.mean(diff)), 2),
        "slope": round(float(fit.slope), 3),
        "r": round(float(fit.rvalue), 3),
        "error_std": round(float(np.std(diff)), 2),
    }


def build_table(all_df: pd.DataFrame) -> tuple[list[str], dict]:
    mocap = all_df["mocap_elbow"].to_numpy()
    straight_mask = mocap > STRAIGHT_ARM_MOCAP_DEG

    lines = ["| variant | n | MAE (deg) | bias (deg, mp-mocap) | slope | r | error_std (deg) | jitter_std (deg/frame) |",
              "|---|---|---|---|---|---|---|---|"]
    all_stats = {}
    for variant in VARIANTS:
        arr = all_df[f"{variant}_elbow"].to_numpy()
        stats = variant_stats(mocap, arr)
        stats["jitter_std"] = round(jitter_std(arr), 3)
        all_stats[variant] = stats
        lines.append(f"| {variant} | {stats.get('n')} | {stats.get('mae')} | {stats.get('bias')} | "
                      f"{stats.get('slope')} | {stats.get('r')} | {stats.get('error_std')} | {stats['jitter_std']} |")

    lines.append("\n### Straight-arm subset only (mocap elbow > 160°)\n")
    lines.append("| variant | n | MAE (deg) | bias (deg, mp-mocap) | slope | r | error_std (deg) | jitter_std (deg/frame, same subset) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    straight_stats = {}
    for variant in VARIANTS:
        arr = all_df[f"{variant}_elbow"].to_numpy()
        stats = variant_stats(mocap[straight_mask], arr[straight_mask])
        stats["jitter_std"] = round(jitter_std(arr[straight_mask]), 3)
        straight_stats[variant] = stats
        lines.append(f"| {variant} | {stats.get('n')} | {stats.get('mae')} | {stats.get('bias')} | "
                      f"{stats.get('slope')} | {stats.get('r')} | {stats.get('error_std')} | {stats['jitter_std']} |")
    return lines, {"all": all_stats, "straight_arm": straight_stats, "straight_arm_n_frames": int(straight_mask.sum())}


def main() -> None:
    commit, dirty = git_provenance()
    print(f"Provenance: commit={commit}  dirty={dirty}")

    manifest, _split = load_manifest_and_split()
    recs = manifest[
        manifest["subject"].isin(SUBJECTS) & (manifest["camera"] == CAMERA) & (manifest["view"] == VIEW)
    ].drop_duplicates(subset=["video_id"])[["subject", "video_id", "video"]]
    print(f"Recordings (subjects {SUBJECTS}, camera={CAMERA}, view={VIEW}): "
          f"{recs['video_id'].tolist()}")

    lines = [
        "# Elbow angle: does dropping depth (z) help? (evidence only, no change proposed)\n",
        f"_commit: `{commit}` (working tree {'dirty' if dirty else 'clean'})_\n",
        f"Calibration subjects **{SUBJECTS} only** (not subject 6 -- it has no cam17/front recording "
        f"in this dataset). Camera={CAMERA!r}, view={VIEW!r}. Evaluation subjects [3, 4, 5, 7, 8, 9] "
        "were never read. Read-only analysis: no src/, config/, pre-existing results/, or split-file "
        "change. Proposes no change to any live threshold, config, or feature.\n",
        f"Recordings used: {recs['video_id'].tolist()} (subjects {sorted(recs['subject'].unique().tolist())}).\n",
    ]

    per_recording_dfs = []
    for _, rec in recs.iterrows():
        video_path = DATA_DIR / "videos" / rec["video"]
        if not video_path.exists():
            print(f"  SKIPPED {rec['video_id']}: video file not found at {video_path}")
            continue
        print(f"  {rec['video_id']} (subject {rec['subject']})...")
        df = per_recording_frame(video_path, rec["video_id"], int(rec["subject"]))
        if df.empty:
            print(f"    SKIPPED {rec['video_id']}: no mocap joints")
            continue
        per_recording_dfs.append(df)

    all_df = pd.concat(per_recording_dfs, ignore_index=True)
    print(f"Pooled frames: {len(all_df)}")

    lines.append("## Pooled frame-level comparison vs mocap (MediaPipe variant - mocap)\n")
    table_lines, stats = build_table(all_df)
    lines.extend(table_lines)
    lines.append(f"\nStraight-arm subset: n={stats['straight_arm_n_frames']} frames with mocap elbow > "
                  f"{STRAIGHT_ARM_MOCAP_DEG}°, out of {len(all_df)} pooled frames total.\n")
    lines.append(f"\nFrames excluded by the {VISIBILITY_MIN} visibility gate (shoulder/elbow/wrist, same "
                  "threshold used live) or missing detection are NaN and dropped from every n above (n "
                  "differs slightly between variants only when 'image_px_xy' additionally needs a "
                  "normalized-landmark frame that happens to be absent while world is present, or vice "
                  "versa -- both landmark sets come from the same MediaPipe call per frame, so this is "
                  "rare).\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "elbow_variants.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS_DIR / 'elbow_variants.md'}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
