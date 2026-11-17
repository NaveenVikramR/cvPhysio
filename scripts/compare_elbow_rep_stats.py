"""F2 step 5: which PER-REP elbow measure tracks the real elbow bend?

The elbow_straight rule makes one decision per rep, so this compares per-rep
MediaPipe statistics with each rep's true (mocap) minimum elbow angle, on
REHAB24-6 Ex1 annotated reps (right arm). Earlier comparisons
(scripts/compare_elbow_variants.py, docs/landmark_fidelity.md finding 2)
were frame by frame.

Candidates: 3 angle variants x 4 per-rep statistics.
  variants (scripts/compare_elbow_variants.elbow_angle_variants, live 0.5
  visibility gate on shoulder/elbow/wrist):
    3d_world     the live angle (world landmarks, x, y, z)
    world_xy     world landmarks with z dropped
    image_px_xy  normalized landmarks in pixels, z dropped
  statistics over the rep's gated frames:
    min          the live default (rules.py "min")
    p10          10th percentile
    median       median over the rep
    hold_median  median within 0.3 s of the peak shoulder angle (rules.py
                 "hold_median")
Truth per rep: the minimum mocap elbow angle over the rep.

SELECTION RULE (fixed before any result was seen):
  1. Primary metric: Spearman rho between the candidate and the truth over
     front-view reps of calibration subjects 1, 2 (6 has no front view).
  2. The candidate with the highest rho is chosen only if it beats the
     current measure (3d_world, min) by >= MIN_RHO_GAIN; else none.
  3. Only then are evaluation subjects (3, 4, 5, 7, 8, 9) read, for the
     chosen candidate and the current measure only. The chosen measure is
     recommended for the app only if, on front-view evaluation reps, its rho
     is above the current measure's AND at least MIN_EVAL_RHO.
Secondary (reported, never used to choose): AUC for telling a bent rep
(true minimum < BENT_DEG) from the rest.

Usage: python scripts/compare_elbow_rep_stats.py
Writes results/elbow_rep_stats.md (gitignored). Changes nothing in src/ or config/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts import landmark_cache
from scripts.compare_elbow_variants import VARIANTS, elbow_angle_variants
from scripts.landmark_fidelity_report import mediapipe_angles, mocap_angles

REPORT_PATH = Path("results/elbow_rep_stats.md")
VIEWS = ("front", "half-profile")
STATISTICS = ("min", "p10", "median", "hold_median")
CURRENT = ("3d_world", "min")
HOLD_WINDOW_S = 0.3
MIN_RHO_GAIN = 0.10
MIN_EVAL_RHO = 0.5
BENT_DEG = 145.0
MIN_GATED_FRAMES = 5


def rep_statistic(values: np.ndarray, shoulder: np.ndarray, fps: float, statistic: str) -> float:
    """One per-rep statistic over gated (non-NaN) frames; NaN if fewer than
    MIN_GATED_FRAMES survive. hold_median centres on the peak shoulder angle."""
    values, shoulder = np.asarray(values, dtype=float), np.asarray(shoulder, dtype=float)
    ok = ~np.isnan(values)
    if ok.sum() < MIN_GATED_FRAMES:
        return np.nan
    if statistic == "min":
        return float(values[ok].min())
    if statistic == "p10":
        return float(np.percentile(values[ok], 10))
    if statistic == "median":
        return float(np.median(values[ok]))
    if np.isnan(shoulder).all():
        return np.nan
    peak = int(np.nanargmax(shoulder))
    window = np.abs(np.arange(len(values)) - peak) <= HOLD_WINDOW_S * fps + 1e-9
    held = values[window & ok]
    return float(np.median(held)) if held.size else np.nan


def collect_reps(manifest: pd.DataFrame, subjects: set[int]) -> pd.DataFrame:
    rows = []
    for rep_idx, rep in manifest.iterrows():
        if rep.view not in VIEWS or int(rep.subject) not in subjects or rep.mocap_erroneous:
            continue
        joints = bm.load_mocap_joints(rep.video_id)
        video = bm.DATA_DIR / "videos" / rep.video
        if joints is None or not video.exists():
            continue
        cache = landmark_cache.get_or_build(video)
        last = min(int(rep.end_frame), cache["frame_count"] - 1, len(joints) - 1)
        frames = range(int(rep.start_frame), last + 1)
        series = {v: [] for v in VARIANTS}
        shoulder, truth = [], []
        for f in frames:
            world = landmark_cache.landmarks_at(cache, f, "world")
            norm = landmark_cache.landmarks_at(cache, f, "normalized")
            for v, a in elbow_angle_variants(world, norm, cache["width"], cache["height"]).items():
                series[v].append(a)
            shoulder.append(mediapipe_angles(world)["mp_shoulder"])
            truth.append(mocap_angles(joints[f])["mocap_elbow"])
        row = {"rep": rep_idx, "subject": int(rep.subject), "view": rep.view, "label": rep.label,
               "truth_min": float(np.nanmin(truth))}
        for v in VARIANTS:
            for s in STATISTICS:
                row[f"{v}:{s}"] = rep_statistic(series[v], shoulder, cache["fps"], s)
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    return float(scipy_stats.spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


def rho_ci(rho: float, n: int) -> str:
    """Fisher-z 95% interval for a Spearman rho (approximate)."""
    if np.isnan(rho) or n < 5 or abs(rho) >= 1:
        return "-"
    z, se = np.arctanh(rho), 1.06 / np.sqrt(n - 3)
    return f"[{np.tanh(z - 1.96 * se):.2f}, {np.tanh(z + 1.96 * se):.2f}]"


def bent_auc(score: pd.Series, truth_min: pd.Series) -> float:
    """AUC for a LOWER score meaning a more bent rep; NaN if one class is empty."""
    ok = score.notna()
    bent = truth_min[ok] < BENT_DEG
    if bent.all() or not bent.any():
        return np.nan
    s = score[ok]
    pos, neg = s[bent].to_numpy(), s[~bent].to_numpy()
    return float(np.mean([(p < n_) + 0.5 * (p == n_) for p in pos for n_ in neg]))


def select(reps: pd.DataFrame) -> tuple[tuple | None, pd.DataFrame]:
    """The selection rule (steps 1-2) on front-view calibration reps."""
    front = reps[reps.view == "front"]
    rows = []
    for v in VARIANTS:
        for s in STATISTICS:
            rho, n = spearman(front[f"{v}:{s}"], front.truth_min)
            rows.append({"variant": v, "statistic": s, "rho": rho, "n": n})
    table = pd.DataFrame(rows)
    current = table[(table.variant == CURRENT[0]) & (table.statistic == CURRENT[1])].rho.iloc[0]
    best = table.sort_values("rho", ascending=False).iloc[0]
    chosen = (best.variant, best.statistic) if best.rho - current >= MIN_RHO_GAIN else None
    return chosen, table


def view_table(reps: pd.DataFrame, candidates: list[tuple]) -> list[str]:
    lines = ["| variant : statistic | view | reps | Spearman rho (95% CI) | bent-rep AUC | bent reps |",
             "|---|---|---|---|---|---|"]
    for v, s in candidates:
        for view in VIEWS:
            g = reps[reps.view == view]
            if g.empty:
                continue
            rho, n = spearman(g[f"{v}:{s}"], g.truth_min)
            auc = bent_auc(g[f"{v}:{s}"], g.truth_min)
            lines.append(f"| {v} : {s} | {view} | {n} | {rho:.2f} {rho_ci(rho, n)} | "
                         f"{'-' if np.isnan(auc) else f'{auc:.2f}'} | {int((g.truth_min < BENT_DEG).sum())} |")
    return lines


def main() -> None:
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()
    calib = collect_reps(manifest, set(split["calibration_subjects"]))
    chosen, table = select(calib)
    cands = [(v, s) for v in VARIANTS for s in STATISTICS]

    lines = ["# Per-rep elbow measures vs mocap (REHAB24-6 Ex1, right arm)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}_", "",
             f"Truth: each rep's minimum mocap elbow angle. Bent rep: truth < {BENT_DEG:g} deg. Selection rule: "
             "scripts/compare_elbow_rep_stats.py docstring. Current measure: 3d_world : min.", "",
             f"## Calibration subjects {split['calibration_subjects']}", "", *view_table(calib, cands), "",
             f"**Chosen (front-view calibration rho, gain >= {MIN_RHO_GAIN:g} over current):** "
             f"{'none' if chosen is None else f'{chosen[0]} : {chosen[1]}'}"]
    if chosen is not None:
        evals = collect_reps(manifest, set(split["evaluation_subjects"]))
        lines += ["", f"## Evaluation subjects {split['evaluation_subjects']}, chosen and current only", "",
                  *view_table(evals, [CURRENT, chosen])]
        front = evals[evals.view == "front"]
        rho_cur, _ = spearman(front[f"{CURRENT[0]}:{CURRENT[1]}"], front.truth_min)
        rho_new, _ = spearman(front[f"{chosen[0]}:{chosen[1]}"], front.truth_min)
        ok = rho_new > rho_cur and rho_new >= MIN_EVAL_RHO
        lines += ["", f"**Recommended for the app:** {'yes' if ok else 'no'} (front-view evaluation rho "
                      f"{rho_new:.2f} vs current {rho_cur:.2f}; needs above current and >= {MIN_EVAL_RHO:g})"]
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    print(text)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
