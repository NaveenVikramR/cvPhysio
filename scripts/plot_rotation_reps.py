"""Diagnostic plot for shoulder_rotation's weak LOSO accuracy (0.625,
confirmed in models/shoulder_rotation.metadata.json). Plots real
correct-vs-incorrect angle-over-time curves from data/ui_prmd/ side by
side, and prints the actual extract_rep_features() aggregates for each
plotted rep, to see whether the 5 scalar summary stats the classifier
trains on actually capture whatever visual difference exists -- or
whether correct/incorrect reps differ in *shape* (a stall, a double-bump,
an asymmetric rise/fall) that simple aggregates wash out.

Purely diagnostic -- no number here is a verdict, no fix is applied here.
Per claude.md's hard rule, nothing is asserted beyond what this run
actually produces.

Usage: python scripts/plot_rotation_reps.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive environment -- save to file, don't plt.show()
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import discover_rep_files, load_rep_frames_fk
from src.feature_extraction import extract_rep_features

RAW_DIR = "data/ui_prmd"
MOVEMENT = 9  # shoulder_rotation
MAX_REPS_PER_LABEL = 5
OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "rotation_reps.png"


def pick_diverse_reps(parsed_files, is_correct: bool, max_count: int):
    """One rep (episode e01) per subject, up to max_count distinct subjects,
    for the given correctness label -- avoids all picks coming from the
    same one or two subjects.
    """
    picked = []
    seen_subjects = set()
    for parsed in parsed_files:
        if parsed.is_correct != is_correct or parsed.rep_id != "e01":
            continue
        if parsed.subject_id in seen_subjects:
            continue
        seen_subjects.add(parsed.subject_id)
        picked.append(parsed)
        if len(picked) >= max_count:
            break
    return picked


def main() -> None:
    all_files = discover_rep_files(RAW_DIR, MOVEMENT, modality="kinect", source="positions")
    if not all_files:
        raise NotImplementedError(
            f"No movement {MOVEMENT} files found under {RAW_DIR}. "
            "This diagnostic needs the real UI-PRMD data on disk."
        )

    correct_reps = pick_diverse_reps(all_files, is_correct=True, max_count=MAX_REPS_PER_LABEL)
    incorrect_reps = pick_diverse_reps(all_files, is_correct=False, max_count=MAX_REPS_PER_LABEL)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax_correct, ax_incorrect = axes

    print(f"=== shoulder_rotation (movement {MOVEMENT}) diagnostic: "
          f"{len(correct_reps)} correct, {len(incorrect_reps)} incorrect reps ===\n")

    for ax, reps, label, color in [
        (ax_correct, correct_reps, "correct", "tab:blue"),
        (ax_incorrect, incorrect_reps, "incorrect", "tab:red"),
    ]:
        print(f"--- {label} ---")
        for parsed in reps:
            frames = load_rep_frames_fk(parsed)
            agg = extract_rep_features(frames)
            pct = frames["frame_idx"] / max(len(frames) - 1, 1)
            ax.plot(pct, frames["angle_deg"], color=color, alpha=0.7,
                     label=f"{parsed.subject_id}")
            print(f"  {parsed.subject_id} ({len(frames)} frames, side={frames['side'].iloc[0]}): "
                  f"max={agg['angle_max_deg']:.1f} min={agg['angle_min_deg']:.1f} "
                  f"mean={agg['angle_mean_deg']:.1f} range={agg['angle_range_deg']:.1f} "
                  f"midpoint={agg['angle_midpoint_deg']:.1f}")
        ax.set_title(f"{label} (n={len(reps)})")
        ax.set_xlabel("fraction of rep elapsed")
        ax.legend(fontsize=8)
        print()

    ax_correct.set_ylabel("angle_deg (FK-derived shoulder rotation angle)")
    fig.suptitle("shoulder_rotation: correct vs incorrect angle-over-time (real UI-PRMD data)")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
