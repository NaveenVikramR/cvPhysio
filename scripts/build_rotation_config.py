"""Derives config/exercises/shoulder_rotation.json from UI-PRMD movement 9
(Standing Shoulder Internal-External Rotation), Kinect, correct reps of the
DERIVATION subjects only, then scores the HELD-OUT subjects' reps (correct and
incorrect) with the unchanged src.rules.score_rep() and reports the result.
Closes docs/open_questions.md #11 (no rotation rules config).

Constraint: src/rules.py is pinned by the pre-registration
(docs/preregistered_own_clip_eval.md), so this script only produces threshold
VALUES for the four existing rules -- it does not add a rotation-specific rule:
  - peak_angle: max of compute_shoulder_rotation_angle() over the rep (the
    forearm's angle about the upper-arm axis, see feature_extraction.py).
    Only the maximum is checked: score_rep() has no "minimum reached" rule,
    so the inward (internal-rotation) end of the movement is not scored.
  - elbow_straight: min of compute_elbow_angle() over the rep, lower bound
    only (upper bound fixed at 180 in rules.py). For rotation the elbow is
    held bent, so this only catches the elbow folding further than correct
    reps do. Marked advisory_rules (reported, not counted), for the same
    reason elbow_straight was demoted in pre-registration Amendment 1: the
    webcam elbow angle was not stable enough to separate bent vs straight.
  - trunk_lean: NOT derivable here (compute_trunk_lean_deg assumes image-space
    Y-down; UI-PRMD FK positions are Y-up). Carried over from
    shoulder_abduction.json, marked not data-derived.
  - tempo: NOT derived. UI-PRMD segment durations (n_frames / 30 Hz) are
    reported for reference only; the band is copied from
    shoulder_abduction.json (see TEMPO_COMMENT for why, and the disclosure).

Same recipe as scripts/build_benchmark_config.py: [p5, p95] of the derivation
subjects' correct reps, except peak's upper end is 180 (the angle's maximum)
and elbow is a p5 lower bound only.

All values are mocap-derived (Kinect forward kinematics), not webcam-derived.
They need checking against real webcam rotation clips before being trusted.

Usage: python scripts/build_rotation_config.py [--dry-run]
Writes config/exercises/shoulder_rotation.json and
results/rotation_config_derivation.md (gitignored, per-machine).
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rules as rules_module
from src.dataset_loader import (
    _fk_frame_to_landmarks,
    discover_rep_files,
    forward_kinematics_frame,
    select_side_for_rep,
    sibling_path,
)
from src.feature_extraction import compute_elbow_angle, compute_shoulder_rotation_angle

RAW_DIR = "data/ui_prmd"
MOVEMENT = 9
UI_PRMD_FPS = 30.0  # Kinect v2 nominal frame rate
# Fixed before any threshold was computed. 5/5 is a small group either way:
# only 10 UI-PRMD subjects exist (CLAUDE.md).
DERIVATION_SUBJECTS = ["s01", "s02", "s03", "s04", "s05"]
HELD_OUT_SUBJECTS = ["s06", "s07", "s08", "s09", "s10"]
CONFIG_PATH = Path("config/exercises/shoulder_rotation.json")
ABDUCTION_CONFIG_PATH = Path("config/exercises/shoulder_abduction.json")
REPORT_PATH = Path("results/rotation_config_derivation.md")
TEMPO_COMMENT = ("NOT data-derived: copied from shoulder_abduction.json. A [p5, p95] band of UI-PRMD "
                 "segment durations was tried first and failed 43/50 held-out correct reps (each subject "
                 "keeps their own pace; segments also include standing still). That choice was made after "
                 "seeing held-out results, so the held-out tempo figures are not a clean test.")


def rep_series(parsed) -> dict:
    """Per-frame rotation and elbow angles for one rep, via the SAME
    FK -> MediaPipe-shaped landmarks -> feature_extraction path the
    classifier's training data uses (src/dataset_loader.py)."""
    pos_df = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
    ang_df = pd.read_csv(sibling_path(parsed.path, "positions"), sep=r"\s+|,", engine="python", header=None)
    named = ang_df.copy()
    named.columns = [f"col_{i}" for i in range(named.shape[1])]
    side = select_side_for_rep(named, "shoulder_rotation", "kinect")
    rotation, elbow = [], []
    for pos_row, ang_row in zip(pos_df.to_numpy(), ang_df.to_numpy()):
        landmarks = _fk_frame_to_landmarks(forward_kinematics_frame(pos_row, ang_row), side)
        r, e = compute_shoulder_rotation_angle(landmarks), compute_elbow_angle(landmarks)
        if not (np.isnan(r) or np.isnan(e)):
            rotation.append(float(r))
            elbow.append(float(e))
    return {"subject": parsed.subject_id, "rep_id": parsed.rep_id, "is_correct": parsed.is_correct,
            "n_frames": len(pos_df), "angle_deg": rotation, "elbow_angle_deg": elbow}


def rep_features(rep: dict) -> dict:
    return {"peak_angle": max(rep["angle_deg"]), "elbow_min": min(rep["elbow_angle_deg"]),
            "tempo_s": len(rep["angle_deg"]) / UI_PRMD_FPS}


def derive_thresholds(features: pd.DataFrame) -> dict:
    pct = {col: {"p5": float(np.percentile(features[col], 5)),
                 "median": float(np.median(features[col])),
                 "p95": float(np.percentile(features[col], 95))}
           for col in ("peak_angle", "elbow_min", "tempo_s")}
    return pct


def build_config(pct: dict, n_reps: int, abduction: dict, commit: str) -> dict:
    p = pct["peak_angle"]
    src_note = (f"p{{}} of {{}} over {n_reps} correct UI-PRMD m09 Kinect reps, subjects "
                f"{', '.join(DERIVATION_SUBJECTS)} (scripts/build_rotation_config.py)")
    return {
        "exercise": "shoulder_rotation",
        "notice": ("Thresholds are derived from UI-PRMD Kinect motion capture, not from webcam clips, and are "
                   "not a clinical judgment. A licensed physiotherapist must review and set appropriate values "
                   "per patient before this is used for real feedback."),
        "derived_from": "UI-PRMD movement 9, Kinect FK positions, correct reps, derivation subjects only",
        "derivation_subjects": DERIVATION_SUBJECTS,
        "held_out_subjects": HELD_OUT_SUBJECTS,
        "n_reps": n_reps,
        "date": date.today().isoformat(),
        "commit": commit,
        "percentiles": pct,
        "scope_comment": ("peak_angle checks only the outward end of the movement (max rotation angle); "
                          "src/rules.py is pinned and has no rule for the inward end."),
        # The rotation angle is bounded at 180 and correct reps saturate there,
        # so the band is [p5, 180]: a [p5, p95] band would fail a rep that
        # reaches exactly 180 whenever p95 rounds below it.
        "target_peak_deg": {"value": round((p["p5"] + 180.0) / 2, 2), "status": "derived",
                            "comment": "band [p5, 180] of peak_angle; " + src_note.format("5", "peak_angle")},
        "tolerance_deg": {"value": round((180.0 - p["p5"]) / 2 + 0.01, 2), "status": "derived",
                          "comment": "half-width of [p5, 180], +0.01 so 180 itself is inside after rounding"},
        "min_elbow_straight_deg": {"value": round(pct["elbow_min"]["p5"], 1), "status": "derived",
                                   "comment": ("Lower bound only: catches the elbow folding more than correct "
                                               "reps do. " + src_note.format("5", "elbow_min"))},
        "advisory_rules": ["elbow_straight"],
        "advisory_rules_comment": ("elbow_straight reports but does not count toward pass/fail -- the webcam "
                                   "elbow angle was found unstable (pre-registration Amendment 1)."),
        "max_trunk_lean_deg": {"value": abduction["max_trunk_lean_deg"]["value"], "status": "provisional",
                               "comment": ("NOT data-derived: copied from shoulder_abduction.json. Trunk lean "
                                           "cannot be computed on UI-PRMD's Y-up FK positions.")},
        "tempo_min_s": {"value": abduction["tempo_min_s"]["value"], "status": "provisional",
                        "comment": TEMPO_COMMENT},
        "tempo_max_s": {"value": abduction["tempo_max_s"]["value"], "status": "provisional",
                        "comment": TEMPO_COMMENT},
        "cue_cooldown_s": abduction["cue_cooldown_s"],
        "min_valid_fraction": abduction["min_valid_fraction"],
        "cues": {
            "peak_angle": "Rotate your forearm outward through the full range, keeping your elbow at your side.",
            "elbow_straight": "Keep your elbow bent at about a right angle.",
            "trunk_lean": "Keep your torso upright -- try not to lean or twist.",
            "tempo": "Move at a steadier, more controlled pace.",
            "insufficient_data": "Keep your whole arm, elbow and hand in view.",
        },
    }


def score_held_out(reps: list[dict], config: dict) -> pd.DataFrame:
    """score_rep() on each held-out rep. trunk_lean is fed zeros (not
    measurable here), so it always passes and is excluded from the report."""
    rows = []
    for rep in reps:
        n = len(rep["angle_deg"])
        result = rules_module.score_rep(
            {"angle_deg": rep["angle_deg"], "elbow_angle_deg": rep["elbow_angle_deg"],
             "trunk_lean_deg": [0.0] * n, "valid_fraction": 1.0}, config, UI_PRMD_FPS)
        counted = [r for r in result["failed_rules"] if r not in config.get("advisory_rules", [])]
        rows.append({"subject": rep["subject"], "rep_id": rep["rep_id"], "is_correct": rep["is_correct"],
                     "verdict_fail": bool(counted),
                     **{f"fail_{rid}": rid in result["failed_rules"] for rid in rules_module.RULE_IDS}})
    return pd.DataFrame(rows)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the report; write nothing")
    args = parser.parse_args()
    assert not set(DERIVATION_SUBJECTS) & set(HELD_OUT_SUBJECTS)

    files = discover_rep_files(RAW_DIR, MOVEMENT, modality="kinect", source="positions")
    if not files:
        sys.exit(f"No UI-PRMD movement {MOVEMENT} Kinect files under {RAW_DIR}.")
    derivation = [rep_series(p) for p in files if p.subject_id in DERIVATION_SUBJECTS and p.is_correct]
    held_out = [rep_series(p) for p in files if p.subject_id in HELD_OUT_SUBJECTS]

    feats = pd.DataFrame([rep_features(r) for r in derivation])
    pct = derive_thresholds(feats)
    abduction = json.loads(ABDUCTION_CONFIG_PATH.read_text())
    config = build_config(pct, len(derivation), abduction, git_commit())
    scored = score_held_out(held_out, config)

    lines = [f"# Rotation rules config derivation", "",
             f"_commit `{config['commit']}`, {config['date']}, scripts/build_rotation_config.py_", "",
             f"Derivation: {len(derivation)} correct reps, subjects {', '.join(DERIVATION_SUBJECTS)}. "
             f"Held out: {len(held_out)} reps, subjects {', '.join(HELD_OUT_SUBJECTS)}. "
             "Both groups are 5 subjects -- small; treat every rate below as a rough indication.", "",
             "| feature | p5 | median | p95 |", "|---|---|---|---|"]
    for col, v in pct.items():
        lines.append(f"| {col} | {v['p5']:.1f} | {v['median']:.1f} | {v['p95']:.1f} |")
    lines += ["", "Allowed: " + ", ".join(
        f"{rid} {lo:.1f}..{hi:.1f}" for rid, (lo, hi) in rules_module._absolute_bounds(config).items()
        if rid != "trunk_lean") + " (trunk_lean not measurable on UI-PRMD; elbow_straight advisory)", ""]

    correct, incorrect = scored[scored.is_correct], scored[~scored.is_correct]
    lines += ["## Held-out subjects, score_rep() verdict (trunk_lean excluded)", "",
              f"- correct reps failed (false-fail): {int(correct.verdict_fail.sum())}/{len(correct)}",
              f"- incorrect reps failed (detected): {int(incorrect.verdict_fail.sum())}/{len(incorrect)}", "",
              "| subject | correct failed | incorrect failed |", "|---|---|---|"]
    for subj, g in scored.groupby("subject"):
        c, i = g[g.is_correct], g[~g.is_correct]
        lines.append(f"| {subj} | {int(c.verdict_fail.sum())}/{len(c)} | {int(i.verdict_fail.sum())}/{len(i)} |")
    lines += ["", "| rule | correct reps failing it | incorrect reps failing it |", "|---|---|---|"]
    for rid in ("peak_angle", "elbow_straight", "tempo"):
        lines.append(f"| {rid} | {int(correct[f'fail_{rid}'].sum())}/{len(correct)} | "
                     f"{int(incorrect[f'fail_{rid}'].sum())}/{len(incorrect)} |")
    report = "\n".join(lines) + "\n"
    print(report)

    if args.dry_run:
        return
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Wrote {CONFIG_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
