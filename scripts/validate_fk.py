"""Validates the forward-kinematics feature pipeline against real data
before trusting it for training. Per movement/subject:
  1. FK-derived abduction angle sweep (min/max) over one rep.
  2. Pearson correlation between the FK-derived angle and the legacy
     Angles-file channel, over the same frames (expect high |r|, not
     equal values -- they're different quantities measuring the same
     motion).
  3. Bone length (shoulder-elbow, elbow-wrist) stability across frames
     (mean, std) -- should be roughly constant if FK is correct.
Checks a confirmed right-arm subject (s01) and a confirmed left-arm
subject (s07), both movement 7.

Usage: python scripts/validate_fk.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import (
    SIDE_ROLE_JOINTS,
    discover_rep_files,
    forward_kinematics_frame,
    select_side_for_rep,
    sibling_path,
    JOINT_COLUMNS,
)

RAW_DIR = "data/ui_prmd"


def validate_rep(subject_id: str, movement: int, exercise: str) -> bool:
    found = [p for p in discover_rep_files(RAW_DIR, movement, modality="kinect", source="positions")
             if p.subject_id == subject_id and p.rep_id == "e01" and p.is_correct]
    if not found:
        print(f"  FAIL: no file found for {subject_id} m{movement:02d} e01 correct")
        return False
    parsed = found[0]

    pos_df = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
    ang_path = sibling_path(parsed.path, "positions")
    ang_df = pd.read_csv(ang_path, sep=r"\s+|,", engine="python", header=None)
    ang_df_named = ang_df.copy()
    ang_df_named.columns = [f"col_{i}" for i in range(ang_df_named.shape[1])]

    side = select_side_for_rep(ang_df_named, exercise, "kinect")
    print(f"  {subject_id} m{movement:02d}: {len(pos_df)} frames, side selected = {side}")

    roles = SIDE_ROLE_JOINTS[side]
    fk_angles = []
    shoulder_elbow_len = []
    elbow_wrist_len = []
    for pos_row, ang_row in zip(pos_df.to_numpy(), ang_df.to_numpy()):
        fk = forward_kinematics_frame(pos_row, ang_row)
        shoulder = fk[roles["shoulder"]]
        elbow = fk[roles["elbow"]]
        wrist = fk[roles["wrist"]]
        hip = fk[roles["hip"]]
        shoulder_elbow_len.append(np.linalg.norm(elbow - shoulder))
        elbow_wrist_len.append(np.linalg.norm(wrist - elbow))

        if exercise == "shoulder_abduction":
            from src.feature_extraction import compute_angle
            fk_angles.append(compute_angle(hip, shoulder, elbow))
        else:
            from src.feature_extraction import shoulder_rotation_angle_from_points
            fk_angles.append(shoulder_rotation_angle_from_points(shoulder, elbow, wrist, hip))

    fk_angles = np.array(fk_angles)
    legacy_col = JOINT_COLUMNS["kinect"][exercise][side]
    legacy_vals = ang_df_named[legacy_col].to_numpy()

    print(f"    FK angle sweep: min={fk_angles.min():.1f} deg, max={fk_angles.max():.1f} deg, range={fk_angles.max()-fk_angles.min():.1f} deg")
    r = np.corrcoef(fk_angles, legacy_vals)[0, 1]
    print(f"    Pearson r(FK angle, legacy Angles-file column) = {r:.4f}")
    se_len = np.array(shoulder_elbow_len)
    ew_len = np.array(elbow_wrist_len)
    print(f"    shoulder-elbow bone length: mean={se_len.mean():.2f} mm, std={se_len.std():.2f} mm")
    print(f"    elbow-wrist bone length:   mean={ew_len.mean():.2f} mm, std={ew_len.std():.2f} mm")

    ok = True
    if exercise == "shoulder_abduction" and not (fk_angles.max() - fk_angles.min() >= 50):
        print("    FAIL: abduction angle range too small (<50 deg)")
        ok = False
    if abs(r) < 0.5:
        print("    FAIL: |correlation| too low (<0.5) vs legacy Angles-file channel")
        ok = False
    if se_len.std() > 0.1 * se_len.mean() or ew_len.std() > 0.1 * ew_len.mean():
        print("    FAIL: bone length not stable (std > 10% of mean)")
        ok = False
    return ok


def main() -> None:
    all_ok = True
    print("=== movement 7 (shoulder_abduction), subject s01 (confirmed right-arm) ===")
    all_ok &= validate_rep("s01", 7, "shoulder_abduction")
    print("\n=== movement 7 (shoulder_abduction), subject s07 (confirmed left-arm) ===")
    all_ok &= validate_rep("s07", 7, "shoulder_abduction")
    print("\n=== movement 9 (shoulder_rotation), subject s01 (confirmed right-arm) ===")
    all_ok &= validate_rep("s01", 9, "shoulder_rotation")
    print("\n=== movement 9 (shoulder_rotation), subject s07 (confirmed left-arm) ===")
    all_ok &= validate_rep("s07", 9, "shoulder_rotation")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- DO NOT TRAIN'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
