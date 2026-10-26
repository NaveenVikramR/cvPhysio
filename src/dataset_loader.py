"""Offline dataset loading for UI-PRMD. Build-time only -- never imported by
the six runtime modules (pose_estimation, feature_extraction, classifier,
feedback, tracking, app.py). Parses UI-PRMD's "Segmented Movements" release
into per-repetition angle features for src/classifier.py to train on.

VERIFIED against real downloaded files (movements 7 and 9, Kinect, all 10
subjects, correct and incorrect -- see data/ui_prmd/) and against the
dataset author's own MATLAB visualization code ("UI-PRMD Visualize.zip",
file Animation.m -- recovered the same way as the data itself, via a
Wayback Machine capture of the dead official site). What was confirmed vs.
corrected from the original (documentation-only) assumptions:

- Filenames: m{MM}_s{SS}_e{EE}_{positions|angles}.txt for correct reps.
  Incorrect reps add an "_inc" suffix before .txt (e.g.
  "m07_s01_e01_angles_inc.txt") -- CONFIRMED, and is_correct is parsed from
  that suffix, not from a "Correct"/"Incorrect" path component (the real
  download folders are named "Segmented Movements" and "Incorrect
  Segmented Movements", neither of which is literally "correct"/
  "incorrect", so path-based detection never actually worked).
- One file = one already-segmented repetition -- CONFIRMED (checked a
  sample file's angle signal: single rise-fall arc, not oscillating).
- Subject numbering (s01-s10) is consistent across movements -- CONFIRMED.
- A single fixed arm per movement -- WRONG as a global constant. 8 of 10
  subjects use the right arm; subjects 07 and 10 consistently use the left
  arm (checked across both m07 and m09). Side is selected dynamically per
  repetition (see select_side_for_rep()), not hardcoded.
- Modality: VICON IS OUT OF SCOPE for this project. The only complete,
  verified download we obtained was Kinect (the Vicon portion of every
  available mirror was incomplete/truncated). This isn't just an expedient
  fallback: Kinect is also the closer analog to this project's actual
  sensor -- a webcam via MediaPipe -- than Vicon's markered optical mocap
  rig, so training on Kinect's noisier, camera-derived skeleton is
  arguably more representative of what live inference will see anyway.
- Joint order (from the dataset paper's Table 2, and matching Animation.m)
  for Kinect's 22 joints, each occupying 3 consecutive columns (axis 0, 1,
  2) in both Positions and Angles files, 66 columns total, no header row,
  comma-delimited:
    Waist, Spine, Chest, Neck, Head, HeadTip, L_Collar, L_UpperArm,
    L_Forearm, L_Hand, R_Collar, R_UpperArm, R_Forearm, R_Hand,
    L_UpperLeg, L_LowerLeg, L_Foot, L_Toes, R_UpperLeg, R_LowerLeg,
    R_Foot, R_Toes

FEATURE SOURCE -- two options, selectable via `feature_source`:

1. "fk_positions" (default, used for the models app.py would load): angle
   is computed by running the SAME feature_extraction.py functions the
   live app uses (compute_angle, select_side, compute_shoulder_abduction_
   angle, compute_shoulder_rotation_angle) on forward-kinematics-derived
   ABSOLUTE 3D joint positions. This is what fixes the original train/
   inference mismatch described below.

2. "angles_file" (legacy, kept only for comparison -- see
   scripts/train_classifier.py --features flag): angle_deg is read
   directly from one Angles-file column (JOINT_COLUMNS below). This was
   the original approach. It produces a clean, high-range signal, but it
   is NOT the quantity feature_extraction.py computes at inference time --
   see "Positions vs. Angles gap" below for the measured size of that
   mismatch. Kept only so the two approaches can be compared head to head.

FORWARD KINEMATICS (confirms Positions are parent-relative offsets, Waist
absolute; recovers Animation.m's exact convention, verified from its
actual code, not guessed):
  - Units: positions in mm, angles in degrees.
  - Coordinate system: +X screen-right, +Y up, +Z out of screen.
  - Rotation: ZYX Euler, R = Rz(axis2) . Ry(axis1) . Rx(axis0), angles in
    radians.
  - Torso chain: pos[child] = rot[child] @ offset[child] + pos[parent],
    rot[child] = rot[parent] @ R(angle[parent]), chained Waist -> Spine ->
    Chest -> Neck -> Head -> HeadTip, rot[Waist] = Identity.
  - VERIFIED QUIRK (literally in Animation.m, not a simplification we are
    introducing): arms and legs do NOT inherit the full torso rotation
    chain. L_Collar/R_Collar rotate only by Chest's own angle ALONE (not
    rot[Chest], i.e. not chained through Waist+Spine+Chest), and L_UpperLeg
    /R_UpperLeg rotate only by Waist's own angle alone. Implemented exactly
    as coded in forward_kinematics_frame() below.
  - The dataset author's own caveat: "the visualization is correct for the
    upper body joints, but it is not entirely correct for all lower body
    joints. The movements that look unnatural are m03, m04, and m05."
    Movements 7 and 9 (arms) are exactly the joints the author says ARE
    correct -- this is why we proceeded instead of stopping at Step 1.
  - Role mapping (mirrors feature_extraction.py's MediaPipe landmark
    roles): shoulder=UpperArm, elbow=Forearm, wrist=Hand, hip=UpperLeg,
    per side. These FK positions are packed into a 33-slot, MediaPipe-
    PoseLandmark-shaped list (see _fk_frame_to_landmarks()) so
    feature_extraction.compute_shoulder_abduction_angle() /
    compute_shoulder_rotation_angle() -- UNMODIFIED -- can be called
    directly. Visibility is set to 1.0 for the side already selected by
    select_side_for_rep() and 0.0 for the other, so feature_extraction's
    own select_side() deterministically repicks the same side.

Positions vs. Angles gap (measured, not fixed): for movement 7 subject 1
rep 1 (63 frames), compute_angle(hip, shoulder, elbow) applied naively to
raw (unrotated) Positions data (R_UpperLeg, R_UpperArm, R_Forearm as
hip/shoulder/elbow) vs. the Angles file's R_UpperArm axis 2 value, frame
by frame:
  - mean |difference| across all 63 frames: 151.5 degrees
  - range of |difference|: 95.0 to 208.2 degrees
The naive-Positions angle barely moved across the whole rep (156-158 deg)
while the Angles-file value swung through its full ~115-degree exercise
range -- because Positions are parent-relative offsets, not world
coordinates, so a 3-point angle computed directly on them is close to
constant bone geometry, not the limb's actual pose. This is why forward
kinematics (reconstructing absolute positions first) is necessary, not
optional -- a footnote-sized gap would not have required this.

Still must verify if the dataset is ever re-downloaded/re-extended:
1. Whether this file layout/column order/FK convention holds for movements
   other than 7 and 9 (not checked -- out of this project's scope, and the
   author's own caveat suggests m03/m04/m05 specifically do NOT hold).
2. If Vicon is ever added: its column layout is a different, unverified
   39-joint order and would need this same verification pass repeated.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_extraction import (
    compute_shoulder_abduction_angle,
    compute_shoulder_rotation_angle,
    extract_rep_features,
)

MOVEMENT_TO_EXERCISE = {7: "shoulder_abduction", 9: "shoulder_rotation"}
EXERCISE_TO_MOVEMENT = {v: k for k, v in MOVEMENT_TO_EXERCISE.items()}

FILENAME_RE = re.compile(
    r"m(\d{2})_s(\d{2})_e(\d{2})_(positions|angles)(_inc)?\.txt$", re.IGNORECASE
)

FEATURE_SOURCES = ("fk_positions", "angles_file")

# Kinect's 22-joint order (paper Table 2, matches Animation.m). Each joint
# = 3 consecutive columns (col_{3i}, col_{3i+1}, col_{3i+2}) in both
# Positions and Angles files -- confirmed against real downloaded files.
KINECT_JOINTS = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "L_Collar", "L_UpperArm", "L_Forearm", "L_Hand",
    "R_Collar", "R_UpperArm", "R_Forearm", "R_Hand",
    "L_UpperLeg", "L_LowerLeg", "L_Foot", "L_Toes",
    "R_UpperLeg", "R_LowerLeg", "R_Foot", "R_Toes",
]
_JOINT_INDEX = {name: i for i, name in enumerate(KINECT_JOINTS)}

# Role mapping for our two exercises, mirroring feature_extraction.py's
# MediaPipe landmark roles (SHOULDER/ELBOW/WRIST/HIP), per side.
SIDE_ROLE_JOINTS = {
    "left": {"shoulder": "L_UpperArm", "elbow": "L_Forearm", "wrist": "L_Hand", "hip": "L_UpperLeg"},
    "right": {"shoulder": "R_UpperArm", "elbow": "R_Forearm", "wrist": "R_Hand", "hip": "R_UpperLeg"},
}

# legacy "angles_file" feature source: which single Angles-file column
# carries each exercise's motion, per side. Values empirically confirmed
# (largest-range column in a sample rep) and anatomically sensible -- see
# module docstring.
JOINT_COLUMNS = {
    "kinect": {
        "shoulder_abduction": {
            "right": f"col_{_JOINT_INDEX['R_UpperArm'] * 3 + 2}",  # col_35
            "left": f"col_{_JOINT_INDEX['L_UpperArm'] * 3 + 2}",   # col_23
        },
        "shoulder_rotation": {
            "right": f"col_{_JOINT_INDEX['R_Forearm'] * 3 + 2}",  # col_38
            "left": f"col_{_JOINT_INDEX['L_Forearm'] * 3 + 2}",   # col_26
        },
    },
    # Vicon is out of scope for this project -- see module docstring.
    "vicon": None,
}


@dataclass
class ParsedFilename:
    movement: int
    subject_id: str
    rep_id: str
    is_correct: bool
    source: str
    path: Path


def parse_filename(path: Path) -> ParsedFilename:
    match = FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Filename does not match UI-PRMD's expected pattern: {path.name}")
    movement, subject, episode, source, inc_suffix = match.groups()

    return ParsedFilename(
        movement=int(movement),
        subject_id=f"s{subject}",
        rep_id=f"e{episode}",
        is_correct=inc_suffix is None,
        source=source.lower(),
        path=path,
    )


def sibling_path(path: Path, source: str) -> Path:
    """Given a Positions (or Angles) file path, return the matching Angles
    (or Positions) file path for the same repetition -- same subject,
    episode, correctness, just the other data channel.
    """
    other = "angles" if source == "positions" else "positions"
    other_cap = "Angles" if other == "angles" else "Positions"
    new_parts = [other_cap if p in ("Positions", "Angles") else p for p in path.parts]
    new_name = path.name.replace(source, other)
    return Path(*new_parts[:-1]) / new_name


def _modality_from_path(path: Path) -> str:
    parts_lower = [p.lower() for p in path.parts]
    for modality in JOINT_COLUMNS:
        if modality in parts_lower:
            return modality
    raise ValueError(f"Could not determine modality (vicon/kinect) from path: {path}")


def discover_rep_files(
    raw_dir: str | Path, movement: int, modality: str = "kinect", source: str = "positions"
) -> list[ParsedFilename]:
    """Find every correct + incorrect Positions (or Angles) file for one
    movement under raw_dir, matching both the plain and "_inc" filename
    forms. Does not read file contents. Returns [] if raw_dir doesn't
    exist or nothing matches (e.g. data not downloaded yet) -- that's
    expected, not an error.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []

    found = []
    for pattern in (f"*_{source}.txt", f"*_{source}_inc.txt"):
        for path in raw_dir.rglob(pattern):
            try:
                parsed = parse_filename(path)
            except ValueError:
                continue
            if parsed.movement != movement:
                continue
            parts_lower = [p.lower() for p in path.parts]
            if modality and modality.lower() not in parts_lower:
                continue
            found.append(parsed)
    return found


def select_side_for_rep(frames: pd.DataFrame, exercise: str, modality: str = "kinect") -> str:
    """Which arm (left/right) performed this repetition. Mocap data has no
    per-frame visibility/confidence signal the way live MediaPipe does, so
    this is a heuristic: compute each side's candidate Angles-file column
    range across the rep's frames, pick whichever moved more. Verified
    against real data: for the confirmed right-arm subjects, the right
    column's range is 20-30x the left column's (motion vs. near-stationary
    noise), so this heuristic cleanly separates the two cases in practice.
    Used both by the legacy "angles_file" feature source directly, and by
    "fk_positions" just to decide which side's visibility=1.0 in the
    FK-derived landmarks (see _fk_frame_to_landmarks()).
    """
    cols = JOINT_COLUMNS[modality][exercise]
    left_range = frames[cols["left"]].max() - frames[cols["left"]].min()
    right_range = frames[cols["right"]].max() - frames[cols["right"]].min()
    return "right" if right_range >= left_range else "left"


# ---------------------------------------------------------------------------
# Forward kinematics (feature_source="fk_positions")
# ---------------------------------------------------------------------------

def euler_zyx_to_matrix(angles_deg: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (degrees, [axis0, axis1, axis2]) -> 3x3 rotation
    matrix, R = Rz(axis2) @ Ry(axis1) @ Rx(axis0). Matches Animation.m's
    eulers_2_rot_matrix() exactly (gamma_x=axis0, beta_y=axis1, alpha_z=axis2).
    """
    gx, by, az = np.radians(angles_deg)
    cx, sx = np.cos(gx), np.sin(gx)
    cy, sy = np.cos(by), np.sin(by)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def forward_kinematics_frame(pos_row: np.ndarray, ang_row: np.ndarray) -> dict[str, np.ndarray]:
    """One frame's raw Positions + Angles row (each length 66, Kinect's
    22 joints x 3) -> dict of joint_name -> absolute (x, y, z) position,
    replicating Animation.m's FK exactly (including the arms/legs-use-
    only-their-immediate-parent's-own-angle quirk documented above).
    """
    offset = {name: pos_row[3 * i:3 * i + 3] for i, name in enumerate(KINECT_JOINTS)}
    ang = {name: ang_row[3 * i:3 * i + 3] for i, name in enumerate(KINECT_JOINTS)}
    R = {name: euler_zyx_to_matrix(ang[name]) for name in KINECT_JOINTS}

    pos = {"Waist": offset["Waist"]}  # root: absolute, given directly

    # torso chain: Waist -> Spine -> Chest -> Neck -> Head -> HeadTip
    rot = np.eye(3)
    for parent, child in [("Waist", "Spine"), ("Spine", "Chest"), ("Chest", "Neck"),
                           ("Neck", "Head"), ("Head", "HeadTip")]:
        rot = rot @ R[parent]
        pos[child] = rot @ offset[child] + pos[parent]

    # left arm: Chest -> L_Collar -> L_UpperArm -> L_Forearm -> L_Hand
    # (collar uses ONLY Chest's own angle, not the full torso chain -- verified quirk)
    rot = R["Chest"]
    pos["L_Collar"] = rot @ offset["L_Collar"] + pos["Chest"]
    for parent, child in [("L_Collar", "L_UpperArm"), ("L_UpperArm", "L_Forearm"), ("L_Forearm", "L_Hand")]:
        rot = rot @ R[parent]
        pos[child] = rot @ offset[child] + pos[parent]

    # right arm (mirror)
    rot = R["Chest"]
    pos["R_Collar"] = rot @ offset["R_Collar"] + pos["Chest"]
    for parent, child in [("R_Collar", "R_UpperArm"), ("R_UpperArm", "R_Forearm"), ("R_Forearm", "R_Hand")]:
        rot = rot @ R[parent]
        pos[child] = rot @ offset[child] + pos[parent]

    # left leg: Waist -> L_UpperLeg -> L_LowerLeg -> L_Foot -> L_Toes
    # (upper leg uses ONLY Waist's own angle, not chained -- verified quirk)
    rot = R["Waist"]
    pos["L_UpperLeg"] = rot @ offset["L_UpperLeg"] + pos["Waist"]
    for parent, child in [("L_UpperLeg", "L_LowerLeg"), ("L_LowerLeg", "L_Foot"), ("L_Foot", "L_Toes")]:
        rot = rot @ R[parent]
        pos[child] = rot @ offset[child] + pos[parent]

    # right leg (mirror)
    rot = R["Waist"]
    pos["R_UpperLeg"] = rot @ offset["R_UpperLeg"] + pos["Waist"]
    for parent, child in [("R_UpperLeg", "R_LowerLeg"), ("R_LowerLeg", "R_Foot"), ("R_Foot", "R_Toes")]:
        rot = rot @ R[parent]
        pos[child] = rot @ offset[child] + pos[parent]

    return pos


def _fk_frame_to_landmarks(fk_pos: dict[str, np.ndarray], active_side: str) -> list[tuple]:
    """Pack FK-derived absolute joint positions into a 33-slot,
    MediaPipe-PoseLandmark-shaped (x, y, z, visibility) list, so
    feature_extraction.compute_shoulder_abduction_angle() /
    compute_shoulder_rotation_angle() can be called UNMODIFIED -- same
    angle math as live inference, no second implementation. Visibility is
    1.0 for active_side's landmarks and 0.0 for the other side, so
    feature_extraction's own select_side() deterministically repicks
    active_side (already determined by select_side_for_rep()).
    """
    landmarks = [(0.0, 0.0, 0.0, 0.0)] * 33
    mp_indices = {"left": {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23},
                  "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24}}
    for side, roles in SIDE_ROLE_JOINTS.items():
        visibility = 1.0 if side == active_side else 0.0
        for role, joint_name in roles.items():
            x, y, z = fk_pos[joint_name]
            landmarks[mp_indices[side][role]] = (float(x), float(y), float(z), visibility)
    return landmarks


def load_rep_frames_fk(parsed_positions: ParsedFilename) -> pd.DataFrame:
    """One rep's Positions file (+ its sibling Angles file, needed for FK
    rotations) -> one DataFrame: frame_idx, metadata columns, side, and
    angle_deg computed via forward kinematics + feature_extraction.py's
    own compute_shoulder_abduction_angle()/compute_shoulder_rotation_angle().
    """
    pos_df = pd.read_csv(parsed_positions.path, sep=r"\s+|,", engine="python", header=None)
    ang_path = sibling_path(parsed_positions.path, "positions")
    ang_df = pd.read_csv(ang_path, sep=r"\s+|,", engine="python", header=None)
    if len(pos_df) != len(ang_df):
        raise ValueError(f"Frame count mismatch between {parsed_positions.path} and {ang_path}")

    modality = _modality_from_path(parsed_positions.path)
    exercise = MOVEMENT_TO_EXERCISE[parsed_positions.movement]

    # side is selected using the Angles-file column heuristic on this rep's
    # own angles, before FK -- same heuristic as the legacy path, just used
    # here to pick which side's visibility=1.0 in the FK landmarks below.
    ang_df_named = ang_df.copy()
    ang_df_named.columns = [f"col_{i}" for i in range(ang_df_named.shape[1])]
    side = select_side_for_rep(ang_df_named, exercise, modality)

    angle_fn = compute_shoulder_abduction_angle if exercise == "shoulder_abduction" else compute_shoulder_rotation_angle

    angles_deg = []
    for pos_row, ang_row in zip(pos_df.to_numpy(), ang_df.to_numpy()):
        fk_pos = forward_kinematics_frame(pos_row, ang_row)
        landmarks = _fk_frame_to_landmarks(fk_pos, side)
        angles_deg.append(angle_fn(landmarks))

    out = pd.DataFrame({"frame_idx": range(len(pos_df)), "angle_deg": angles_deg})
    out["subject_id"] = parsed_positions.subject_id
    out["movement"] = parsed_positions.movement
    out["rep_id"] = parsed_positions.rep_id
    out["is_correct"] = parsed_positions.is_correct
    out["side"] = side
    return out


# ---------------------------------------------------------------------------
# Legacy direct-column feature source (feature_source="angles_file")
# ---------------------------------------------------------------------------

def load_rep_frames_angles_file(parsed: ParsedFilename) -> pd.DataFrame:
    """One Angles file -> one DataFrame: frame_idx, raw columns
    (col_0..col_65 for Kinect), metadata, side, and angle_deg -- the
    single joint-axis column (see JOINT_COLUMNS) that carries this
    exercise's motion, selected per rep via select_side_for_rep(). Legacy
    path, kept only for comparison against feature_source="fk_positions"
    -- see module docstring for why this isn't what app.py's models use.
    """
    raw = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
    raw.columns = [f"col_{i}" for i in range(raw.shape[1])]
    raw.insert(0, "frame_idx", range(len(raw)))
    raw["subject_id"] = parsed.subject_id
    raw["movement"] = parsed.movement
    raw["rep_id"] = parsed.rep_id
    raw["is_correct"] = parsed.is_correct

    modality = _modality_from_path(parsed.path)
    exercise = MOVEMENT_TO_EXERCISE[parsed.movement]
    side = select_side_for_rep(raw, exercise, modality)
    raw["side"] = side
    raw["angle_deg"] = raw[JOINT_COLUMNS[modality][exercise][side]]
    return raw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset_for_exercise(
    raw_dir: str | Path,
    exercise: str,
    modality: str = "kinect",
    feature_source: str = "fk_positions",
) -> pd.DataFrame:
    """Discovers every rep for the exercise, computes angle_deg per frame
    (via forward kinematics + feature_extraction.py, or via the legacy
    Angles-file column -- see feature_source), aggregates each rep via
    feature_extraction.extract_rep_features(). Returns the rep-level
    DataFrame src/classifier.py trains on: subject_id, movement, rep_id,
    is_correct, angle_max_deg, angle_min_deg, angle_mean_deg,
    angle_range_deg, angle_midpoint_deg -- one row per repetition.
    """
    if exercise not in EXERCISE_TO_MOVEMENT:
        raise ValueError(f"Unknown exercise '{exercise}'. Must be one of {list(EXERCISE_TO_MOVEMENT)}")
    if feature_source not in FEATURE_SOURCES:
        raise ValueError(f"Unknown feature_source '{feature_source}'. Must be one of {FEATURE_SOURCES}")
    movement = EXERCISE_TO_MOVEMENT[exercise]

    source = "positions" if feature_source == "fk_positions" else "angles"
    loader = load_rep_frames_fk if feature_source == "fk_positions" else load_rep_frames_angles_file

    rows = []
    for parsed in discover_rep_files(raw_dir, movement, modality=modality, source=source):
        frames = loader(parsed)
        features = extract_rep_features(frames, angle_col="angle_deg")
        rows.append({
            "subject_id": parsed.subject_id,
            "movement": parsed.movement,
            "rep_id": parsed.rep_id,
            "is_correct": parsed.is_correct,
            **features,
        })

    if not rows:
        raise NotImplementedError(
            f"No UI-PRMD movement {movement} files found under {raw_dir} (modality={modality}). "
            "Download UI-PRMD into data/ui_prmd/ before calling load_dataset_for_exercise()."
        )
    return pd.DataFrame(rows)
