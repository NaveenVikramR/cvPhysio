"""Offline dataset loading for UI-PRMD. Build-time only -- never imported by
the six runtime modules (pose_estimation, feature_extraction, classifier,
feedback, tracking, app.py). Parses UI-PRMD's "Segmented Movements" release
into per-repetition angle features for src/classifier.py to train on.

VERIFIED against real downloaded files (movements 7 and 9, Kinect, all 10
subjects, correct and incorrect -- see data/ui_prmd/). What was confirmed
vs. corrected from the original (documentation-only) assumptions:

- Filenames: m{MM}_s{SS}_e{EE}_{positions|angles}.txt for correct reps.
  Incorrect reps add an "_inc" suffix before .txt (e.g.
  "m07_s01_e01_angles_inc.txt") -- CONFIRMED, and is_correct is now parsed
  from that suffix, not from a "Correct"/"Incorrect" path component (the
  real download folders are named "Segmented Movements" and "Incorrect
  Segmented Movements", neither of which is literally "correct"/
  "incorrect", so path-based detection never actually worked).
- One file = one already-segmented repetition -- CONFIRMED (checked a
  sample file's angle signal: single rise-fall arc, not oscillating).
- Subject numbering (s01-s10) is consistent across movements -- CONFIRMED.
- A single fixed arm per movement (DEFAULT_SIDE) -- WRONG as a global
  constant. 8 of 10 subjects use the right arm; subjects 07 and 10
  consistently use the left arm (checked across both m07 and m09). Side is
  now selected dynamically per repetition (see select_side_for_rep()),
  not hardcoded.
- Modality: VICON IS OUT OF SCOPE for this project. The only complete,
  verified download we obtained was Kinect (the Vicon portion of every
  available mirror was incomplete/truncated). This isn't just an
  expedient fallback: Kinect is also the closer analog to this project's
  actual sensor -- a webcam via MediaPipe -- than Vicon's markered optical
  mocap rig, so training on Kinect's noisier, camera-derived skeleton is
  arguably more representative of what live inference will see anyway.
- Positions vs. Angles as the angle_deg source: SWITCHED from Positions
  (+ compute_angle()) to Angles (direct column selection). Kinect's
  Positions values turned out to be bone-local offsets relative to the
  parent joint, not world coordinates -- e.g. during a 115-degree-range
  abduction, the right upper arm position only moves ~2mm. There's no
  usable displacement to compute a 3-point angle from without
  reconstructing world space through the full kinematic chain (out of
  scope). The Angles file's joint-axis columns are directly usable and
  empirically clean (single-arc, large range, matching the exercise).
  KNOWN GAP: this means dataset_loader.py's angle_deg is no longer
  computed via the same compute_angle() formula used at live inference
  time by feature_extraction.py -- see the sanity-check measurement below
  for how large that definition gap actually is.
- Joint order (from the dataset's paper, Table 2) for Kinect's 22 joints,
  each occupying 3 consecutive columns (axis 0, 1, 2) in both Positions
  and Angles files, 66 columns total, no header row, comma-delimited:
    Waist, Spine, Chest, Neck, Head, HeadTip, L_Collar, L_UpperArm,
    L_Forearm, L_Hand, R_Collar, R_UpperArm, R_Forearm, R_Hand,
    L_UpperLeg, L_LowerLeg, L_Foot, L_Toes, R_UpperLeg, R_LowerLeg,
    R_Foot, R_Toes
  For shoulder_abduction, R_UpperArm axis 2 (col 35) / L_UpperArm axis 2
  (col 23) carries the exercise's motion (empirically the largest-range
  column in a sample m07 file). For shoulder_rotation, R_Forearm axis 2
  (col 38) / L_Forearm axis 2 (col 26) carries it (largest-range column
  in a sample m09 file) -- anatomically sensible, since internal/external
  rotation is forearm rotation about the humeral axis.

Sanity check (measured, not fixed): for movement 7 subject 1 rep 1 (63
frames), compute_angle(hip, shoulder, elbow) on Positions data
(R_UpperLeg, R_UpperArm, R_Forearm as hip/shoulder/elbow) vs. the Angles
file's R_UpperArm axis 2 value, frame by frame:
  - mean |difference| across all 63 frames: 151.5 degrees
  - range of |difference|: 95.0 to 208.2 degrees
  - at the rep's midpoint frame: compute_angle() = 156.5 deg vs.
    Angles-file = -51.7 deg (diff = 208.2 deg)
The Positions-derived angle barely moves across the whole rep (156-158
deg, ~2 deg of "motion" total) while the Angles-file value swings through
its full ~115-degree exercise range -- confirming these are not two
measurements of the same quantity offset by a constant, but genuinely
different signals (bone-local geometry vs. the true joint rotation). This
is not a footnote-sized gap; it's why Positions + compute_angle() was
abandoned in favor of the Angles file above, not merely a documented
imperfection.

Still must verify if the dataset is ever re-downloaded/re-extended:
1. Whether this file layout/column order holds for movements other than
   7 and 9 (not checked -- out of this project's scope).
2. If Vicon is ever added: its column layout is a different, unverified
   39-joint order and would need this same verification pass repeated.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.feature_extraction import extract_rep_features

MOVEMENT_TO_EXERCISE = {7: "shoulder_abduction", 9: "shoulder_rotation"}
EXERCISE_TO_MOVEMENT = {v: k for k, v in MOVEMENT_TO_EXERCISE.items()}

FILENAME_RE = re.compile(
    r"m(\d{2})_s(\d{2})_e(\d{2})_(positions|angles)(_inc)?\.txt$", re.IGNORECASE
)

# Kinect's 22-joint order (paper Table 2). Each joint = 3 consecutive
# columns (col_{3i}, col_{3i+1}, col_{3i+2}) in both Positions and Angles
# files -- confirmed against real downloaded files.
KINECT_JOINTS = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "L_Collar", "L_UpperArm", "L_Forearm", "L_Hand",
    "R_Collar", "R_UpperArm", "R_Forearm", "R_Hand",
    "L_UpperLeg", "L_LowerLeg", "L_Foot", "L_Toes",
    "R_UpperLeg", "R_LowerLeg", "R_Foot", "R_Toes",
]


def _joint_col(joint_name: str, axis: int) -> str:
    return f"col_{KINECT_JOINTS.index(joint_name) * 3 + axis}"


# Which single Angles-file column carries each exercise's motion, per side.
# Values empirically confirmed (largest-range column in a sample rep) and
# anatomically sensible -- see module docstring.
JOINT_COLUMNS = {
    "kinect": {
        "shoulder_abduction": {
            "right": _joint_col("R_UpperArm", 2),  # col_35
            "left": _joint_col("L_UpperArm", 2),   # col_23
        },
        "shoulder_rotation": {
            "right": _joint_col("R_Forearm", 2),  # col_38
            "left": _joint_col("L_Forearm", 2),   # col_26
        },
    },
    # Vicon is out of scope for this project -- see module docstring.
    "vicon": None,
}

ANGLE_SOURCE = "angles"  # Angles file, not Positions -- see module docstring.


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


def _modality_from_path(path: Path) -> str:
    parts_lower = [p.lower() for p in path.parts]
    for modality in JOINT_COLUMNS:
        if modality in parts_lower:
            return modality
    raise ValueError(f"Could not determine modality (vicon/kinect) from path: {path}")


def discover_rep_files(
    raw_dir: str | Path, movement: int, modality: str = "kinect", source: str = ANGLE_SOURCE
) -> list[ParsedFilename]:
    """Find every correct + incorrect Angles (or Positions) file for one
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
    this is a heuristic: compute each side's candidate angle column range
    across the rep's frames, pick whichever moved more. Verified against
    real data: for the confirmed right-arm subjects, the right column's
    range is 20-30x the left column's (motion vs. near-stationary noise),
    so this heuristic cleanly separates the two cases in practice.
    """
    cols = JOINT_COLUMNS[modality][exercise]
    left_range = frames[cols["left"]].max() - frames[cols["left"]].min()
    right_range = frames[cols["right"]].max() - frames[cols["right"]].min()
    return "right" if right_range >= left_range else "left"


def load_rep_frames(parsed: ParsedFilename) -> pd.DataFrame:
    """One Angles file -> one DataFrame: frame_idx, raw columns
    (col_0..col_65 for Kinect), subject_id/movement/rep_id/is_correct
    metadata, the dynamically-selected side, and angle_deg -- the single
    joint-axis column (see JOINT_COLUMNS) that carries this exercise's
    motion, selected per rep via select_side_for_rep().
    """
    raw = pd.read_csv(parsed.path, sep=r"\s+|,", engine="python", header=None)
    raw.columns = [f"col_{i}" for i in range(raw.shape[1])]
    raw.insert(0, "frame_idx", range(len(raw)))
    raw["subject_id"] = parsed.subject_id
    raw["movement"] = parsed.movement
    raw["rep_id"] = parsed.rep_id
    raw["is_correct"] = parsed.is_correct

    modality = _modality_from_path(parsed.path)
    if JOINT_COLUMNS.get(modality) is None:
        raise NotImplementedError(
            f"JOINT_COLUMNS['{modality}'] is not available -- see module docstring "
            "(only 'kinect' is verified for this project)."
        )
    exercise = MOVEMENT_TO_EXERCISE[parsed.movement]
    side = select_side_for_rep(raw, exercise, modality)
    raw["side"] = side
    raw["angle_deg"] = raw[JOINT_COLUMNS[modality][exercise][side]]
    return raw


def load_dataset_for_exercise(
    raw_dir: str | Path, exercise: str, modality: str = "kinect"
) -> pd.DataFrame:
    """Discovers every rep's Angles file for the exercise, loads + selects
    angle_deg per frame, aggregates each rep via
    feature_extraction.extract_rep_features(). Returns the rep-level
    DataFrame src/classifier.py trains on: subject_id, movement, rep_id,
    is_correct, angle_max_deg, angle_min_deg, angle_mean_deg,
    angle_range_deg, angle_midpoint_deg -- one row per repetition.
    """
    if exercise not in EXERCISE_TO_MOVEMENT:
        raise ValueError(f"Unknown exercise '{exercise}'. Must be one of {list(EXERCISE_TO_MOVEMENT)}")
    movement = EXERCISE_TO_MOVEMENT[exercise]

    rows = []
    for parsed in discover_rep_files(raw_dir, movement, modality=modality, source=ANGLE_SOURCE):
        frames = load_rep_frames(parsed)
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
