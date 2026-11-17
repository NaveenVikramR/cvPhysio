"""Biomechanical feature extraction. Input: pose_estimation's 33-tuple
landmark list (x, y, z, visibility), indexed per MediaPipe's PoseLandmark
enum. Output: feature dict, or None if confidence gating fails. Must not
import pose_estimation or classifier.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MIN_VISIBILITY = 0.5

ANGLE_CORRECTION_PATH = Path("config/angle_correction.json")

EXERCISES = {
    "shoulder_abduction": "Standing Shoulder Abduction",  # UI-PRMD movement 7
    "shoulder_rotation": "Standing Shoulder Internal-External Rotation",  # UI-PRMD movement 9
}

LEFT = {"shoulder": 11, "elbow": 13, "wrist": 15, "hip": 23}
RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16, "hip": 24}

# Default joint roles required for each exercise's OWN primary angle
# (compute_shoulder_abduction_angle / compute_shoulder_rotation_angle)
# only -- NOT what a rules-based scorer needs, which depends on which
# rules are active and is derived from there (see src/rules.py's
# required_joint_roles()), not hardcoded here a second time. Used as the
# fallback when no explicit required_roles is given, e.g. the classifier
# scorer path, which never touches elbow/trunk features.
BASE_REQUIRED_ROLES = {
    "shoulder_abduction": {"shoulder", "elbow", "hip"},
    "shoulder_rotation": {"shoulder", "elbow", "wrist", "hip"},
}


def compute_angle(a, b, c) -> float:
    """Angle at point b, in degrees, between vectors b->a and b->c.
    a, b, c are (x, y, z, ...) points; any fields past index 2 are ignored.
    Returns NaN (not a crash) if b coincides with a or with c -- a
    zero-length vector has no defined direction, so the angle is
    undefined. Callers must treat a NaN result as an invalid frame,
    the same way a confidence-gating failure is treated (see
    extract_features() and app.py's process_frame()).
    """
    a = np.asarray(a[:3], dtype=float)
    b = np.asarray(b[:3], dtype=float)
    c = np.asarray(c[:3], dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return float("nan")
    cos_theta = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def select_side(landmarks: list[tuple[float, float, float, float]]) -> dict[str, int]:
    left_visibility = sum(landmarks[i][3] for i in LEFT.values()) / len(LEFT)
    right_visibility = sum(landmarks[i][3] for i in RIGHT.values()) / len(RIGHT)
    return LEFT if left_visibility >= right_visibility else RIGHT


SIDES = {"LEFT": LEFT, "RIGHT": RIGHT}


def side_joints(landmarks: list[tuple[float, float, float, float]], side: str | None = None) -> dict[str, int]:
    """The joint indices for `side` ("LEFT"/"RIGHT") when the caller has
    locked an arm, else select_side()'s per-frame pick by visibility.
    Locking matters for one-arm exercises: with both arms resting at the
    sides, the per-frame pick can flip to the idle arm (measured in
    docs/landmark_fidelity.md, finding 5)."""
    if side is None:
        return select_side(landmarks)
    if side not in SIDES:
        raise ValueError(f"side={side!r}; must be one of {list(SIDES)} or None")
    return SIDES[side]


def compute_shoulder_abduction_angle(landmarks: list[tuple[float, float, float, float]],
                                     side: str | None = None) -> float:
    """Standing Shoulder Abduction angle (UI-PRMD movement 7): the angle at
    the shoulder between the torso (shoulder->hip) and the upper arm
    (shoulder->elbow) — ~0 deg with the arm at the side, ~90 deg raised to
    horizontal. Uses `side` if given, else whichever arm (left/right) has
    higher average landmark visibility.
    """
    joints = side_joints(landmarks, side)
    shoulder = landmarks[joints["shoulder"]]
    elbow = landmarks[joints["elbow"]]
    hip = landmarks[joints["hip"]]
    return compute_angle(hip, shoulder, elbow)


def shoulder_rotation_angle_from_points(shoulder, elbow, wrist, hip) -> float:
    """Core geometry behind the shoulder rotation angle: projects the
    forearm (elbow->wrist) and a torso reference (shoulder->hip) into the
    plane perpendicular to the upper-arm axis (shoulder->elbow), then
    measures the angle between them via compute_angle(). Takes raw
    (x, y, z, ...) points directly (not a MediaPipe landmark list) so
    dataset_loader.py can reuse this exact formula on UI-PRMD joint
    positions, keeping training-time and inference-time angles
    geometrically consistent.
    """
    shoulder = np.asarray(shoulder[:3], dtype=float)
    elbow = np.asarray(elbow[:3], dtype=float)
    wrist = np.asarray(wrist[:3], dtype=float)
    hip = np.asarray(hip[:3], dtype=float)

    humeral_axis = elbow - shoulder
    humeral_axis = humeral_axis / np.linalg.norm(humeral_axis)

    forearm = wrist - elbow
    torso_ref = hip - shoulder
    forearm_proj = forearm - np.dot(forearm, humeral_axis) * humeral_axis
    torso_ref_proj = torso_ref - np.dot(torso_ref, humeral_axis) * humeral_axis

    return compute_angle(elbow + forearm_proj, elbow, elbow + torso_ref_proj)


def compute_shoulder_rotation_angle(landmarks: list[tuple[float, float, float, float]],
                                    side: str | None = None) -> float:
    """Standing Shoulder Internal-External Rotation angle (UI-PRMD movement
    9): the angle, measured in the plane perpendicular to the upper-arm
    axis (shoulder->elbow), between the forearm (elbow->wrist) and a torso
    reference (shoulder->hip) projected into that same plane — this
    isolates rotation about the humeral axis from shoulder flexion/
    abduction. Uses `side` if given, else whichever arm (left/right) has
    higher average landmark visibility.
    """
    joints = side_joints(landmarks, side)
    return shoulder_rotation_angle_from_points(
        landmarks[joints["shoulder"]], landmarks[joints["elbow"]],
        landmarks[joints["wrist"]], landmarks[joints["hip"]],
    )


def compute_elbow_angle(landmarks: list[tuple[float, float, float, float]], side: str | None = None) -> float:
    """Elbow angle: angle at the elbow between the upper arm (elbow->
    shoulder) and the forearm (elbow->wrist) -- ~180 deg for a straight
    arm, smaller as the elbow bends. Uses `side` if given, else whichever
    arm (left/right) has higher average landmark visibility. Live-pipeline
    helper for src/rules.py's "elbow stays roughly straight" rule.
    """
    joints = side_joints(landmarks, side)
    shoulder = landmarks[joints["shoulder"]]
    elbow = landmarks[joints["elbow"]]
    wrist = landmarks[joints["wrist"]]
    return compute_angle(shoulder, elbow, wrist)


def compute_trunk_lean_deg(landmarks: list[tuple[float, float, float, float]]) -> float:
    """Trunk lean from vertical, in degrees: the angle between the trunk
    axis (hip midpoint -> shoulder midpoint) and straight up. 0 deg is an
    upright trunk; larger values mean more lean/compensation. Uses the
    midpoint of both shoulders and both hips (bilateral), not select_side
    -- trunk posture isn't a single-arm measurement. "Up" is -Y because
    this operates on raw MediaPipe image-space landmarks (Y increases
    downward), so this must only be called on live landmarks, never on
    UI-PRMD's Y-up FK positions. Live-pipeline helper for src/rules.py's
    "trunk lean below limit" rule.
    """
    shoulder_mid = (np.asarray(landmarks[LEFT["shoulder"]][:3], dtype=float)
                     + np.asarray(landmarks[RIGHT["shoulder"]][:3], dtype=float)) / 2
    hip_mid = (np.asarray(landmarks[LEFT["hip"]][:3], dtype=float)
               + np.asarray(landmarks[RIGHT["hip"]][:3], dtype=float)) / 2
    up_point = hip_mid + np.array([0.0, -1.0, 0.0])
    return compute_angle(shoulder_mid, hip_mid, up_point)


def extract_rep_features(sequence_df: pd.DataFrame, angle_col: str = "angle_deg") -> dict:
    """One repetition's frame sequence -> aggregate feature dict for
    training. sequence_df is one rep's frames in original frame order, with
    a numeric angle_col column already computed per frame (e.g. by
    dataset_loader.py, via compute_angle() on the same joint triplet used
    by compute_shoulder_abduction_angle/compute_shoulder_rotation_angle, so
    train-time and inference-time angles stay geometrically consistent).

    angle_recovery_deg: how far the rep's final angle sits from its
    starting angle -- added after a diagnostic plot of real shoulder_
    rotation reps (scripts/plot_rotation_reps.py) showed an incorrect rep
    that dipped lower than every correct rep shown but recovered more
    slowly, ending well short of where it started. Magnitude alone
    (max/min/range) can't distinguish that from a good rep; distance from
    the starting position can.
    """
    angles = sequence_df[angle_col]
    midpoint_idx = len(angles) // 2
    return {
        "angle_max_deg": float(angles.max()),
        "angle_min_deg": float(angles.min()),
        "angle_mean_deg": float(angles.mean()),
        "angle_range_deg": float(angles.max() - angles.min()),
        "angle_midpoint_deg": float(angles.iloc[midpoint_idx]),
        "angle_recovery_deg": float(abs(angles.iloc[-1] - angles.iloc[0])),
    }


def check_confidence(
    landmarks: list[tuple[float, float, float, float]], side: dict[str, int], exercise: str,
    required_roles: set[str] | None = None,
) -> tuple[bool, list[int]]:
    """required_roles, if given, overrides exercise's BASE_REQUIRED_ROLES
    default -- pass src.rules.required_joint_roles() here so the gate
    checks exactly the joints the active rules' features depend on,
    instead of a second, independently-hardcoded list (that mismatch is
    exactly what let a 0.11-visibility wrist corrupt elbow_straight
    undetected before this was fixed).
    """
    roles = required_roles if required_roles is not None else BASE_REQUIRED_ROLES[exercise]
    indices = [side[role] for role in roles]
    failed = [i for i in indices if landmarks[i][3] < MIN_VISIBILITY]
    return (len(failed) == 0, failed)


def load_angle_correction(path: Path | str = ANGLE_CORRECTION_PATH) -> dict[str, float]:
    """{exercise: offset_deg} from config/angle_correction.json. Exercises
    without an entry get no correction."""
    data = json.loads(Path(path).read_text())
    return {ex: float(data[ex]["offset_deg"]) for ex in EXERCISES if ex in data}


def correct_angle(angle: float, exercise: str, correction: dict[str, float] | None) -> float:
    """raw - offset for `exercise`, clipped to the angle's [0, 180] range.
    Returns `angle` unchanged when correction is None or has no entry for
    the exercise (the default everywhere: correction is opt-in)."""
    if not correction or exercise not in correction:
        return angle
    return min(180.0, max(0.0, angle - correction[exercise]))


def extract_features(
    landmarks: list[tuple[float, float, float, float]] | None, exercise: str, side: str | None = None,
    correction: dict[str, float] | None = None,
) -> dict | None:
    """`side` ("LEFT"/"RIGHT") locks the arm; None keeps the per-frame
    select_side() pick. `correction` ({exercise: offset_deg}, see
    load_angle_correction()) shifts angle_deg by a fitted constant; None
    (the default) leaves the raw angle."""
    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise '{exercise}'. Must be one of {list(EXERCISES)}")
    if landmarks is None:
        return None

    joints = side_joints(landmarks, side)
    ok, failed = check_confidence(landmarks, joints, exercise)
    if not ok:
        return None

    side_name = "LEFT" if joints is LEFT else "RIGHT"
    if exercise == "shoulder_abduction":
        angle = compute_shoulder_abduction_angle(landmarks, side_name)
    else:
        angle = compute_shoulder_rotation_angle(landmarks, side_name)

    if math.isnan(angle):
        # Degenerate frame (e.g. two landmarks coincide) -- treated the
        # same as a confidence-gating failure: exclude it, don't return
        # an unusable angle.
        return None

    return {"exercise": exercise, "side": side_name, "angle_deg": correct_angle(angle, exercise, correction)}
