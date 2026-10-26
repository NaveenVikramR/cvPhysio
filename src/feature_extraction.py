"""Biomechanical feature extraction. Input: pose_estimation's 33-tuple
landmark list (x, y, z, visibility), indexed per MediaPipe's PoseLandmark
enum. Output: feature dict, or None if confidence gating fails. Must not
import pose_estimation or classifier.
"""

import numpy as np
import pandas as pd

MIN_VISIBILITY = 0.5

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
    """
    a = np.asarray(a[:3], dtype=float)
    b = np.asarray(b[:3], dtype=float)
    c = np.asarray(c[:3], dtype=float)
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def select_side(landmarks: list[tuple[float, float, float, float]]) -> dict[str, int]:
    left_visibility = sum(landmarks[i][3] for i in LEFT.values()) / len(LEFT)
    right_visibility = sum(landmarks[i][3] for i in RIGHT.values()) / len(RIGHT)
    return LEFT if left_visibility >= right_visibility else RIGHT


def compute_shoulder_abduction_angle(landmarks: list[tuple[float, float, float, float]]) -> float:
    """Standing Shoulder Abduction angle (UI-PRMD movement 7): the angle at
    the shoulder between the torso (shoulder->hip) and the upper arm
    (shoulder->elbow) — ~0 deg with the arm at the side, ~90 deg raised to
    horizontal. Uses whichever arm (left/right) has higher average
    landmark visibility.
    """
    side = select_side(landmarks)
    shoulder = landmarks[side["shoulder"]]
    elbow = landmarks[side["elbow"]]
    hip = landmarks[side["hip"]]
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


def compute_shoulder_rotation_angle(landmarks: list[tuple[float, float, float, float]]) -> float:
    """Standing Shoulder Internal-External Rotation angle (UI-PRMD movement
    9): the angle, measured in the plane perpendicular to the upper-arm
    axis (shoulder->elbow), between the forearm (elbow->wrist) and a torso
    reference (shoulder->hip) projected into that same plane — this
    isolates rotation about the humeral axis from shoulder flexion/
    abduction. Uses whichever arm (left/right) has higher average landmark
    visibility.
    """
    side = select_side(landmarks)
    return shoulder_rotation_angle_from_points(
        landmarks[side["shoulder"]], landmarks[side["elbow"]],
        landmarks[side["wrist"]], landmarks[side["hip"]],
    )


def compute_elbow_angle(landmarks: list[tuple[float, float, float, float]]) -> float:
    """Elbow angle: angle at the elbow between the upper arm (elbow->
    shoulder) and the forearm (elbow->wrist) -- ~180 deg for a straight
    arm, smaller as the elbow bends. Uses whichever arm (left/right) has
    higher average landmark visibility. Live-pipeline helper for
    src/rules.py's "elbow stays roughly straight" rule.
    """
    side = select_side(landmarks)
    shoulder = landmarks[side["shoulder"]]
    elbow = landmarks[side["elbow"]]
    wrist = landmarks[side["wrist"]]
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


def extract_features(
    landmarks: list[tuple[float, float, float, float]] | None, exercise: str
) -> dict | None:
    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise '{exercise}'. Must be one of {list(EXERCISES)}")
    if landmarks is None:
        return None

    side = select_side(landmarks)
    ok, failed = check_confidence(landmarks, side, exercise)
    if not ok:
        return None

    if exercise == "shoulder_abduction":
        angle = compute_shoulder_abduction_angle(landmarks)
    else:
        angle = compute_shoulder_rotation_angle(landmarks)

    side_name = "LEFT" if side is LEFT else "RIGHT"
    return {"exercise": exercise, "side": side_name, "angle_deg": angle}
