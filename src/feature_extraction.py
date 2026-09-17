"""Biomechanical feature extraction. Input: pose_estimation's per-frame landmark
dict (name -> object with .x/.y/.z/.visibility). Output: feature dict, or None
if confidence gating fails. Must not import pose_estimation or classifier.
"""

from typing import Protocol

import numpy as np

MIN_VISIBILITY = 0.5

EXERCISES = {
    "shoulder_abduction": "Standing Shoulder Abduction",  # UI-PRMD movement 7
    "shoulder_rotation": "Standing Shoulder Internal-External Rotation",  # UI-PRMD movement 9
}

REQUIRED_LANDMARKS = {
    "shoulder_abduction": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "{side}_ELBOW", "{side}_WRIST"],
    "shoulder_rotation": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", "{side}_ELBOW", "{side}_WRIST"],
}


class LandmarkLike(Protocol):
    x: float
    y: float
    z: float
    visibility: float


def _vec(lm: LandmarkLike) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def required_landmarks_for(exercise: str, side: str) -> list[str]:
    return [name.format(side=side) for name in REQUIRED_LANDMARKS[exercise]]


def check_confidence(landmarks: dict[str, LandmarkLike], required_names: list[str]) -> tuple[bool, list[str]]:
    failed = [
        name for name in required_names
        if name not in landmarks or landmarks[name].visibility < MIN_VISIBILITY
    ]
    return (len(failed) == 0, failed)


def _shoulder_abduction_angle(landmarks: dict[str, LandmarkLike], side: str) -> float:
    shoulder = _vec(landmarks[f"{side}_SHOULDER"])
    elbow = _vec(landmarks[f"{side}_ELBOW"])
    hip = _vec(landmarks[f"{side}_HIP"])
    upper_arm = elbow - shoulder
    torso_down = hip - shoulder
    return _angle_between(upper_arm, torso_down)


def _shoulder_rotation_angle(landmarks: dict[str, LandmarkLike], side: str) -> float:
    shoulder = _vec(landmarks[f"{side}_SHOULDER"])
    elbow = _vec(landmarks[f"{side}_ELBOW"])
    wrist = _vec(landmarks[f"{side}_WRIST"])
    hip = _vec(landmarks[f"{side}_HIP"])

    humeral_axis = elbow - shoulder
    humeral_axis /= np.linalg.norm(humeral_axis)

    forearm = wrist - elbow
    reference = hip - shoulder

    # Project both vectors onto the plane perpendicular to the humeral axis,
    # then measure the angle between them to isolate rotation about the arm.
    forearm_proj = forearm - np.dot(forearm, humeral_axis) * humeral_axis
    reference_proj = reference - np.dot(reference, humeral_axis) * humeral_axis
    return _angle_between(forearm_proj, reference_proj)


def extract_features(
    landmarks: dict[str, LandmarkLike], exercise: str, side: str = "RIGHT"
) -> dict | None:
    if exercise not in EXERCISES:
        raise ValueError(f"Unknown exercise '{exercise}'. Must be one of {list(EXERCISES)}")

    required = required_landmarks_for(exercise, side)
    ok, failed = check_confidence(landmarks, required)
    if not ok:
        return None

    if exercise == "shoulder_abduction":
        angle = _shoulder_abduction_angle(landmarks, side)
    else:
        angle = _shoulder_rotation_angle(landmarks, side)

    return {"exercise": exercise, "side": side, "angle_deg": angle}
