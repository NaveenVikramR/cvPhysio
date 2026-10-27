"""Tests for src/feature_extraction.py's NaN guard on degenerate
(coincident-point) frames."""

import math

from src.feature_extraction import (
    LEFT,
    compute_angle,
    compute_elbow_angle,
    compute_shoulder_abduction_angle,
    extract_features,
)


def make_landmarks(overrides: dict[int, tuple[float, float, float, float]] | None = None):
    """33 landmarks, all distinct and fully visible by default, spaced
    out along x so no two coincide unless explicitly overridden."""
    landmarks = [(0.1 * i, 0.5, 0.0, 1.0) for i in range(33)]
    for idx, point in (overrides or {}).items():
        landmarks[idx] = point
    return landmarks


def test_compute_angle_returns_nan_when_b_coincides_with_a():
    a = (1.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)  # same as a
    c = (0.0, 1.0, 0.0)
    result = compute_angle(a, b, c)
    assert math.isnan(result)


def test_compute_angle_returns_nan_when_b_coincides_with_c():
    a = (0.0, 1.0, 0.0)
    b = (1.0, 0.0, 0.0)
    c = (1.0, 0.0, 0.0)  # same as b
    result = compute_angle(a, b, c)
    assert math.isnan(result)


def test_compute_angle_is_normal_for_distinct_points():
    # sanity check the guard doesn't affect the ordinary case
    a, b, c = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    result = compute_angle(a, b, c)
    assert abs(result - 90.0) < 1e-9


def test_compute_shoulder_abduction_angle_is_nan_when_shoulder_equals_elbow():
    landmarks = make_landmarks({
        LEFT["shoulder"]: (0.5, 0.5, 0.0, 1.0),
        LEFT["elbow"]: (0.5, 0.5, 0.0, 1.0),  # coincides with shoulder
    })
    result = compute_shoulder_abduction_angle(landmarks)
    assert math.isnan(result)


def test_compute_elbow_angle_is_nan_when_elbow_equals_wrist():
    landmarks = make_landmarks({
        LEFT["elbow"]: (0.5, 0.5, 0.0, 1.0),
        LEFT["wrist"]: (0.5, 0.5, 0.0, 1.0),  # coincides with elbow
    })
    result = compute_elbow_angle(landmarks)
    assert math.isnan(result)


def test_extract_features_excludes_degenerate_frame_as_invalid():
    # Confidence-wise this frame is fine (every landmark fully visible),
    # but the angle itself is undefined -- extract_features() must treat
    # it the same as a confidence-gating failure, not return a NaN angle.
    landmarks = make_landmarks({
        LEFT["shoulder"]: (0.5, 0.5, 0.0, 1.0),
        LEFT["elbow"]: (0.5, 0.5, 0.0, 1.0),
    })
    result = extract_features(landmarks, "shoulder_abduction")
    assert result is None


def test_extract_features_returns_normal_dict_for_a_healthy_frame():
    landmarks = make_landmarks()
    result = extract_features(landmarks, "shoulder_abduction")
    assert result is not None
    assert not math.isnan(result["angle_deg"])
