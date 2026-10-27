"""Tests for src/pose_estimation.py's landmark_space selection
(config/pose.json's "landmark_space" key): world (default) vs normalized.
Mocks _landmarker.detect_for_video() directly -- no real video/webcam
needed -- so these run on synthetic landmark sets, including a case that
proves the world path doesn't depend on frame aspect ratio the way the
normalized path does (results/diag_coords.md's whole finding).
"""

import numpy as np

from src import pose_estimation


class _FakeLandmark:
    def __init__(self, x, y, z, visibility):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _FakeResult:
    def __init__(self, normalized_triples, world_triples):
        self.pose_landmarks = [[_FakeLandmark(x, y, z, 1.0) for x, y, z in normalized_triples]]
        self.pose_world_landmarks = [[_FakeLandmark(x, y, z, 1.0) for x, y, z in world_triples]]


def _dummy_frame():
    return np.zeros((10, 10, 3), dtype=np.uint8)


# One fixed WORLD (metric) scene: shoulder at origin, elbow straight out
# to the side (horizontal upper arm), hip straight down -- a 90 deg
# abduction angle, same geometry regardless of any frame's aspect ratio.
WORLD_PTS = [(0.0, 0.0, 0.0)] * 33
WORLD_PTS[11] = (0.0, 0.0, 0.0)    # left shoulder
WORLD_PTS[13] = (0.3, 0.0, 0.0)    # left elbow
WORLD_PTS[15] = (0.3, -0.25, 0.0)  # left wrist
WORLD_PTS[23] = (0.0, -0.5, 0.0)   # left hip

# Two DIFFERENT normalized representations of frames with different
# (pretend) aspect ratios -- deliberately scaled anisotropically, the
# way a real 1920x1080 frame's x/width vs y/height normalization would.
NORMALIZED_PTS_SQUARE = [(0.5 + x, 0.5 + y, z) for x, y, z in WORLD_PTS]
NORMALIZED_PTS_WIDE = [(0.5 + x * 0.3, 0.5 + y * 1.4, z) for x, y, z in WORLD_PTS]


def _install_fake_result(monkeypatch, normalized_triples, world_triples):
    fake_result = _FakeResult(normalized_triples, world_triples)
    monkeypatch.setattr(pose_estimation._landmarker, "detect_for_video", lambda *a, **k: fake_result)


def test_world_landmarks_dont_depend_on_frame_aspect_ratio(monkeypatch):
    monkeypatch.setattr(pose_estimation, "load_config", lambda: {"landmark_space": "world"})

    _install_fake_result(monkeypatch, NORMALIZED_PTS_SQUARE, WORLD_PTS)
    landmarks_square = pose_estimation.get_pose_landmarks(_dummy_frame())

    _install_fake_result(monkeypatch, NORMALIZED_PTS_WIDE, WORLD_PTS)
    landmarks_wide = pose_estimation.get_pose_landmarks(_dummy_frame())

    assert landmarks_square == landmarks_wide  # same world scene -> same result, regardless of frame shape


def test_normalized_path_reproduces_old_pose_landmarks_values(monkeypatch):
    monkeypatch.setattr(pose_estimation, "load_config", lambda: {"landmark_space": "normalized"})
    _install_fake_result(monkeypatch, NORMALIZED_PTS_WIDE, WORLD_PTS)

    landmarks = pose_estimation.get_pose_landmarks(_dummy_frame())

    expected = [(x, y, z, 1.0) for x, y, z in NORMALIZED_PTS_WIDE]
    assert landmarks == expected  # exactly result.pose_landmarks[0] -- the pre-this-change behavior


def test_landmark_space_flag_switches_which_set_is_returned(monkeypatch):
    _install_fake_result(monkeypatch, NORMALIZED_PTS_WIDE, WORLD_PTS)

    monkeypatch.setattr(pose_estimation, "load_config", lambda: {"landmark_space": "world"})
    world_result = pose_estimation.get_pose_landmarks(_dummy_frame())

    monkeypatch.setattr(pose_estimation, "load_config", lambda: {"landmark_space": "normalized"})
    normalized_result = pose_estimation.get_pose_landmarks(_dummy_frame())

    assert world_result == [(x, y, z, 1.0) for x, y, z in WORLD_PTS]
    assert normalized_result == [(x, y, z, 1.0) for x, y, z in NORMALIZED_PTS_WIDE]
    assert world_result != normalized_result


def test_unknown_landmark_space_raises(monkeypatch):
    monkeypatch.setattr(pose_estimation, "load_config", lambda: {"landmark_space": "bogus"})
    _install_fake_result(monkeypatch, NORMALIZED_PTS_WIDE, WORLD_PTS)

    import pytest
    with pytest.raises(ValueError):
        pose_estimation.get_pose_landmarks(_dummy_frame())


def test_world_angle_matches_known_90deg_abduction():
    """Sanity check the synthetic WORLD_PTS scene really is a 90 deg
    abduction angle under the SAME compute_shoulder_abduction_angle()
    the live path uses, before relying on it in the tests above."""
    from src.feature_extraction import compute_angle

    shoulder, elbow, hip = WORLD_PTS[11], WORLD_PTS[13], WORLD_PTS[23]
    angle = compute_angle(hip, shoulder, elbow)
    assert abs(angle - 90.0) < 1e-6
