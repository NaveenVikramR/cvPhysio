"""Pure helpers in scripts/diagnose_rotation.py, on synthetic landmarks."""

import math

import pytest

from scripts.diagnose_rotation import forearm_z_share, image_proxy, segment_summary
from src.feature_extraction import LEFT, RIGHT


def landmarks(side: dict, elbow, wrist, vis_side=1.0):
    lm = [(0.0, 0.0, 0.0, 0.0)] * 33
    other = LEFT if side is RIGHT else RIGHT
    for role in ("shoulder", "hip"):
        lm[side[role]] = (0.5, 0.2 if role == "shoulder" else 0.6, 0.0, vis_side)
        lm[other[role]] = (0.5, 0.5, 0.0, 0.0)
    lm[side["elbow"]] = (*elbow, vis_side)
    lm[side["wrist"]] = (*wrist, vis_side)
    lm[other["elbow"]] = (0.5, 0.5, 0.0, 0.0)
    lm[other["wrist"]] = (0.5, 0.5, 0.0, 0.0)
    return lm


def test_z_share_forearm_toward_camera_is_one():
    lm = landmarks(RIGHT, (0.0, 0.0, 0.0), (0.0, 0.0, -0.3))
    assert forearm_z_share(lm) == pytest.approx(1.0)


def test_z_share_forearm_in_image_plane_is_zero():
    lm = landmarks(RIGHT, (0.0, 0.0, 0.0), (0.3, 0.0, 0.0))
    assert forearm_z_share(lm) == pytest.approx(0.0)


def test_image_proxy_sign_is_outward_for_both_arms():
    # Subject faces the camera: right arm's outward is image-left (-x),
    # left arm's outward is image-right (+x).
    right_out = landmarks(RIGHT, (0.4, 0.5, 0.0), (0.3, 0.5, 0.0))
    left_out = landmarks(LEFT, (0.6, 0.5, 0.0), (0.7, 0.5, 0.0))
    assert image_proxy(right_out, 640, 480) == pytest.approx(1.0)
    assert image_proxy(left_out, 640, 480) == pytest.approx(1.0)


def test_image_proxy_uses_pixels_not_normalized_units():
    # Equal normalized dx and dy on a 2:1 frame -> 45 deg only in normalized
    # units; in pixels the horizontal part dominates.
    lm = landmarks(RIGHT, (0.5, 0.5, 0.0), (0.4, 0.6, 0.0))
    assert image_proxy(lm, 2000, 1000) == pytest.approx(2 / math.sqrt(5))


def test_image_proxy_degenerate_is_nan():
    lm = landmarks(RIGHT, (0.5, 0.5, 0.0), (0.5, 0.5, 0.2))
    assert math.isnan(image_proxy(lm, 640, 480))


def test_segment_summary_direction():
    assert segment_summary([100, 140, 105], 100.0)["direction"] == "outward(above)"
    assert segment_summary([100, 60, 95], 100.0)["direction"] == "inward(below)"
    s = segment_summary([100, 140, 105], 100.0)
    assert (s["n_frames"], s["range"]) == (3, 40.0)
