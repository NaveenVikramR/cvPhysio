"""Pure helpers in scripts/landmark_fidelity_report.py."""

import math

import numpy as np
import pytest

from scripts.landmark_fidelity_report import MOCAP, fidelity, mediapipe_angles, mocap_angles
from src.feature_extraction import LEFT, RIGHT


def test_fidelity_perfect_offset():
    mocap = np.array([10.0, 20.0, 30.0, 40.0])
    stats = fidelity(mocap - 5.0, mocap, rep_ids=np.zeros(4))
    assert stats["bias"] == pytest.approx(-5.0)
    assert stats["mae"] == pytest.approx(5.0)
    assert stats["mae_debiased"] == pytest.approx(0.0)
    assert stats["r"] == pytest.approx(1.0) and stats["slope"] == pytest.approx(1.0)
    assert stats["jitter"] == pytest.approx(0.0)


def test_fidelity_drops_nan_and_handles_too_few():
    assert fidelity([1.0, np.nan], [1.0, 2.0])["n"] == 1
    assert math.isnan(fidelity([1.0, np.nan], [1.0, 2.0])["bias"])


def test_jitter_ignores_rep_boundaries():
    mocap = np.zeros(4)
    mp = np.array([0.0, 0.0, 10.0, 10.0])  # error jumps only across the rep boundary
    stats = fidelity(mp, mocap, rep_ids=np.array([0, 0, 1, 1]))
    assert stats["jitter"] == pytest.approx(0.0)
    assert math.isnan(stats["r"]) and math.isnan(stats["slope"])  # constant mocap: undefined


def _mocap_frame():
    j = np.zeros((26, 3))
    j[MOCAP["hips_root"]] = (0.0, 1.0, 0.0)
    j[MOCAP["r_hip"]] = (-0.1, 1.0, 0.0)
    j[MOCAP["l_hip"]] = (0.1, 1.0, 0.0)
    j[MOCAP["r_shoulder"]] = (-0.2, 1.5, 0.0)
    j[MOCAP["l_shoulder"]] = (0.2, 1.5, 0.0)
    j[MOCAP["r_elbow"]] = (-0.5, 1.5, 0.0)  # arm horizontal
    j[MOCAP["r_wrist"]] = (-0.8, 1.5, 0.0)  # straight elbow
    return j


def test_mocap_angles_geometry():
    a = mocap_angles(_mocap_frame())
    assert a["mocap_elbow"] == pytest.approx(180.0)
    assert a["mocap_trunk"] == pytest.approx(0.0, abs=1e-6)  # Y-up, upright
    assert 60.0 < a["mocap_shoulder"] < 120.0
    assert a["mocap_shoulder"] != pytest.approx(a["mocap_shoulder_root"])


def test_mediapipe_angles_gate_and_side():
    lm = [(0.0, 0.0, 0.0, 1.0)] * 33
    lm[RIGHT["hip"]] = (-0.1, 0.0, 0.0, 1.0)
    lm[LEFT["hip"]] = (0.1, 0.0, 0.0, 1.0)
    lm[RIGHT["shoulder"]] = (-0.2, -0.5, 0.0, 1.0)
    lm[LEFT["shoulder"]] = (0.2, -0.5, 0.0, 1.0)
    lm[RIGHT["elbow"]] = (-0.5, -0.5, 0.0, 1.0)
    lm[RIGHT["wrist"]] = (-0.8, -0.5, 0.0, 0.2)  # wrist below the gate
    out = mediapipe_angles(lm)
    assert not math.isnan(out["mp_shoulder"]) and not math.isnan(out["mp_trunk"])
    assert math.isnan(out["mp_elbow"])
    assert mediapipe_angles(None)["detected"] is False
