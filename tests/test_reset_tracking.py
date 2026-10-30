"""Per-clip pose tracking reset (Amendment 4). Uses the real pose model on generated
frames; the frame content doesn't matter for these checks."""

import ast
import inspect

import numpy as np

from scripts import evaluate_own_clips as eoc
from scripts import qc_clip
from src import pose_estimation


def test_reset_gives_a_new_model_instance_and_restarts_the_clock():
    frame = np.zeros((48, 64, 3), np.uint8)
    pose_estimation.get_pose_landmarks_pair(frame)
    before = pose_estimation._landmarker
    assert pose_estimation._next_timestamp_ms > 0
    pose_estimation.reset_tracking()
    assert pose_estimation._landmarker is not before
    assert pose_estimation._next_timestamp_ms == 0


def test_inference_still_works_after_a_reset():
    frame = np.zeros((48, 64, 3), np.uint8)
    pose_estimation.reset_tracking()
    assert pose_estimation.get_pose_landmarks_pair(frame) == (None, None)
    assert pose_estimation._next_timestamp_ms == 33


def test_reset_keeps_the_configured_model(monkeypatch):
    used = []
    real = pose_estimation.create_landmarker
    monkeypatch.setattr(pose_estimation, "create_landmarker", lambda model: (used.append(model), real(model))[1])
    pose_estimation.reset_tracking()
    assert used == [pose_estimation.configured_pose_model()]


def calls_reset(function) -> bool:
    tree = ast.parse(inspect.getsource(function))
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "reset_tracking"
               for n in ast.walk(tree))


def test_evaluator_and_qc_both_reset_for_every_clip():
    assert calls_reset(eoc.process_clip_reps)
    assert calls_reset(qc_clip.collect_detector_trace)
