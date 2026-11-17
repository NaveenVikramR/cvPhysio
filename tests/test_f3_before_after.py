"""scripts/f3_before_after.py summaries and benchmark_rehab24.recording_view
(recordings holding two blocks of reps from different views)."""

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_rehab24 import recording_view
from scripts.f3_before_after import angle_summary, count_summary, pipelines
from src.feature_extraction import load_angle_correction


def test_recording_view_labels_mixed_recordings():
    m = pd.DataFrame({"video_id": ["A"] * 3 + ["B"] * 2, "camera": ["cam17"] * 5,
                      "view": ["half-profile", "front", "front", "front", "front"]})
    assert recording_view(m, "A", "cam17") == "front+half-profile"  # ordered, not by first rep
    assert recording_view(m, "B", "cam17") == "front"


def test_pipelines_are_f1_and_current_app():
    p = pipelines()
    assert p["before"] == {"side": "per_frame", "correction": None}
    assert p["after"] == {"side": "auto", "correction": load_angle_correction()}


def result(frames, detected, annotated, mixed=0, rid="r1"):
    return {"rec": {"id": rid, "annotated": annotated}, "frames": pd.DataFrame(frames),
            "detected": detected, "mixed": mixed}


def test_angle_summary_uses_each_frames_view_and_right_arm_only():
    frames = [
        {"frame": 0, "rep": 1, "view": "front", "angle": 50.0, "arm": "RIGHT", "truth": 55.0},
        {"frame": 1, "rep": 1, "view": "front", "angle": 60.0, "arm": "RIGHT", "truth": 65.0},
        {"frame": 2, "rep": 1, "view": "front", "angle": 70.0, "arm": "RIGHT", "truth": 75.0},
        {"frame": 3, "rep": 1, "view": "front", "angle": 10.0, "arm": "LEFT", "truth": 80.0},
        {"frame": 4, "rep": 2, "view": "half-profile", "angle": 0.0, "arm": "RIGHT", "truth": 90.0},
    ]
    a = angle_summary([result(frames, 1, 1)], "front")
    assert a["frames"] == 4 and a["wrong_arm"] == pytest.approx(0.25)
    assert a["bias"] == pytest.approx(-5.0) and a["mae_debiased"] == pytest.approx(0.0)


def test_count_summary():
    rs = [result([], 10, 10, mixed=1, rid="a"), result([], 12, 10, rid="b"), result([], 9, 10, rid="c")]
    c = count_summary(rs)
    assert (c["recordings"], c["exact"], c["abs_error"], c["mixed"], c["reps"]) == (3, 1, 3, 1, 31)
    assert c["mean_signed"] == pytest.approx(np.mean([0, 2, -1]))
