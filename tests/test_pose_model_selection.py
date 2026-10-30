"""The pose_model config key and scripts/compare_pose_models.py's subject guard
and metrics, on synthetic inputs. No REHAB24 data needed."""

import json

import pandas as pd
import pytest

from scripts import compare_pose_models as cpm
from src import pose_estimation

EVALUATION_SUBJECTS = {3, 4, 5, 7, 8, 9}


def write_config(tmp_path, **keys):
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(keys))
    return path


def test_shipped_default_is_lite():
    assert pose_estimation.configured_pose_model() == "lite"


def test_missing_key_means_lite(tmp_path):
    assert pose_estimation.configured_pose_model(write_config(tmp_path)) == "lite"


def test_heavy_can_be_selected(tmp_path):
    assert pose_estimation.configured_pose_model(write_config(tmp_path, pose_model="heavy")) == "heavy"


def test_unknown_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pose_model"):
        pose_estimation.configured_pose_model(write_config(tmp_path, pose_model="full"))


def test_model_path_and_url_follow_the_name():
    assert pose_estimation.model_path("heavy").name == "pose_landmarker_heavy.task"
    assert pose_estimation.model_url("heavy").endswith("pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def test_missing_model_file_error_names_the_download(monkeypatch, tmp_path):
    monkeypatch.setattr(pose_estimation, "model_path", lambda model: tmp_path / "absent.task")
    with pytest.raises(FileNotFoundError, match="pose_landmarker_heavy.task"):
        pose_estimation.create_landmarker("heavy")


def test_comparison_selects_only_calibration_front_reps():
    manifest = pd.DataFrame([
        {"subject": s, "view": v} for s in range(1, 11) for v in ("front", "profile")
    ])
    selected = cpm.calibration_front_reps(manifest)
    assert sorted(selected["subject"]) == [1, 2]
    assert set(selected["view"]) == {"front"}
    assert not EVALUATION_SUBJECTS & set(selected["subject"])
    assert not EVALUATION_SUBJECTS & set(cpm.CALIBRATION_SUBJECTS)


def test_fidelity_on_a_known_relationship():
    mocap = [0.0, 30.0, 60.0, 90.0]
    result = cpm.fidelity([2.0 * m + 10.0 for m in mocap], mocap)
    assert result["slope"] == pytest.approx(2.0)
    assert result["r"] == pytest.approx(1.0)
    assert result["mae"] == pytest.approx(55.0)
    assert result["n_frames"] == 4


def test_fidelity_with_too_few_frames_is_nan_not_an_error():
    result = cpm.fidelity([1.0], [1.0])
    assert result["n_frames"] == 1
    assert result["mae"] != result["mae"]  # NaN


def test_table_rows_drop_unpaired_frames_per_joint():
    frames = pd.DataFrame([
        {"subject": 1, "shoulder_mp": 10.0, "shoulder_mocap": 12.0, "elbow_mp": None, "elbow_mocap": None},
        {"subject": 1, "shoulder_mp": 20.0, "shoulder_mocap": 22.0, "elbow_mp": 150.0, "elbow_mocap": 160.0},
        {"subject": 2, "shoulder_mp": 30.0, "shoulder_mocap": 32.0, "elbow_mp": 170.0, "elbow_mocap": 165.0},
    ])
    rows = {(r["subjects"], r["angle"]): r for r in cpm.table_rows("lite", frames, 25.0)}
    assert rows[("1+2 pooled", "shoulder")]["n_frames"] == 3
    assert rows[("1+2 pooled", "elbow")]["n_frames"] == 2
    assert rows[("1", "elbow")]["n_frames"] == 1
    assert rows[("1+2 pooled", "shoulder")]["inference_fps"] == 25.0
