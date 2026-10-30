"""Tests for src/tracking.py's SessionTracker/RepRecord/DiscardRecord."""

from dataclasses import fields

from src.tracking import RepRecord, SessionTracker


def test_save_writes_a_header_even_with_zero_reps(tmp_path):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    path = tracker.save(logs_dir=tmp_path)

    assert path.exists()
    assert path.stat().st_size > 0  # not a 0-byte file

    content = path.read_text()
    lines = content.strip().splitlines()
    assert len(lines) == 1  # header only, no data rows

    expected_header = ",".join(f.name for f in fields(RepRecord))
    assert lines[0] == expected_header


def test_save_header_matches_real_rows_when_reps_exist(tmp_path):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    tracker.add_rep(RepRecord(
        exercise="shoulder_abduction", side="RIGHT", angle_deg=88.1,
        status="pass", confidence=None,
    ))
    path = tracker.save(logs_dir=tmp_path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2  # header + 1 data row
    expected_header = ",".join(f.name for f in fields(RepRecord))
    assert lines[0] == expected_header


def test_summary_handles_zero_reps_without_error(tmp_path):
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    summary = tracker.summary()
    assert summary["total_reps"] == 0
    assert summary["accuracy_pct"] is None
    assert summary["insufficient_data_rate_pct"] is None


# --- camera_info (resolution/fps actually received) ----------------------

def test_summary_omits_camera_key_when_no_camera_info_given():
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction")
    assert "camera" not in tracker.summary()


def test_summary_includes_camera_info_when_given():
    camera_info = {"width": 1280, "height": 720, "fps": 29.97}
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction", camera_info=camera_info)
    assert tracker.summary()["camera"] == camera_info


def test_camera_info_is_written_into_the_summary_json_sidecar(tmp_path):
    camera_info = {"width": 640, "height": 480, "fps": 30.0}
    tracker = SessionTracker(user_id="demo", exercise="shoulder_abduction", camera_info=camera_info)
    path = tracker.save(logs_dir=tmp_path)

    import json
    summary_path = path.with_suffix(".summary.json")
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text())
    assert saved["camera"] == camera_info
