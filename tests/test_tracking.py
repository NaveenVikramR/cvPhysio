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
