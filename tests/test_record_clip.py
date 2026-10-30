"""Tests for scripts/record_clip.py on synthetic inputs -- no camera, and
every path is under tmp_path, so data/recordings/ and data/own_manifest.csv
are never touched."""

import argparse
import csv
import json

import cv2
import numpy as np
import pytest

from scripts import record_clip as rc
from scripts.evaluate_own_clips import read_manifest


def make_args(**overrides):
    values = dict(role="eval", intended="fault_trunk", expected_reps=3, subject="demo",
                  notes="window light", camera=0, seconds=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def make_landmarks(x=0.5, y=0.5, visibility=0.9):
    return [(x, y, 0.0, visibility)] * 33


def write_spool(path, n_frames, width=64, height=48):
    with path.open("wb") as f:
        for i in range(n_frames):
            f.write(np.full((height, width, 3), i * 5, np.uint8).tobytes())


# --- naming and auto-increment ---------------------------------------------

def test_clip_stem_follows_the_protocol_naming():
    assert rc.clip_stem("reference", "correct", 1) == "abd_reference_correct_01"
    assert rc.clip_stem("eval", "fault_lowarm", 12) == "abd_eval_fault_lowarm_12"


def test_next_index_starts_at_one_when_empty(tmp_path):
    assert rc.next_clip_index(tmp_path, "eval", "correct") == 1


def test_next_index_is_one_past_the_highest_and_never_refills_a_gap(tmp_path):
    (tmp_path / "abd_eval_correct_01.mp4").touch()
    (tmp_path / "abd_eval_correct_04.mp4").touch()
    assert rc.next_clip_index(tmp_path, "eval", "correct") == 5


def test_next_index_counts_an_orphan_sidecar(tmp_path):
    (tmp_path / "abd_eval_correct_07.json").touch()
    assert rc.next_clip_index(tmp_path, "eval", "correct") == 8


def test_next_index_is_per_role_and_intended(tmp_path):
    (tmp_path / "abd_eval_correct_03.mp4").touch()
    (tmp_path / "abd_calibration_correct_09.mp4").touch()
    (tmp_path / "abd_eval_fault_fast_05.mp4").touch()
    (tmp_path / "abd_eval_correct_notes.mp4").touch()
    assert rc.next_clip_index(tmp_path, "eval", "correct") == 4
    assert rc.next_clip_index(tmp_path, "eval", "fault_trunk") == 1


def test_next_index_goes_past_two_digits(tmp_path):
    (tmp_path / "abd_eval_correct_99.mp4").touch()
    assert rc.clip_stem("eval", "correct", rc.next_clip_index(tmp_path, "eval", "correct")) == "abd_eval_correct_100"


# --- manifest --------------------------------------------------------------

ROW = {"clip": "data/recordings/a.mp4", "role": "eval", "intended": "correct",
       "expected_reps": 6, "subject": "demo", "notes": "lamp, red, sleeve"}


def test_manifest_created_with_header_and_readable_by_the_evaluator(tmp_path):
    path = tmp_path / "sub" / "own_manifest.csv"
    rc.append_manifest_row(path, ROW)
    assert path.read_text().splitlines()[0] == "clip,role,intended,expected_reps,subject,notes"
    assert read_manifest(path) == [{**ROW, "expected_reps": "6"}]


def test_manifest_appends_without_repeating_the_header(tmp_path):
    path = tmp_path / "own_manifest.csv"
    rc.append_manifest_row(path, ROW)
    rc.append_manifest_row(path, {**ROW, "clip": "data/recordings/b.mp4"})
    rows = read_manifest(path)
    assert [r["clip"] for r in rows] == ["data/recordings/a.mp4", "data/recordings/b.mp4"]


def test_manifest_append_survives_a_missing_final_newline(tmp_path):
    path = tmp_path / "own_manifest.csv"
    path.write_text("clip,role,intended,expected_reps,subject,notes\nx.mp4,eval,correct,1,s,n")
    rc.append_manifest_row(path, ROW)
    assert [r["clip"] for r in read_manifest(path)] == ["x.mp4", "data/recordings/a.mp4"]


def test_manifest_with_a_different_header_is_refused_and_left_unchanged(tmp_path):
    path = tmp_path / "own_manifest.csv"
    path.write_text("clip,role\n")
    with pytest.raises(ValueError, match="expected"):
        rc.append_manifest_row(path, ROW)
    assert path.read_text() == "clip,role\n"


# --- saving a take ---------------------------------------------------------

def test_measured_fps_uses_frame_intervals():
    assert rc.measured_fps([10.0, 10.5, 11.0, 11.5, 12.0]) == pytest.approx(2.0)


@pytest.mark.parametrize("timestamps", [[], [1.0], [1.0, 1.0]])
def test_measured_fps_refuses_unmeasurable_input(timestamps):
    with pytest.raises(ValueError):
        rc.measured_fps(timestamps)


def test_save_take_writes_clip_sidecar_and_manifest_at_the_measured_fps(tmp_path):
    recordings = tmp_path / "data" / "recordings"
    recordings.mkdir(parents=True)
    manifest = tmp_path / "data" / "own_manifest.csv"
    spool = recordings / "spool.raw"
    write_spool(spool, 25)
    timestamps = [i / 12.5 for i in range(25)]

    video = rc.save_take(spool, timestamps, 64, 48, make_args(), "2026-10-30T10:00:00+05:30",
                         recordings_dir=recordings, manifest_path=manifest, root=tmp_path)

    assert video.name == "abd_eval_fault_trunk_01.mp4"
    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(12.5, abs=0.01)
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 25
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (64, 48)
    cap.release()

    sidecar = json.loads((recordings / "abd_eval_fault_trunk_01.json").read_text())
    assert sidecar["measured_fps"] == pytest.approx(12.5)
    assert sidecar["resolution"] == [64, 48]
    assert sidecar["frames"] == 25
    assert sidecar["start_time"] == "2026-10-30T10:00:00+05:30"
    assert sidecar["camera_index"] == 0
    assert sidecar["notes"] == "window light"

    with manifest.open(newline="") as f:
        assert list(csv.DictReader(f)) == [{
            "clip": "data/recordings/abd_eval_fault_trunk_01.mp4", "role": "eval", "intended": "fault_trunk",
            "expected_reps": "3", "subject": "demo", "notes": "window light",
        }]


def test_save_take_adds_no_manifest_row_when_frames_are_lost(tmp_path):
    spool = tmp_path / "spool.raw"
    write_spool(spool, 5)
    manifest = tmp_path / "own_manifest.csv"
    with pytest.raises(RuntimeError, match="captured 6"):
        rc.save_take(spool, [i / 10 for i in range(6)], 64, 48, make_args(), "t",
                     recordings_dir=tmp_path, manifest_path=manifest, root=tmp_path)
    assert not manifest.exists()


# --- preview warnings ------------------------------------------------------

def test_wrist_warning_only_below_the_threshold():
    landmarks = make_landmarks()
    assert rc.low_wrist_warnings(landmarks) == []
    landmarks[15] = (0.5, 0.5, 0.0, 0.49)
    assert rc.low_wrist_warnings(landmarks) == ["LOW VISIBILITY: left wrist 0.49"]


def test_no_pose_is_a_warning():
    assert rc.low_wrist_warnings(None) == ["NO POSE DETECTED"]


def test_outside_box_catches_an_off_image_joint_with_high_visibility():
    landmarks = make_landmarks()
    assert rc.outside_box(landmarks) == []
    landmarks[23] = (0.5, 1.4, 0.0, 0.95)
    assert rc.outside_box(landmarks) == ["left hip"]


def test_draw_preview_leaves_the_recorded_frame_clean():
    frame = np.zeros((48, 64, 3), np.uint8)
    view = rc.draw_preview(frame, make_landmarks(), "status", (255, 255, 255))
    assert view.any()
    assert not frame.any()


# --- args ------------------------------------------------------------------

BASE_ARGS = ["--role", "eval", "--intended", "correct", "--expected-reps", "6", "--subject", "demo"]


def test_args_defaults():
    args = rc.parse_args(BASE_ARGS)
    assert args.camera == rc.DEFAULT_CAMERA_INDEX
    assert args.seconds is None
    assert args.notes == ""


@pytest.mark.parametrize("bad", [
    ["--role", "reference", "--intended", "fault_fast", "--expected-reps", "3", "--subject", "demo"],
    ["--role", "other", "--intended", "correct", "--expected-reps", "3", "--subject", "demo"],
    BASE_ARGS + ["--seconds", "0"],
    ["--role", "eval", "--intended", "correct", "--expected-reps", "0", "--subject", "demo"],
    BASE_ARGS + ["--notes", "café"],
])
def test_args_refused(bad):
    with pytest.raises(SystemExit):
        rc.parse_args(bad)
