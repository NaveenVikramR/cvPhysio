"""Tests for scripts/evaluate_own_clips.py on SYNTHETIC data only: manifest
parser and pure metrics functions, plus clip timing on small generated mp4s
with MediaPipe stubbed out (the frame's brightness says where the arm is).
No real recording or real pose inference is exercised here.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import evaluate_own_clips as eoc
from src import rules as rules_module
from scripts.evaluate_own_clips import (
    derive_tolerances,
    fault_detection_rates,
    false_fail_rate,
    insufficient_data_rate,
    manifest_hash,
    one_sided_deviation,
    rate_with_ci,
    read_manifest,
    rep_count_metrics,
    wilson_interval,
    wrong_rules_on_correct,
)


# --- manifest parser -----------------------------------------------------

def write_manifest(tmp_path, rows):
    path = tmp_path / "own_manifest.csv"
    header = "clip,role,intended,expected_reps,subject,notes"
    lines = [header] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_manifest_parses_synthetic_rows(tmp_path):
    path = write_manifest(tmp_path, [
        ("clips/ref.mp4", "reference", "correct", "3", "demo", "ref clip"),
        ("clips/eval_lowarm.mp4", "eval", "fault_lowarm", "5", "demo", "lowarm fault"),
    ])
    rows = read_manifest(path)
    assert len(rows) == 2
    assert rows[0]["role"] == "reference"
    assert rows[0]["clip"] == "clips/ref.mp4"
    assert rows[1]["intended"] == "fault_lowarm"
    assert rows[1]["expected_reps"] == "5"


def test_read_manifest_returns_empty_list_when_missing(tmp_path):
    assert read_manifest(tmp_path / "does_not_exist.csv") == []


def test_manifest_hash_is_deterministic_and_changes_with_content(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("a,b\n1,2\n")
    h1 = manifest_hash(path)
    h2 = manifest_hash(path)
    assert h1 == h2
    path.write_text("a,b\n1,3\n")
    h3 = manifest_hash(path)
    assert h1 != h3


def test_manifest_hash_is_a_fixed_string_when_file_missing(tmp_path):
    assert manifest_hash(tmp_path / "missing.csv") == "no-manifest"


# --- wilson_interval / rate_with_ci --------------------------------------

def test_wilson_interval_zero_n_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert lo != lo and hi != hi  # nan != nan


def test_wilson_interval_bounds_are_sane():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0


def test_rate_with_ci_formats_n_zero():
    assert rate_with_ci(0, 0) == "N/A (n=0)"


def test_rate_with_ci_contains_fraction_and_percent():
    s = rate_with_ci(5, 5)
    assert "5/5" in s and "100.0%" in s


# --- one_sided_deviation --------------------------------------------------

def test_one_sided_deviation_peak_below_reference_is_positive():
    measured = {"peak_angle": 80.0}
    ref = {"peak_angle": 100.0}
    assert one_sided_deviation("peak_angle", measured, ref) == 20.0


def test_one_sided_deviation_peak_above_reference_is_zero():
    measured = {"peak_angle": 120.0}
    ref = {"peak_angle": 100.0}
    assert one_sided_deviation("peak_angle", measured, ref) == 0.0


def test_one_sided_deviation_elbow_below_reference_is_positive():
    measured = {"elbow_min": 140.0}
    ref = {"elbow_min": 150.0}
    assert one_sided_deviation("elbow_min", measured, ref) == 10.0


def test_one_sided_deviation_trunk_above_reference_is_positive():
    measured = {"trunk_max": 25.0}
    ref = {"trunk_max": 10.0}
    assert one_sided_deviation("trunk_max", measured, ref) == 15.0


def test_one_sided_deviation_trunk_below_reference_is_zero():
    measured = {"trunk_max": 5.0}
    ref = {"trunk_max": 10.0}
    assert one_sided_deviation("trunk_max", measured, ref) == 0.0


def test_one_sided_deviation_tempo_is_two_sided_fraction():
    measured = {"tempo_s": 4.0}
    ref = {"tempo_s": 5.0}
    assert one_sided_deviation("tempo_s", measured, ref) == pytest.approx(0.2)
    measured_fast = {"tempo_s": 6.0}
    assert one_sided_deviation("tempo_s", measured_fast, ref) == pytest.approx(0.2)


def test_one_sided_deviation_raises_on_unknown_feature():
    with pytest.raises(ValueError):
        one_sided_deviation("not_a_feature", {}, {})


# --- derive_tolerances ----------------------------------------------------

def make_synthetic_rep(peak=100.0, elbow=150.0, trunk=10.0, n_frames=90, fps=30.0):
    angle = [0.0] + [peak] * n_frames + [0.0]
    elbow_seq = [elbow] * len(angle)
    trunk_seq = [trunk] * len(angle)
    return {"angle_deg": angle, "elbow_angle_deg": elbow_seq, "trunk_lean_deg": trunk_seq, "fps": fps}


def test_derive_tolerances_on_synthetic_reps():
    reference_medians = {"peak_angle": 100.0, "elbow_min": 150.0, "trunk_max": 10.0, "tempo_s": 3.0}
    reps = [
        make_synthetic_rep(peak=90.0, elbow=145.0, trunk=15.0, n_frames=88),   # tempo 3.0s
        make_synthetic_rep(peak=85.0, elbow=140.0, trunk=20.0, n_frames=88),
    ]
    result = derive_tolerances(reps, reference_medians=reference_medians)
    assert result["n"] == 2
    assert result["tol_peak_deg"] >= 10.0  # p95 of [10, 15] deviations is close to 15
    assert result["tol_trunk_deg"] >= 5.0


def test_derive_tolerances_times_each_rep_at_its_own_clips_fps():
    reference_medians = {"peak_angle": 100.0, "elbow_min": 150.0, "trunk_max": 10.0, "tempo_s": 3.0}
    reps = [
        make_synthetic_rep(n_frames=88, fps=30.0),  # 90 frames / 30 fps = 3.0 s
        make_synthetic_rep(n_frames=43, fps=15.0),  # 45 frames / 15 fps = 3.0 s
    ]
    assert derive_tolerances(reps, reference_medians)["tol_tempo_frac"] == pytest.approx(0.0)


def test_derive_tolerances_empty_reps_returns_n_zero_and_none_tolerances():
    result = derive_tolerances([], reference_medians={"peak_angle": 1, "elbow_min": 1, "trunk_max": 1, "tempo_s": 1})
    assert result["n"] == 0
    assert result["tol_peak_deg"] is None


# --- rep_count_metrics -----------------------------------------------------

def test_rep_count_metrics_exact_match():
    r = rep_count_metrics(detected=10, expected=10)
    assert r["exact_match"] is True
    assert r["within_1"] is True
    assert r["signed_error"] == 0


def test_rep_count_metrics_within_one_but_not_exact():
    r = rep_count_metrics(detected=9, expected=10)
    assert r["exact_match"] is False
    assert r["within_1"] is True
    assert r["signed_error"] == -1


def test_rep_count_metrics_outside_one():
    r = rep_count_metrics(detected=7, expected=10)
    assert r["exact_match"] is False
    assert r["within_1"] is False
    assert r["signed_error"] == -3


# --- false_fail_rate / fault_detection_rates / wrong_rules_on_correct /
#     insufficient_data_rate -- all on synthetic score_rep()-shaped dicts

def result(status, failed_rules=None):
    return {"status": status, "failed_rules": failed_rules or []}


def test_false_fail_rate_counts_fail_among_scorable_only():
    correct = [result("pass"), result("fail", ["peak_angle"]), result("insufficient_data")]
    k, n = false_fail_rate(correct)
    assert (k, n) == (1, 2)  # insufficient_data excluded from the denominator


def test_fault_detection_rates_any_vs_expected_rule():
    by_intended = {
        "fault_lowarm": [
            result("fail", ["peak_angle"]),
            result("fail", ["trunk_lean"]),  # flagged, but NOT by the expected rule
            result("pass"),
        ],
    }
    rates = fault_detection_rates(by_intended)
    assert rates["fault_lowarm"]["rule_id"] == "peak_angle"
    assert rates["fault_lowarm"]["any"] == (2, 3)
    assert rates["fault_lowarm"]["expected_rule"] == (1, 3)


def test_fault_detection_rates_excludes_insufficient_data_from_denominator():
    by_intended = {"fault_trunk": [result("fail", ["trunk_lean"]), result("insufficient_data")]}
    rates = fault_detection_rates(by_intended)
    assert rates["fault_trunk"]["any"] == (1, 1)


def test_fault_detection_rates_covers_all_four_fault_types():
    rates = fault_detection_rates({})
    assert set(rates.keys()) == {"fault_lowarm", "fault_trunk", "fault_elbow", "fault_fast"}
    for r in rates.values():
        assert r["any"] == (0, 0)


def test_wrong_rules_on_correct_only_counts_failed_reps():
    correct = [result("pass"), result("fail", ["peak_angle", "tempo"]), result("fail", ["peak_angle"])]
    counts = wrong_rules_on_correct(correct)
    assert counts["peak_angle"] == 2
    assert counts["tempo"] == 1


def test_insufficient_data_rate():
    all_results = [result("pass"), result("fail"), result("insufficient_data"), result("insufficient_data")]
    k, n = insufficient_data_rate(all_results)
    assert (k, n) == (2, 4)


# --- clip fps and rep timing, on generated clips ---------------------------

REST_S = 2.0
REP_S = 2.0  # a whole number of frames at every fps tested
N_REPS = 2


def write_clip(path, fps, sidecar_fps=None, header_fps=None):
    """Rest, then N_REPS bright stretches of exactly REP_S seconds at fps.
    header_fps lets a clip's mp4 header disagree with its real frame rate."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), header_fps or fps, (64, 48))
    assert writer.isOpened()
    dark, bright = np.zeros((48, 64, 3), np.uint8), np.full((48, 64, 3), 255, np.uint8)
    for _ in range(N_REPS):
        for _ in range(round(REST_S * fps)):
            writer.write(dark)
        for _ in range(round(REP_S * fps)):
            writer.write(bright)
    for _ in range(round(REST_S * fps)):
        writer.write(dark)
    writer.release()
    if sidecar_fps is not None:
        path.with_suffix(".json").write_text(json.dumps({"measured_fps": sidecar_fps}))


def fake_landmarks(frame):
    """Arm down on a dark frame, out to the side on a bright one."""
    elbow = (0.3, 0.0, 0.0, 0.9) if frame.mean() > 127 else (0.0, 0.3, 0.0, 0.9)
    points = [(0.0, 0.0, 0.0, 0.9)] * 33
    for i, point in {11: (0.0, 0.0, 0.0, 0.9), 12: (-0.3, 0.0, 0.0, 0.9), 13: elbow, 14: (-0.3, 0.3, 0.0, 0.9),
                     15: (elbow[0] * 2, elbow[1] * 2, 0.0, 0.9), 16: (-0.3, 0.6, 0.0, 0.9),
                     23: (0.0, 0.5, 0.0, 0.9), 24: (-0.3, 0.5, 0.0, 0.9)}.items():
        points[i] = point
    return points


@pytest.fixture
def stub_pose(monkeypatch):
    monkeypatch.setattr(eoc.pose_estimation, "get_pose_landmarks", fake_landmarks)
    monkeypatch.setattr(eoc, "build_smoother", lambda: None)


@pytest.mark.parametrize("fps", [15.0, 30.0])
def test_measured_tempo_matches_the_known_duration_at_each_fps(tmp_path, stub_pose, fps):
    clip = tmp_path / "clip.mp4"
    write_clip(clip, fps, sidecar_fps=fps)
    reps, clip_fps, _processing_fps = eoc.process_clip_reps(clip, eoc.EXERCISE)
    assert clip_fps == fps
    assert len(reps) == N_REPS
    for rep in reps:
        tempo = rules_module.rep_summary_features(rep, rep["fps"])["tempo_s"]
        # The rep buffer also holds the frame that returns to rest.
        assert tempo == pytest.approx(REP_S + 1 / fps, abs=1e-9)


def test_tempo_does_not_depend_on_processing_speed(tmp_path, stub_pose, monkeypatch):
    clip = tmp_path / "clip.mp4"
    write_clip(clip, 30.0, sidecar_fps=30.0)
    fast = eoc.process_clip_reps(clip, eoc.EXERCISE)

    def slow_landmarks(frame):
        time.sleep(0.002)
        return fake_landmarks(frame)

    monkeypatch.setattr(eoc.pose_estimation, "get_pose_landmarks", slow_landmarks)
    slow = eoc.process_clip_reps(clip, eoc.EXERCISE)
    assert slow[2] < fast[2]  # genuinely different processing speeds
    assert slow[1] == fast[1] == 30.0


def test_sidecar_fps_wins_over_the_video_header(tmp_path, stub_pose, capsys):
    clip = tmp_path / "clip.mp4"
    write_clip(clip, 15.0, sidecar_fps=15.0, header_fps=30.0)
    reps, clip_fps, _processing_fps = eoc.process_clip_reps(clip, eoc.EXERCISE)
    assert clip_fps == 15.0
    assert reps[0]["fps"] == 15.0
    assert "clip fps used = 15.000 (from sidecar)" in capsys.readouterr().out


@pytest.mark.parametrize("sidecar_text", [None, "not json", "{}", '{"measured_fps": 0}', '{"measured_fps": null}'])
def test_falls_back_to_the_video_fps_without_a_usable_sidecar(tmp_path, sidecar_text):
    clip = tmp_path / "clip.mp4"
    write_clip(clip, 20.0)
    if sidecar_text is not None:
        clip.with_suffix(".json").write_text(sidecar_text)
    cap = cv2.VideoCapture(str(clip))
    fps, source = eoc.resolve_clip_fps(clip, cap)
    cap.release()
    assert fps == pytest.approx(20.0)
    assert source == "video"


def test_no_usable_fps_raises_instead_of_guessing(tmp_path):
    class NoFps:
        def get(self, _prop):
            return 0.0

    with pytest.raises(RuntimeError, match="No usable fps"):
        eoc.resolve_clip_fps(tmp_path / "clip.mp4", NoFps())


def test_main_times_reference_calibration_and_eval_at_each_clips_own_fps(tmp_path, stub_pose, monkeypatch, capsys):
    clips = {"ref.mp4": 15.0, "cal.mp4": 30.0, "ev.mp4": 20.0}
    for name, fps in clips.items():
        write_clip(tmp_path / name, fps, sidecar_fps=fps)
    # N_REFERENCE_REPS is 3 but each generated clip holds N_REPS.
    monkeypatch.setattr(eoc, "N_REFERENCE_REPS", N_REPS)
    manifest = tmp_path / "own_manifest.csv"
    manifest.write_text(
        "clip,role,intended,expected_reps,subject,notes\n"
        f"{(tmp_path / 'ref.mp4').as_posix()},reference,correct,{N_REPS},s,\n"
        f"{(tmp_path / 'cal.mp4').as_posix()},calibration,correct,{N_REPS},s,\n"
        f"{(tmp_path / 'ev.mp4').as_posix()},eval,correct,{N_REPS},s,\n"
    )
    monkeypatch.setattr(eoc, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(eoc, "RESULTS_DIR", tmp_path / "results")

    build_reference_fps, score_fps = [], []
    real_build, real_score = rules_module.build_reference, rules_module.score_rep

    def spy_build(name, exercise, reps, fps, *args):
        build_reference_fps.append(fps)
        return real_build(name, exercise, reps, fps, *args)

    def spy_score(rep, config, fps, reference=None):
        score_fps.append(fps)
        return real_score(rep, config, fps, reference=reference)

    monkeypatch.setattr(rules_module, "build_reference", spy_build)
    monkeypatch.setattr(rules_module, "score_rep", spy_score)

    eoc.main()

    assert build_reference_fps == [15.0]
    assert score_fps == [20.0] * (2 * N_REPS)  # both configs, per eval rep
    out = capsys.readouterr().out
    for name, fps in clips.items():
        assert f"{name}: clip fps used = {fps:.3f} (from sidecar)" in out
