"""Tests for src/rules.py's "reference_relative" scoring mode (STEP 2 of
the reference-scoring task) and its fallback to "absolute" mode when no
Reference is supplied, plus build_reference()/save_reference()/
load_reference()'s round trip.
"""

import pytest

from src.rules import build_reference, load_reference, save_reference, score_rep

FPS = 30.0


def make_reference_config(tol_peak_deg=10.0, tol_elbow_deg=10.0, tol_trunk_deg=10.0,
                           tol_tempo_frac=0.2, min_valid_fraction=0.7):
    return {
        "mode": "reference_relative",
        "tol_peak_deg": {"value": tol_peak_deg, "status": "derived"},
        "tol_elbow_deg": {"value": tol_elbow_deg, "status": "derived"},
        "tol_trunk_deg": {"value": tol_trunk_deg, "status": "derived"},
        "tol_tempo_frac": {"value": tol_tempo_frac, "status": "derived"},
        # absolute fallback keys -- required by score_rep() whenever
        # mode=="reference_relative" but no reference is supplied.
        "target_peak_deg": {"value": 90, "status": "provisional"},
        "tolerance_deg": {"value": 15, "status": "provisional"},
        "min_elbow_straight_deg": {"value": 150, "status": "provisional"},
        "max_trunk_lean_deg": {"value": 15, "status": "provisional"},
        "tempo_min_s": {"value": 1.0, "status": "provisional"},
        "tempo_max_s": {"value": 6.0, "status": "provisional"},
        "cue_cooldown_s": {"value": 8, "status": "provisional"},
        "min_valid_fraction": {"value": min_valid_fraction, "status": "provisional"},
        "cues": {
            "peak_angle": "peak cue", "elbow_straight": "elbow cue",
            "trunk_lean": "trunk cue", "tempo": "tempo cue",
            "insufficient_data": "visibility cue",
        },
    }


def make_reference(peak_angle=100.0, elbow_min=160.0, trunk_max=10.0, tempo_s=3.0):
    return {
        "name": "test_ref", "exercise": "shoulder_abduction", "k_reps": 3,
        "created": "2026-01-01T00:00:00+00:00",
        "medians": {"peak_angle": peak_angle, "elbow_min": elbow_min, "trunk_max": trunk_max, "tempo_s": tempo_s},
    }


def make_rep(n_frames, peak=90.0, elbow=170.0, trunk=5.0, valid_fraction=1.0):
    angle = [0.0] + [peak] * n_frames + [0.0]
    elbow_seq = [elbow] * len(angle)
    trunk_seq = [trunk] * len(angle)
    return {
        "angle_deg": angle, "elbow_angle_deg": elbow_seq, "trunk_lean_deg": trunk_seq,
        "valid_fraction": valid_fraction,
    }


# --- peak_angle: lower bound only (ref - tol), no upper bound ----------

def test_reference_relative_peak_passes_at_ref_minus_tol():
    config = make_reference_config(tol_peak_deg=10.0)
    reference = make_reference(peak_angle=100.0)
    rep = make_rep(n_frames=90, peak=90.0)  # exactly ref(100) - tol(10)
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "peak_angle" not in result["failed_rules"]
    assert result["rules"]["peak_angle"]["allowed"] == [90.0, float("inf")]


def test_reference_relative_peak_fails_below_ref_minus_tol():
    config = make_reference_config(tol_peak_deg=10.0)
    reference = make_reference(peak_angle=100.0)
    rep = make_rep(n_frames=90, peak=85.0)  # below ref(100) - tol(10) = 90
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "peak_angle" in result["failed_rules"]


def test_reference_relative_peak_has_no_upper_bound():
    config = make_reference_config(tol_peak_deg=10.0)
    reference = make_reference(peak_angle=100.0)
    rep = make_rep(n_frames=90, peak=170.0)  # way above reference -- still fine
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "peak_angle" not in result["failed_rules"]


# --- elbow_straight: lower bound only (ref - tol) -----------------------

def test_reference_relative_elbow_passes_above_ref_minus_tol():
    config = make_reference_config(tol_elbow_deg=10.0)
    reference = make_reference(elbow_min=160.0)
    rep = make_rep(n_frames=90, elbow=155.0)  # above ref(160) - tol(10) = 150
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "elbow_straight" not in result["failed_rules"]


def test_reference_relative_elbow_fails_below_ref_minus_tol():
    config = make_reference_config(tol_elbow_deg=10.0)
    reference = make_reference(elbow_min=160.0)
    rep = make_rep(n_frames=90, elbow=140.0)  # below 150
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "elbow_straight" in result["failed_rules"]


# --- trunk_lean: upper bound only (ref + tol) ---------------------------

def test_reference_relative_trunk_passes_at_or_below_ref_plus_tol():
    config = make_reference_config(tol_trunk_deg=10.0)
    reference = make_reference(trunk_max=10.0)
    rep = make_rep(n_frames=90, trunk=20.0)  # exactly ref(10) + tol(10)
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "trunk_lean" not in result["failed_rules"]
    assert result["rules"]["trunk_lean"]["allowed"] == [float("-inf"), 20.0]


def test_reference_relative_trunk_fails_above_ref_plus_tol():
    config = make_reference_config(tol_trunk_deg=10.0)
    reference = make_reference(trunk_max=10.0)
    rep = make_rep(n_frames=90, trunk=25.0)  # above 20
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "trunk_lean" in result["failed_rules"]


def test_reference_relative_trunk_has_no_lower_bound():
    config = make_reference_config(tol_trunk_deg=10.0)
    reference = make_reference(trunk_max=10.0)
    rep = make_rep(n_frames=90, trunk=0.0)  # much less lean than reference -- fine
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "trunk_lean" not in result["failed_rules"]


# --- tempo: +/- fraction of reference tempo -----------------------------

def test_reference_relative_tempo_passes_within_fraction():
    config = make_reference_config(tol_tempo_frac=0.2)
    reference = make_reference(tempo_s=3.0)  # allowed [2.4, 3.6]
    rep = make_rep(n_frames=88)  # (88+2)/30 = 3.0s
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "tempo" not in result["failed_rules"]


def test_reference_relative_tempo_fails_too_fast():
    config = make_reference_config(tol_tempo_frac=0.2)
    reference = make_reference(tempo_s=3.0)  # allowed [2.4, 3.6]
    rep = make_rep(n_frames=58)  # (58+2)/30 = 2.0s -- below 2.4
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "tempo" in result["failed_rules"]


def test_reference_relative_tempo_fails_too_slow():
    config = make_reference_config(tol_tempo_frac=0.2)
    reference = make_reference(tempo_s=3.0)  # allowed [2.4, 3.6]
    rep = make_rep(n_frames=148)  # (148+2)/30 = 5.0s -- above 3.6
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert "tempo" in result["failed_rules"]


# --- fallback to absolute mode when no reference is supplied -----------

def test_reference_relative_without_reference_falls_back_to_absolute_and_warns():
    config = make_reference_config()
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=5.0)  # passes absolute defaults
    result = score_rep(rep, config, fps=FPS, reference=None)
    assert result["warning"] is not None
    assert "falling back to absolute" in result["warning"]
    assert result["status"] == "pass"
    assert result["rules"]["peak_angle"]["allowed"] == [75, 105]  # absolute target=90 +/- tolerance=15


def test_absolute_mode_has_no_warning():
    config = make_reference_config()
    config["mode"] = "absolute"
    reference = make_reference()
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert result["warning"] is None


def test_absolute_mode_ignores_a_supplied_reference():
    """Even if a reference happens to be passed in, "absolute" mode
    must score against its own fixed thresholds, not the reference."""
    config = make_reference_config()
    config["mode"] = "absolute"
    reference = make_reference(peak_angle=9999.0)  # would fail everything if (wrongly) used
    rep = make_rep(n_frames=60, peak=90.0, elbow=170.0, trunk=5.0)
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert result["status"] == "pass"
    assert result["rules"]["peak_angle"]["allowed"] == [75, 105]


def test_reference_relative_insufficient_data_still_overrides_pass_fail():
    config = make_reference_config(min_valid_fraction=0.7)
    reference = make_reference(peak_angle=100.0, elbow_min=160.0, trunk_max=10.0, tempo_s=3.0)
    rep = make_rep(n_frames=88, peak=90.0, elbow=155.0, trunk=15.0, valid_fraction=0.5)
    result = score_rep(rep, config, fps=FPS, reference=reference)
    assert result["status"] == "insufficient_data"
    assert result["cue"] == "visibility cue"


# --- build_reference / save_reference / load_reference -----------------

def test_build_reference_takes_median_over_k_reps():
    reps = [
        make_rep(n_frames=58, peak=90.0, elbow=160.0, trunk=10.0),   # tempo (58+2)/30=2.0
        make_rep(n_frames=88, peak=100.0, elbow=150.0, trunk=20.0),  # tempo 3.0
        make_rep(n_frames=118, peak=110.0, elbow=140.0, trunk=30.0),  # tempo 4.0
    ]
    reference = build_reference("test_patient", "shoulder_abduction", reps, fps=FPS)
    assert reference["k_reps"] == 3
    assert reference["medians"]["peak_angle"] == 100.0
    assert reference["medians"]["elbow_min"] == 150.0
    assert reference["medians"]["trunk_max"] == 20.0
    assert reference["medians"]["tempo_s"] == pytest.approx(3.0)


def test_build_reference_raises_on_empty_reps():
    with pytest.raises(ValueError):
        build_reference("empty", "shoulder_abduction", [], fps=FPS)


def test_save_and_load_reference_round_trip(tmp_path):
    reps = [make_rep(n_frames=88, peak=100.0, elbow=150.0, trunk=20.0)]
    reference = build_reference("roundtrip_patient", "shoulder_abduction", reps, fps=FPS)

    saved_path = save_reference(reference, references_dir=tmp_path)
    assert saved_path == tmp_path / "roundtrip_patient.json"
    assert saved_path.exists()

    loaded = load_reference(saved_path)
    assert loaded == reference
