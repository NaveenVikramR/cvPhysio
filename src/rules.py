"""Rule-based live scorer. Checks measured per-rep features against
physiotherapist-tunable thresholds in config/exercises/{exercise}.json --
no model, no training data, no UI-PRMD dependency. Built because
scripts/diagnose_misclassification.py showed the UI-PRMD-trained
classifier (src/classifier.py) is systematically biased on live webcam
reps; this module is the default scorer for app.py instead. Must not
import classifier.py or pose_estimation.py (module independence).

Two scoring modes, selected by config["mode"]:
  "absolute" (default, omitted "mode" key means this too): target_peak_deg
    +/- tolerance_deg, min_elbow_straight_deg, max_trunk_lean_deg,
    tempo_min_s/tempo_max_s -- fixed thresholds, same for every patient.
  "reference_relative": tol_peak_deg/tol_elbow_deg/tol_trunk_deg/
    tol_tempo_frac, applied relative to a Reference (median peak_angle/
    elbow_min/trunk_max/tempo_s over K reps -- see build_reference()/
    save_reference()/load_reference() below). Falls back to "absolute"
    (using the SAME config's absolute keys) if no reference is supplied
    -- score_rep() reports this fallback via the returned "warning" key,
    it never raises for a missing reference.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.statuses import RuleStatus

CONFIG_DIR = Path("config/exercises")
REFERENCE_DIR = Path("data/references")
DEFAULT_REFERENCE_K = 3

RULE_IDS = ["peak_angle", "elbow_straight", "trunk_lean", "tempo"]

# config["elbow_statistic"]: what elbow_straight measures per rep. "min" (the
# default, also when the key is absent) is the worst gated frame anywhere in
# the rep. "hold_median" is the median around the peak: steadier against a
# one-frame glitch, but blind to a bend outside the hold window.
ELBOW_STATISTICS = ("min", "hold_median")
HOLD_WINDOW_S = 0.3

# Which joint roles each rule's underlying feature depends on -- the
# single source of truth for what the confidence gate must check before
# a frame is trusted for scoring. feature_extraction.check_confidence()
# takes required_joint_roles() instead of re-hardcoding this list, so
# the two can't drift apart the way they did before (elbow_straight
# depends on compute_elbow_angle(), which needs the wrist, but the old
# confidence gate for shoulder_abduction never checked it).
REQUIRED_ROLES_BY_RULE = {
    "peak_angle": {"shoulder", "elbow", "hip"},
    "elbow_straight": {"shoulder", "elbow", "wrist"},
    "trunk_lean": {"shoulder", "hip"},
    "tempo": set(),
}


def required_joint_roles() -> set[str]:
    """Union of every active rule's required joint roles."""
    roles = set()
    for rule_id in RULE_IDS:
        roles |= REQUIRED_ROLES_BY_RULE[rule_id]
    return roles


def load_config(exercise: str) -> dict:
    path = CONFIG_DIR / f"{exercise}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No rules config for exercise '{exercise}' at {path} -- "
            f"only exercises with a config/exercises/{{exercise}}.json file "
            f"can use --scorer rules."
        )
    return json.loads(path.read_text())


def elbow_statistic(config: dict) -> str:
    """The per-rep elbow statistic this config selects (default "min")."""
    statistic = config.get("elbow_statistic", "min")
    if statistic not in ELBOW_STATISTICS:
        raise ValueError(f"elbow_statistic={statistic!r}; must be one of {ELBOW_STATISTICS}")
    return statistic


def hold_window_elbow(rep: dict, fps: float) -> float:
    """Median elbow angle over the rep's gated frames within HOLD_WINDOW_S of
    its peak shoulder angle. Times come from rep["frame_t"] when present,
    because gated-out frames leave gaps; otherwise frame index / fps."""
    angle = rep["angle_deg"]
    elbow = rep["elbow_angle_deg"]
    frame_t = rep.get("frame_t")
    times = frame_t if frame_t and len(frame_t) == len(angle) else [i / fps for i in range(len(angle))]
    peak_t = times[max(range(len(angle)), key=angle.__getitem__)]
    # Small slack so a frame exactly on the window's edge isn't lost to float rounding.
    return statistics.median(e for e, t in zip(elbow, times) if abs(t - peak_t) <= HOLD_WINDOW_S + 1e-9)


def rep_elbow_value(rep: dict, fps: float, statistic: str = "min") -> float:
    if statistic == "hold_median":
        return hold_window_elbow(rep, fps)
    return min(rep["elbow_angle_deg"])


def active_scoring_mode(config: dict, reference: dict | None) -> str:
    """The mode score_rep() actually applies: a reference only takes effect
    when the config itself is in reference_relative mode."""
    if config.get("mode", "absolute") == "reference_relative" and reference is not None:
        return "reference_relative"
    return "absolute"


def reference_ignored_message(config: dict, reference: dict | None) -> str | None:
    """Warning text when a reference was selected but won't affect scoring,
    otherwise None."""
    if reference is None or active_scoring_mode(config, reference) == "reference_relative":
        return None
    return (f"Reference '{reference['name']}' is selected but NOT applied: this exercise's config scores in "
            "absolute mode, so reps are scored against its fixed thresholds.")


def _value(config: dict, key: str) -> float:
    return config[key]["value"]


def _absolute_bounds(config: dict) -> dict:
    target_peak = _value(config, "target_peak_deg")
    tolerance = _value(config, "tolerance_deg")
    min_elbow = _value(config, "min_elbow_straight_deg")
    max_trunk = _value(config, "max_trunk_lean_deg")
    tempo_min = _value(config, "tempo_min_s")
    tempo_max = _value(config, "tempo_max_s")
    return {
        "peak_angle": (target_peak - tolerance, target_peak + tolerance),
        "elbow_straight": (min_elbow, 180.0),
        "trunk_lean": (0.0, max_trunk),
        "tempo": (tempo_min, tempo_max),
    }


def _reference_relative_bounds(config: dict, reference: dict) -> dict:
    medians = reference["medians"]
    tol_peak = _value(config, "tol_peak_deg")
    tol_elbow = _value(config, "tol_elbow_deg")
    tol_trunk = _value(config, "tol_trunk_deg")
    tol_tempo_frac = _value(config, "tol_tempo_frac")
    ref_tempo = medians["tempo_s"]
    return {
        # peak/elbow: lower bound only (no upper bound -- more range of
        # motion than the reference is never penalized here).
        "peak_angle": (medians["peak_angle"] - tol_peak, float("inf")),
        "elbow_straight": (medians["elbow_min"] - tol_elbow, float("inf")),
        # trunk: upper bound only (less lean than the reference is fine).
        "trunk_lean": (float("-inf"), medians["trunk_max"] + tol_trunk),
        "tempo": (ref_tempo * (1 - tol_tempo_frac), ref_tempo * (1 + tol_tempo_frac)),
    }


def score_rep(rep: dict, config: dict, fps: float, reference: dict | None = None) -> dict:
    """rep: {"angle_deg": [...], "elbow_angle_deg": [...], "trunk_lean_deg":
    [...], "valid_fraction": float}. The three feature lists are one
    value per VALID (confidence-gated) frame of the completed repetition,
    all equal length -- frames below the visibility gate are excluded
    before this is called (see app.py's process_frame()).
    valid_fraction is valid_frames / total_frames_in_the_rep's_span
    (including frames the gate rejected), supplied by the caller since
    only it tracks the rejected frames.
    fps: frames/sec of the source video or webcam, used to convert the
    rep's frame count into a tempo in seconds.
    reference: a Reference dict (see build_reference()/load_reference()),
    used only when config["mode"] == "reference_relative". Ignored
    entirely in "absolute" mode (the default). If mode is
    "reference_relative" but reference is None, falls back to this same
    config's absolute keys (target_peak_deg etc.) and reports that via
    the returned "warning" key -- never raises for a missing reference.

    Returns {"status": "pass" | "fail" | "insufficient_data",
    "failed_rules": [rule_id, ...], "rules": {rule_id: {"measured":
    float, "allowed": [lo, hi], "passed": bool}}, "valid_fraction": float,
    "cue": str | None, "warning": str | None}. Rules are still computed
    even when status is "insufficient_data" (useful for calibration-set
    reporting), but that status itself must never be read as a pass or
    a fail.
    """
    angle = rep["angle_deg"]
    elbow = rep["elbow_angle_deg"]
    trunk = rep["trunk_lean_deg"]
    valid_fraction = rep["valid_fraction"]
    if not (len(angle) == len(elbow) == len(trunk)) or len(angle) == 0:
        raise ValueError("rep's angle_deg/elbow_angle_deg/trunk_lean_deg must be equal-length and non-empty")

    min_valid_fraction = _value(config, "min_valid_fraction")
    mode = config.get("mode", "absolute")

    statistic = elbow_statistic(config)
    warning = None
    if mode == "reference_relative" and reference is not None:
        bounds = _reference_relative_bounds(config, reference)
        if reference.get("elbow_statistic", "min") != statistic:
            warning = (f"Reference elbow value was built with {reference.get('elbow_statistic', 'min')!r} but this "
                       f"config scores with {statistic!r} -- re-record the reference.")
    else:
        if mode == "reference_relative" and reference is None:
            warning = "reference_relative mode requested but no reference was supplied -- falling back to absolute thresholds."
        bounds = _absolute_bounds(config)

    peak = max(angle)
    elbow_value = rep_elbow_value(rep, fps, statistic)
    worst_trunk = max(trunk)
    duration_s = len(angle) / fps
    measured = {"peak_angle": peak, "elbow_straight": elbow_value, "trunk_lean": worst_trunk, "tempo": duration_s}

    rules = {
        rule_id: {
            "measured": measured[rule_id],
            "allowed": [lo, hi],
            "passed": lo <= measured[rule_id] <= hi,
        }
        for rule_id, (lo, hi) in bounds.items()
    }

    failed_rules = [rule_id for rule_id, r in rules.items() if not r["passed"]]

    if valid_fraction < min_valid_fraction:
        return {
            "status": RuleStatus.INSUFFICIENT_DATA.value, "failed_rules": failed_rules, "rules": rules,
            "valid_fraction": valid_fraction, "cue": config["cues"].get("insufficient_data"), "warning": warning,
        }
    return {
        "status": (RuleStatus.PASS if not failed_rules else RuleStatus.FAIL).value,
        "failed_rules": failed_rules, "rules": rules,
        "valid_fraction": valid_fraction, "cue": None, "warning": warning,
    }


# ---------------------------------------------------------------------------
# Reference: median peak_angle/elbow_min/trunk_max/tempo_s over K reps,
# used by "reference_relative" mode above.
# ---------------------------------------------------------------------------

def rep_summary_features(rep: dict, fps: float, statistic: str = "min") -> dict:
    """peak_angle/elbow_min/trunk_max/tempo_s for one rep -- the same
    per-rep aggregates score_rep() computes internally, exposed here so
    a Reference can be built from K real reps' own measurements. The
    "elbow_min" key holds whichever elbow statistic was asked for."""
    return {
        "peak_angle": max(rep["angle_deg"]), "elbow_min": rep_elbow_value(rep, fps, statistic),
        "trunk_max": max(rep["trunk_lean_deg"]), "tempo_s": len(rep["angle_deg"]) / fps,
    }


def build_reference(name: str, exercise: str, reps: list[dict], fps: float, statistic: str = "min") -> dict:
    """Median of rep_summary_features() over `reps` (K repetitions --
    e.g. captured once at the start of a patient's session, or a
    therapist's own demonstration). A rep carrying its own "fps" is timed
    with that (live reps: see src.pipeline.live_rep_fps()), otherwise with
    fps. Raises ValueError if reps is empty -- a reference needs at least
    one rep to be meaningful."""
    if not reps:
        raise ValueError("build_reference() needs at least one rep")
    summaries = [rep_summary_features(rep, rep.get("fps", fps), statistic) for rep in reps]
    medians = {
        col: float(statistics.median(s[col] for s in summaries))
        for col in ("peak_angle", "elbow_min", "trunk_max", "tempo_s")
    }
    return {
        "name": name, "exercise": exercise, "k_reps": len(reps),
        "created": datetime.now(timezone.utc).isoformat(), "medians": medians, "elbow_statistic": statistic,
    }


def save_reference(reference: dict, references_dir: Path = REFERENCE_DIR) -> Path:
    references_dir.mkdir(parents=True, exist_ok=True)
    path = references_dir / f"{reference['name']}.json"
    path.write_text(json.dumps(reference, indent=2))
    return path


def load_reference(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


class CueCooldown:
    """Tracks the last time each cue id fired so the same cue isn't
    repeated more than once per cooldown_s seconds."""

    def __init__(self, cooldown_s: float):
        self.cooldown_s = cooldown_s
        self._last_fired: dict[str, float] = {}

    def should_fire(self, cue_id: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        last = self._last_fired.get(cue_id)
        return last is None or (now - last) >= self.cooldown_s

    def record(self, cue_id: str, now: float | None = None) -> None:
        self._last_fired[cue_id] = time.monotonic() if now is None else now

    def cues_to_speak(self, failed_rules: list[str], cue_text: dict, now: float | None = None) -> list[str]:
        """Given a rep's failed rule ids, return the cue strings allowed
        to fire right now (not in cooldown) and mark them as fired."""
        now = time.monotonic() if now is None else now
        messages = []
        for rule_id in failed_rules:
            if rule_id in cue_text and self.should_fire(rule_id, now):
                messages.append(cue_text[rule_id])
                self.record(rule_id, now)
        return messages
