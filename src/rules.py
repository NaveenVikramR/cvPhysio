"""Rule-based live scorer. Checks measured per-rep features against
physiotherapist-tunable thresholds in config/exercises/{exercise}.json --
no model, no training data, no UI-PRMD dependency. Built because
scripts/diagnose_misclassification.py showed the UI-PRMD-trained
classifier (src/classifier.py) is systematically biased on live webcam
reps; this module is the default scorer for app.py instead. Must not
import classifier.py or pose_estimation.py (module independence).
"""

import json
import time
from pathlib import Path

CONFIG_DIR = Path("config/exercises")

RULE_IDS = ["peak_angle", "elbow_straight", "trunk_lean", "tempo"]

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


def _value(config: dict, key: str) -> float:
    return config[key]["value"]


def score_rep(rep: dict, config: dict, fps: float) -> dict:
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

    Returns {"status": "pass" | "fail" | "insufficient_data",
    "failed_rules": [rule_id, ...], "rules": {rule_id: {"measured":
    float, "allowed": [lo, hi], "passed": bool}}, "valid_fraction": float,
    "cue": str | None}. Rules are still computed even when status is
    "insufficient_data" (useful for calibration-set reporting), but that
    status itself must never be read as a pass or a fail.
    """
    angle = rep["angle_deg"]
    elbow = rep["elbow_angle_deg"]
    trunk = rep["trunk_lean_deg"]
    valid_fraction = rep["valid_fraction"]
    if not (len(angle) == len(elbow) == len(trunk)) or len(angle) == 0:
        raise ValueError("rep's angle_deg/elbow_angle_deg/trunk_lean_deg must be equal-length and non-empty")

    target_peak = _value(config, "target_peak_deg")
    tolerance = _value(config, "tolerance_deg")
    min_elbow = _value(config, "min_elbow_straight_deg")
    max_trunk = _value(config, "max_trunk_lean_deg")
    tempo_min = _value(config, "tempo_min_s")
    tempo_max = _value(config, "tempo_max_s")
    min_valid_fraction = _value(config, "min_valid_fraction")

    peak = max(angle)
    worst_elbow = min(elbow)
    worst_trunk = max(trunk)
    duration_s = len(angle) / fps

    peak_lo, peak_hi = target_peak - tolerance, target_peak + tolerance
    rules = {
        "peak_angle": {
            "measured": peak,
            "allowed": [peak_lo, peak_hi],
            "passed": peak_lo <= peak <= peak_hi,
        },
        "elbow_straight": {
            "measured": worst_elbow,
            "allowed": [min_elbow, 180.0],
            "passed": worst_elbow >= min_elbow,
        },
        "trunk_lean": {
            "measured": worst_trunk,
            "allowed": [0.0, max_trunk],
            "passed": worst_trunk <= max_trunk,
        },
        "tempo": {
            "measured": duration_s,
            "allowed": [tempo_min, tempo_max],
            "passed": tempo_min <= duration_s <= tempo_max,
        },
    }

    failed_rules = [rule_id for rule_id, r in rules.items() if not r["passed"]]

    if valid_fraction < min_valid_fraction:
        return {
            "status": "insufficient_data", "failed_rules": failed_rules, "rules": rules,
            "valid_fraction": valid_fraction, "cue": config["cues"].get("insufficient_data"),
        }
    return {
        "status": "pass" if not failed_rules else "fail", "failed_rules": failed_rules, "rules": rules,
        "valid_fraction": valid_fraction, "cue": None,
    }


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
