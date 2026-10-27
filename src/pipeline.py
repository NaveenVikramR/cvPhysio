"""Shared live-pipeline orchestration: RepDetector (the rep-boundary
state machine), process_frame() (one frame through confidence-gating +
NaN guard + RepDetector, keeping the elbow/trunk parallel buffers
aligned), build_smoother(), load_classifier_for_live_use() (the
classifier + metadata guard), and Session (the Start/Stop session
lifecycle that gates whether a frame ever reaches RepDetector at all).
No Streamlit, no cv2 window/webcam code, no work at import time --
app.py and every scripts/*.py diagnostic that needs RepDetector/
process_frame import directly from here, instead of each independently
exec()-slicing these out of app.py's source text (the previous
approach, dropped because every new name this code referenced had to
be re-added to every single loader's namespace by hand, and frequently
wasn't).
"""

import json
import math
import statistics
from collections import Counter, deque
from pathlib import Path

from src import smoothing as smoothing_module
from src.classifier import FEATURE_COLUMNS, CorrectnessClassifier
from src.feature_extraction import (
    check_confidence,
    compute_elbow_angle,
    compute_trunk_lean_deg,
    extract_features,
    select_side,
)
from src.feedback import LANDMARK_LABELS
from src.statuses import DiscardReason

MODELS_DIR = Path("models")
# The only feature_source src/dataset_loader.py's FK positions produce and
# the live path actually computes -- see that module's docstring for why
# "angles_file" (the other feature_source it supports) is not what this
# pipeline needs. A metadata.json recorded with any other value was not
# trained on what this pipeline feeds the classifier at inference time.
LIVE_FEATURE_SOURCE = "fk_positions"

# Rep-boundary state machine constants -- see the dev plan for rationale.
# "Rest" is not a fixed angle (confirmed real sweeps: abduction rests near
# ~20 deg, rotation near ~60 deg), so the baseline is estimated live.
# MIN_REP_RANGE_DEG=25 is set from a real test clip: a genuine noise blip
# measured only 4 deg of range (should be rejected), a real rep measured
# 51 deg (must be accepted), training reps swept 110-140 deg -- 25 sits
# with real headroom above the noise and below every real rep observed.
CALIBRATION_FRAMES = 15
CALIBRATION_MAX_STD_DEG = 6.0
REST_BAND_DEG = 10.0
BASELINE_WINDOW = 10
MIN_REP_FRAMES = 15
MIN_REP_RANGE_DEG = 25.0
MAX_REP_FRAMES = 450
CONFIDENCE_TOLERANCE_FRAMES = 10
CALIBRATION_STALL_FRAMES = 150  # ~a few seconds at a plausible live fps
# Used only by the "already near rest" tracking-lost rescue below, NOT
# REST_BAND_DEG itself: any angle that actually lands inside REST_BAND_DEG
# while still visible completes (or too-short-discards) synchronously, in
# the same update() call -- so by the time a rep reaches the tracking-lost
# branch, its last visible angle is, by construction, always still outside
# REST_BAND_DEG. A wider tolerance is what makes "was it already heading
# back into rest" an actually answerable question from rep_buffer alone.
RESCUE_REST_BAND_DEG = 2 * REST_BAND_DEG


class RepDetector:
    """CALIBRATING -> RESTING <-> MOVING. Buffers per-frame angle_deg values
    for one repetition and reports when a rep completes, so the caller runs
    extract_rep_features() + classifies exactly once per rep, not per frame.
    """

    def __init__(self):
        self.state = "CALIBRATING"
        self.baseline = None
        self.calib_window = deque(maxlen=CALIBRATION_FRAMES)
        self.calib_fail_counts = Counter()
        self.calib_attempt_frames = 0
        self.rest_window = deque(maxlen=BASELINE_WINDOW)
        self.rep_buffer = []
        self.side_votes = []
        self.bad_confidence_streak = 0

    def update(self, visible: bool, angle_deg: float | None, side: str | None,
               failed: list[int] | None = None):
        """Advance by one frame. Returns (event, payload):
        ("none", None) -- nothing to report this frame
        ("rep_complete", {"angle_deg": [...], "side": [...]}) -- classify this
        ("rep_discarded", {"reason_code": "too_short"|"too_long"|
        "lost_tracking", "message": str}) -- a buffered rep was thrown away
        `failed` (landmark indices failing confidence this frame) is only
        used while CALIBRATING, to drive calibration_hint() below.
        """
        if self.state == "CALIBRATING":
            self.calib_attempt_frames += 1
            if not visible:
                self.calib_fail_counts.update(failed or [])
                return "none", None
            self.calib_window.append(angle_deg)
            if len(self.calib_window) < CALIBRATION_FRAMES:
                return "none", None
            if statistics.pstdev(self.calib_window) > CALIBRATION_MAX_STD_DEG:
                self.calib_window.popleft()
                return "none", None
            self.baseline = statistics.fmean(self.calib_window)
            self.calib_window.clear()
            self.state = "RESTING"
            return "none", None

        if self.state == "RESTING":
            if not visible:
                return "none", None
            if abs(angle_deg - self.baseline) <= REST_BAND_DEG:
                self.rest_window.append(angle_deg)
                self.baseline = statistics.fmean(self.rest_window)
                return "none", None
            self.rep_buffer = [angle_deg]
            self.side_votes = [side]
            self.bad_confidence_streak = 0
            self.state = "MOVING"
            return "none", None

        # self.state == "MOVING"
        if not visible:
            self.bad_confidence_streak += 1
            if self.bad_confidence_streak > CONFIDENCE_TOLERANCE_FRAMES:
                # Tracking was lost, but if the LAST valid frame had already
                # returned near the rest band and the rep otherwise qualifies,
                # that rep already happened -- emit it instead of discarding
                # it just because confirmation came too late. Only discard
                # when tracking is lost BEFORE the rep actually completed.
                if self.rep_buffer and abs(self.rep_buffer[-1] - self.baseline) <= RESCUE_REST_BAND_DEG:
                    rep_range = max(self.rep_buffer) - min(self.rep_buffer)
                    if len(self.rep_buffer) >= MIN_REP_FRAMES and rep_range >= MIN_REP_RANGE_DEG:
                        buffer, votes = self.rep_buffer, self.side_votes
                        self.rest_window.append(self.rep_buffer[-1])
                        self.baseline = statistics.fmean(self.rest_window)
                        self._reset_to_resting()
                        return "rep_complete", {"angle_deg": buffer, "side": votes}
                self._reset_to_resting()
                return "rep_discarded", {"reason_code": DiscardReason.LOST_TRACKING.value, "message": "Rep lost -- tracking interrupted."}
            return "none", None

        self.bad_confidence_streak = 0
        self.rep_buffer.append(angle_deg)
        self.side_votes.append(side)

        if len(self.rep_buffer) > MAX_REP_FRAMES:
            self._reset_to_resting()
            return "rep_discarded", {"reason_code": DiscardReason.TOO_LONG.value, "message": "Movement took too long -- discarded."}

        if abs(angle_deg - self.baseline) > REST_BAND_DEG:
            return "none", None

        # back within the rest band
        rep_range = max(self.rep_buffer) - min(self.rep_buffer)
        if len(self.rep_buffer) < MIN_REP_FRAMES or rep_range < MIN_REP_RANGE_DEG:
            self._reset_to_resting()
            return "rep_discarded", {"reason_code": DiscardReason.TOO_SHORT.value, "message": "Movement too brief/small -- discarded."}

        buffer, votes = self.rep_buffer, self.side_votes
        self.rest_window.append(angle_deg)
        self.baseline = statistics.fmean(self.rest_window)
        self._reset_to_resting()
        return "rep_complete", {"angle_deg": buffer, "side": votes}

    def _reset_to_resting(self) -> None:
        self.rep_buffer = []
        self.side_votes = []
        self.bad_confidence_streak = 0
        self.state = "RESTING"

    def calibration_hint(self) -> str | None:
        """Stable, actionable hint once CALIBRATING has stalled for a
        while -- names the single most-frequently-failing landmark instead
        of the per-frame low-visibility message flickering between
        whichever landmark happens to fail on a given frame. Returns None
        before the stall threshold, or once no longer calibrating.
        """
        if self.state != "CALIBRATING" or self.calib_attempt_frames < CALIBRATION_STALL_FRAMES:
            return None
        if not self.calib_fail_counts:
            return "Having trouble getting a stable reading — please hold still."
        worst_idx, _ = self.calib_fail_counts.most_common(1)[0]
        label = LANDMARK_LABELS.get(worst_idx, str(worst_idx))
        return f"Step back so your {label} stays visible, then hold still."


def process_frame(detector: RepDetector, parallel: dict, landmarks, exercise: str,
                   smoother=None, t: float | None = None, required_roles: set[str] | None = None):
    """One frame through confidence-gating + RepDetector, plus a parallel
    elbow_angle_deg/trunk_lean_deg buffer kept aligned with RepDetector's
    own internal rep_buffer -- RepDetector only tracks a single angle_deg
    series, but src.rules.score_rep() needs elbow/trunk too, for exactly
    the same frames. `parallel` is {"elbow": [...], "trunk": [...],
    "total_frames": int}, mutated in place; on "rep_complete" the aligned
    sequences and valid_fraction (valid/total frames over the rep's whole
    span, including ones the confidence gate rejected) are added to the
    returned payload.

    `smoother`, if given (src.smoothing.LandmarkSmoother), runs between
    pose and features: raw landmarks in, smoothed (or held-over, or None
    if a dropout outlasted its hold budget) landmarks out, before any
    angle is computed from them. `t` is this frame's timestamp in
    seconds (required if smoother is given -- see src/smoothing.py).

    `required_roles`, if given, is passed to check_confidence() instead
    of its exercise-based default -- pass src.rules.required_joint_roles()
    when scoring with rules, so the gate checks exactly what the active
    rules' features depend on (e.g. the wrist, for elbow_straight).

    Mirrors RepDetector.update()'s own append rule: a frame is part of
    the rep buffer iff it's visible AND (the detector was already MOVING,
    which also covers the final frame that completes the rep, OR this
    frame is the RESTING->MOVING transition itself).
    """
    if smoother is not None:
        landmarks = smoother.update(landmarks, t)

    visible, failed, features = False, [], None
    elbow_val = trunk_val = None
    if landmarks is not None:
        visible, failed = check_confidence(landmarks, select_side(landmarks), exercise, required_roles)
        features = extract_features(landmarks, exercise) if visible else None
        elbow_val = compute_elbow_angle(landmarks)
        trunk_val = compute_trunk_lean_deg(landmarks)
        if visible and (features is None or math.isnan(elbow_val) or math.isnan(trunk_val)):
            # extract_features() already excludes a NaN angle_deg (e.g. a
            # degenerate/coincident-landmark frame) by returning None; a
            # NaN elbow/trunk reading is caught here instead, since those
            # are computed outside extract_features(). Either way, treat
            # this frame as if confidence had failed -- it must not reach
            # RepDetector or the parallel elbow/trunk buffers (score_rep
            # requires those three lists to stay equal-length).
            visible = False
            features = None

    angle_deg = features["angle_deg"] if features else None
    side = features["side"] if features else None

    prev_state = detector.state
    event, payload = detector.update(visible, angle_deg, side, failed)
    new_state = detector.state

    in_rep_span = prev_state == "MOVING" or (prev_state == "RESTING" and new_state == "MOVING")
    if in_rep_span:
        parallel["total_frames"] += 1
        if visible:
            parallel["elbow"].append(elbow_val)
            parallel["trunk"].append(trunk_val)

    if event == "rep_complete":
        payload["elbow_angle_deg"] = list(parallel["elbow"])
        payload["trunk_lean_deg"] = list(parallel["trunk"])
        payload["valid_fraction"] = (
            len(parallel["elbow"]) / parallel["total_frames"] if parallel["total_frames"] else 0.0
        )

    if prev_state == "MOVING" and new_state == "RESTING":
        parallel["elbow"].clear()
        parallel["trunk"].clear()
        parallel["total_frames"] = 0

    return event, payload, visible, failed


def build_smoother():
    """None if config/smoothing.json's "enabled" is false (the shipped
    default -- see that file's "enabled_comment" for why) or no tuned
    params exist yet; otherwise a fresh LandmarkSmoother from it."""
    config = smoothing_module.load_config()
    if not config.get("enabled"):
        return None
    return smoothing_module.LandmarkSmoother(
        min_cutoff=config["min_cutoff"], beta=config["beta"],
        d_cutoff=config["d_cutoff"], max_hold_frames=config["max_hold_frames"],
    )


def load_classifier_for_live_use(
    exercise: str, models_dir: Path = MODELS_DIR,
) -> tuple[CorrectnessClassifier | None, dict | None]:
    """Loads {models_dir}/{exercise}.pkl, but only after its sidecar
    {models_dir}/{exercise}.metadata.json confirms it was trained on the
    SAME feature names (and order) and feature_source the live path
    actually computes (FEATURE_COLUMNS, LIVE_FEATURE_SOURCE) -- refusing
    with a clear RuntimeError otherwise, rather than silently scoring
    with a model trained on different features than this pipeline
    produces. `models_dir` defaults to the real MODELS_DIR; overridable
    so tests can point it at a fixture directory.

    Returns (None, None) if no model file exists at all (the existing,
    unchanged "no trained model" case -- not an error, just nothing to
    load yet). Returns (classifier, correct_feature_ranges) on success.
    """
    model_path = models_dir / f"{exercise}.pkl"
    metadata_path = models_dir / f"{exercise}.metadata.json"
    if not model_path.exists():
        return None, None

    if not metadata_path.exists():
        raise RuntimeError(
            f"Refusing to load {model_path}: no sidecar metadata at {metadata_path} "
            f"to verify it was trained on the features this pipeline actually computes."
        )

    metadata = json.loads(metadata_path.read_text())
    metadata_features = metadata.get("feature_names")
    if metadata_features != FEATURE_COLUMNS:
        raise RuntimeError(
            f"Refusing to load {model_path}: metadata feature_names {metadata_features} "
            f"do not match the live path's {FEATURE_COLUMNS}."
        )
    metadata_source = metadata.get("feature_source")
    if metadata_source != LIVE_FEATURE_SOURCE:
        raise RuntimeError(
            f"Refusing to load {model_path}: metadata feature_source {metadata_source!r} "
            f"!= {LIVE_FEATURE_SOURCE!r} (the only source the live pipeline computes)."
        )

    classifier = CorrectnessClassifier()
    classifier.load(model_path)
    return classifier, metadata.get("correct_feature_ranges")


# ===========================================================================
# Start/Stop session lifecycle
# ===========================================================================

class SessionState:
    """IDLE -> CALIBRATING -> ACTIVE -> STOPPED. CALIBRATING/ACTIVE are
    DERIVED from RepDetector's own, unmodified internal state (CALIBRATING
    mirrors RepDetector.state == "CALIBRATING"; ACTIVE covers both RESTING
    and MOVING) -- Session adds no new segmentation logic of its own, it
    only decides whether a frame reaches RepDetector.update() at all.
    """
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    ACTIVE = "ACTIVE"
    STOPPED = "STOPPED"


class Session:
    """Wraps one RepDetector with a Start/Stop lifecycle: IDLE ->
    CALIBRATING -> ACTIVE -> STOPPED. Frames fed in while IDLE or
    STOPPED are ignored by the rep detector entirely -- they never reach
    RepDetector.update() or process_frame() -- so a frame before Start
    or after Stop can never be counted into a rep, discarded, or used
    for calibration. start() constructs a fresh RepDetector and begins
    calibration (the exact same CALIBRATING behavior RepDetector always
    had); stop() is a one-way transition to STOPPED (no restart).
    app.py's Streamlit Start/Stop buttons and --span (headless mode)
    both drive this same class, so "pressing Start/Stop" and "--span
    START END" are the same mechanism, not two independent ones.
    """

    def __init__(self, exercise: str):
        self.exercise = exercise
        self.detector: RepDetector | None = None
        self.parallel: dict | None = None
        self._stopped = False

    @property
    def session_state(self) -> str:
        if self._stopped:
            return SessionState.STOPPED
        if self.detector is None:
            return SessionState.IDLE
        return SessionState.CALIBRATING if self.detector.state == "CALIBRATING" else SessionState.ACTIVE

    def start(self) -> None:
        """IDLE -> CALIBRATING. A no-op once already started (CALIBRATING/
        ACTIVE) or stopped -- there is no restart after Stop."""
        if self.session_state != SessionState.IDLE:
            return
        self.detector = RepDetector()
        self.parallel = {"elbow": [], "trunk": [], "total_frames": 0}

    def stop(self) -> None:
        """CALIBRATING/ACTIVE -> STOPPED. One-way, idempotent."""
        self._stopped = True

    def process_frame(self, landmarks, smoother=None, t: float | None = None,
                       required_roles: set[str] | None = None):
        """Same return shape as the module-level process_frame():
        (event, payload, visible, failed). While IDLE or STOPPED, returns
        ("idle", None, False, []) without calling RepDetector/
        process_frame() at all -- the frame is ignored, not merely
        discarded after being looked at.
        """
        if self.session_state in (SessionState.IDLE, SessionState.STOPPED):
            return "idle", None, False, []
        return process_frame(self.detector, self.parallel, landmarks, self.exercise, smoother, t, required_roles)
