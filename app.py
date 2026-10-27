"""Streamlit UI: wires pose_estimation -> feature_extraction -> classifier
-> feedback -> tracking into a live webcam session.

Classification happens once per completed repetition, not once per frame --
RepDetector below detects rep boundaries automatically from the live angle
signal, since the trained models (src/classifier.py's FEATURE_COLUMNS) need
a whole rep's worth of frames aggregated via feature_extraction.extract_rep_
features(), not a single frame's angle_deg.
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src import rules as rules_module
from src import smoothing as smoothing_module
from src.classifier import FEATURE_COLUMNS, CorrectnessClassifier, is_within_correct_range
from src.feature_extraction import (
    EXERCISES,
    check_confidence,
    compute_elbow_angle,
    compute_trunk_lean_deg,
    extract_features,
    extract_rep_features,
    select_side,
)
from src.feedback import LANDMARK_LABELS, SpeechFeedback, draw_overlay, message_for_result
from src.pose_estimation import get_pose_landmarks, open_webcam
from src.tracking import DiscardRecord, RepRecord, SessionTracker

MODELS_DIR = Path("models")
CLASSIFIER_WARNING = (
    "WARNING: --scorer classifier was trained on UI-PRMD Kinect recordings, "
    "not webcam video, and is not calibrated for webcam input "
    "(see scripts/diagnose_misclassification.py). Treat its output with caution."
)
# The only feature_source src/dataset_loader.py's FK positions produce and
# the live path actually computes -- see that module's docstring for why
# "angles_file" (the other feature_source it supports) is not what app.py
# needs. A metadata.json recorded with any other value was not trained on
# what this pipeline feeds the classifier at inference time.
LIVE_FEATURE_SOURCE = "fk_positions"


def parse_args():
    parser = argparse.ArgumentParser(description="cvPhysio rehab coach")
    parser.add_argument("--scorer", choices=["rules", "classifier"], default="rules")
    parser.add_argument("--video", type=str, default=None,
                         help="Path to a video file to score headlessly instead of launching the Streamlit webcam UI.")
    parser.add_argument("--exercise", choices=list(EXERCISES), default="shoulder_abduction")
    return parser.parse_args(sys.argv[1:])


ARGS = parse_args()

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
                return "rep_discarded", {"reason_code": "lost_tracking", "message": "Rep lost -- tracking interrupted."}
            return "none", None

        self.bad_confidence_streak = 0
        self.rep_buffer.append(angle_deg)
        self.side_votes.append(side)

        if len(self.rep_buffer) > MAX_REP_FRAMES:
            self._reset_to_resting()
            return "rep_discarded", {"reason_code": "too_long", "message": "Movement took too long -- discarded."}

        if abs(angle_deg - self.baseline) > REST_BAND_DEG:
            return "none", None

        # back within the rest band
        rep_range = max(self.rep_buffer) - min(self.rep_buffer)
        if len(self.rep_buffer) < MIN_REP_FRAMES or rep_range < MIN_REP_RANGE_DEG:
            self._reset_to_resting()
            return "rep_discarded", {"reason_code": "too_short", "message": "Movement too brief/small -- discarded."}

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


def run_headless(video_path: str, exercise: str, scorer: str) -> None:
    """--video path: score a video file frame-by-frame with no Streamlit
    UI, printing per-rep results to stdout. Used for verification against
    real recorded clips (no webcam available in that context)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config = None
    classifier = None
    correct_feature_ranges = None
    required_roles = None
    if scorer == "rules":
        try:
            config = rules_module.load_config(exercise)
        except FileNotFoundError as exc:
            print(str(exc))
            cap.release()
            return
        required_roles = rules_module.required_joint_roles()
    else:
        print(CLASSIFIER_WARNING)
        try:
            classifier, correct_feature_ranges = load_classifier_for_live_use(exercise)
        except RuntimeError as exc:
            print(str(exc))
            cap.release()
            return
        if classifier is None:
            print(f"No trained model at {MODELS_DIR / f'{exercise}.pkl'} -- cannot score with --scorer classifier.")
            cap.release()
            return

    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    smoother = build_smoother()
    rep_count = 0
    frame_idx = 0
    print(f"Scoring {video_path} (exercise={exercise}, scorer={scorer}, fps={fps:.2f}, smoothing={'on' if smoother else 'off'})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        frame_idx += 1
        landmarks = get_pose_landmarks(frame)
        event, payload, _visible, _failed = process_frame(
            detector, parallel, landmarks, exercise, smoother, t, required_roles,
        )

        if event == "rep_complete":
            rep_count += 1
            if scorer == "rules":
                rep = {
                    "angle_deg": payload["angle_deg"],
                    "elbow_angle_deg": payload["elbow_angle_deg"],
                    "trunk_lean_deg": payload["trunk_lean_deg"],
                    "valid_fraction": payload["valid_fraction"],
                }
                result = rules_module.score_rep(rep, config, fps)
                print(f"Rep {rep_count}: {result['status'].upper()}  failed_rules={result['failed_rules']}  "
                      f"valid_fraction={result['valid_fraction']:.2f}")
                for rule_id, r in result["rules"].items():
                    print(f"    {rule_id}: measured={r['measured']:.1f}  "
                          f"allowed=[{r['allowed'][0]:.1f}, {r['allowed'][1]:.1f}]  passed={r['passed']}")
            else:
                agg = extract_rep_features(pd.DataFrame({"angle_deg": payload["angle_deg"]}))
                result = classifier.predict(agg) if classifier else None
                if result:
                    note = ""
                    if correct_feature_ranges and not is_within_correct_range(agg, correct_feature_ranges):
                        note = "  (outside training-correct range)"
                    print(f"Rep {rep_count}: classifier is_correct={result['is_correct']}  "
                          f"confidence={result['confidence']:.2f}{note}")
                else:
                    print(f"Rep {rep_count}: no trained model available.")
        elif event == "rep_discarded":
            print(f"  (rep discarded: {payload['reason_code']} -- {payload['message']})")

    cap.release()
    if rep_count == 0:
        print("No completed reps detected in this clip.")


if ARGS.video:
    run_headless(ARGS.video, ARGS.exercise, ARGS.scorer)
    raise SystemExit(0)

st.set_page_config(page_title="cvPhysio", layout="centered")
st.title("cvPhysio — AI Rehabilitation Coach")

exercise_label = st.selectbox("Exercise", list(EXERCISES.values()))
exercise = next(key for key, label in EXERCISES.items() if label == exercise_label)
user_id = st.text_input("User ID", value="demo")

classifier = None
correct_feature_ranges = None
rules_config = None
if ARGS.scorer == "classifier":
    st.warning(CLASSIFIER_WARNING)
    try:
        classifier, correct_feature_ranges = load_classifier_for_live_use(exercise)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    if classifier is None:
        st.warning(f"No trained model at {MODELS_DIR / f'{exercise}.pkl'} — showing pose and angle only, no correctness label.")
else:
    try:
        rules_config = rules_module.load_config(exercise)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
if col1.button("Start session"):
    st.session_state.running = True
if col2.button("Stop session"):
    st.session_state.running = False

frame_slot = st.empty()
status_slot = st.empty()

if st.session_state.running:
    capture = open_webcam()
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = SessionTracker(user_id=user_id, exercise=exercise)
    detector = RepDetector()
    parallel = {"elbow": [], "trunk": [], "total_frames": 0}
    smoother = build_smoother()
    required_roles = rules_module.required_joint_roles() if rules_config else None
    speech = SpeechFeedback()
    cue_cooldown = rules_module.CueCooldown(rules_config["cue_cooldown_s"]["value"]) if rules_config else None
    rep_count = 0
    last_message = "Ready — begin your repetition."
    last_is_correct = None
    try:
        while st.session_state.running:
            ok, frame = capture.read()
            if not ok:
                st.error("Could not read from camera.")
                break

            landmarks = get_pose_landmarks(frame)
            t = time.monotonic()  # live webcam: real arrival time, not an assumed fixed rate
            event, payload, visible, failed = process_frame(
                detector, parallel, landmarks, exercise, smoother, t, required_roles,
            )

            if event == "rep_complete":
                rep_count += 1
                if ARGS.scorer == "rules":
                    rep = {
                        "angle_deg": payload["angle_deg"],
                        "elbow_angle_deg": payload["elbow_angle_deg"],
                        "trunk_lean_deg": payload["trunk_lean_deg"],
                        "valid_fraction": payload["valid_fraction"],
                    }
                    result = rules_module.score_rep(rep, rules_config, fps)
                    side_value = Counter(payload["side"]).most_common(1)[0][0]
                    tracker.add_rep(RepRecord(
                        exercise=exercise,
                        side=side_value,
                        angle_deg=max(rep["angle_deg"]),
                        status=result["status"],
                        confidence=None,
                    ))
                    if result["status"] == "insufficient_data":
                        status_text = f"Not enough visible tracking to score this rep ({result['valid_fraction']:.0%} valid)."
                    elif result["status"] == "pass":
                        status_text = "Good form!"
                    else:
                        status_text = "Needs adjustment: " + ", ".join(result["failed_rules"])
                    last_message = f"Rep {rep_count}: {status_text}"
                    last_is_correct = {"pass": True, "fail": False, "insufficient_data": None}[result["status"]]
                    cues = cue_cooldown.cues_to_speak(result["failed_rules"], rules_config["cues"]) if result["status"] == "fail" else []
                    if result["status"] == "insufficient_data" and result["cue"] and cue_cooldown.should_fire("insufficient_data"):
                        cues = [result["cue"]]
                        cue_cooldown.record("insufficient_data")
                    for cue in cues:
                        speech.speak(cue)
                    if not cues:
                        speech.speak(last_message)
                else:
                    rep_frames = pd.DataFrame({"angle_deg": payload["angle_deg"]})
                    agg = extract_rep_features(rep_frames)
                    result = classifier.predict(agg) if classifier else None
                    side_value = Counter(payload["side"]).most_common(1)[0][0]
                    if classifier and result:
                        tracker.add_rep(RepRecord(
                            exercise=exercise,
                            side=side_value,
                            angle_deg=agg["angle_max_deg"],
                            status="pass" if result["is_correct"] else "fail",
                            confidence=result["confidence"],
                        ))
                        last_message = f"Rep {rep_count}: {message_for_result(result, None)}"
                        if correct_feature_ranges and not is_within_correct_range(agg, correct_feature_ranges):
                            last_message += (
                                " (Note: this rep's motion was outside the range seen in training "
                                "data labeled correct — treat this result with extra caution.)"
                            )
                        last_is_correct = result["is_correct"]
                    else:
                        last_message = f"Rep {rep_count} complete — no trained model, correctness not shown."
                        last_is_correct = None
                    speech.speak(last_message)  # spoken cue once per completed rep, not per frame
            elif event == "rep_discarded":
                tracker.add_discard(DiscardRecord(
                    exercise=exercise, reason_code=payload["reason_code"], message=payload["message"],
                ))
                st.warning(payload["message"])

            stall_hint = detector.calibration_hint() if detector.state == "CALIBRATING" else None
            if stall_hint:
                # Stable, actionable hint once stuck calibrating -- takes
                # priority over the flickery per-frame low-visibility message.
                message, overlay_is_correct = stall_hint, None
            elif landmarks is None:
                message, overlay_is_correct = "No person detected.", None
            elif not visible:
                message, overlay_is_correct = message_for_result(None, failed), None
            elif detector.state == "CALIBRATING":
                message, overlay_is_correct = "Calibrating rest position — please hold still...", None
            elif detector.state == "MOVING":
                message, overlay_is_correct = "Repetition in progress...", None
            else:  # RESTING
                message, overlay_is_correct = last_message, last_is_correct

            status_slot.write(message)
            annotated = draw_overlay(frame, message, overlay_is_correct)
            frame_slot.image(annotated, channels="BGR")
    finally:
        capture.release()
        if tracker.reps or tracker.discards:
            tracker.save()
            st.json(tracker.summary())
