"""Streamlit UI: wires pose_estimation -> feature_extraction -> classifier
-> feedback -> tracking into a live webcam session.

Classification happens once per completed repetition, not once per frame --
RepDetector below detects rep boundaries automatically from the live angle
signal, since the trained models (src/classifier.py's FEATURE_COLUMNS) need
a whole rep's worth of frames aggregated via feature_extraction.extract_rep_
features(), not a single frame's angle_deg.
"""

import json
import statistics
from collections import Counter, deque
from pathlib import Path

import pandas as pd
import streamlit as st

from src.classifier import CorrectnessClassifier, is_within_correct_range
from src.feature_extraction import (
    EXERCISES,
    check_confidence,
    extract_features,
    extract_rep_features,
    select_side,
)
from src.feedback import LANDMARK_LABELS, SpeechFeedback, draw_overlay, message_for_result
from src.pose_estimation import get_pose_landmarks, open_webcam
from src.tracking import RepRecord, SessionTracker

MODELS_DIR = Path("models")

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
        ("rep_discarded", reason: str) -- a buffered rep was thrown away
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
                self._reset_to_resting()
                return "rep_discarded", "Rep lost -- tracking interrupted."
            return "none", None

        self.bad_confidence_streak = 0
        self.rep_buffer.append(angle_deg)
        self.side_votes.append(side)

        if len(self.rep_buffer) > MAX_REP_FRAMES:
            self._reset_to_resting()
            return "rep_discarded", "Movement took too long -- discarded."

        if abs(angle_deg - self.baseline) > REST_BAND_DEG:
            return "none", None

        # back within the rest band
        rep_range = max(self.rep_buffer) - min(self.rep_buffer)
        if len(self.rep_buffer) < MIN_REP_FRAMES or rep_range < MIN_REP_RANGE_DEG:
            self._reset_to_resting()
            return "none", None  # silent discard -- flinch, not a real rep

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


st.set_page_config(page_title="cvPhysio", layout="centered")
st.title("cvPhysio — AI Rehabilitation Coach")

exercise_label = st.selectbox("Exercise", list(EXERCISES.values()))
exercise = next(key for key, label in EXERCISES.items() if label == exercise_label)
user_id = st.text_input("User ID", value="demo")

model_path = MODELS_DIR / f"{exercise}.pkl"
metadata_path = MODELS_DIR / f"{exercise}.metadata.json"
classifier = None
correct_feature_ranges = None
if model_path.exists():
    classifier = CorrectnessClassifier()
    classifier.load(model_path)
    if metadata_path.exists():
        correct_feature_ranges = json.loads(metadata_path.read_text()).get("correct_feature_ranges")
else:
    st.warning(f"No trained model at {model_path} — showing pose and angle only, no correctness label.")

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
    tracker = SessionTracker(user_id=user_id, exercise=exercise)
    detector = RepDetector()
    speech = SpeechFeedback()
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
            visible, failed, features = False, [], None
            if landmarks is not None:
                visible, failed = check_confidence(landmarks, select_side(landmarks), exercise)
                features = extract_features(landmarks, exercise) if visible else None

            angle_deg = features["angle_deg"] if features else None
            side = features["side"] if features else None
            event, payload = detector.update(visible, angle_deg, side, failed)

            if event == "rep_complete":
                rep_frames = pd.DataFrame({"angle_deg": payload["angle_deg"]})
                agg = extract_rep_features(rep_frames)
                result = classifier.predict(agg) if classifier else None
                side_value = Counter(payload["side"]).most_common(1)[0][0]
                rep_count += 1
                if classifier and result:
                    tracker.add_rep(RepRecord(
                        exercise=exercise,
                        side=side_value,
                        angle_deg=agg["angle_max_deg"],
                        is_correct=result["is_correct"],
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
                st.warning(payload)

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
        if tracker.reps:
            tracker.save()
            st.json(tracker.summary())
