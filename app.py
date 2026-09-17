"""Streamlit UI: wires pose_estimation -> feature_extraction -> classifier
-> feedback -> tracking into a live webcam session.
"""

from pathlib import Path

import cv2
import streamlit as st

from src.classifier import CorrectnessClassifier
from src.feature_extraction import EXERCISES, check_confidence, extract_features, required_landmarks_for
from src.feedback import draw_overlay, message_for_result
from src.pose_estimation import PoseEstimator, open_webcam
from src.tracking import RepRecord, SessionTracker

MODELS_DIR = Path("models")

st.set_page_config(page_title="cvPhysio", layout="centered")
st.title("cvPhysio — AI Rehabilitation Coach")

exercise_label = st.selectbox("Exercise", list(EXERCISES.values()))
exercise = next(key for key, label in EXERCISES.items() if label == exercise_label)
side = st.radio("Arm", ["RIGHT", "LEFT"], horizontal=True)
user_id = st.text_input("User ID", value="demo")

model_path = MODELS_DIR / f"{exercise}.pkl"
classifier = None
if model_path.exists():
    classifier = CorrectnessClassifier()
    classifier.load(model_path)
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
    estimator = PoseEstimator()
    capture = open_webcam()
    tracker = SessionTracker(user_id=user_id, exercise=exercise)
    try:
        while st.session_state.running:
            ok, frame = capture.read()
            if not ok:
                st.error("Could not read from camera.")
                break

            landmarks = estimator.process_frame(frame)
            if landmarks is None:
                status_slot.info("No person detected.")
                frame_slot.image(frame, channels="BGR")
                continue

            required = required_landmarks_for(exercise, side)
            visible, failed = check_confidence(landmarks, required)
            features = extract_features(landmarks, exercise, side) if visible else None

            result = classifier.predict(features) if (classifier and features) else None
            message = message_for_result(result, failed)
            status_slot.write(message)

            if result is not None and features is not None:
                tracker.add_rep(RepRecord(
                    exercise=exercise,
                    side=side,
                    angle_deg=features["angle_deg"],
                    is_correct=result["is_correct"],
                    confidence=result["confidence"],
                ))

            annotated = draw_overlay(frame, message, result["is_correct"] if result else None)
            frame_slot.image(annotated, channels="BGR")
    finally:
        capture.release()
        estimator.close()
        if tracker.reps:
            tracker.save()
            st.json(tracker.summary())
