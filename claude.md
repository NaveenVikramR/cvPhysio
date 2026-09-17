# cvPhysio — AI Rehabilitation Coach

## What this is
Real-time webcam-based rehabilitation coach. Pose estimation → joint angle
computation → correctness classification → feedback. University semester
project, Review 2 stage — design is finalized, implementation just starting.

## Scope (do not expand without asking)
Exactly two exercises, matching UI-PRMD's official names:
- Standing Shoulder Abduction (UI-PRMD movement 7)
- Standing Shoulder Internal-External Rotation (UI-PRMD movement 9)
No other exercises. No lower-body movements. This is a deliberate scope cut
for camera-framing and elderly-cohort safety reasons — do not suggest adding
squats/lunges/leg raises back in.

## Architecture (6 modules, do not restructure)
1. Video Capture & Pose Estimation — src/pose_estimation.py — MediaPipe Pose
2. Biomechanical Feature Extraction — src/feature_extraction.py — joint angles
3. Correctness Classification — src/classifier.py — Random Forest on UI-PRMD
4. Real-Time Feedback Engine — src/feedback.py — visual overlay + TTS
5. Progress Tracking — src/tracking.py — session logs
6. UI — app.py — Streamlit

## Dataset
UI-PRMD (https://webpages.uidaho.edu/ui-prmd/) — 10 subjects, each movement
performed both correctly and incorrectly. Use ONLY movements 7 and 9.
Cross-subject splits only — never split by repetition, subjects must not
appear in more than one of train/validation/test. Only 10 subjects total,
so say so if the split leaves any subject group very small.

## Hard rules
- Never report an accuracy/precision/recall number that wasn't just produced
  by actually running the code. If asked "what's the accuracy", run it and
  paste the real output — do not estimate or round to a "reasonable" number.
- Confidence gating: if MediaPipe visibility score for a required landmark
  is below 0.5, do not classify that rep — say so instead.
- Keep modules independent — feature_extraction.py should not import
  anything from classifier.py, etc. Each module's output is the next
  module's only input.

## Stack
Python, OpenCV, MediaPipe, scikit-learn, pandas, Streamlit, pyttsx3.
No deep learning (no PyTorch/TensorFlow) — classical ML only, this is a
deliberate scope decision, not a gap to fill.