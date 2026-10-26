"""Video capture & pose estimation. Output: a 33-tuple landmark list per
frame (MediaPipe's PoseLandmark order), nothing else.

Uses MediaPipe's Tasks API (PoseLandmarker), not the older
mediapipe.solutions.pose API -- the legacy "solutions" module ships no
compiled extension at all in MediaPipe's Windows wheels for Python 3.13
(confirmed: absent in 1.1.1, 1.1.0, 1.0.x and 0.10.35 alike), so it isn't a
version-pin problem, the Tasks API is the only one available here. It's
also the actively-maintained API going forward -- "solutions" is
deprecated upstream, not just broken on this machine.

Needs a pose-landmarker model bundle at models/pose_landmarker_lite.task
(downloaded once from Google's model zoo: https://storage.googleapis.com/
mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/
pose_landmarker_lite.task -- gitignored, not committed, ~5.8MB).

Runs in VIDEO mode (not IMAGE) so MediaPipe reuses temporal context
between frames the way the old solutions API implicitly did for a video
stream -- IMAGE mode re-detects from scratch every call with no tracking
continuity, which would be jitterier for RepDetector's rest-baseline
calibration in app.py.
"""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"

_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
))

_next_timestamp_ms = 0


def get_pose_landmarks(frame) -> list[tuple[float, float, float, float]] | None:
    """Run MediaPipe Pose on a BGR frame and return the 33 landmarks as
    (x, y, z, visibility) tuples, indexed per MediaPipe's PoseLandmark
    enum (e.g. 11=left shoulder, 23=left hip). Returns None if no pose
    was detected in the frame.
    """
    global _next_timestamp_ms
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = _landmarker.detect_for_video(mp_image, _next_timestamp_ms)
    _next_timestamp_ms += 33  # monotonic tick; VIDEO mode needs increasing values, not real elapsed time

    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    return [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return capture
