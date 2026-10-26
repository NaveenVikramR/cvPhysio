"""Video capture & pose estimation. Output: a 33-tuple landmark list per
frame (MediaPipe's PoseLandmark order), nothing else.
"""

import cv2
import mediapipe as mp

_mp_pose = mp.solutions.pose
_pose = _mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def get_pose_landmarks(frame) -> list[tuple[float, float, float, float]] | None:
    """Run MediaPipe Pose on a BGR frame and return the 33 landmarks as
    (x, y, z, visibility) tuples, indexed per MediaPipe's PoseLandmark
    enum (e.g. 11=left shoulder, 23=left hip). Returns None if no pose
    was detected in the frame.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = _pose.process(frame_rgb)
    if not result.pose_landmarks:
        return None
    return [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks.landmark]


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return capture
