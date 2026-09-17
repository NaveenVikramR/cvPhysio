"""Video capture & pose estimation. Output: per-frame landmark dict, nothing else."""

from dataclasses import dataclass

import cv2
import mediapipe as mp

LANDMARK_NAMES = [lm.name for lm in mp.solutions.pose.PoseLandmark]


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr) -> dict[str, Landmark] | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return None
        return {
            name: Landmark(lm.x, lm.y, lm.z, lm.visibility)
            for name, lm in zip(LANDMARK_NAMES, result.pose_landmarks.landmark)
        }

    def close(self) -> None:
        self._pose.close()


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return capture
