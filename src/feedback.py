"""Real-time feedback engine. Input: classifier's result dict (or a
confidence-gating failure). Output: annotated frame + spoken feedback.
"""

import threading

import cv2
import pyttsx3

CORRECT_MESSAGE = "Good form, keep going."
INCORRECT_MESSAGE = "Adjust your form."

LANDMARK_LABELS = {
    11: "left shoulder", 12: "right shoulder",
    13: "left elbow", 14: "right elbow",
    15: "left wrist", 16: "right wrist",
    23: "left hip", 24: "right hip",
}


def low_visibility_message(failed_landmarks: list[int]) -> str:
    readable = ", ".join(LANDMARK_LABELS.get(i, str(i)) for i in failed_landmarks)
    return f"Can't clearly see your {readable}. Please step back or adjust lighting."


def draw_overlay(frame, text: str, is_correct: bool | None):
    color = (0, 200, 0) if is_correct else (0, 0, 220) if is_correct is False else (0, 165, 255)
    return cv2.putText(
        frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA
    )


class SpeechFeedback:
    def __init__(self):
        self._engine = pyttsx3.init()
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        def _run():
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()

        threading.Thread(target=_run, daemon=True).start()


def message_for_result(result: dict | None, failed_landmarks: list[int] | None = None) -> str:
    if result is None:
        return low_visibility_message(failed_landmarks or [])
    return CORRECT_MESSAGE if result["is_correct"] else INCORRECT_MESSAGE


def message_for_rules_result(result: dict) -> str:
    """The one status-text builder for src.rules.score_rep()'s 3-way
    result, shared by app.py's Streamlit path and run_headless() --
    previously each built this text inline, independently, and could
    drift apart (see open_questions.md #7)."""
    if result["status"] == "insufficient_data":
        return f"Not enough visible tracking to score this rep ({result['valid_fraction']:.0%} valid)."
    if result["status"] == "pass":
        return "Good form!"
    return "Needs adjustment: " + ", ".join(result["failed_rules"])
