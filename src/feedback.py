"""Real-time feedback engine. Input: classifier's result dict (or a
confidence-gating failure). Output: annotated frame + spoken feedback.
"""

import os
import queue
import threading
import time

import cv2
import pyttsx3

CORRECT_MESSAGE = "Good form, keep going."
INCORRECT_MESSAGE = "Adjust your form."

NO_VOICE_ENV_VAR = "CVPHYSIO_NO_VOICE"
DEFAULT_SPEECH_TIMEOUT_S = 15.0
DEFAULT_SPEECH_MAX_RESTARTS = 2

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


class _SpeechWorker:
    """One daemon thread that owns one TTS engine. The engine must be
    created on the thread that uses it: a pyttsx3 engine created on one
    thread hung forever in runAndWait() on another (measured on Windows
    SAPI5, speaking nothing further and leaking a thread per call)."""

    def __init__(self, engine_factory):
        self._engine_factory = engine_factory
        self._queue = queue.Queue(maxsize=1)
        self.busy_since = None  # monotonic time the cue in hand was accepted
        self.abandoned = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text: str) -> None:
        self.busy_since = time.monotonic()
        self._queue.put_nowait(text)

    def _run(self) -> None:
        engine = None
        while not self.abandoned:
            text = self._queue.get()
            try:
                if engine is None:
                    engine = self._engine_factory()
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:  # a failed cue must never reach the frame loop
                engine = None
                print(f"Speech failed: {exc}")
            self.busy_since = None


class SpeechFeedback:
    """speak() never blocks the caller: a single worker speaks one cue at a
    time, and a cue arriving while it's busy is dropped, not queued, so
    feedback can't lag behind the movement it's about. A cue still unfinished
    after timeout_s counts as hung: the next speak() abandons that worker
    (a hung thread can't be killed) and starts a fresh one, up to max_restarts
    times before voice turns itself off. CVPHYSIO_NO_VOICE=1 disables voice
    entirely, with no engine created."""

    def __init__(self, engine_factory=None, timeout_s: float = DEFAULT_SPEECH_TIMEOUT_S,
                 max_restarts: int = DEFAULT_SPEECH_MAX_RESTARTS):
        self._engine_factory = engine_factory or pyttsx3.init
        self._timeout_s = timeout_s
        self._restarts_left = max_restarts
        self._enabled = os.environ.get(NO_VOICE_ENV_VAR) != "1"
        self._worker = None
        self._lock = threading.Lock()

    def speak(self, text: str) -> bool:
        """True if the cue was handed to the worker, False if it was dropped
        (worker busy or hung) or voice is disabled."""
        with self._lock:
            if not self._enabled:
                return False
            if self._worker is None:
                self._worker = _SpeechWorker(self._engine_factory)
            busy_since = self._worker.busy_since
            if busy_since is not None:
                if time.monotonic() - busy_since < self._timeout_s:
                    return False
                self._worker.abandoned = True
                if self._restarts_left == 0:
                    self._enabled = False
                    print("Speech keeps hanging -- voice disabled for this session.")
                    return False
                self._restarts_left -= 1
                print(f"Speech cue exceeded {self._timeout_s:g} s -- restarting the speech engine.")
                self._worker = _SpeechWorker(self._engine_factory)
            self._worker.submit(text)
            return True


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
