"""Tests for src/feedback.py's shared rules-result message builder and
SpeechFeedback's worker (fake engines only -- no audio)."""

import threading
import time

import pytest

from src.feedback import NO_VOICE_ENV_VAR, SpeechFeedback, message_for_rules_result

WAIT_S = 5.0


class FakeEngine:
    """runAndWait() blocks until release is set; records what it spoke."""

    def __init__(self, release=None):
        self.release = release or threading.Event()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.spoken = []
        self.created_on = threading.current_thread()
        self.used_on = None
        self._pending = None

    def say(self, text):
        self._pending = text

    def runAndWait(self):
        self.used_on = threading.current_thread()
        self.started.set()
        assert self.release.wait(WAIT_S)
        self.spoken.append(self._pending)
        self.finished.set()


def wait_until(condition):
    deadline = time.monotonic() + WAIT_S
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


@pytest.fixture(autouse=True)
def voice_enabled(monkeypatch):
    monkeypatch.delenv(NO_VOICE_ENV_VAR, raising=False)


def test_speak_returns_while_the_engine_is_still_speaking():
    engine = FakeEngine()
    speech = SpeechFeedback(engine_factory=lambda: engine)
    start = time.perf_counter()
    assert speech.speak("hello") is True
    assert time.perf_counter() - start < 0.5
    assert engine.started.wait(WAIT_S)
    assert engine.spoken == []  # still inside runAndWait
    engine.release.set()
    assert engine.finished.wait(WAIT_S)
    assert engine.spoken == ["hello"]


def test_cue_is_dropped_while_the_worker_is_busy_then_speech_resumes():
    engine = FakeEngine()
    speech = SpeechFeedback(engine_factory=lambda: engine)
    assert speech.speak("first") is True
    assert engine.started.wait(WAIT_S)
    assert speech.speak("dropped") is False
    engine.release.set()
    assert engine.finished.wait(WAIT_S)
    wait_until(lambda: speech.speak("second"))
    wait_until(lambda: len(engine.spoken) == 2)
    assert engine.spoken == ["first", "second"]


def test_cue_is_dropped_before_the_worker_has_even_started_it():
    release = threading.Event()
    engine = FakeEngine(release)
    speech = SpeechFeedback(engine_factory=lambda: engine)
    assert speech.speak("first") is True
    assert speech.speak("immediately after") is False
    release.set()
    wait_until(lambda: engine.spoken == ["first"])


def test_disabled_flag_creates_no_engine_and_speaks_nothing(monkeypatch):
    monkeypatch.setenv(NO_VOICE_ENV_VAR, "1")
    created = []
    speech = SpeechFeedback(engine_factory=lambda: created.append(1))
    assert speech.speak("hello") is False
    time.sleep(0.05)
    assert created == []


def test_engine_is_created_and_used_on_the_same_worker_thread():
    # The real-engine hang this guards against: created on one thread, run on another.
    engine_box = []

    def factory():
        engine_box.append(FakeEngine())
        engine_box[0].release.set()
        return engine_box[0]

    speech = SpeechFeedback(engine_factory=factory)
    speech.speak("hello")
    wait_until(lambda: engine_box and engine_box[0].finished.is_set())
    assert engine_box[0].used_on is engine_box[0].created_on
    assert engine_box[0].used_on is not threading.current_thread()


def test_hung_cue_past_the_timeout_restarts_with_a_fresh_engine():
    hung = FakeEngine()
    fresh = FakeEngine()
    fresh.release.set()
    engines = iter([hung, fresh])
    speech = SpeechFeedback(engine_factory=lambda: next(engines), timeout_s=0.05)
    assert speech.speak("hangs") is True
    assert hung.started.wait(WAIT_S)
    time.sleep(0.1)
    assert speech.speak("after restart") is True
    assert fresh.finished.wait(WAIT_S)
    assert fresh.spoken == ["after restart"]
    hung.release.set()


def test_voice_turns_itself_off_after_the_restart_budget():
    release = threading.Event()
    engines = []

    def factory():
        engines.append(FakeEngine(release))
        return engines[-1]

    speech = SpeechFeedback(engine_factory=factory, timeout_s=0.01, max_restarts=1)
    assert speech.speak("one") is True
    wait_until(lambda: len(engines) == 1 and engines[0].started.is_set())
    time.sleep(0.03)
    assert speech.speak("two") is True
    wait_until(lambda: len(engines) == 2 and engines[1].started.is_set())
    time.sleep(0.03)
    assert speech.speak("three") is False
    assert speech.speak("four") is False
    assert len(engines) == 2
    release.set()


def test_engine_failure_never_reaches_the_caller_and_the_next_cue_retries():
    good = FakeEngine()
    good.release.set()
    results = iter([RuntimeError("no audio device"), good])

    def factory():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    speech = SpeechFeedback(engine_factory=factory)
    assert speech.speak("fails") is True
    wait_until(lambda: speech.speak("works"))
    assert good.finished.wait(WAIT_S)
    assert good.spoken == ["works"]


def test_pass_status_gives_good_form_message():
    result = {"status": "pass", "failed_rules": [], "valid_fraction": 1.0}
    assert message_for_rules_result(result) == "Good form!"


def test_fail_status_lists_failed_rules():
    result = {"status": "fail", "failed_rules": ["peak_angle", "tempo"], "valid_fraction": 1.0}
    message = message_for_rules_result(result)
    assert message.startswith("Needs adjustment: ")
    assert "peak_angle" in message
    assert "tempo" in message


def test_insufficient_data_status_reports_valid_fraction_as_percent():
    result = {"status": "insufficient_data", "failed_rules": ["peak_angle"], "valid_fraction": 0.42}
    message = message_for_rules_result(result)
    assert "42%" in message
    assert "Not enough visible tracking" in message


def test_insufficient_data_wins_over_failed_rules_in_the_message():
    # status == insufficient_data must produce the insufficient-data
    # message even if failed_rules is non-empty -- it's a distinct
    # status, not a worse "fail".
    result = {"status": "insufficient_data", "failed_rules": ["peak_angle", "tempo"], "valid_fraction": 0.1}
    message = message_for_rules_result(result)
    assert "Not enough visible tracking" in message
    assert "Needs adjustment" not in message
