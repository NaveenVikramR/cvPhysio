"""Tests for src/feedback.py's shared rules-result message builder."""

from src.feedback import message_for_rules_result


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
