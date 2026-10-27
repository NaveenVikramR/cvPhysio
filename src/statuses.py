"""Shared status/reason-code vocabulary -- a leaf module with no
behavior, so src/rules.py and src/tracking.py can both use the SAME
named constants (instead of duplicating or hardcoding the strings
independently in each) without either importing the other. See
open_questions.md #8: RepRecord.status and DiscardRecord.reason_code
were plain, unvalidated strings; app.py's RepDetector and src/rules.py's
score_rep() each hardcoded their own literal strings for the same
values.
"""

from enum import Enum


class RuleStatus(str, Enum):
    """src.rules.score_rep()'s 3-way verdict -- a str subclass so it
    stores, compares, and (de)serializes (CSV, JSON) exactly like the
    plain strings it replaces."""
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


class DiscardReason(str, Enum):
    """Why app.py's RepDetector threw away a buffered segment instead
    of completing it."""
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LOST_TRACKING = "lost_tracking"
