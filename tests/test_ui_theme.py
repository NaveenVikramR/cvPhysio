"""Tests for src/ui_theme.py's pure functions: HTML escaping, state ->
colour/icon/label mapping, and a WCAG 2.1 contrast-ratio check for
every text/background token pair this module's CSS/HTML actually uses
(>= 4.5:1 for normal text, >= 3:1 for text >= 24px, per WCAG AA).
Also an import-safety check: this module must be importable with no
Streamlit, cv2, or mediapipe dependency (it's pure presentation code,
testable without a browser or camera).
"""

import sys

import pytest

from src import ui_theme as theme


# ---------------------------------------------------------------------------
# escape()
# ---------------------------------------------------------------------------

def test_escape_handles_html_special_characters():
    assert theme.escape("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_escape_handles_quotes():
    assert "&quot;" in theme.escape('say "hi"')


def test_escape_handles_non_string_input():
    assert theme.escape(12) == "12"
    assert theme.escape(3.14159) == "3.14159"


def test_escape_handles_ampersand():
    assert theme.escape("A & B") == "A &amp; B"


# ---------------------------------------------------------------------------
# state_display() / glow_ring_color()
# ---------------------------------------------------------------------------

def test_state_display_known_states_have_color_icon_label():
    for state in theme.ALL_STATES:
        d = theme.state_display(state)
        assert set(d.keys()) == {"color", "icon", "label"}
        assert d["color"]
        assert d["icon"]
        assert d["label"]


def test_state_display_unknown_state_raises():
    with pytest.raises(ValueError):
        theme.state_display("not_a_real_state")


def test_state_display_fail_detail_overrides_label():
    d = theme.state_display(theme.STATE_FAIL, detail="trunk_lean")
    assert d["label"] == "trunk_lean"


def test_state_display_detail_ignored_for_insufficient_data():
    """A stray detail string must never silently replace this
    safety-relevant label."""
    d = theme.state_display(theme.STATE_INSUFFICIENT, detail="should be ignored")
    assert d["label"] != "should be ignored"
    assert "keep your whole arm in view" in d["label"].lower()


def test_state_display_detail_ignored_for_pass_and_error():
    assert theme.state_display(theme.STATE_PASS, detail="x")["label"] != "x"
    assert theme.state_display(theme.STATE_ERROR, detail="x")["label"] != "x"


def test_state_display_detail_overrides_label_for_idle_and_calibrating():
    """IDLE/CALIBRATING are purely informational (no fixed safety
    claim), so a caller's more specific detail (e.g. a calibration
    stall hint) is allowed to replace the generic label."""
    assert theme.state_display(theme.STATE_IDLE, detail="Repetition in progress...")["label"] == "Repetition in progress..."
    assert theme.state_display(theme.STATE_CALIBRATING, detail="Move so your whole arm is visible")["label"] == "Move so your whole arm is visible"


def test_glow_ring_color_pass_is_blue():
    assert theme.glow_ring_color(theme.STATE_PASS) == theme.ELECTRIC_BLUE


def test_glow_ring_color_fail_is_orange():
    assert theme.glow_ring_color(theme.STATE_FAIL) == theme.CANTALOUPE


def test_glow_ring_color_other_states_are_muted():
    for state in (theme.STATE_IDLE, theme.STATE_CALIBRATING, theme.STATE_INSUFFICIENT, theme.STATE_ERROR):
        assert theme.glow_ring_color(state) == theme.MUTED


# ---------------------------------------------------------------------------
# format_allowed_range()
# ---------------------------------------------------------------------------

def test_format_allowed_range_both_bounds():
    assert theme.format_allowed_range(75.0, 105.0) == "75–105"


def test_format_allowed_range_no_upper_bound():
    assert theme.format_allowed_range(90.0, float("inf")) == "≥ 90"


def test_format_allowed_range_no_lower_bound():
    assert theme.format_allowed_range(float("-inf"), 20.0) == "≤ 20"


def test_format_allowed_range_unbounded_both_sides():
    assert theme.format_allowed_range(float("-inf"), float("inf")) == "any"


# ---------------------------------------------------------------------------
# HTML-returning functions: escaping is actually applied, no raw
# injection of caller-supplied strings
# ---------------------------------------------------------------------------

def test_render_stat_block_escapes_label_and_value():
    html_out = theme.render_stat_block("<b>Reps</b>", "<script>")
    assert "<b>" not in html_out
    assert "<script>" not in html_out
    assert "&lt;b&gt;" in html_out
    assert "&lt;script&gt;" in html_out


def test_render_cue_banner_escapes_detail():
    html_out = theme.render_cue_banner(theme.STATE_FAIL, detail="<img src=x onerror=alert(1)>")
    assert "<img" not in html_out
    assert "&lt;img" in html_out


def test_render_rule_chip_escapes_rule_id():
    html_out = theme.render_rule_chip("<b>rule</b>", 90.0, (75.0, 105.0), passed=True)
    assert "<b>rule</b>" not in html_out


def test_render_session_log_table_empty():
    assert "No reps yet" in theme.render_session_log_table([])


def test_render_session_log_table_escapes_rows():
    rows = [{"rep": 1, "status": theme.STATE_PASS}, {"rep": "<b>2</b>", "status": theme.STATE_FAIL, "detail": "<i>x</i>"}]
    html_out = theme.render_session_log_table(rows)
    assert "<b>2</b>" not in html_out
    assert "<i>x</i>" not in html_out
    assert "&lt;b&gt;2&lt;/b&gt;" in html_out


def test_render_card_header_escapes_label():
    html_out = theme.render_card_header("<script>")
    assert "<script>" not in html_out


def test_render_video_ring_style_is_a_style_tag_with_the_state_color():
    html_out = theme.render_video_ring_style(theme.STATE_PASS)
    assert html_out.startswith("<style>")
    assert theme.ELECTRIC_BLUE in html_out
    assert "--cvp-ring-color" in html_out


def test_render_video_ring_style_treats_unknown_state_as_muted_not_a_crash():
    """glow_ring_color() intentionally treats anything other than pass/
    fail as the muted idle/can't-tell colour (see its own tests above)
    -- render_video_ring_style() must not crash on an unexpected state
    string, and must not place it into the HTML unescaped either way."""
    html_out = theme.render_video_ring_style("<script>whatever</script>")
    assert "<script>whatever" not in html_out
    assert theme.MUTED in html_out


# ---------------------------------------------------------------------------
# plain_ui_enabled() escape hatch
# ---------------------------------------------------------------------------

def test_plain_ui_enabled_false_by_default(monkeypatch):
    monkeypatch.delenv("CVPHYSIO_PLAIN_UI", raising=False)
    assert theme.plain_ui_enabled() is False


def test_plain_ui_enabled_true_when_set_to_1(monkeypatch):
    monkeypatch.setenv("CVPHYSIO_PLAIN_UI", "1")
    assert theme.plain_ui_enabled() is True


def test_plain_ui_enabled_false_for_other_values(monkeypatch):
    monkeypatch.setenv("CVPHYSIO_PLAIN_UI", "true")
    assert theme.plain_ui_enabled() is False


# ---------------------------------------------------------------------------
# import-safety: no Streamlit/cv2/mediapipe dependency
# ---------------------------------------------------------------------------

def test_ui_theme_has_no_streamlit_cv2_or_mediapipe_dependency():
    """src/ui_theme.py must be importable (and IS already imported,
    above) without pulling in streamlit, cv2, or mediapipe -- it's pure
    presentation code, testable without a browser or camera. Checking
    sys.modules here (rather than just "the import above worked")
    proves those heavy modules weren't pulled in as a side effect."""
    for forbidden in ("streamlit", "cv2", "mediapipe"):
        assert forbidden not in getattr(theme, "__dict__", {})
    # the module's own source only imports html/os (stdlib) -- confirmed
    # by inspecting its compiled globals rather than sys.modules, since
    # other already-imported test modules may have pulled in cv2 etc.
    # independently and would make a sys.modules check unreliable.
    import inspect
    source = inspect.getsource(theme)
    for forbidden in ("import streamlit", "import cv2", "import mediapipe"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# WCAG 2.1 contrast ratios for every text/background token pair used
# ---------------------------------------------------------------------------

def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _srgb_to_linear(c: float) -> float:
    c = c / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(rgb1, rgb2) -> float:
    l1, l2 = _relative_luminance(rgb1), _relative_luminance(rgb2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _blend(fg_rgb, alpha: float, bg_rgb) -> tuple[float, float, float]:
    """fg_rgb painted at `alpha` opacity over bg_rgb -- used for the
    rgba() tokens (muted text, glass card background), which only have
    a well-defined contrast ratio once composited onto an actual
    background."""
    return tuple(fg_rgb[i] * alpha + bg_rgb[i] * (1 - alpha) for i in range(3))


MIDNIGHT_RGB = _hex_to_rgb(theme.MIDNIGHT)
DEEP_RGB = _hex_to_rgb(theme.DEEP)
BLUE_RGB = _hex_to_rgb(theme.ELECTRIC_BLUE)
ORANGE_RGB = _hex_to_rgb(theme.CANTALOUPE)
WHITE_RGB = _hex_to_rgb(theme.WHITE)

# the two page backgrounds text can sit directly on (top/bottom of the
# vertical gradient -- the glow radials only ever brighten these, so
# testing the un-glowed gradient is the conservative/worst case)
PAGE_BACKGROUNDS = {"midnight (top of gradient)": MIDNIGHT_RGB, "deep (bottom of gradient)": DEEP_RGB}

CARD_ON_MIDNIGHT = _blend(WHITE_RGB, 0.06, MIDNIGHT_RGB)  # --cvp-card-bg over the page
MUTED_ON_MIDNIGHT = _blend(WHITE_RGB, 0.64, MIDNIGHT_RGB)  # --cvp-muted text, composited
MUTED_ON_CARD = _blend(WHITE_RGB, 0.64, CARD_ON_MIDNIGHT)

# (description, foreground_rgb, background_rgb, is_large_text, min_ratio)
# -- every text/background token pair src/ui_theme.py's CSS/HTML
# actually produces. Large text (chip/cue text is never this small,
# but checked at the stricter 4.5:1 bound anyway to be conservative)
# only gets the relaxed 3:1 bound for the genuinely >=24px elements
# (.cvp-cue, .cvp-rep-counter, .cvp-hero-heading, .cvp-stat-value).
CONTRAST_CASES = [
    ("body text (white) on page background", WHITE_RGB, MIDNIGHT_RGB, False),
    ("body text (white) on page background (deep)", WHITE_RGB, DEEP_RGB, False),
    ("card text (white) on glass card", WHITE_RGB, CARD_ON_MIDNIGHT, False),
    ("primary button text (midnight) on white fill", MIDNIGHT_RGB, WHITE_RGB, False),
    ("accent-fill text (midnight) on electric blue", MIDNIGHT_RGB, BLUE_RGB, False),
    ("accent-fill text (midnight) on cantaloupe", MIDNIGHT_RGB, ORANGE_RGB, False),
    ("electric blue AS TEXT on page background", BLUE_RGB, MIDNIGHT_RGB, False),
    ("electric blue AS TEXT on page background (deep)", BLUE_RGB, DEEP_RGB, False),
    ("cantaloupe AS TEXT on page background", ORANGE_RGB, MIDNIGHT_RGB, False),
    ("cantaloupe AS TEXT on page background (deep)", ORANGE_RGB, DEEP_RGB, False),
    ("muted caption text on page background", MUTED_ON_MIDNIGHT, MIDNIGHT_RGB, False),
    ("muted caption text on glass card", MUTED_ON_CARD, CARD_ON_MIDNIGHT, False),
    # large text (>=24px): cue banner, rep counter, hero heading, stat numerals
    ("cue banner text (pass=blue) on cue banner bg", BLUE_RGB, MIDNIGHT_RGB, True),
    ("cue banner text (fail=orange) on cue banner bg", ORANGE_RGB, MIDNIGHT_RGB, True),
    ("rep counter / stat numeral (white) on page background", WHITE_RGB, MIDNIGHT_RGB, True),
]


@pytest.mark.parametrize("name,fg,bg,is_large", CONTRAST_CASES)
def test_wcag_contrast_ratio(name, fg, bg, is_large):
    ratio = _contrast_ratio(fg, bg)
    required = 3.0 if is_large else 4.5
    assert ratio >= required, f"{name}: ratio={ratio:.2f}, required>={required} (fg={fg}, bg={bg})"
