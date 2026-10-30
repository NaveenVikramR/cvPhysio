"""Presentation-only Streamlit theme for cvPhysio: a CSS string (`css()`,
injected ONCE at app start via `st.markdown(..., unsafe_allow_html=True)`,
never inside the per-frame loop) plus small pure functions that return
HTML snippets. No pipeline/scoring/session logic lives here, no
Streamlit widget calls -- everything here is testable without
Streamlit, a browser, or a camera (see tests/test_ui_theme.py).

Escape hatch: if the CVPHYSIO_PLAIN_UI environment variable is "1",
plain_ui_enabled() returns True and app.py skips calling css() (and
every render_* call) entirely, falling back to the previous default
Streamlit look -- see app.py for exactly where that branch is.

Design tokens (colours) are the exact hexes given in the design brief;
every other value (spacing, radii beyond the stated 22px/999px, glow
size/position) is this module's own reasonable estimate, since the
brief says those are "estimated by eye" from a reference image this
module never saw directly.
"""

import html
import os

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

MIDNIGHT = "#02070A"
DEEP = "#04152E"  # bottom of the page's vertical gradient
ELECTRIC_BLUE = "#0297DE"
CANTALOUPE = "#FC734D"
WHITE = "#FFFFFF"
MUTED = "rgba(255,255,255,.64)"
HAIRLINE = "rgba(255,255,255,.14)"
CARD_BG = "rgba(255,255,255,.06)"

CARD_RADIUS_PX = 22
PILL_RADIUS_PX = 999

SERIF_STACK = '"Instrument Serif", "Playfair Display", Georgia, "Times New Roman", serif'

STATE_PASS = "pass"
STATE_FAIL = "fail"
STATE_INSUFFICIENT = "insufficient_data"
STATE_IDLE = "idle"
STATE_CALIBRATING = "calibrating"
STATE_ERROR = "error"

ALL_STATES = (STATE_PASS, STATE_FAIL, STATE_INSUFFICIENT, STATE_IDLE, STATE_CALIBRATING, STATE_ERROR)

# icon + colour + default label per state. Colour is NEVER the only
# signal -- every state also gets a distinct icon and text, per the
# brief's STATE SEMANTICS section.
_STATE_DISPLAY = {
    STATE_PASS: {"color": ELECTRIC_BLUE, "icon": "✓", "label": "On target"},
    STATE_FAIL: {"color": CANTALOUPE, "icon": "✕", "label": "Needs adjustment"},
    STATE_INSUFFICIENT: {"color": MUTED, "icon": "\U0001F441", "label": "Can’t tell — keep your whole arm in view"},
    STATE_IDLE: {"color": MUTED, "icon": "○", "label": "Idle"},
    STATE_CALIBRATING: {"color": WHITE, "icon": "◐", "label": "Calibrating — hold still"},
    STATE_ERROR: {"color": CANTALOUPE, "icon": "⚠", "label": "Camera not available"},
}


def plain_ui_enabled() -> bool:
    """True iff CVPHYSIO_PLAIN_UI=1 -- the escape hatch that disables
    every custom CSS/HTML call in this module."""
    return os.environ.get("CVPHYSIO_PLAIN_UI") == "1"


def escape(value) -> str:
    """html.escape() on str(value) -- every dynamic string this module
    places into an HTML snippet goes through this first."""
    return html.escape(str(value), quote=True)


# States whose label a caller's `detail` may override: FAIL (names the
# failed rule) and the two purely-informational states IDLE/CALIBRATING
# (e.g. a specific calibration_hint() string). PASS, INSUFFICIENT_DATA,
# and ERROR keep their fixed, safety-relevant label no matter what a
# caller passes, so a stray detail string can never silently replace
# "Can't tell -- keep your whole arm in view" or similar.
_DETAIL_OVERRIDABLE_STATES = (STATE_FAIL, STATE_IDLE, STATE_CALIBRATING)


def state_display(state: str, detail: str | None = None) -> dict:
    """{"color", "icon", "label"} for one of ALL_STATES. `detail`, if
    given, overrides the label only for states in
    _DETAIL_OVERRIDABLE_STATES -- every other state ignores it, so a
    stray detail string can never silently replace a safety-relevant
    label like the insufficient_data text."""
    if state not in _STATE_DISPLAY:
        raise ValueError(f"Unknown state {state!r}; must be one of {ALL_STATES}")
    result = dict(_STATE_DISPLAY[state])
    if state in _DETAIL_OVERRIDABLE_STATES and detail:
        result["label"] = detail
    return result


def glow_ring_color(state: str) -> str:
    """The video feed's glow-ring colour: blue=good (pass), orange=
    attention (fail), white/grey=idle or can't tell (every other
    state) -- matches the brief exactly; never any other colour."""
    if state == STATE_PASS:
        return ELECTRIC_BLUE
    if state == STATE_FAIL:
        return CANTALOUPE
    return MUTED


def format_allowed_range(lo: float, hi: float) -> str:
    """'90-105', '>= 150', '<= 15' -- renders an open bound (+/-inf,
    from reference_relative mode's one-sided rules) without printing
    'inf'."""
    lo_open = lo == float("-inf")
    hi_open = hi == float("inf")
    if lo_open and hi_open:
        return "any"
    if hi_open:
        return f"≥ {lo:.0f}"
    if lo_open:
        return f"≤ {hi:.0f}"
    return f"{lo:.0f}–{hi:.0f}"


# ---------------------------------------------------------------------------
# HTML-returning pure functions
# ---------------------------------------------------------------------------

def format_rep_tempo(duration_s: float | None) -> str:
    """The Tempo stat: the last rep's real duration, the same figure the tempo
    rule scored -- not session time / reps, which counts rests and pauses."""
    return f"{duration_s:.1f}s" if duration_s is not None else "—"


def render_stat_block(label: str, value: str) -> str:
    """One stat block: a serif numeral + a small uppercase caption.
    `value` is already formatted by the caller (e.g. "12", "84%", "—")
    -- this function never invents or defaults a number itself."""
    return (
        '<div class="cvp-stat">'
        f'<div class="cvp-stat-value">{escape(value)}</div>'
        f'<div class="cvp-stat-label">{escape(label)}</div>'
        '</div>'
    )


def render_rep_counter(value: str) -> str:
    return f'<div class="cvp-rep-counter">{escape(value)}</div>'


def render_cue_banner(state: str, detail: str | None = None) -> str:
    d = state_display(state, detail)
    dashed = ' cvp-cue-dashed' if state == STATE_INSUFFICIENT else ''
    return (
        f'<div class="cvp-cue cvp-cue-{escape(state)}{dashed}" style="--cvp-accent:{d["color"]}">'
        f'<span class="cvp-cue-icon" aria-hidden="true">{d["icon"]}</span>'
        f'<span class="cvp-cue-text">{escape(d["label"])}</span>'
        '</div>'
    )


def render_rule_chip(rule_id: str, measured: float, allowed: tuple[float, float], passed: bool) -> str:
    state = STATE_PASS if passed else STATE_FAIL
    d = state_display(state, rule_id.replace("_", " ") if not passed else None)
    allowed_text = format_allowed_range(allowed[0], allowed[1])
    return (
        f'<span class="cvp-chip cvp-chip-{escape(state)}" style="--cvp-accent:{d["color"]}">'
        f'<span class="cvp-chip-icon" aria-hidden="true">{d["icon"]}</span>'
        f'<span class="cvp-chip-text">{escape(rule_id)}: {escape(f"{measured:.1f}")} '
        f'(allowed {escape(allowed_text)})</span>'
        '</span>'
    )


def render_session_log_table(rows: list[dict]) -> str:
    """rows: [{"rep": int, "status": one of ALL_STATES, "detail": str|None}, ...]
    -- one row per completed rep, most-recent-last (caller's order is
    preserved, not re-sorted here)."""
    if not rows:
        return '<div class="cvp-muted">No reps yet.</div>'
    body_rows = []
    for row in rows:
        d = state_display(row["status"], row.get("detail"))
        body_rows.append(
            '<tr>'
            f'<td class="cvp-log-rep">{escape(row["rep"])}</td>'
            f'<td><span class="cvp-chip cvp-chip-{escape(row["status"])}" style="--cvp-accent:{d["color"]}">'
            f'<span class="cvp-chip-icon" aria-hidden="true">{d["icon"]}</span> '
            f'<span class="cvp-chip-text">{escape(d["label"])}</span></span></td>'
            '</tr>'
        )
    return '<table class="cvp-log"><tbody>' + "".join(body_rows) + '</tbody></table>'


def render_card_header(label: str, dot_color: str = MUTED) -> str:
    """The small dot + label heading used at the top of each of the 3
    glass cards (Reference / Live feedback / Session log)."""
    return (
        '<div class="cvp-card-header">'
        f'<span class="cvp-dot" style="background:{dot_color}"></span>'
        f'<span class="cvp-card-label">{escape(label)}</span>'
        '</div>'
    )


VIDEO_RING_CONTAINER_KEY = "cvp-video-ring"


def render_video_ring_style(state: str) -> str:
    """A tiny <style> snippet overriding the --cvp-ring-color custom
    property that .st-key-cvp-video-ring (see css()) reads for its glow.

    Callers must create `st.container(key=ui_theme.VIDEO_RING_CONTAINER_KEY)`
    ONCE, before any per-frame loop, and place the video's st.empty()
    placeholder INSIDE it -- st.container(key=...) is the only way to
    get Streamlit to actually nest a widget inside custom-styled markup;
    splitting a raw <div> open/close tag across two separate
    st.markdown() calls does NOT nest a real widget placed in between
    (confirmed against the installed Streamlit version), so an
    open/close-div-pair function (an earlier version of this one) can't
    actually draw a ring around the video and was replaced by this.

    Call this once per detected state CHANGE, not unconditionally every
    frame, and render it into its own small st.empty() placeholder --
    the colour is the only thing that updates per frame-loop iteration;
    the stylesheet itself (css()) still only renders once at start."""
    return f'<style>:root{{--cvp-ring-color:{escape(glow_ring_color(state))}}}</style>'


# ---------------------------------------------------------------------------
# CSS (injected once; see plain_ui_enabled() for the escape hatch)
# ---------------------------------------------------------------------------

def css() -> str:
    return f"""<style>
:root {{
  --cvp-midnight: {MIDNIGHT};
  --cvp-deep: {DEEP};
  --cvp-blue: {ELECTRIC_BLUE};
  --cvp-orange: {CANTALOUPE};
  --cvp-white: {WHITE};
  --cvp-muted: {MUTED};
  --cvp-hairline: {HAIRLINE};
  --cvp-card-bg: {CARD_BG};
}}

.stApp {{
  background:
    radial-gradient(ellipse 60% 40% at 50% 0%, rgba(2,151,222,.45), transparent 70%),
    radial-gradient(ellipse 70% 35% at 50% 100%, rgba(252,115,77,.5), transparent 70%),
    linear-gradient(180deg, var(--cvp-midnight), var(--cvp-deep));
  color: var(--cvp-white);
}}

.cvp-serif {{
  font-family: {SERIF_STACK};
}}

.cvp-topbar {{
  display: flex;
  align-items: center;
  padding: 8px 0 20px 0;
}}
.cvp-topbar-brand {{
  font-size: clamp(18px, 2vw, 22px);
  font-weight: 600;
}}

.cvp-hero-heading {{
  font-family: {SERIF_STACK};
  font-size: clamp(28px, 5vw, 56px);
  line-height: 1.1;
  text-align: center;
  margin: 0 0 8px 0;
}}
.cvp-hero-heading em {{
  font-style: italic;
  color: var(--cvp-blue);
}}
.cvp-hero-subtitle {{
  text-align: center;
  color: var(--cvp-muted);
  font-size: clamp(14px, 1.4vw, 17px);
  margin-bottom: 24px;
}}

.cvp-label-sm {{
  font-size: clamp(11px, 1vw, 12px);
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--cvp-muted);
}}

/* Glass cards: static elements only -- backdrop-filter never applied
   to anything overlapping the video feed (see .cvp-video-ring below). */
.cvp-card {{
  background: var(--cvp-card-bg);
  border: 1px solid var(--cvp-hairline);
  border-radius: {CARD_RADIUS_PX}px;
  box-shadow: 0 12px 40px rgba(0,0,0,.45);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  padding: 20px;
}}
.cvp-card-header {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}}
.cvp-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex: none;
}}
.cvp-card-label {{
  font-size: clamp(11px, 1vw, 12px);
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--cvp-muted);
}}

/* Pills / buttons */
.cvp-pill {{
  border-radius: {PILL_RADIUS_PX}px;
  padding: 10px 22px;
  font-weight: 600;
  display: inline-block;
  border: 1px solid var(--cvp-hairline);
}}
.cvp-pill-primary {{
  background: var(--cvp-white);
  color: var(--cvp-midnight);
  border: none;
}}
.cvp-pill-secondary {{
  background: var(--cvp-card-bg);
  color: var(--cvp-white);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
}}
.cvp-pill-stop {{
  background: transparent;
  color: var(--cvp-orange);
  border: 1px solid var(--cvp-orange);
}}

/* Re-skin Streamlit's OWN native button/segmented-control widgets as
   pills, instead of replacing them with raw HTML -- native widgets are
   what actually wire clicks back to Python; raw HTML buttons can't,
   without a custom component (ruled out: "no new dependencies"). These
   selectors target Streamlit 1.66's data-testid/kind attributes; a
   future Streamlit upgrade could rename them, in which case buttons
   still work, they just lose this re-skin (CVPHYSIO_PLAIN_UI=1 always
   gets Streamlit's own default button look regardless). */
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stButton"] button[data-testid="stBaseButton-primary"] {{
  background: var(--cvp-white) !important;
  color: var(--cvp-midnight) !important;
  border: none !important;
  border-radius: {PILL_RADIUS_PX}px !important;
  font-weight: 600 !important;
}}
div[data-testid="stButton"] button[kind="secondary"],
div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {{
  background: transparent !important;
  color: var(--cvp-orange) !important;
  border: 1px solid var(--cvp-orange) !important;
  border-radius: {PILL_RADIUS_PX}px !important;
}}
/* st.segmented_control's selected option -- blue fill, midnight text
   (an accent fill, so WCAG requires midnight text, same rule as any
   other accent-filled pill). */
div[data-testid="stSegmentedControl"] label[data-checked="true"],
div[data-testid="stSegmentedControl"] [aria-checked="true"] {{
  background: var(--cvp-blue) !important;
  color: var(--cvp-midnight) !important;
}}

/* Video feed: rounded corners + a glow ring on the st.container(key=
   VIDEO_RING_CONTAINER_KEY) WRAPPER, never a filter/backdrop-filter/
   animation on the video element itself (costs real fps) -- the ring
   colour is a CSS custom property updated by render_video_ring_style()
   on state change only, not unconditionally every frame. */
[class*="st-key-cvp-video-ring"] {{
  border-radius: {CARD_RADIUS_PX}px;
  padding: 6px;
  box-shadow: 0 0 60px 10px var(--cvp-ring-color, {MUTED});
  display: inline-block;
}}
[class*="st-key-cvp-video-ring"] img {{
  border-radius: {CARD_RADIUS_PX - 6}px;
  display: block;
}}

/* Stats */
.cvp-stat {{
  text-align: center;
}}
.cvp-stat-value {{
  font-family: {SERIF_STACK};
  font-size: clamp(32px, 4vw, 48px);
  line-height: 1;
}}
.cvp-stat-label {{
  font-size: clamp(11px, 1vw, 12px);
  text-transform: uppercase;
  letter-spacing: .06em;
  color: var(--cvp-muted);
  margin-top: 4px;
}}

/* Rep counter -- always >= 96px, legible from 2m */
.cvp-rep-counter {{
  font-family: {SERIF_STACK};
  font-size: clamp(96px, 10vw, 140px);
  line-height: 1;
  text-align: center;
}}

/* Cue banner -- always >= 32px at 1080p, legible from 2m */
.cvp-cue {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  font-size: clamp(32px, 3vw, 44px);
  font-weight: 600;
  color: var(--cvp-accent, {WHITE});
  padding: 16px;
  border-radius: {CARD_RADIUS_PX}px;
  border: 1px solid var(--cvp-hairline);
}}
.cvp-cue-dashed {{
  border: 2px dashed var(--cvp-accent, {MUTED});
}}
.cvp-cue-icon {{
  font-size: 1em;
}}

/* Chips (per-rule, per-rep-status) */
.cvp-chip {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: {PILL_RADIUS_PX}px;
  padding: 4px 12px;
  font-size: 13px;
  color: var(--cvp-accent, {WHITE});
  border: 1px solid var(--cvp-accent, {HAIRLINE});
  background: rgba(255,255,255,.04);
}}
.cvp-chip-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}}

/* Session log table */
.cvp-log {{
  width: 100%;
  border-collapse: collapse;
}}
.cvp-log td {{
  padding: 8px 4px;
  border-bottom: 1px solid var(--cvp-hairline);
  font-size: 14px;
}}
.cvp-log-rep {{
  color: var(--cvp-muted);
  width: 2.5em;
}}

.cvp-muted {{
  color: var(--cvp-muted);
}}

.cvp-footer {{
  text-align: center;
  color: var(--cvp-muted);
  font-size: 12px;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid var(--cvp-hairline);
}}

/* Respect the user's motion preference -- no transition/animation
   anywhere in this stylesheet is load-bearing, but keep this rule as
   a hard guarantee for anything a future edit might add. */
@media (prefers-reduced-motion: reduce) {{
  * {{
    animation: none !important;
    transition: none !important;
  }}
}}
</style>"""
