"""Streamlit preview page for the cvPhysio UI theme (src/ui_theme.py):
renders the whole restyled layout -- hero, video/glow ring, stats,
cue banner, the three glass cards, footer -- against MOCK data, with
NO camera and NO MediaPipe import anywhere in this file. A state
selector lets you flip through every state the real app can show
(idle, calibrating, active-pass, active-fail with failed-rule chips,
insufficient_data, error) without a webcam or a browser automation
tool.

Run with: streamlit run ui_preview.py

Every number on this page is made up for layout purposes -- the
"PREVIEW - mock data" banner at the top is there so a screenshot of
this page can never be mistaken for a real session's results.
"""

import numpy as np
import streamlit as st

from src import ui_theme
from src.feedback import draw_overlay

MOCK_FRAME_SIZE = (480, 640)  # (height, width) -- an arbitrary placeholder resolution

STATE_PRESETS = {
    "idle": {
        "cue_state": ui_theme.STATE_IDLE,
        "cue_detail": "Press Start session to begin.",
        "message": "Ready — begin your repetition.",
        "overlay_is_correct": None,
        "reps": None,
        "form_pct": None,
        "tempo_s": None,
        "fps": None,
        "rules": None,
        "log": [],
    },
    "calibrating": {
        "cue_state": ui_theme.STATE_CALIBRATING,
        "cue_detail": "Step back so your left wrist stays visible, then hold still.",
        "message": "Step back so your left wrist stays visible, then hold still.",
        "overlay_is_correct": None,
        "reps": None,
        "form_pct": None,
        "tempo_s": None,
        "fps": None,
        "rules": None,
        "log": [],
    },
    "active-pass": {
        "cue_state": ui_theme.STATE_PASS,
        "cue_detail": None,
        "message": "Rep 3: Good form!",
        "overlay_is_correct": True,
        "reps": 3,
        "form_pct": 100,
        "tempo_s": 4.2,
        "fps": 24.8,
        "rules": {
            "peak_angle": {"measured": 92.3, "allowed": [75.0, 105.0], "passed": True},
            "elbow_straight": {"measured": 168.0, "allowed": [150.0, 180.0], "passed": True},
            "trunk_lean": {"measured": 4.1, "allowed": [0.0, 15.0], "passed": True},
            "tempo": {"measured": 3.8, "allowed": [2.0, 6.0], "passed": True},
        },
        "log": [
            {"rep": 1, "status": ui_theme.STATE_PASS},
            {"rep": 2, "status": ui_theme.STATE_PASS},
            {"rep": 3, "status": ui_theme.STATE_PASS},
        ],
    },
    "active-fail": {
        "cue_state": ui_theme.STATE_FAIL,
        "cue_detail": "Rep 4: Needs adjustment: elbow_straight, trunk_lean",
        "message": "Rep 4: Needs adjustment: elbow_straight, trunk_lean",
        "overlay_is_correct": False,
        "reps": 4,
        "form_pct": 75,
        "tempo_s": 3.9,
        "fps": 23.1,
        "rules": {
            "peak_angle": {"measured": 88.0, "allowed": [75.0, 105.0], "passed": True},
            "elbow_straight": {"measured": 131.0, "allowed": [150.0, 180.0], "passed": False},
            "trunk_lean": {"measured": 21.4, "allowed": [0.0, 15.0], "passed": False},
            "tempo": {"measured": 4.0, "allowed": [2.0, 6.0], "passed": True},
        },
        "log": [
            {"rep": 1, "status": ui_theme.STATE_PASS},
            {"rep": 2, "status": ui_theme.STATE_PASS},
            {"rep": 3, "status": ui_theme.STATE_PASS},
            {"rep": 4, "status": ui_theme.STATE_FAIL},
        ],
    },
    "insufficient_data": {
        "cue_state": ui_theme.STATE_INSUFFICIENT,
        "cue_detail": None,  # fixed safety label -- a caller's detail is never honoured for this state
        "message": "Rep 2: Not enough visible tracking to score this rep (41% valid).",
        "overlay_is_correct": None,
        "reps": 2,
        "form_pct": 100,
        "tempo_s": 4.5,
        "fps": 22.0,
        "rules": None,
        "log": [
            {"rep": 1, "status": ui_theme.STATE_PASS},
            {"rep": 2, "status": ui_theme.STATE_INSUFFICIENT},
        ],
    },
    "error: camera not available": {
        "cue_state": ui_theme.STATE_ERROR,
        "cue_detail": None,
        "message": "Could not open the camera: Could not open camera index 0",
        "overlay_is_correct": None,
        "reps": None,
        "form_pct": None,
        "tempo_s": None,
        "fps": None,
        "rules": None,
        "log": [],
        "camera_error": True,
    },
}


def mock_frame() -> np.ndarray:
    """A flat mid-grey placeholder frame -- no camera, no real pixels."""
    h, w = MOCK_FRAME_SIZE
    frame = np.full((h, w, 3), 60, dtype=np.uint8)
    return frame


def stat_value(value, suffix: str = "") -> str:
    """"—" for a None value (e.g. before any rep exists) -- mirrors
    app.py's own rule: never display a hard-coded number."""
    return "—" if value is None else f"{value}{suffix}"


st.set_page_config(page_title="cvPhysio -- UI preview", layout="wide")
st.markdown(ui_theme.css(), unsafe_allow_html=True)

st.markdown(
    '<div style="background:var(--cvp-orange);color:var(--cvp-midnight);'
    'text-align:center;font-weight:600;letter-spacing:.06em;text-transform:uppercase;'
    'padding:8px;border-radius:999px;margin-bottom:16px;">'
    'PREVIEW -- MOCK DATA, no camera, no real session'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="cvp-topbar"><span class="cvp-serif cvp-topbar-brand">cvPhysio</span></div>'
    '<h1 class="cvp-hero-heading">Your rehab coach, <em>in real time.</em></h1>'
    '<p class="cvp-hero-subtitle">Live, on-device pose feedback while you move -- not a diagnosis.</p>',
    unsafe_allow_html=True,
)

state_name = st.selectbox("Preview state", list(STATE_PRESETS))
preset = STATE_PRESETS[state_name]

st.caption(f"Mock message this state would show in the real app: “{preset['message']}”")

if preset.get("camera_error"):
    st.error(preset["message"])
    st.markdown(ui_theme.render_video_ring_style(ui_theme.STATE_ERROR), unsafe_allow_html=True)
    st.markdown(ui_theme.render_cue_banner(ui_theme.STATE_ERROR), unsafe_allow_html=True)
else:
    stats_left_col, video_col, stats_right_col = st.columns([1, 2, 1])
    with stats_left_col:
        st.markdown(ui_theme.render_stat_block("Reps", stat_value(preset["reps"])), unsafe_allow_html=True)
        st.markdown(ui_theme.render_stat_block("Form", stat_value(preset["form_pct"], "%")), unsafe_allow_html=True)
    with video_col:
        with st.container(key=ui_theme.VIDEO_RING_CONTAINER_KEY):
            annotated = draw_overlay(mock_frame(), preset["message"], preset["overlay_is_correct"])
            st.image(annotated, channels="BGR")
        st.markdown(ui_theme.render_video_ring_style(preset["cue_state"]), unsafe_allow_html=True)
    with stats_right_col:
        st.markdown(ui_theme.render_stat_block("Tempo (s/rep)", stat_value(preset["tempo_s"], "s")), unsafe_allow_html=True)
        st.markdown(ui_theme.render_stat_block("FPS", stat_value(preset["fps"])), unsafe_allow_html=True)

    st.markdown(
        ui_theme.render_cue_banner(preset["cue_state"], detail=preset["cue_detail"]),
        unsafe_allow_html=True,
    )

    reference_card, live_feedback_card, session_log_card = st.columns(3)
    with reference_card:
        with st.container(key="cvp-card-reference"):
            st.markdown(ui_theme.render_card_header("Reference"), unsafe_allow_html=True)
            st.caption("demo_patient_1 -- K=3 reps")
    with live_feedback_card:
        with st.container(key="cvp-card-live-feedback"):
            st.markdown(ui_theme.render_card_header("Live feedback"), unsafe_allow_html=True)
            if preset["rules"]:
                chips = "".join(
                    ui_theme.render_rule_chip(rule_id, r["measured"], tuple(r["allowed"]), r["passed"])
                    for rule_id, r in preset["rules"].items()
                )
                st.markdown(f'<div class="cvp-chip-row">{chips}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cvp-muted">No reps yet.</div>', unsafe_allow_html=True)
    with session_log_card:
        with st.container(key="cvp-card-session-log"):
            st.markdown(ui_theme.render_card_header("Session log"), unsafe_allow_html=True)
            st.markdown(ui_theme.render_session_log_table(preset["log"]), unsafe_allow_html=True)

st.markdown(
    f'<div class="cvp-footer">{ui_theme.escape("cvPhysio is a research prototype, not a medical device. "
                                                  "It does not diagnose or treat. Stop if you feel pain. "
                                                  "Video is processed on this device and is not stored.")}</div>',
    unsafe_allow_html=True,
)
