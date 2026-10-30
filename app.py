"""Streamlit UI: wires pose_estimation -> feature_extraction -> classifier
-> feedback -> tracking into a live webcam session.

Classification happens once per completed repetition, not once per frame --
RepDetector below detects rep boundaries automatically from the live angle
signal, since the trained models (src/classifier.py's FEATURE_COLUMNS) need
a whole rep's worth of frames aggregated via feature_extraction.extract_rep_
features(), not a single frame's angle_deg.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from src import rules as rules_module
from src import ui_theme
from src.classifier import is_within_correct_range
from src.feature_extraction import EXERCISES, extract_rep_features
from src.feedback import SpeechFeedback, draw_overlay, message_for_result, message_for_rules_result
from src.pipeline import (
    MODELS_DIR,
    Session,
    build_smoother,
    load_classifier_for_live_use,
)
from src.pose_estimation import get_pose_landmarks, open_webcam
from src.statuses import RuleStatus
from src.tracking import DiscardRecord, RepRecord, SessionTracker, load_history

CLASSIFIER_WARNING = (
    "WARNING: --scorer classifier was trained on UI-PRMD Kinect recordings, "
    "not webcam video, and is not calibrated for webcam input "
    "(see scripts/diagnose_misclassification.py). Treat its output with caution."
)


def parse_args():
    parser = argparse.ArgumentParser(description="cvPhysio rehab coach")
    parser.add_argument("--scorer", choices=["rules", "classifier"], default="rules")
    parser.add_argument("--video", type=str, default=None,
                         help="Path to a video file to score headlessly instead of launching the Streamlit webcam UI.")
    parser.add_argument("--exercise", choices=list(EXERCISES), default="shoulder_abduction")
    parser.add_argument("--span", type=int, nargs=2, default=None, metavar=("START", "END"),
                         help="Frame indices [START, END] (inclusive), only with --video: equivalent to "
                              "pressing Start at frame START and Stop right after frame END. Frames "
                              "outside the span are ignored by the rep detector (src.pipeline.Session). "
                              "Omit to score the whole clip (Start before frame 0, never Stop) -- the "
                              "pre-this-flag behavior.")
    parser.add_argument("--reference", type=str, default=None,
                         help="Path to a Reference JSON (src.rules.load_reference()) to score against in "
                              "'reference_relative' mode. Ignored if the active rules config's mode isn't "
                              "'reference_relative'; if that mode is active but this is omitted, score_rep() "
                              "falls back to absolute thresholds and reports a one-line warning per rep.")
    return parser.parse_args(sys.argv[1:])


def run_headless(video_path: str, exercise: str, scorer: str, span: tuple[int, int] | None = None,
                  reference_path: str | None = None) -> None:
    """--video path: score a video file frame-by-frame with no Streamlit
    UI, printing per-rep results to stdout. Used for verification against
    real recorded clips (no webcam available in that context).

    `span`, if given, is (start_frame, end_frame) inclusive: Session.
    start() is called exactly at frame_idx == start_frame, Session.
    stop() right after processing frame_idx == end_frame -- frames
    outside that span are ignored by the rep detector entirely. If
    omitted, Session.start() is called before frame 0 and never
    stopped, which is exactly the pre-this-flag behavior (every frame
    reaches the rep detector)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    config = None
    classifier = None
    correct_feature_ranges = None
    required_roles = None
    if scorer == "rules":
        try:
            config = rules_module.load_config(exercise)
        except FileNotFoundError as exc:
            print(str(exc))
            cap.release()
            return
        required_roles = rules_module.required_joint_roles()
    else:
        if reference_path:
            print("WARNING: --reference is only used with --scorer rules; ignored with --scorer classifier.")
        print(CLASSIFIER_WARNING)
        try:
            classifier, correct_feature_ranges = load_classifier_for_live_use(exercise)
        except RuntimeError as exc:
            print(str(exc))
            cap.release()
            return
        if classifier is None:
            print(f"No trained model at {MODELS_DIR / f'{exercise}.pkl'} -- cannot score with --scorer classifier.")
            cap.release()
            return

    reference = None
    if scorer == "rules" and reference_path:
        reference = rules_module.load_reference(reference_path)
        print(f"Reference: {reference_path} (name={reference['name']}, k_reps={reference['k_reps']})")

    session = Session(exercise)
    smoother = build_smoother()
    cue_cooldown = rules_module.CueCooldown(config["cue_cooldown_s"]["value"]) if scorer == "rules" else None
    rep_count = 0
    frame_idx = 0
    last_hint_printed = None
    if span is None:
        session.start()  # Start before frame 0, never Stop -- identical to the pre-this-flag behavior
    print(f"Scoring {video_path} (exercise={exercise}, scorer={scorer}, fps={fps:.2f}, smoothing={'on' if smoother else 'off'}"
          f"{f', span={span}' if span else ''})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if span is not None and frame_idx == span[0]:
            session.start()
        landmarks = get_pose_landmarks(frame)
        event, payload, _visible, _failed = session.process_frame(landmarks, smoother, t, required_roles)
        if span is not None and frame_idx == span[1]:
            session.stop()
        frame_idx += 1

        if session.detector is not None and session.detector.state == "CALIBRATING":
            hint = session.detector.calibration_hint()
            if hint and hint != last_hint_printed:
                print(f"  (calibration stalled at t={t:.1f}s: {hint})")
                last_hint_printed = hint
        else:
            last_hint_printed = None

        if event == "rep_complete":
            rep_count += 1
            if scorer == "rules":
                rep = {
                    "angle_deg": payload["angle_deg"],
                    "elbow_angle_deg": payload["elbow_angle_deg"],
                    "trunk_lean_deg": payload["trunk_lean_deg"],
                    "valid_fraction": payload["valid_fraction"],
                }
                result = rules_module.score_rep(rep, config, fps, reference=reference)
                print(f"Rep {rep_count}: {result['status'].upper()}  failed_rules={result['failed_rules']}  "
                      f"valid_fraction={result['valid_fraction']:.2f}")
                for rule_id, r in result["rules"].items():
                    print(f"    {rule_id}: measured={r['measured']:.1f}  "
                          f"allowed=[{r['allowed'][0]:.1f}, {r['allowed'][1]:.1f}]  passed={r['passed']}")
                print(f"    message: {message_for_rules_result(result)}")
                if result["warning"]:
                    print(f"    warning: {result['warning']}")
                cues = cue_cooldown.cues_to_speak(result["failed_rules"], config["cues"], now=t) if result["status"] == RuleStatus.FAIL.value else []
                if result["status"] == RuleStatus.INSUFFICIENT_DATA.value and result["cue"] and cue_cooldown.should_fire(RuleStatus.INSUFFICIENT_DATA.value, now=t):
                    cues = [result["cue"]]
                    cue_cooldown.record(RuleStatus.INSUFFICIENT_DATA.value, now=t)
                if cues:
                    print(f"    cues: {cues}")
            else:
                agg = extract_rep_features(pd.DataFrame({"angle_deg": payload["angle_deg"]}))
                result = classifier.predict(agg) if classifier else None
                if result:
                    note = ""
                    if correct_feature_ranges and not is_within_correct_range(agg, correct_feature_ranges):
                        note = "  (outside training-correct range)"
                    print(f"Rep {rep_count}: classifier is_correct={result['is_correct']}  "
                          f"confidence={result['confidence']:.2f}{note}")
                else:
                    print(f"Rep {rep_count}: no trained model available.")
        elif event == "rep_discarded":
            print(f"  (rep discarded: {payload['reason_code']} -- {payload['message']})")

    cap.release()
    if rep_count == 0:
        print("No completed reps detected in this clip.")


if __name__ == "__main__":
    ARGS = parse_args()

    if ARGS.video:
        run_headless(ARGS.video, ARGS.exercise, ARGS.scorer, tuple(ARGS.span) if ARGS.span else None, ARGS.reference)
        raise SystemExit(0)

    PLAIN_UI = ui_theme.plain_ui_enabled()

    st.set_page_config(page_title="cvPhysio", layout="centered" if PLAIN_UI else "wide")
    if not PLAIN_UI:
        st.markdown(ui_theme.css(), unsafe_allow_html=True)  # injected once, never inside the frame loop below

    FOOTER_TEXT = (
        "cvPhysio is a research prototype, not a medical device. It does not diagnose or treat. "
        "Stop if you feel pain. Video is processed on this device and is not stored."
    )

    if PLAIN_UI:
        st.title("cvPhysio — AI Rehabilitation Coach")
    else:
        st.markdown(
            '<div class="cvp-topbar"><span class="cvp-serif cvp-topbar-brand">cvPhysio</span></div>'
            '<h1 class="cvp-hero-heading">Your rehab coach, <em>in real time.</em></h1>'
            '<p class="cvp-hero-subtitle">Live, on-device pose feedback while you move -- not a diagnosis.</p>',
            unsafe_allow_html=True,
        )

    exercise_labels = list(EXERCISES.values())
    if PLAIN_UI:
        exercise_label = st.selectbox("Exercise", exercise_labels)
    else:
        exercise_label = st.segmented_control(
            "Exercise", exercise_labels, default=exercise_labels[0],
        ) or exercise_labels[0]
    exercise = next(key for key, label in EXERCISES.items() if label == exercise_label)
    user_id = st.text_input("User ID", value="demo")

    tab_session, tab_reference, tab_history, tab_about = st.tabs(["Session", "Reference", "History", "About"])

    with tab_session:
        classifier = None
        correct_feature_ranges = None
        rules_config = None
        if ARGS.scorer == "classifier":
            st.warning(CLASSIFIER_WARNING)
            try:
                classifier, correct_feature_ranges = load_classifier_for_live_use(exercise)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()
            if classifier is None:
                st.warning(f"No trained model at {MODELS_DIR / f'{exercise}.pkl'} — showing pose and angle only, no correctness label.")
        else:
            try:
                rules_config = rules_module.load_config(exercise)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()

        # Per-patient reference, chosen at session start (before Start is
        # pressed) -- only meaningful for --scorer rules, and only takes
        # effect if rules_config["mode"] == "reference_relative"; score_rep()
        # itself ignores it otherwise, so picking one is always safe.
        # Positioned before the Start/Stop pills (not in the "Reference"
        # card below the feed, as the design brief's card layout implies)
        # because its value must be read before the camera loop starts --
        # a card placed visually below the feed can only be POPULATED
        # after that loop exits (Streamlit renders containers in the order
        # they're first created in the script), so the choice itself has
        # to live in the setup area; the card below the feed instead shows
        # a read-only recap of the active choice. Noted in the final report.
        reference = None
        if ARGS.scorer == "rules":
            reference_dir = Path("data/references")
            reference_files = sorted(reference_dir.glob("*.json")) if reference_dir.exists() else []
            reference_choice = st.selectbox("Patient reference (optional)", ["None"] + [f.stem for f in reference_files])
            if reference_choice != "None":
                reference = rules_module.load_reference(reference_dir / f"{reference_choice}.json")
                st.info(f"Using reference '{reference_choice}' (k_reps={reference['k_reps']}, "
                        f"config mode={rules_config.get('mode', 'absolute')!r})")

        if "running" not in st.session_state:
            st.session_state.running = False

        col1, col2 = st.columns(2)
        if col1.button("Start session", type="primary", use_container_width=True):
            st.session_state.running = True
        if col2.button("Stop session", type="secondary", use_container_width=True):
            st.session_state.running = False

        if PLAIN_UI:
            frame_slot = st.empty()
            status_slot = st.empty()
        else:
            stats_left_col, video_col, stats_right_col = st.columns([1, 2, 1])
            with stats_left_col:
                reps_stat_slot = st.empty()
                form_stat_slot = st.empty()
            with video_col:
                video_ring = st.container(key=ui_theme.VIDEO_RING_CONTAINER_KEY)
                with video_ring:
                    frame_slot = st.empty()
                ring_style_slot = st.empty()
            with stats_right_col:
                tempo_stat_slot = st.empty()
                fps_stat_slot = st.empty()
            cue_slot = st.empty()
            reference_card, live_feedback_card, session_log_card = st.columns(3)
            with reference_card:
                reference_status_slot = st.container(key="cvp-card-reference")
            with live_feedback_card:
                live_feedback_header_slot = st.container(key="cvp-card-live-feedback")
                live_feedback_slot = st.empty()
            with session_log_card:
                session_log_header_slot = st.container(key="cvp-card-session-log")
                session_log_slot = st.empty()
            with reference_status_slot:
                st.markdown(ui_theme.render_card_header("Reference"), unsafe_allow_html=True)
                if reference:
                    st.caption(f"{reference_choice} -- K={reference['k_reps']} reps")
                else:
                    st.caption("None selected -- scoring against fixed thresholds.")
            with live_feedback_header_slot:
                st.markdown(ui_theme.render_card_header("Live feedback"), unsafe_allow_html=True)
            with session_log_header_slot:
                st.markdown(ui_theme.render_card_header("Session log"), unsafe_allow_html=True)
            # Reps/Form/Tempo/FPS/cue/chips/log start as "no session yet" --
            # every value below comes from session state, never a hard-coded
            # number, and reads "--" until that state actually exists.
            reps_stat_slot.markdown(ui_theme.render_stat_block("Reps", "—"), unsafe_allow_html=True)
            form_stat_slot.markdown(ui_theme.render_stat_block("Form", "—"), unsafe_allow_html=True)
            tempo_stat_slot.markdown(ui_theme.render_stat_block("Tempo (s/rep)", "—"), unsafe_allow_html=True)
            fps_stat_slot.markdown(ui_theme.render_stat_block("FPS", "—"), unsafe_allow_html=True)
            ring_style_slot.markdown(ui_theme.render_video_ring_style(ui_theme.STATE_IDLE), unsafe_allow_html=True)
            cue_slot.markdown(
                ui_theme.render_cue_banner(ui_theme.STATE_IDLE, detail="Press Start session to begin."),
                unsafe_allow_html=True,
            )
            live_feedback_slot.markdown('<div class="cvp-muted">No session running.</div>', unsafe_allow_html=True)
            session_log_slot.markdown(ui_theme.render_session_log_table([]), unsafe_allow_html=True)

        if st.session_state.running:
            try:
                capture = open_webcam()
            except RuntimeError as exc:
                st.error(f"Could not open the camera: {exc}")
                st.session_state.running = False
                if not PLAIN_UI:
                    ring_style_slot.markdown(ui_theme.render_video_ring_style(ui_theme.STATE_ERROR), unsafe_allow_html=True)
                    cue_slot.markdown(ui_theme.render_cue_banner(ui_theme.STATE_ERROR), unsafe_allow_html=True)
                st.stop()
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            camera_info = {
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(fps, 2),
            }
            tracker = SessionTracker(user_id=user_id, exercise=exercise, camera_info=camera_info)
            session = Session(exercise)
            session.start()  # "Start session" button already gated entry into this block -- begin calibrating now
            smoother = build_smoother()
            required_roles = rules_module.required_joint_roles() if rules_config else None
            speech = SpeechFeedback()
            cue_cooldown = rules_module.CueCooldown(rules_config["cue_cooldown_s"]["value"]) if rules_config else None
            rep_count = 0
            last_message = "Ready — begin your repetition."
            last_is_correct = None
            last_rule_results = None  # most recent rep's result["rules"], for the per-rule chips
            session_start_t = time.monotonic()  # presentation only: drives the Tempo/FPS stats below
            frame_count = 0
            last_ring_state = None  # only re-render the ring's <style> override on an actual state change
            try:
                while st.session_state.running:
                    ok, frame = capture.read()
                    if not ok:
                        st.error("Could not read from camera.")
                        break

                    landmarks = get_pose_landmarks(frame)
                    t = time.monotonic()  # live webcam: real arrival time, not an assumed fixed rate
                    event, payload, visible, failed = session.process_frame(landmarks, smoother, t, required_roles)

                    if event == "rep_complete":
                        rep_count += 1
                        if ARGS.scorer == "rules":
                            rep = {
                                "angle_deg": payload["angle_deg"],
                                "elbow_angle_deg": payload["elbow_angle_deg"],
                                "trunk_lean_deg": payload["trunk_lean_deg"],
                                "valid_fraction": payload["valid_fraction"],
                            }
                            result = rules_module.score_rep(rep, rules_config, fps, reference=reference)
                            last_rule_results = result["rules"]
                            side_value = Counter(payload["side"]).most_common(1)[0][0]
                            tracker.add_rep(RepRecord(
                                exercise=exercise,
                                side=side_value,
                                angle_deg=max(rep["angle_deg"]),
                                status=result["status"],
                                confidence=None,
                            ))
                            last_message = f"Rep {rep_count}: {message_for_rules_result(result)}"
                            if result["warning"]:
                                st.warning(result["warning"])
                            last_is_correct = {
                                RuleStatus.PASS.value: True, RuleStatus.FAIL.value: False, RuleStatus.INSUFFICIENT_DATA.value: None,
                            }[result["status"]]
                            cues = cue_cooldown.cues_to_speak(result["failed_rules"], rules_config["cues"]) if result["status"] == RuleStatus.FAIL.value else []
                            if result["status"] == RuleStatus.INSUFFICIENT_DATA.value and result["cue"] and cue_cooldown.should_fire(RuleStatus.INSUFFICIENT_DATA.value):
                                cues = [result["cue"]]
                                cue_cooldown.record(RuleStatus.INSUFFICIENT_DATA.value)
                            for cue in cues:
                                speech.speak(cue)
                            if not cues:
                                speech.speak(last_message)
                        else:
                            rep_frames = pd.DataFrame({"angle_deg": payload["angle_deg"]})
                            agg = extract_rep_features(rep_frames)
                            result = classifier.predict(agg) if classifier else None
                            side_value = Counter(payload["side"]).most_common(1)[0][0]
                            if classifier and result:
                                tracker.add_rep(RepRecord(
                                    exercise=exercise,
                                    side=side_value,
                                    angle_deg=agg["angle_max_deg"],
                                    status=(RuleStatus.PASS if result["is_correct"] else RuleStatus.FAIL).value,
                                    confidence=result["confidence"],
                                ))
                                last_message = f"Rep {rep_count}: {message_for_result(result, None)}"
                                if correct_feature_ranges and not is_within_correct_range(agg, correct_feature_ranges):
                                    last_message += (
                                        " (Note: this rep's motion was outside the range seen in training "
                                        "data labeled correct — treat this result with extra caution.)"
                                    )
                                last_is_correct = result["is_correct"]
                            else:
                                last_message = f"Rep {rep_count} complete — no trained model, correctness not shown."
                                last_is_correct = None
                            speech.speak(last_message)  # spoken cue once per completed rep, not per frame
                    elif event == "rep_discarded":
                        tracker.add_discard(DiscardRecord(
                            exercise=exercise, reason_code=payload["reason_code"], message=payload["message"],
                        ))
                        st.warning(payload["message"])

                    stall_hint = (
                        session.detector.calibration_hint()
                        if session.detector is not None and session.detector.state == "CALIBRATING" else None
                    )
                    # cue_state/cue_detail are presentation-only derivations of the
                    # exact same branches below (never a new pipeline/session
                    # decision) -- they pick which ui_theme state/colour/icon the
                    # Streamlit-side cue banner and video glow ring show; the
                    # in-frame cv2 overlay (draw_overlay, message/overlay_is_correct)
                    # is unchanged, same call, same args, same place as before.
                    if stall_hint:
                        # Stable, actionable hint once stuck calibrating -- takes
                        # priority over the flickery per-frame low-visibility message.
                        message, overlay_is_correct = stall_hint, None
                        cue_state, cue_detail = ui_theme.STATE_CALIBRATING, stall_hint
                    elif landmarks is None:
                        message, overlay_is_correct = "No person detected.", None
                        cue_state, cue_detail = ui_theme.STATE_INSUFFICIENT, None
                    elif not visible:
                        message, overlay_is_correct = message_for_result(None, failed), None
                        cue_state, cue_detail = ui_theme.STATE_INSUFFICIENT, None
                    elif session.detector is not None and session.detector.state == "CALIBRATING":
                        message, overlay_is_correct = "Calibrating rest position — please hold still...", None
                        cue_state, cue_detail = ui_theme.STATE_CALIBRATING, None
                    elif session.detector is not None and session.detector.state == "MOVING":
                        message, overlay_is_correct = "Repetition in progress...", None
                        cue_state, cue_detail = ui_theme.STATE_IDLE, message
                    else:  # RESTING
                        message, overlay_is_correct = last_message, last_is_correct
                        cue_state = {
                            True: ui_theme.STATE_PASS, False: ui_theme.STATE_FAIL, None: ui_theme.STATE_INSUFFICIENT,
                        }[last_is_correct]
                        cue_detail = message if cue_state == ui_theme.STATE_FAIL else None

                    if PLAIN_UI:
                        status_slot.write(message)
                    else:
                        frame_count += 1
                        if cue_state != last_ring_state:
                            ring_style_slot.markdown(ui_theme.render_video_ring_style(cue_state), unsafe_allow_html=True)
                            last_ring_state = cue_state
                        cue_slot.markdown(ui_theme.render_cue_banner(cue_state, detail=cue_detail), unsafe_allow_html=True)

                        scored_reps = [r for r in tracker.reps if r.status in (RuleStatus.PASS.value, RuleStatus.FAIL.value)]
                        passed_reps = sum(1 for r in scored_reps if r.status == RuleStatus.PASS.value)
                        form_value = f"{round(100 * passed_reps / len(scored_reps))}%" if scored_reps else "—"
                        elapsed = t - session_start_t
                        tempo_value = f"{elapsed / rep_count:.1f}s" if rep_count else "—"
                        fps_value = f"{frame_count / elapsed:.1f}" if elapsed > 0 else "—"
                        reps_stat_slot.markdown(ui_theme.render_stat_block("Reps", str(rep_count) if rep_count else "—"), unsafe_allow_html=True)
                        form_stat_slot.markdown(ui_theme.render_stat_block("Form", form_value), unsafe_allow_html=True)
                        tempo_stat_slot.markdown(ui_theme.render_stat_block("Tempo (s/rep)", tempo_value), unsafe_allow_html=True)
                        fps_stat_slot.markdown(ui_theme.render_stat_block("FPS", fps_value), unsafe_allow_html=True)

                        if last_rule_results:
                            chips = "".join(
                                ui_theme.render_rule_chip(rule_id, r["measured"], tuple(r["allowed"]), r["passed"])
                                for rule_id, r in last_rule_results.items()
                            )
                            live_feedback_slot.markdown(
                                f'<div class="cvp-chip-row">{chips}</div>', unsafe_allow_html=True,
                            )
                        else:
                            live_feedback_slot.markdown('<div class="cvp-muted">No reps yet.</div>', unsafe_allow_html=True)

                        log_rows = [{"rep": i + 1, "status": r.status} for i, r in enumerate(tracker.reps)]
                        session_log_slot.markdown(ui_theme.render_session_log_table(log_rows), unsafe_allow_html=True)

                    annotated = draw_overlay(frame, message, overlay_is_correct)
                    frame_slot.image(annotated, channels="BGR")
            finally:
                session.stop()  # loop exited (camera failure, or "Stop session" interrupting this script run) --
                # any rep still MOVING at this moment is simply never completed: its buffered frames live only
                # on session.detector, which is about to go out of scope, so it's discarded by construction,
                # never scored, never written anywhere.
                capture.release()
                tracker.save()  # always, even a totally empty session -- writes a header-only CSV, never a 0-byte file
                st.json(tracker.summary())

    with tab_reference:
        st.subheader("Record a reference")
        st.caption("Captures K repetitions (Start -> K reps -> saved) and stores their median "
                   "peak angle/elbow minimum/trunk-lean maximum/tempo as a named Reference "
                   "(data/references/<name>.json), for 'reference_relative' scoring mode.")
        reference_name = st.text_input("Reference name", value=f"{user_id}_{exercise}")
        k_reps = int(st.number_input("Number of reference reps (K)", min_value=1, max_value=10,
                                      value=rules_module.DEFAULT_REFERENCE_K, step=1))

        if "recording_reference" not in st.session_state:
            st.session_state.recording_reference = False
        col1, col2 = st.columns(2)
        if col1.button("Start recording", type="primary", use_container_width=True):
            st.session_state.recording_reference = True
        if col2.button("Stop recording", type="secondary", use_container_width=True):
            st.session_state.recording_reference = False

        if PLAIN_UI:
            frame_slot = st.empty()
            status_slot = st.empty()
        else:
            video_ring = st.container(key="cvp-video-ring-reference")
            with video_ring:
                frame_slot = st.empty()
            ring_style_slot = st.empty()
            cue_slot = st.empty()
            ring_style_slot.markdown(ui_theme.render_video_ring_style(ui_theme.STATE_IDLE), unsafe_allow_html=True)
            cue_slot.markdown(
                ui_theme.render_cue_banner(ui_theme.STATE_IDLE, detail="Press Start recording to begin."),
                unsafe_allow_html=True,
            )

        if st.session_state.recording_reference:
            try:
                capture = open_webcam()
            except RuntimeError as exc:
                st.error(f"Could not open the camera: {exc}")
                st.session_state.recording_reference = False
                if not PLAIN_UI:
                    ring_style_slot.markdown(ui_theme.render_video_ring_style(ui_theme.STATE_ERROR), unsafe_allow_html=True)
                    cue_slot.markdown(ui_theme.render_cue_banner(ui_theme.STATE_ERROR), unsafe_allow_html=True)
                st.stop()
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            session = Session(exercise)
            session.start()
            smoother = build_smoother()
            required_roles = rules_module.required_joint_roles()
            collected_reps = []
            last_ring_state = None
            try:
                while st.session_state.recording_reference and len(collected_reps) < k_reps:
                    ok, frame = capture.read()
                    if not ok:
                        st.error("Could not read from camera.")
                        break

                    landmarks = get_pose_landmarks(frame)
                    t = time.monotonic()
                    event, payload, visible, failed = session.process_frame(landmarks, smoother, t, required_roles)

                    if event == "rep_complete":
                        collected_reps.append({
                            "angle_deg": payload["angle_deg"], "elbow_angle_deg": payload["elbow_angle_deg"],
                            "trunk_lean_deg": payload["trunk_lean_deg"],
                        })
                        if PLAIN_UI:
                            status_slot.write(f"Captured rep {len(collected_reps)}/{k_reps}")
                    elif event == "rep_discarded":
                        st.warning(payload["message"])

                    stall_hint = (
                        session.detector.calibration_hint()
                        if session.detector is not None and session.detector.state == "CALIBRATING" else None
                    )
                    if stall_hint:
                        message, cue_state, cue_detail = stall_hint, ui_theme.STATE_CALIBRATING, stall_hint
                    elif session.detector is not None and session.detector.state == "CALIBRATING":
                        message = "Calibrating rest position — please hold still..."
                        cue_state, cue_detail = ui_theme.STATE_CALIBRATING, None
                    else:
                        message = f"Captured {len(collected_reps)}/{k_reps} reference reps -- keep going..."
                        cue_state, cue_detail = ui_theme.STATE_IDLE, message

                    if PLAIN_UI:
                        status_slot.write(message)
                    else:
                        if cue_state != last_ring_state:
                            ring_style_slot.markdown(ui_theme.render_video_ring_style(cue_state), unsafe_allow_html=True)
                            last_ring_state = cue_state
                        cue_slot.markdown(ui_theme.render_cue_banner(cue_state, detail=cue_detail), unsafe_allow_html=True)

                    annotated = draw_overlay(frame, message, None)
                    frame_slot.image(annotated, channels="BGR")
            finally:
                session.stop()
                capture.release()

            if len(collected_reps) >= k_reps:
                st.session_state.recording_reference = False
                reference = rules_module.build_reference(reference_name, exercise, collected_reps, fps)
                saved_path = rules_module.save_reference(reference)
                st.success(f"Saved reference '{reference_name}' ({len(collected_reps)} reps) to {saved_path}")
                st.json(reference["medians"])

    with tab_history:
        history = load_history(user_id)
        if not history:
            st.caption("No saved sessions yet for this user.")
        else:
            st.table(history)

    with tab_about:
        st.markdown(
            "**cvPhysio** is a university rehabilitation-coaching project covering two exercises: "
            "Standing Shoulder Abduction and Standing Shoulder Internal/External Rotation "
            "(UI-PRMD movements 7 and 9). A webcam feed is turned into joint angles via "
            "MediaPipe Pose, scored against configurable rule thresholds (or an optional "
            "trained classifier), and summarised into a per-session log."
        )
        st.caption(
            f"CVPHYSIO_PLAIN_UI={'1 (plain Streamlit look)' if PLAIN_UI else '0/unset (styled UI)'} -- "
            "set the environment variable to 1 to disable all custom CSS."
        )

    if PLAIN_UI:
        st.caption(FOOTER_TEXT)
    else:
        st.markdown(f'<div class="cvp-footer">{ui_theme.escape(FOOTER_TEXT)}</div>', unsafe_allow_html=True)
