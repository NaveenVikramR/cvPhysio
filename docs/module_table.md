# Module table

One row per module. Derived from `docs/module_interfaces.md` and
`docs/config_schema.md` (both themselves derived from the code) --
kept short enough for a slide; see those two files for full detail.

| Module | Purpose | Inputs | Outputs | Config keys |
|---|---|---|---|---|
| `src/pose_estimation.py` | Video capture & pose estimation | BGR frame | 33x(x,y,z,visibility), selected landmark space | `config/pose.json`: `landmark_space` |
| `src/feature_extraction.py` | Biomechanical feature extraction | 33-tuple landmarks, exercise name | `angle_deg`/`side`, or `None`; `elbow_angle_deg`, `trunk_lean_deg` | none (caller passes `exercise`/`required_roles`) |
| `src/pipeline.py` | Live-pipeline orchestration: RepDetector, process_frame, Session (Start/Stop), build_smoother, load_classifier_for_live_use | per-frame landmarks + exercise | rep-boundary events (`rep_complete`/`rep_discarded`/`none`), smoothed landmarks, loaded classifier | `config/smoothing.json` (via `build_smoother`) |
| `src/rules.py` | Rule-based scorer (absolute or reference_relative) + Reference build/save/load | per-rep angle/elbow/trunk sequences, config, optional Reference | `{status, failed_rules, rules, valid_fraction, cue, warning}`; Reference JSON | `config/exercises/{exercise}.json`: `target_peak_deg`, `tolerance_deg`, `min_elbow_straight_deg`, `max_trunk_lean_deg`, `tempo_min_s`, `tempo_max_s`, `tol_peak_deg`, `tol_elbow_deg`, `tol_trunk_deg`, `tol_tempo_frac`, `mode`, `cue_cooldown_s`, `min_valid_fraction`, `cues` |
| `src/smoothing.py` | One-Euro landmark smoothing | raw landmarks + timestamp | smoothed/held-over/`None` landmarks | `config/smoothing.json`: `enabled`, `min_cutoff`, `beta`, `d_cutoff`, `max_hold_frames` |
| `src/feedback.py` | Overlay + spoken feedback | a scorer result dict | annotated frame, spoken text | none |
| `src/tracking.py` | Session logging | one `RepRecord`/`DiscardRecord` per event | `{user}_{exercise}_{stamp}.csv` + `.discards.csv` + `.summary.json` | none |
| `src/classifier.py` | UI-PRMD-trained correctness classifier (cross-domain, not the default scorer) | aggregate rep feature dict | `{is_correct, confidence}` | none (reads `models/{exercise}.pkl` + sidecar `.metadata.json`) |
| `app.py` | UI: wires every module above into a live webcam session or headless `--video` run | webcam frames or a video file; CLI args | live overlay + spoken cues + session logs, or per-rep stdout | reads `config/exercises/{exercise}.json` via `src/rules.py` |
