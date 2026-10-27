import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_app_rep_detector():
    """exec()-slice app.py's RepDetector class + its constants out of its
    source text, instead of importing app.py directly -- app.py parses
    CLI args and launches Streamlit/headless logic at import time, which
    would run against pytest's own argv and crash. Same technique used by
    scripts/diagnose_elbow_angle.py, scripts/measure_smoothing_effect.py
    and scripts/evaluate_rules.py.
    """
    import statistics
    from collections import Counter, deque

    source = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    start = source.index("CALIBRATION_FRAMES = 15")
    end = source.index("\ndef process_frame(detector")
    namespace = {"statistics": statistics, "Counter": Counter, "deque": deque}
    exec(source[start:end], namespace)
    return namespace["RepDetector"]


def load_app_process_frame():
    """Same exec()-slice technique as load_app_rep_detector(), but
    extended to also capture process_frame() -- needed for tests that
    exercise the full per-frame pipeline (confidence gate + NaN guard +
    RepDetector), not just RepDetector in isolation."""
    import math
    import statistics
    from collections import Counter, deque

    from src.feature_extraction import (
        check_confidence,
        compute_elbow_angle,
        compute_trunk_lean_deg,
        extract_features,
        select_side,
    )

    source = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    start = source.index("CALIBRATION_FRAMES = 15")
    end = source.index("\ndef build_smoother")
    namespace = {
        "math": math, "statistics": statistics, "Counter": Counter, "deque": deque,
        "check_confidence": check_confidence, "extract_features": extract_features,
        "select_side": select_side, "compute_elbow_angle": compute_elbow_angle,
        "compute_trunk_lean_deg": compute_trunk_lean_deg,
    }
    exec(source[start:end], namespace)
    return namespace["RepDetector"], namespace["process_frame"]


def load_app_classifier_loader():
    """Same exec()-slice technique, extended through build_smoother()'s
    end, to also capture load_classifier_for_live_use() -- needed to
    test the classifier metadata validation without importing app.py
    directly (see load_app_rep_detector())."""
    import json
    import statistics
    from collections import Counter, deque

    from src.classifier import FEATURE_COLUMNS, CorrectnessClassifier

    source = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    start = source.index("CALIBRATION_FRAMES = 15")
    end = source.index("\ndef run_headless(video_path")
    namespace = {
        "json": json, "statistics": statistics, "Counter": Counter, "deque": deque,
        "Path": Path, "FEATURE_COLUMNS": FEATURE_COLUMNS, "CorrectnessClassifier": CorrectnessClassifier,
        "MODELS_DIR": Path("models"), "LIVE_FEATURE_SOURCE": "fk_positions",
    }
    exec(source[start:end], namespace)
    return namespace["load_classifier_for_live_use"]
