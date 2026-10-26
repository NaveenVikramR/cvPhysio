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
