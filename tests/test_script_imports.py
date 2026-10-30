"""Every script that uses RepDetector/process_frame now imports them
straight from src.pipeline instead of exec()-slicing app.py's source
text (see src/pipeline.py's docstring for why that technique was
dropped). This test proves two things hold for every such script:

1. It can be imported at all, with no CLI argv parsed and no Streamlit/
   webcam/video work triggered merely by importing it (each has an
   `if __name__ == "__main__":` guard around its real work).
2. app.py itself imports cleanly under the same conditions -- it must
   not re-run argparse or launch Streamlit/headless logic at import
   time either, only inside its own __main__ guard.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCRIPT_MODULES = [
    "scripts.benchmark_rehab24",
    "scripts.diagnose_elbow_angle",
    "scripts.diagnose_misclassification",
    "scripts.diagnose_rehab24",
    "scripts.diagnose_coord_space",
    "scripts.diagnose_coords_C",
    "scripts.build_benchmark_config",
    "scripts.per_subject_fidelity",
    "scripts.derive_reference_tolerances",
    "scripts.evaluate_reference_scoring",
    "scripts.evaluate_oracle_ceiling",
    "scripts.build_review3_summary",
    "scripts.compare_AB_sessions",
    "scripts.compare_pose_models",
    "scripts.compute_primary_outcomes",
    "scripts.evaluate_own_clips",
    "scripts.measure_elbow_noise",
    "scripts.check_environment",
    "scripts.evaluate_rules",
    "scripts.measure_smoothing_effect",
    "scripts.rescore_session",
    "scripts.rotation_proxy_experiment",
    "scripts.tune_smoothing",
]


@pytest.mark.parametrize("module_name", SCRIPT_MODULES)
def test_script_module_imports_without_running_main(module_name):
    module = importlib.import_module(module_name)
    assert hasattr(module, "main")


def test_app_imports_without_parsing_argv_or_launching_anything():
    import app
    assert hasattr(app, "parse_args")
    assert hasattr(app, "run_headless")
    assert not hasattr(app, "ARGS")  # only set inside `if __name__ == "__main__":`


def test_ui_preview_imports_without_mediapipe_or_camera():
    """ui_preview.py has no __main__ guard (Streamlit scripts run their
    whole body when launched via `streamlit run`), but importing it as
    an ordinary module must still succeed, and its own source must
    never mention mediapipe or VideoCapture/open_webcam -- checked
    against source text, not sys.modules, since other test modules in
    the same pytest run legitimately import mediapipe-dependent code
    first (see tests/test_ui_preview.py for the full AppTest-driven
    per-state render check)."""
    import inspect
    import ui_preview
    assert hasattr(ui_preview, "STATE_PRESETS")
    source = inspect.getsource(ui_preview)
    assert "mediapipe" not in source
    assert "VideoCapture" not in source
    assert "open_webcam" not in source


def test_pipeline_importable_standalone():
    from src.pipeline import (
        RepDetector,
        Session,
        SessionState,
        build_smoother,
        load_classifier_for_live_use,
        process_frame,
    )
    assert RepDetector is not None
    assert callable(process_frame)
    assert callable(build_smoother)
    assert callable(load_classifier_for_live_use)
    assert Session is not None
    assert SessionState.IDLE == "IDLE"
