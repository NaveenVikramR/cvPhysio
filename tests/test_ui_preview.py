"""ui_preview.py (the mock-data "PREVIEW" Streamlit page for src/ui_theme.py)
must render every state with no exception and without ever importing
mediapipe or opening a camera -- it's a layout preview, not a live
session. Uses streamlit.testing.v1.AppTest to run the script headlessly
(no real browser, no camera).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui_preview import STATE_PRESETS

STATE_NAMES = list(STATE_PRESETS)
UI_PREVIEW_PATH = str(Path(__file__).resolve().parent.parent / "ui_preview.py")


def test_ui_preview_renders_without_exception_on_load():
    at = AppTest.from_file(UI_PREVIEW_PATH, default_timeout=30)
    at.run()
    assert not at.exception


@pytest.mark.parametrize("state_name", STATE_NAMES)
def test_ui_preview_renders_every_state_without_exception(state_name):
    at = AppTest.from_file(UI_PREVIEW_PATH, default_timeout=30)
    at.run()
    at.selectbox[0].set_value(state_name).run()
    assert not at.exception


def test_ui_preview_never_imports_mediapipe():
    """Checked against the module's own source text, not sys.modules --
    in a full pytest run, other test files legitimately import
    mediapipe-dependent modules (e.g. src.pose_estimation) first, which
    would make a sys.modules check pass or fail based on test order
    rather than on what ui_preview.py itself imports (same reasoning as
    tests/test_ui_theme.py's equivalent check)."""
    import inspect

    import ui_preview

    source = inspect.getsource(ui_preview)
    assert "mediapipe" not in source
    assert "import cv2" not in source


def test_ui_preview_has_no_videocapture_call_in_its_own_source():
    """No cv2.VideoCapture/open_webcam call anywhere in this file's own
    source -- the only frame it ever shows is the synthetic mock_frame()
    array below, never a real camera."""
    import inspect

    import ui_preview

    source = inspect.getsource(ui_preview)
    assert "VideoCapture" not in source
    assert "open_webcam" not in source


def test_every_preset_has_the_fields_the_page_reads():
    required_keys = {"cue_state", "cue_detail", "message", "overlay_is_correct", "reps", "form_pct", "tempo_s", "fps", "rules", "log"}
    for name, preset in STATE_PRESETS.items():
        assert required_keys <= set(preset.keys()), f"{name} is missing one of {required_keys}"
