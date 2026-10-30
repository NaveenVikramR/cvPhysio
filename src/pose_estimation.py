"""Video capture & pose estimation. Output: a 33-tuple landmark list per
frame (MediaPipe's PoseLandmark order), nothing else.

Uses MediaPipe's Tasks API (PoseLandmarker), not the older
mediapipe.solutions.pose API -- the legacy "solutions" module ships no
compiled extension at all in MediaPipe's Windows wheels for Python 3.13
(confirmed: absent in 1.1.1, 1.1.0, 1.0.x and 0.10.35 alike), so it isn't a
version-pin problem, the Tasks API is the only one available here. It's
also the actively-maintained API going forward -- "solutions" is
deprecated upstream, not just broken on this machine.

Needs a pose-landmarker model bundle at models/pose_landmarker_lite.task
(downloaded once from Google's model zoo: https://storage.googleapis.com/
mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/
pose_landmarker_lite.task -- gitignored, not committed, ~5.8MB).
config/pose.json's "pose_model" selects "lite" (default) or "heavy"
(models/pose_landmarker_heavy.task, same URL pattern, also gitignored).

Runs in VIDEO mode (not IMAGE) so MediaPipe reuses temporal context
between frames the way the old solutions API implicitly did for a video
stream -- IMAGE mode re-detects from scratch every call with no tracking
continuity, which would be jitterier for RepDetector's rest-baseline
calibration in app.py.

MediaPipe's PoseLandmarkerResult carries TWO landmark sets per frame:
pose_landmarks (NormalizedLandmark -- x/y each independently divided by
that axis's own frame dimension, landing in roughly [0, 1]) and
pose_world_landmarks (Landmark -- metric meters, hip-centered, NOT
dependent on frame width/height). results/diag_coords.md found that
feeding the normalized set straight into compute_angle() on a non-square
frame is an anisotropic distortion of the real geometry; world
landmarks don't have that problem, so they're the default here
(config/pose.json's "landmark_space" key) -- see get_pose_landmarks().
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

_REPO_ROOT = Path(__file__).resolve().parent.parent
POSE_MODELS = ("lite", "heavy")
CONFIG_PATH = Path("config/pose.json")


def model_path(model: str) -> Path:
    return _REPO_ROOT / "models" / f"pose_landmarker_{model}.task"


def model_url(model: str) -> str:
    return (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{model}/float16/latest/pose_landmarker_{model}.task"
    )


def configured_pose_model(config_path: Path | None = None) -> str:
    """config/pose.json's "pose_model" (default "lite"). Anchored to the repo
    root by default because it's read at import time, from any working directory."""
    config = json.loads((config_path or _REPO_ROOT / CONFIG_PATH).read_text())
    model = config.get("pose_model", "lite")
    if model not in POSE_MODELS:
        raise ValueError(f"config/pose.json pose_model={model!r}; must be one of {POSE_MODELS}")
    return model


def create_landmarker(model: str) -> PoseLandmarker:
    path = model_path(model)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pose-landmarker model bundle at {path}. "
            f"Download it from {model_url(model)} and save it to that exact path "
            f"(gitignored, not committed -- see docs/laptop_setup.md)."
        )
    return PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))


_POSE_MODEL = configured_pose_model()
_MODEL_PATH = model_path(_POSE_MODEL)
_landmarker = create_landmarker(_POSE_MODEL)

_next_timestamp_ms = 0


def reset_tracking() -> None:
    """Starts a fresh pose-model instance, so the next frame has no tracking state
    from earlier frames. For offline runs over several clips, where a clip's results
    would otherwise depend on which clip came before it."""
    global _landmarker, _next_timestamp_ms
    _landmarker.close()
    _landmarker = create_landmarker(_POSE_MODEL)
    _next_timestamp_ms = 0


LANDMARK_SPACES = ("world", "normalized")


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def get_pose_landmarks_pair(frame) -> tuple[
    list[tuple[float, float, float, float]] | None,
    list[tuple[float, float, float, float]] | None,
]:
    """Run MediaPipe Pose on a BGR frame ONCE and return BOTH landmark
    sets that single inference call produces: (normalized, world).
    normalized is MediaPipe's own NormalizedLandmark set, in image/pixel
    space (kept available for drawing overlays on the frame); world is
    pose_world_landmarks, in metric space. Either side is None if no
    pose was detected. visibility is identical between the two sets for
    the same joint (confirmed empirically on a real frame), so
    confidence gating gives the same answer regardless of which set it
    runs on.
    """
    global _next_timestamp_ms
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = _landmarker.detect_for_video(mp_image, _next_timestamp_ms)
    _next_timestamp_ms += 33  # monotonic tick; VIDEO mode needs increasing values, not real elapsed time

    if not result.pose_landmarks:
        return None, None
    normalized = [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_landmarks[0]]
    world = (
        [(lm.x, lm.y, lm.z, lm.visibility) for lm in result.pose_world_landmarks[0]]
        if result.pose_world_landmarks else None
    )
    return normalized, world


def get_pose_landmarks(frame) -> list[tuple[float, float, float, float]] | None:
    """The single landmark set every existing caller (feature
    extraction, confidence gating) uses -- whichever coordinate space
    config/pose.json's "landmark_space" selects ("world" by default,
    "normalized" to reproduce the pre-this-change behavior). The
    selection happens here, not at the call site, so every caller's own
    code is unchanged. Returns None if no pose was detected, or if the
    selected space wasn't available for this frame (treated the same as
    "no pose detected" -- never silently falls back to the other space).
    """
    landmark_space = load_config().get("landmark_space", "world")
    if landmark_space not in LANDMARK_SPACES:
        raise ValueError(f"config/pose.json landmark_space={landmark_space!r}; must be one of {LANDMARK_SPACES}")
    normalized, world = get_pose_landmarks_pair(frame)
    return world if landmark_space == "world" else normalized


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    return capture
