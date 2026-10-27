"""Per-video MediaPipe landmark cache for scripts/benchmark_rehab24.py.
Caches BOTH landmark spaces (normalized + world) plus a per-frame
detected flag, keyed by (video filename, file size, MediaPipe version,
pose-landmarker model file hash) -- a cache entry is invalidated
automatically if the video file changes, MediaPipe is upgraded, or the
model bundle is swapped, with no manual cache-busting needed. Reused if
present; --rescore-only (scripts/benchmark_rehab24.py's CLI flag) must
never fall back to running MediaPipe when the cache is missing -- see
get_or_build()'s rescore_only branch.

Stored at results/cache/{key}.npz (the landmark arrays) + results/
cache/{key}.json (a small sidecar: fps/frame_count/width/height and the
raw key components, for human debugging only -- never read back by any
code, unlike the .npz). results/cache/ is gitignored, same as the rest
of results/.
"""

import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.pose_estimation import _MODEL_PATH, get_pose_landmarks_pair

CACHE_DIR = Path("results/cache")

_model_hash_cache: str | None = None


def model_file_hash(model_path: Path = _MODEL_PATH) -> str:
    """sha256 of the pose-landmarker model bundle, memoized -- it's a
    ~5.8MB file and this is called once per video, not once per frame."""
    global _model_hash_cache
    if _model_hash_cache is None:
        _model_hash_cache = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return _model_hash_cache


def mediapipe_version() -> str:
    return mp.__version__


def cache_key(video_path: Path) -> str:
    size = video_path.stat().st_size
    raw = f"{video_path.name}|{size}|{mediapipe_version()}|{model_file_hash()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _paths_for(video_path: Path) -> tuple[Path, Path]:
    key = cache_key(video_path)
    return CACHE_DIR / f"{key}.npz", CACHE_DIR / f"{key}.json"


def load_cache(video_path: Path) -> dict | None:
    """None if no cache entry exists (or it's unreadable -- treated the
    same as absent, never raises)."""
    npz_path, json_path = _paths_for(video_path)
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        data = np.load(npz_path)
        meta = json.loads(json_path.read_text())
        return {
            "normalized": data["normalized"], "world": data["world"], "detected": data["detected"],
            "fps": meta["fps"], "frame_count": meta["frame_count"],
            "width": meta["width"], "height": meta["height"],
        }
    except Exception:
        return None


def save_cache(video_path: Path, normalized: np.ndarray, world: np.ndarray, detected: np.ndarray,
               fps: float, width: int, height: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = _paths_for(video_path)
    np.savez_compressed(npz_path, normalized=normalized, world=world, detected=detected)
    json_path.write_text(json.dumps({
        "video_name": video_path.name, "video_size": video_path.stat().st_size,
        "mediapipe_version": mediapipe_version(), "model_hash": model_file_hash(),
        "fps": fps, "frame_count": int(len(detected)), "width": width, "height": height,
    }, indent=2))


def build_cache(video_path: Path) -> dict:
    """Runs MediaPipe over every frame of video_path and returns the same
    dict shape load_cache() does, WITHOUT saving -- get_or_build() below
    is the only place that both builds and saves, so a direct caller of
    build_cache() can never accidentally skip the save step inconsistently.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # read before release() -- a released capture returns -1 (truthy!)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    normalized_rows, world_rows, detected_rows = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        normalized, world = get_pose_landmarks_pair(frame)
        if normalized is None or world is None:
            normalized_rows.append(np.full((33, 4), np.nan))
            world_rows.append(np.full((33, 4), np.nan))
            detected_rows.append(False)
        else:
            normalized_rows.append(np.asarray(normalized, dtype=float))
            world_rows.append(np.asarray(world, dtype=float))
            detected_rows.append(True)
    cap.release()
    return {
        "normalized": np.stack(normalized_rows), "world": np.stack(world_rows),
        "detected": np.asarray(detected_rows, dtype=bool),
        "fps": fps, "frame_count": len(detected_rows), "width": width, "height": height,
    }


def get_or_build(video_path: Path, rescore_only: bool = False) -> dict:
    """The one entry point callers should use. Cache hit: returned
    as-is, MediaPipe never runs. Cache miss + rescore_only=True: raises
    RuntimeError -- never silently falls back to MediaPipe. Cache miss +
    rescore_only=False: builds via MediaPipe, saves, returns."""
    cached = load_cache(video_path)
    if cached is not None:
        return cached
    if rescore_only:
        raise RuntimeError(
            f"--rescore-only is set but no cache entry exists for {video_path} "
            f"(key would be {cache_key(video_path)}) -- run once without --rescore-only first."
        )
    fresh = build_cache(video_path)
    save_cache(video_path, fresh["normalized"], fresh["world"], fresh["detected"],
               fresh["fps"], fresh["width"], fresh["height"])
    return fresh


def landmarks_at(cache: dict, frame_idx: int, landmark_space: str) -> list[tuple[float, float, float, float]] | None:
    """Same return shape as src.pose_estimation.get_pose_landmarks(): a
    33-tuple (x,y,z,visibility) list, or None if no pose was detected at
    this frame (or frame_idx is out of range)."""
    if frame_idx >= cache["frame_count"] or not cache["detected"][frame_idx]:
        return None
    arr = cache["world"] if landmark_space == "world" else cache["normalized"]
    return [tuple(row) for row in arr[frame_idx]]
