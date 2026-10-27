"""Tests for scripts/landmark_cache.py's save/load round trip, on
SYNTHETIC arrays (no real video, no MediaPipe) -- plus the
--rescore-only contract: MediaPipe (build_cache) must never run when a
cache entry already exists, and must never run at all under
--rescore-only even when the cache is missing.
"""

import numpy as np
import pytest

from scripts import landmark_cache


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(landmark_cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(landmark_cache, "model_file_hash", lambda model_path=None: "fakehash")
    monkeypatch.setattr(landmark_cache, "mediapipe_version", lambda: "0.0.0-test")


def make_fake_video_path(tmp_path, name="fake.mp4", size_bytes=1234):
    path = tmp_path / name
    path.write_bytes(b"0" * size_bytes)
    return path


def test_save_and_load_round_trip_on_synthetic_arrays(tmp_path):
    video_path = make_fake_video_path(tmp_path)
    n_frames = 5
    rng = np.random.default_rng(0)
    normalized = rng.random((n_frames, 33, 4))
    world = rng.random((n_frames, 33, 4))
    detected = np.array([True, True, False, True, True])

    landmark_cache.save_cache(video_path, normalized, world, detected, fps=29.97, width=1920, height=1080)
    loaded = landmark_cache.load_cache(video_path)

    assert loaded is not None
    np.testing.assert_array_equal(loaded["normalized"], normalized)
    np.testing.assert_array_equal(loaded["world"], world)
    np.testing.assert_array_equal(loaded["detected"], detected)
    assert loaded["fps"] == pytest.approx(29.97)
    assert loaded["frame_count"] == n_frames
    assert loaded["width"] == 1920
    assert loaded["height"] == 1080


def test_load_cache_returns_none_when_absent(tmp_path):
    video_path = make_fake_video_path(tmp_path, name="never_cached.mp4")
    assert landmark_cache.load_cache(video_path) is None


def test_get_or_build_reuses_cache_without_calling_build_cache_again(tmp_path, monkeypatch):
    video_path = make_fake_video_path(tmp_path)
    normalized = np.zeros((3, 33, 4))
    world = np.ones((3, 33, 4))
    detected = np.array([True, True, True])
    landmark_cache.save_cache(video_path, normalized, world, detected, fps=30.0, width=640, height=480)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("build_cache() must not run when a cache entry already exists")
    monkeypatch.setattr(landmark_cache, "build_cache", _should_not_be_called)

    result = landmark_cache.get_or_build(video_path, rescore_only=False)
    np.testing.assert_array_equal(result["world"], world)

    # --rescore-only must also succeed (not raise) when a cache entry already exists
    result2 = landmark_cache.get_or_build(video_path, rescore_only=True)
    np.testing.assert_array_equal(result2["world"], world)


def test_get_or_build_with_rescore_only_and_no_cache_raises_without_touching_mediapipe(tmp_path, monkeypatch):
    video_path = make_fake_video_path(tmp_path, name="uncached.mp4")

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("build_cache() (MediaPipe) must never run under --rescore-only")
    monkeypatch.setattr(landmark_cache, "build_cache", _should_not_be_called)

    with pytest.raises(RuntimeError, match="rescore-only"):
        landmark_cache.get_or_build(video_path, rescore_only=True)


def test_landmarks_at_returns_none_for_undetected_or_out_of_range_frame():
    cache = {
        "normalized": np.arange(2 * 33 * 4, dtype=float).reshape(2, 33, 4),
        "world": np.arange(2 * 33 * 4, dtype=float).reshape(2, 33, 4) * 10,
        "detected": np.array([True, False]),
        "frame_count": 2,
    }
    world_landmarks = landmark_cache.landmarks_at(cache, 0, "world")
    assert world_landmarks is not None
    assert len(world_landmarks) == 33
    assert world_landmarks[0] == tuple(cache["world"][0, 0])

    normalized_landmarks = landmark_cache.landmarks_at(cache, 0, "normalized")
    assert normalized_landmarks[0] == tuple(cache["normalized"][0, 0])

    assert landmark_cache.landmarks_at(cache, 1, "world") is None  # not detected
    assert landmark_cache.landmarks_at(cache, 5, "world") is None  # out of range


def test_cache_key_changes_with_file_size_and_dependency_versions(tmp_path, monkeypatch):
    small = make_fake_video_path(tmp_path, name="a.mp4", size_bytes=100)
    key_small = landmark_cache.cache_key(small)  # computed BEFORE the file is overwritten below

    big = make_fake_video_path(tmp_path, name="a.mp4", size_bytes=200)  # same name/path, different size
    key_big = landmark_cache.cache_key(big)
    assert key_small != key_big

    key_before = landmark_cache.cache_key(small)
    monkeypatch.setattr(landmark_cache, "mediapipe_version", lambda: "9.9.9-different")
    key_after = landmark_cache.cache_key(small)
    assert key_before != key_after
