"""Tests for app.py's load_classifier_for_live_use(): refuses to load a
classifier whose sidecar metadata doesn't match what the live pipeline
actually computes, instead of silently scoring with it.
"""

import json

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from conftest import load_app_classifier_loader
from src.classifier import FEATURE_COLUMNS

load_classifier_for_live_use = load_app_classifier_loader()


def write_fixture_model(models_dir, exercise="shoulder_abduction"):
    """A trivially-fitted RandomForestClassifier, saved the same way
    CorrectnessClassifier.save() does (joblib.dump of the sklearn model
    only)."""
    model = RandomForestClassifier(n_estimators=2, random_state=0)
    model.fit([[0, 0, 0, 0, 0, 0], [180, 0, 90, 180, 90, 0]], [False, True])
    joblib.dump(model, models_dir / f"{exercise}.pkl")


def write_metadata(models_dir, exercise="shoulder_abduction", **overrides):
    metadata = {
        "exercise": exercise,
        "feature_names": FEATURE_COLUMNS,
        "feature_source": "fk_positions",
        "correct_feature_ranges": {},
    }
    metadata.update(overrides)
    (models_dir / f"{exercise}.metadata.json").write_text(json.dumps(metadata))


def test_loads_successfully_when_metadata_matches(tmp_path):
    write_fixture_model(tmp_path)
    write_metadata(tmp_path)

    classifier, ranges = load_classifier_for_live_use("shoulder_abduction", models_dir=tmp_path)
    assert classifier is not None
    assert ranges == {}


def test_refuses_when_feature_names_mismatch(tmp_path):
    write_fixture_model(tmp_path)
    write_metadata(tmp_path, feature_names=["angle_max_deg", "angle_min_deg"])  # wrong/incomplete

    with pytest.raises(RuntimeError, match="feature_names"):
        load_classifier_for_live_use("shoulder_abduction", models_dir=tmp_path)


def test_refuses_when_feature_names_order_differs(tmp_path):
    write_fixture_model(tmp_path)
    reordered = list(reversed(FEATURE_COLUMNS))
    write_metadata(tmp_path, feature_names=reordered)

    with pytest.raises(RuntimeError, match="feature_names"):
        load_classifier_for_live_use("shoulder_abduction", models_dir=tmp_path)


def test_refuses_when_feature_source_mismatch(tmp_path):
    write_fixture_model(tmp_path)
    write_metadata(tmp_path, feature_source="angles_file")  # the legacy, non-live source

    with pytest.raises(RuntimeError, match="feature_source"):
        load_classifier_for_live_use("shoulder_abduction", models_dir=tmp_path)


def test_refuses_when_metadata_file_missing(tmp_path):
    write_fixture_model(tmp_path)
    # no metadata file written at all

    with pytest.raises(RuntimeError, match="no sidecar metadata"):
        load_classifier_for_live_use("shoulder_abduction", models_dir=tmp_path)


def test_returns_none_none_when_model_file_missing(tmp_path):
    # no model, no metadata -- the existing, unchanged "nothing to load yet" case
    classifier, ranges = load_classifier_for_live_use("shoulder_abduction", models_dir=tmp_path)
    assert classifier is None
    assert ranges is None
