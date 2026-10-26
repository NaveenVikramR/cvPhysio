"""Correctness classification. Input: feature_extraction's feature dict
(or a batch of them as rows for training). Output: label + confidence.
Trained on UI-PRMD movements 7 and 9 only, with cross-subject splits.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLUMNS = ["angle_max_deg", "angle_min_deg", "angle_mean_deg", "angle_range_deg",
                    "angle_midpoint_deg", "angle_recovery_deg"]
LABEL_COLUMN = "is_correct"
GROUP_COLUMN = "subject_id"


class CorrectnessClassifier:
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self._model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self._is_fitted = False

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self._model.fit(features[FEATURE_COLUMNS], labels)
        self._is_fitted = True

    def predict(self, features: dict) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained or loaded yet")
        row = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
        label = bool(self._model.predict(row)[0])
        confidence = float(self._model.predict_proba(row)[0].max())
        return {"is_correct": label, "confidence": confidence}

    def predict_batch(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model is not trained or loaded yet")
        rows = features[FEATURE_COLUMNS]
        labels = self._model.predict(rows).astype(bool)
        confidences = self._model.predict_proba(rows).max(axis=1)
        return pd.DataFrame({"is_correct": labels, "confidence": confidences}, index=features.index)

    def save(self, path: str | Path) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str | Path) -> None:
        self._model = joblib.load(path)
        self._is_fitted = True


def compute_correct_feature_ranges(dataset: pd.DataFrame) -> dict:
    """Per-feature (min, max) across only the "correct"-labeled training
    rows. Used to flag a live prediction as outside anything the model was
    actually shown as correct -- e.g. a healthy person's unusually full
    range of motion, or an injured person's genuinely limited range, can
    both fall outside this narrow healthy-volunteer band and get
    confidently (and wrongly) called "incorrect" by the classifier alone.
    """
    correct_rows = dataset[dataset[LABEL_COLUMN]]
    return {col: [float(correct_rows[col].min()), float(correct_rows[col].max())] for col in FEATURE_COLUMNS}


def is_within_correct_range(features: dict, correct_feature_ranges: dict) -> bool:
    """True only if every feature value falls within the training data's
    observed "correct"-class range for that feature. False means the
    classifier's label (whatever it predicted) isn't well supported by
    what it actually saw labeled correct during training -- a reason to
    treat the prediction with extra caution, not a reason to flip it.
    """
    return all(
        lo <= features[col] <= hi
        for col, (lo, hi) in correct_feature_ranges.items()
        if col in features
    )


def cross_subject_split(
    dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_subjects = dataset[GROUP_COLUMN].nunique()
    if n_subjects < 5:
        print(
            f"Warning: only {n_subjects} subjects in dataset; a held-out test "
            "group will be very small. Treat any resulting metric with caution."
        )
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(dataset, groups=dataset[GROUP_COLUMN]))
    return dataset.iloc[train_idx], dataset.iloc[test_idx]
