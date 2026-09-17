"""Correctness classification. Input: feature_extraction's feature dict
(or a batch of them as rows for training). Output: label + confidence.
Trained on UI-PRMD movements 7 and 9 only, with cross-subject splits.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLUMNS = ["angle_deg"]
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

    def save(self, path: str | Path) -> None:
        joblib.dump(self._model, path)

    def load(self, path: str | Path) -> None:
        self._model = joblib.load(path)
        self._is_fitted = True


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


def load_dataset(raw_dir: str | Path, exercise: str) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    if not any(raw_dir.iterdir()) if raw_dir.exists() else True:
        raise NotImplementedError(
            f"No UI-PRMD files found under {raw_dir}. Download UI-PRMD movements "
            "7 and 9 into data/raw/ before calling load_dataset()."
        )
    raise NotImplementedError("UI-PRMD parsing not yet implemented for exercise: " + exercise)
