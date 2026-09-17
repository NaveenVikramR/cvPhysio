"""Progress tracking. Input: one rep record per completed rep (exercise, side,
angle, correctness, confidence). Output: session log files + summaries.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path("logs")


@dataclass
class RepRecord:
    exercise: str
    side: str
    angle_deg: float
    is_correct: bool
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionTracker:
    def __init__(self, user_id: str, exercise: str):
        self.user_id = user_id
        self.exercise = exercise
        self.started_at = datetime.now(timezone.utc)
        self.reps: list[RepRecord] = []

    def add_rep(self, rep: RepRecord) -> None:
        self.reps.append(rep)

    def summary(self) -> dict:
        total = len(self.reps)
        correct = sum(1 for r in self.reps if r.is_correct)
        return {
            "user_id": self.user_id,
            "exercise": self.exercise,
            "total_reps": total,
            "correct_reps": correct,
            "accuracy_pct": round(100 * correct / total, 1) if total else None,
        }

    def save(self, logs_dir: Path | str = LOGS_DIR) -> Path:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        path = logs_dir / f"{self.user_id}_{self.exercise}_{stamp}.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.reps[0]).keys()) if self.reps else [])
            if self.reps:
                writer.writeheader()
                for rep in self.reps:
                    writer.writerow(asdict(rep))
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(self.summary(), indent=2))
        return path


def load_history(user_id: str, logs_dir: Path | str = LOGS_DIR) -> list[dict]:
    logs_dir = Path(logs_dir)
    summaries = []
    for summary_file in sorted(logs_dir.glob(f"{user_id}_*.summary.json")):
        summaries.append(json.loads(summary_file.read_text()))
    return summaries
