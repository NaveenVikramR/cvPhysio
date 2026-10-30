"""Progress tracking. Input: one rep record per completed rep (exercise, side,
angle, correctness, confidence). Output: session log files + summaries.
"""

import csv
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from src.statuses import RuleStatus

LOGS_DIR = Path("logs")


@dataclass
class RepRecord:
    exercise: str
    side: str
    angle_deg: float
    status: str  # one of RuleStatus's values (classifier scorer only ever produces PASS/FAIL)
    confidence: float | None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DiscardRecord:
    """A repetition the segmenter started buffering but never completed.
    reason_code is one of DiscardReason's values (see app.py's RepDetector)."""
    exercise: str
    reason_code: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SessionTracker:
    def __init__(self, user_id: str, exercise: str, camera_info: dict | None = None):
        """camera_info, if given: {"width": int, "height": int, "fps":
        float} -- the resolution/fps the live webcam actually reported
        (cv2.VideoCapture.get(), after opening it), not a configured or
        assumed value. None (the default) for any caller that doesn't
        have a live camera (e.g. a future headless/video-file tracker),
        kept out of summary() entirely in that case."""
        self.user_id = user_id
        self.exercise = exercise
        self.camera_info = camera_info
        self.started_at = datetime.now(timezone.utc)
        self.reps: list[RepRecord] = []
        self.discards: list[DiscardRecord] = []

    def add_rep(self, rep: RepRecord) -> None:
        self.reps.append(rep)

    def add_discard(self, discard: DiscardRecord) -> None:
        self.discards.append(discard)

    def summary(self) -> dict:
        total = len(self.reps)
        passed = sum(1 for r in self.reps if r.status == RuleStatus.PASS)
        failed = sum(1 for r in self.reps if r.status == RuleStatus.FAIL)
        insufficient = sum(1 for r in self.reps if r.status == RuleStatus.INSUFFICIENT_DATA)
        scored = passed + failed
        result = {
            "user_id": self.user_id,
            "exercise": self.exercise,
            "total_reps": total,
            "passed_reps": passed,
            "failed_reps": failed,
            "insufficient_data_reps": insufficient,
            "accuracy_pct": round(100 * passed / scored, 1) if scored else None,
            "insufficient_data_rate_pct": round(100 * insufficient / total, 1) if total else None,
            "discarded_segments": len(self.discards),
        }
        if self.camera_info is not None:
            result["camera"] = self.camera_info
        return result

    def save(self, logs_dir: Path | str = LOGS_DIR) -> Path:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        path = logs_dir / f"{self.user_id}_{self.exercise}_{stamp}.csv"
        with path.open("w", newline="") as f:
            # Field names come from the dataclass itself, not from an
            # instance -- so a header row is always written, even for a
            # zero-rep session (previously: fieldnames=[] and no
            # writeheader() call when self.reps was empty, producing a
            # silent 0-byte file).
            writer = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(RepRecord)])
            writer.writeheader()
            for rep in self.reps:
                writer.writerow(asdict(rep))
        if self.discards:
            discards_path = path.with_suffix(".discards.csv")
            with discards_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.discards[0]).keys()))
                writer.writeheader()
                for discard in self.discards:
                    writer.writerow(asdict(discard))
        summary_path = path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(self.summary(), indent=2))
        return path


def load_history(user_id: str, logs_dir: Path | str = LOGS_DIR) -> list[dict]:
    logs_dir = Path(logs_dir)
    summaries = []
    for summary_file in sorted(logs_dir.glob(f"{user_id}_*.summary.json")):
        summaries.append(json.loads(summary_file.read_text()))
    return summaries
