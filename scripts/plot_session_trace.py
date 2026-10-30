"""Report figure A1.3: shoulder angle against time for one saved live session, with
rep boundaries and each rep's peak marked and the allowed peak band shaded.
Read-only apart from the output image.

    python scripts/plot_session_trace.py                  # newest session with traces
    python scripts/plot_session_trace.py logs/<session>.csv

Every plotted value comes from the session's .traces.json, unsmoothed and
unresampled. Rep start times come from the session CSV's rep_start_timestamp,
because the trace's frame_t is relative to each rep. Frames between reps were never
logged, so reps are separate line segments and nothing is drawn across the gaps.

Output: results/figures/A1_3_shoulder_trace.png (gitignored).
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import rules as rules_module
from src.tracking import LOGS_DIR

OUTPUT_PATH = REPO_ROOT / "results" / "figures" / "A1_3_shoulder_trace.png"
LINE_COLOUR = "#1F6FB2"
BAND_COLOUR = "#D9D9D9"
BOUNDARY_COLOUR = "#8C8C8C"
TEXT_COLOUR = "#333333"
MAX_LABELLED_REPS = 15


class SessionError(Exception):
    pass


def find_latest_traces(logs_dir: Path) -> Path | None:
    traces = sorted(logs_dir.glob("*.traces.json"), key=lambda p: p.stat().st_mtime)
    return traces[-1] if traces else None


def traces_path_for(session: Path) -> Path:
    if session.name.endswith(".traces.json"):
        return session
    return session.with_suffix(".traces.json")


def load_session(traces_path: Path) -> tuple[dict, list[dict]]:
    """(traces file contents, session CSV rows)."""
    if not traces_path.exists():
        raise SessionError(f"No traces at {traces_path}. Sessions saved before traces were added can't be plotted.")
    data = json.loads(traces_path.read_text())
    if not data.get("reps"):
        raise SessionError(f"{traces_path} holds no reps.")
    csv_path = traces_path.with_name(traces_path.name.removesuffix(".traces.json") + ".csv")
    if not csv_path.exists():
        raise SessionError(f"No session CSV at {csv_path}, so rep start times are unknown. Nothing plotted.")
    with csv_path.open(newline="") as f:
        return data, list(csv.DictReader(f))


def rep_series(traces: list[dict], rows: list[dict]) -> list[dict]:
    """One {"rep", "t", "angle", "peak_t", "peak_angle"} per rep, with t in seconds
    since the first rep's start. Values are the trace's own, unchanged."""
    starts = {}
    for trace in traces:
        if trace["rep"] > len(rows) or not rows[trace["rep"] - 1].get("rep_start_timestamp"):
            raise SessionError(f"Rep {trace['rep']} has no rep_start_timestamp in the session CSV. Nothing plotted.")
        starts[trace["rep"]] = datetime.fromisoformat(rows[trace["rep"] - 1]["rep_start_timestamp"]).timestamp()
    origin = min(starts.values())

    series = []
    for trace in traces:
        frame_t, angle = trace["frame_t"], trace["angle_deg"]
        if not frame_t or len(frame_t) != len(angle):
            raise SessionError(f"Rep {trace['rep']} has unusable frame times in the trace. Nothing plotted.")
        t = [starts[trace["rep"]] - origin + value for value in frame_t]
        peak = max(range(len(angle)), key=angle.__getitem__)
        series.append({"rep": trace["rep"], "t": t, "angle": list(angle), "peak_t": t[peak], "peak_angle": angle[peak]})
    return series


def build_figure(series: list[dict], band: tuple[float, float], title: str):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    ax.axhspan(band[0], band[1], color=BAND_COLOUR, alpha=0.6, linewidth=0, zorder=0,
               label=f"Allowed peak band, {band[0]:g}-{band[1]:g}°")
    top = max(max(s["angle"]) for s in series)
    for i, s in enumerate(series):
        for edge in (s["t"][0], s["t"][-1]):
            ax.axvline(edge, color=BOUNDARY_COLOUR, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1,
                       label="Rep start and end" if i == 0 and edge == s["t"][0] else None)
        ax.plot(s["t"], s["angle"], color=LINE_COLOUR, linewidth=2, solid_capstyle="round", zorder=2,
                label="Shoulder angle (gated frames)" if i == 0 else None)
        ax.plot(s["peak_t"], s["peak_angle"], "o", color=LINE_COLOUR, markersize=7, markeredgecolor="white",
                markeredgewidth=1.5, zorder=3, label="Rep peak" if i == 0 else None)
        if len(series) <= MAX_LABELLED_REPS:
            ax.annotate(str(s["rep"]), (s["peak_t"], s["peak_angle"]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT_COLOUR)
    ax.set_xlabel("Time since first rep started (s)", color=TEXT_COLOUR)
    ax.set_ylabel("Shoulder angle (degrees)", color=TEXT_COLOUR)
    ax.set_ylim(bottom=0, top=max(top, band[1]) + 15)
    ax.set_title(title, fontsize=10, color=TEXT_COLOUR, loc="left")
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOUR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot report figure A1.3 from a saved session's trace.")
    parser.add_argument("session", nargs="?", type=Path, default=None, help="session CSV or .traces.json (default: newest)")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    traces_path = traces_path_for(args.session) if args.session else find_latest_traces(args.logs_dir)
    if traces_path is None:
        print(f"No session with traces in {args.logs_dir}. Nothing plotted.")
        return 1
    try:
        data, rows = load_session(traces_path)
        series = rep_series(data["reps"], rows)
    except SessionError as exc:
        print(exc)
        return 1

    config = rules_module.load_config(data["exercise"])
    target, tolerance = config["target_peak_deg"]["value"], config["tolerance_deg"]["value"]
    band = (target - tolerance, target + tolerance)
    session_name = traces_path.name.removesuffix(".traces.json")
    fig, _ax = build_figure(series, band, f"Session {session_name}: {len(series)} reps. Gaps between reps weren't logged.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)

    in_band = sum(band[0] <= s["peak_angle"] <= band[1] for s in series)
    print(f"Session: {session_name}  reps: {len(series)}  peaks in {band[0]:g}-{band[1]:g} deg band: {in_band}/{len(series)}")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
