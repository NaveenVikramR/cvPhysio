"""Copies result files into team_results/<date>_<label>/ with a
provenance.json, so results made on one machine can be committed and read on
another (results/ itself is gitignored and per-machine).

Refuses videos and anything over MAX_FILE_BYTES, so a recording can't be
committed by mistake (docs/RECORDING_GUIDE.md, rule 2).

Usage: python scripts/package_results.py <label> --machine <name> <file> [<file> ...]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEAM_RESULTS_DIR = REPO_ROOT / "team_results"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".raw"}
MAX_FILE_BYTES = 5 * 1024 * 1024
LABEL_RE = re.compile(r"[A-Za-z0-9-]+")


def git_state(root: Path = REPO_ROOT) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "dirty": None}


def check_files(files: list[Path]) -> None:
    for f in files:
        if not f.is_file():
            raise ValueError(f"Not a file: {f}")
        if f.suffix.lower() in VIDEO_SUFFIXES:
            raise ValueError(f"Refusing to package a video: {f} -- videos go to the shared drive, never git")
        if f.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{f} is over {MAX_FILE_BYTES // (1024 * 1024)} MB -- share it through the drive")
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise ValueError("Two files share a name; package them separately")


def package(label: str, machine: str, files: list[Path], out_root: Path = TEAM_RESULTS_DIR,
            today: date | None = None, root: Path = REPO_ROOT) -> Path:
    if not LABEL_RE.fullmatch(label) or not LABEL_RE.fullmatch(machine):
        raise ValueError("label and --machine may use only letters, digits and '-'")
    check_files(files)
    base = out_root / f"{(today or date.today()).isoformat()}_{label}"
    out_dir, n = base, 2
    while out_dir.exists():  # never overwrite an earlier package
        out_dir = base.with_name(f"{base.name}-{n}")
        n += 1
    out_dir.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, out_dir / f.name)
    provenance = {"label": label, "machine": machine,
                  "packaged_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                  **git_state(root), "files": [f.name for f in files]}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label", help="short name, letters/digits/'-', e.g. camera-angle-test")
    parser.add_argument("--machine", required=True, help="e.g. laptop or pc")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        out_dir = package(args.label, args.machine, args.files)
    except ValueError as exc:
        print(f"Not packaged: {exc}")
        return 1
    print(f"Packaged {len(args.files)} file(s) into {out_dir.relative_to(REPO_ROOT)}. Commit and push that folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
