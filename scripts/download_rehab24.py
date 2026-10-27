"""PART 0 of the REHAB24-6 benchmark task: downloads the 5 needed files
from Zenodo record 13305826 (CC BY-NC 4.0, academic/non-commercial use
only -- see data/README.md), verifies their md5 against the Zenodo API's
own reported checksums, and extracts them into data/rehab24_6/
(gitignored). Does not re-download a file whose md5 already matches.

Only these 5 of the record's 9 files are used (see data/README.md for
why): videos.zip, Segmentation.csv, Segmentation.txt, joints_names.txt,
3d_joints.zip. Total ~3.2GB -- check free disk space first.
"""

import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD_API = "https://zenodo.org/api/records/13305826"
DATA_DIR = Path("data/rehab24_6")
NEEDED_FILES = {"videos.zip", "Segmentation.csv", "Segmentation.txt", "joints_names.txt", "3d_joints.zip"}
MIN_FREE_BYTES = 5 * 1024 ** 3  # 5GB headroom for download + extraction


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_record() -> dict:
    with urllib.request.urlopen(RECORD_API, timeout=30) as resp:
        return json.loads(resp.read())


def download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def main() -> None:
    free_bytes = shutil.disk_usage(".").free
    print(f"Free disk space: {free_bytes / 1024**3:.1f} GB")
    if free_bytes < MIN_FREE_BYTES:
        print(f"Less than {MIN_FREE_BYTES / 1024**3:.0f}GB free -- stopping before download.")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_dir = DATA_DIR / "_download"
    download_dir.mkdir(exist_ok=True)

    print("Fetching file list + md5s from the Zenodo API...")
    record = fetch_record()
    files_by_name = {f["key"]: f for f in record["files"]}
    missing = NEEDED_FILES - set(files_by_name)
    if missing:
        print(f"Zenodo record is missing expected files: {missing} -- stopping.")
        sys.exit(1)

    for name in sorted(NEEDED_FILES):
        meta = files_by_name[name]
        expected_md5 = meta["checksum"].split(":", 1)[1]
        dest = download_dir / name
        if dest.exists() and md5sum(dest) == expected_md5:
            print(f"  {name}: already present, md5 OK -- skipping download.")
            continue
        url = meta["links"]["self"]
        print(f"  {name}: downloading from {url} ({meta['size'] / 1024**2:.1f} MB)...")
        try:
            download_file(url, dest)
        except Exception as exc:
            print(f"  {name}: download failed ({exc}). Manual URL: {url}")
            sys.exit(1)
        actual_md5 = md5sum(dest)
        if actual_md5 != expected_md5:
            print(f"  {name}: MD5 MISMATCH (expected {expected_md5}, got {actual_md5}) -- stopping.")
            sys.exit(1)
        print(f"  {name}: md5 verified ({actual_md5}).")

    for zip_name in ("videos.zip", "3d_joints.zip"):
        zip_path = download_dir / zip_name
        extract_dir = DATA_DIR / zip_name.removesuffix(".zip")
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"  {zip_name}: already extracted to {extract_dir} -- skipping.")
            continue
        print(f"  Extracting {zip_name} to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    for flat_name in ("Segmentation.csv", "Segmentation.txt", "joints_names.txt"):
        shutil.copy(download_dir / flat_name, DATA_DIR / flat_name)

    print(f"\nDone. Dataset ready under {DATA_DIR}/")


if __name__ == "__main__":
    main()
