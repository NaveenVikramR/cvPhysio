"""scripts/package_results.py, entirely under tmp_path."""

import json
from datetime import date

import pytest

from scripts import package_results as pr


def make(tmp_path, name, size=10):
    f = tmp_path / name
    f.write_bytes(b"x" * size)
    return f


def test_packages_files_with_provenance(tmp_path):
    report = make(tmp_path, "diag_rotation.md")
    plot = make(tmp_path, "plot.png")
    out = pr.package("camera-angle-test", "laptop", [report, plot], out_root=tmp_path / "team_results",
                     today=date(2026, 11, 17))
    assert out.name == "2026-11-17_camera-angle-test"
    assert (out / "diag_rotation.md").exists() and (out / "plot.png").exists()
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["machine"] == "laptop" and prov["files"] == ["diag_rotation.md", "plot.png"]
    assert "commit" in prov and "dirty" in prov


def test_never_overwrites_an_earlier_package(tmp_path):
    f = make(tmp_path, "a.md")
    root = tmp_path / "team_results"
    first = pr.package("x", "pc", [f], out_root=root, today=date(2026, 11, 17))
    second = pr.package("x", "pc", [f], out_root=root, today=date(2026, 11, 17))
    assert first != second and second.name == "2026-11-17_x-2"


@pytest.mark.parametrize("name", ["take.mp4", "take.MOV", "spool.raw", "clip.webm"])
def test_refuses_videos(tmp_path, name):
    with pytest.raises(ValueError, match="video"):
        pr.package("x", "pc", [make(tmp_path, name)], out_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_large_files(tmp_path):
    big = make(tmp_path, "big.csv", size=pr.MAX_FILE_BYTES + 1)
    with pytest.raises(ValueError, match="MB"):
        pr.package("x", "pc", [big], out_root=tmp_path / "out")


@pytest.mark.parametrize("label,machine", [("bad label", "pc"), ("ok", "my_pc"), ("../up", "pc")])
def test_refuses_unsafe_names(tmp_path, label, machine):
    with pytest.raises(ValueError):
        pr.package(label, machine, [make(tmp_path, "a.md")], out_root=tmp_path / "out")


def test_refuses_duplicate_names(tmp_path):
    a = make(tmp_path, "r.md")
    (tmp_path / "sub").mkdir()
    b = make(tmp_path / "sub", "r.md")
    with pytest.raises(ValueError, match="share a name"):
        pr.package("x", "pc", [a, b], out_root=tmp_path / "out")
