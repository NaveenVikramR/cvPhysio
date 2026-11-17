"""F3 of the landmark-fidelity work: the live pipeline before and after F2,
measured against REHAB24-6 Ex1 motion capture on the same frames.

  before  the app as measured in F1: Session(side="per_frame"), raw angle
  after   the app now: Session(side="auto") with the shoulder offset from
          config/angle_correction.json (what app.py uses with the rules scorer)
Unchanged between the two, so not compared: smoothing (off, F2 step 3), the
rep detector (original, F2 step 4), the elbow and trunk measures (F2 step 5).

For every recording span made of front and half-profile reps (some
recordings hold one block of each: bm.recording_view), both pipelines run over
the same cached landmarks. Measured:
  - the shoulder angle the pipeline would show on each annotated rep frame
    (before: the per-frame arm pick; after: the arm the session locked to,
    with the correction), compared with the mocap right-arm angle
    (anatomical hip); only frames where that arm passes the live gate
  - the share of annotated frames measured on the wrong (left) arm
  - rep counts against the annotated counts, and reps measured on both arms
No setting is chosen here: everything was fixed in F2. Calibration (1, 2, 6)
and evaluation (3, 4, 5, 7, 8, 9) subjects are reported separately.

Usage: python scripts/f3_before_after.py
Writes results/f3_before_after.md (gitignored).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.benchmark_rehab24 as bm
from scripts import landmark_cache
from scripts.landmark_fidelity_report import fidelity, mocap_angles
from src import pose_estimation
from src import rules as rules_module
from src.feature_extraction import extract_features, load_angle_correction
from src.pipeline import Session

REPORT_PATH = Path("results/f3_before_after.md")
VIEWS = ("front", "half-profile")


def pipelines() -> dict:
    return {"before": {"side": "per_frame", "correction": None},
            "after": {"side": "auto", "correction": load_angle_correction()}}


def run_recording(rec: dict, side: str, correction: dict | None, landmark_space: str) -> dict:
    """Rep counts from the session, then the angle it would show on every
    annotated frame (per-frame pick before; the locked arm after)."""
    cache, start, end = rec["cache"], rec["span"][0], min(rec["span"][1], rec["cache"]["frame_count"] - 1)
    session = Session(bm.EXERCISE, side=side, correction=correction)
    roles = rules_module.required_joint_roles()
    reps = mixed = 0
    for f in range(start, end + 1):
        if f == start:
            session.start()
        event, payload, *_ = session.process_frame(landmark_cache.landmarks_at(cache, f, landmark_space), None,
                                                    f / cache["fps"], roles)
        if event == "rep_complete":
            reps += 1
            mixed += len(set(payload["side"])) > 1
    arm = session.locked_side  # None for per_frame
    rows = []
    for f in rec["frames"]:
        feats = extract_features(landmark_cache.landmarks_at(cache, f, landmark_space), bm.EXERCISE, arm, correction)
        rows.append({"frame": f, "rep": rec["rep_of"][f], "view": rec["rep_view"][rec["rep_of"][f]],
                     "angle": feats["angle_deg"] if feats else np.nan,
                     "arm": feats["side"] if feats else None,
                     "truth": mocap_angles(rec["joints"][f])["mocap_shoulder"]})
    return {"frames": pd.DataFrame(rows), "detected": reps, "mixed": mixed, "locked": arm}


def load_recordings(manifest: pd.DataFrame, split: dict) -> list[dict]:
    calib = set(split["calibration_subjects"])
    out = []
    for _, rec in manifest.drop_duplicates(["video_id", "camera"]).iterrows():
        video = bm.DATA_DIR / "videos" / rec.video
        joints = bm.load_mocap_joints(rec.video_id)
        view = bm.recording_view(manifest, rec.video_id, rec.camera)
        if not set(view.split("+")) <= set(VIEWS) or joints is None or not video.exists():
            continue
        all_reps = manifest[(manifest.video_id == rec.video_id) & (manifest.camera == rec.camera)]
        cache = landmark_cache.get_or_build(video)
        rep_of = {f: i for i, r in all_reps[~all_reps.mocap_erroneous].iterrows()
                  for f in range(int(r.start_frame), int(r.end_frame) + 1)
                  if f < min(cache["frame_count"], len(joints))}
        out.append({"id": f"{rec.video_id}/{rec.camera}", "view": view, "subject": int(rec.subject),
                    "rep_view": all_reps["view"].to_dict(),
                    "group": "calibration" if int(rec.subject) in calib else "evaluation",
                    "cache": cache, "joints": joints, "annotated": len(all_reps), "rep_of": rep_of,
                    "frames": sorted(rep_of), "span": bm.recording_span(manifest, rec.video_id, rec.camera)})
    return out


def angle_summary(results: list[dict], view: str) -> dict:
    """Shoulder angle and wrong-arm share over annotated frames of `view`."""
    frames = pd.concat([r["frames"].assign(recording=r["rec"]["id"]) for r in results], ignore_index=True)
    frames = frames[frames.view == view]
    measured = frames[frames.arm.notna()]
    right = measured[measured.arm == "RIGHT"]
    s = fidelity(right.angle, right.truth, right.recording + ":" + right.rep.astype(str))
    return {"frames": len(frames), "wrong_arm": float((measured.arm == "LEFT").mean()) if len(measured) else np.nan,
            "bias": s["bias"], "mae": s["mae"], "mae_debiased": s["mae_debiased"], "r": s["r"]}


def count_summary(results: list[dict]) -> dict:
    """Rep counts per recording (a recording's count covers all its reps)."""
    err = [r["detected"] - r["rec"]["annotated"] for r in results]
    return {"recordings": len(results), "exact": sum(e == 0 for e in err), "mean_signed": float(np.mean(err)),
            "abs_error": int(np.sum(np.abs(err))), "mixed": sum(r["mixed"] for r in results),
            "reps": sum(r["detected"] for r in results)}


RECORDING_VIEWS = ("front", "half-profile", "front+half-profile")


def main() -> None:
    landmark_space = pose_estimation.load_config().get("landmark_space", "world")
    manifest, split = bm.load_manifest_and_split()
    commit, dirty = bm.git_provenance()
    recordings = load_recordings(manifest, split)
    configs = pipelines()
    results = {name: [] for name in configs}
    for rec in recordings:
        for name, cfg in configs.items():
            results[name].append({"rec": rec, **run_recording(rec, cfg["side"], cfg["correction"], landmark_space)})

    lines = ["# F3: live pipeline before and after F2 (REHAB24-6 Ex1, right arm)", "",
             f"_commit `{commit}`{' (dirty)' if dirty else ''}, landmark_space `{landmark_space}`, smoothing off_", "",
             "before = per-frame arm pick, raw angle (the app at F1). after = arm locked automatically, shoulder offset "
             f"{configs['after']['correction']['shoulder_abduction']:+.2f} deg removed (the app now).", "",
             "## Shoulder angle, by each rep's view", "",
             "Frames measured on the right arm that pass the live gate, vs mocap with the anatomical hip.", "",
             "| view | group | pipeline | annotated frames | wrong-arm frames | bias | MAE | MAE debiased | r |",
             "|---|---|---|---|---|---|---|---|---|"]
    for view in VIEWS:
        for group in ("calibration", "evaluation"):
            for name in configs:
                rs = [r for r in results[name] if r["rec"]["group"] == group]
                a = angle_summary(rs, view)
                if not a["frames"]:
                    continue
                lines.append(f"| {view} | {group} | {name} | {a['frames']} | {a['wrong_arm']:.1%} | {a['bias']:+.2f} | "
                             f"{a['mae']:.2f} | {a['mae_debiased']:.2f} | {a['r']:.3f} |")
    lines += ["", "## Rep counts, by recording", "",
              "Some recordings hold a front block and a half-profile block of reps (`front+half-profile`); a "
              "recording's count covers all its reps.", "",
              "| recording view | group | pipeline | recordings | exact rep counts | mean signed count error | "
              "total abs count error | reps measured on both arms |", "|---|---|---|---|---|---|---|---|"]
    for view in RECORDING_VIEWS:
        for group in ("calibration", "evaluation"):
            for name in configs:
                rs = [r for r in results[name] if r["rec"]["view"] == view and r["rec"]["group"] == group]
                if not rs:
                    continue
                c = count_summary(rs)
                lines.append(f"| {view} | {group} | {name} | {c['recordings']} | "
                             f"{bm.rate_with_ci(c['exact'], c['recordings'])} | {c['mean_signed']:+.2f} | "
                             f"{c['abs_error']} | {c['mixed']}/{c['reps']} |")
    locked = [r["locked"] for r in results["after"]]
    lines += ["", f"after: auto locked the RIGHT arm in {locked.count('RIGHT')}/{len(locked)} recordings "
                  f"(not locked: {locked.count(None)})."]
    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    print(text)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
