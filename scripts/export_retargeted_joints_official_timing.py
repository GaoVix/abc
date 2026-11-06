import argparse
import gzip
import hashlib
import json
import multiprocessing
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.export_retargeted_joints import (
    IK_MAX_CALLS,
    IK_POSITION_TOL_MM,
    IK_ROTATION_TOL_DEG,
    retarget,
    write_json,
)
from scripts.select_calibrated_episodes import link


def conversion_signature(calibration, source, real_model, sim_model):
    digest = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode())
    for path in (Path(__file__), REPO / "scripts/export_retargeted_joints.py",
                 REPO / "scripts/verify_camera_fk_ik.py", REPO / "deploy/robot/control/mink_ik.py",
                 real_model, sim_model):
        digest.update(path.read_bytes())
    digest.update((source / "episode_metadata.json").read_bytes())
    binary = source / "states_actions.bin"
    info = binary.stat()
    digest.update(str((binary, info.st_size, info.st_mtime_ns)).encode())
    return digest.hexdigest()


def convert_episode(record, source_root, output):
    episode = record["episode"]
    started = time.monotonic()
    try:
        relative = Path(f"{record['split']}_real") / episode
        source, destination = source_root / relative, output / relative
        metadata = json.loads((source / "episode_metadata.json").read_text())
        if "joint_conversion" in metadata:
            raise ValueError("Source is already retargeted; use the original official export")
        calibration = {key: record["calibration"][key]
                       for key in ("base_poses", "left_base_to_camera")}
        real_model = REPO / "abc_sim/models/yam_bimanual_empty.xml"
        sim_model = REPO / "abc_sim/models/yam_dishrack_base.xml"
        signature = conversion_signature(calibration, source, real_model, sim_model)
        if destination.exists():
            report = json.loads((destination / "joint_conversion_report.json").read_text())
            if report.get("signature") != signature or report.get("status") != "complete":
                raise ValueError("Existing output uses different conversion inputs; choose a new output directory")
            return {"episode": episode, "split": record["split"], "status": "complete",
                    "resumed": True, "num_steps": report["num_steps"]}
        rows = np.fromfile(source / "states_actions.bin", dtype=np.float64).reshape(-1, 28)
        converted, statistics = retarget(rows, calibration, real_model, sim_model)
        work = output / ".work" / relative
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        for path in source.iterdir():
            if path.name not in ("states_actions.bin", "episode_metadata.json"):
                link(path, work / path.name)
        converted.tofile(work / "states_actions.bin")
        metadata["joint_conversion"] = {
            "report": "joint_conversion_report.json", "solver": "MinkArmController",
            "time_offset_s": 0.0, "interpolation": False,
            "sampling": "unchanged official export", "target_camera": "simulation top",
        }
        write_json(work / "episode_metadata.json", metadata)
        report = {
            "status": "complete", "signature": signature, "episode": episode,
            "num_steps": len(converted), "source_episode": str(source),
            "calibration": calibration, "real_model": str(real_model), "sim_model": str(sim_model),
            "time_offset_s": 0.0, "interpolation": False,
            "ik_solver": "MinkArmController", "ik_dt": 1 / 60, "ik_iterations": 4,
            "ik_position_tolerance_mm": IK_POSITION_TOL_MM,
            "ik_rotation_tolerance_deg": IK_ROTATION_TOL_DEG,
            "ik_max_solve_calls": IK_MAX_CALLS,
            "grippers": "official sampled values preserved",
            "statistics": statistics, "seconds": time.monotonic() - started,
        }
        write_json(work / "joint_conversion_report.json", report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        work.rename(destination)
        return {"episode": episode, "split": record["split"], "status": "complete",
                "resumed": False, "num_steps": len(converted), "seconds": report["seconds"]}
    except Exception as error:
        return {"episode": episode, "split": record["split"], "status": "failed",
                "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description="Retarget official exported joints without changing time sampling")
    parser.add_argument("--source", type=Path, required=True, help="Original official exported dataset root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO / "scripts/dishrack_joint_retarget_calibrations.json.gz")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--episodes", nargs="*")
    args = parser.parse_args()
    source, output = args.source.expanduser().resolve(), args.output.expanduser().resolve()
    with gzip.open(args.manifest, "rt", encoding="utf-8") as stream:
        records = json.load(stream)["records"]
    selected = [r for r in records if r["calibration_quality"] == "accepted"]
    skipped = [r for r in records if r["calibration_quality"] != "accepted"]
    if args.episodes:
        selected = [r for r in selected if r["episode"] in args.episodes]
    output.mkdir(parents=True, exist_ok=True)
    for name in ("train_sim", "val_sim"):
        link(source / name, output / name)
    stats = source / "norm_stats.json"
    if not stats.exists():
        stats = source.parent / "norm_stats.json"
    if stats.exists():
        link(stats, output / "norm_stats.json")
    write_json(output / "skipped.json", skipped)
    print(f"{len(selected)} episodes, unchanged official timing -> {output}", flush=True)
    started, results = time.monotonic(), []

    def progress(status):
        value = {"status": status, "selected": len(selected), "skipped": len(skipped),
                 "completed": sum(r["status"] == "complete" for r in results),
                 "failed": sum(r["status"] == "failed" for r in results),
                 "elapsed_seconds": time.monotonic() - started}
        write_json(output / "progress.json", value)
        return value

    progress("running")
    with (output / "results.jsonl").open("w") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(convert_episode, r, source, output) for r in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            value = progress("running")
            print(f"[{len(results)}/{len(selected)}] {result['status']} {result['episode']}"
                  f" complete={value['completed']} failed={value['failed']}", flush=True)
            if result["status"] == "failed":
                print(result["error"], flush=True)
    value = progress("complete" if all(r["status"] == "complete" for r in results) else "finished_with_errors")
    write_json(output / "summary.json", {**value, "results": results})
    if value["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
