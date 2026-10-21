import argparse
import gzip
import hashlib
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from abc_minimal.export_mcap import ACTION_TOPICS, STATE_TOPICS, export_episode, floor_indices
from scripts.verify_camera_fk_ik import Robot, pose


IK_POSITION_TOL_MM = 0.1
IK_ROTATION_TOL_DEG = 0.1
IK_MAX_CALLS = 10


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def aligned_rows(mcap, metadata, offset_seconds):
    topics = STATE_TOPICS + ACTION_TOPICS
    messages = {name: [] for name, _ in topics}
    with mcap.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, value in reader.iter_decoded_messages(topics=list(messages), log_time_order=False):
            messages[channel.topic].append((message.log_time, list(value.position)))
    ticks = metadata["t0_ns"] + (np.arange(metadata["num_steps"], dtype=np.int64) + 1) * metadata["tick_ns"]
    offset = round(offset_seconds * 1_000_000_000)
    state_query = np.asarray(metadata["top_camera_timestamps_ns"], dtype=np.int64) + offset
    columns, held = [], {}
    for name, dim in topics:
        values = sorted(messages[name], key=lambda pair: pair[0])
        if not values:
            raise ValueError(f"Missing training-data topic: {name}")
        timestamps = np.array([row[0] for row in values], dtype=np.int64)
        positions = np.array([row[1] for row in values], dtype=np.float64).reshape(-1, dim)
        if (name, dim) in STATE_TOPICS:
            keep = np.concatenate((np.diff(timestamps) != 0, [True]))
            timestamps, positions = timestamps[keep], positions[keep]
            query = state_query
            columns.append(np.column_stack([
                np.interp(query - timestamps[0], timestamps - timestamps[0], positions[:, index])
                for index in range(dim)
            ]))
        else:
            query = ticks + offset
            columns.append(positions[floor_indices(timestamps, query)])
        held[name] = int(np.sum((query < timestamps[0]) | (query > timestamps[-1])))
    return np.concatenate(columns, axis=1), held


def retarget(rows, calibration, real_model, sim_model):
    real = Robot(real_model, calibration["base_poses"])
    sim = Robot(sim_model)
    sim.set_state(rows[0, :14])
    mujoco.mj_forward(sim.model, sim.data)
    sim.setup_ik(sim_model, rows[0, :14])
    world_from_camera = pose(sim.data.camera("top"))
    world_from_camera[:3, :3] @= np.diag([1., -1., -1.])
    world_from_real_base = world_from_camera @ np.asarray(calibration["left_base_to_camera"])
    converted = rows.copy()
    errors = np.empty((len(rows), 2, 2, 2))
    solve_calls = np.empty((len(rows), 2, 2), dtype=np.int16)
    seeds = [rows[0, :14].copy(), rows[0, 14:].copy()]
    for index, row in enumerate(rows):
        for stream in range(2):
            values = row[stream * 14:(stream + 1) * 14]
            real.set_state(values)
            sim.set_state(seeds[stream])
            result = values.copy()
            for arm in range(2):
                segment = slice(arm * 7, arm * 7 + 6)
                target = world_from_real_base @ real.tcp(arm)
                seed = seeds[stream][segment]
                for attempt in range(IK_MAX_CALLS):
                    joints, error = sim.solve(arm, target, seed)
                    if error[0] <= IK_POSITION_TOL_MM and error[1] <= IK_ROTATION_TOL_DEG:
                        break
                    seed = joints
                solve_calls[index, stream, arm] = attempt + 1
                result[segment] = joints
                errors[index, stream, arm] = error
            converted[index, stream * 14:(stream + 1) * 14] = result
            seeds[stream] = result
    statistics = {}
    for stream, name in enumerate(("state", "action")):
        statistics[name] = {}
        for arm, side in enumerate(("left", "right")):
            error = errors[:, stream, arm]
            joints = converted[:, stream * 14 + arm * 7:stream * 14 + arm * 7 + 6]
            statistics[name][side] = {
                "mean_position_error_mm": float(error[:, 0].mean()),
                "max_position_error_mm": float(error[:, 0].max()),
                "mean_rotation_error_deg": float(error[:, 1].mean()),
                "max_rotation_error_deg": float(error[:, 1].max()),
                "frames_not_converged": int(np.sum(
                    (error[:, 0] > IK_POSITION_TOL_MM) | (error[:, 1] > IK_ROTATION_TOL_DEG))),
                "frames_above_1mm_or_0_1deg": int(np.sum((error[:, 0] > 1) | (error[:, 1] > 0.1))),
                "max_joint_step_rad": float(np.abs(np.diff(joints, axis=0)).max(initial=0)),
                "mean_solve_calls": float(solve_calls[:, stream, arm].mean()),
                "max_solve_calls": int(solve_calls[:, stream, arm].max()),
                "one_call_fraction": float((solve_calls[:, stream, arm] == 1).mean()),
            }
    return converted, statistics


def conversion_signature(record, mcap, real_model, sim_model):
    digest = hashlib.sha256()
    digest.update(json.dumps(record, sort_keys=True, allow_nan=False).encode())
    for path in (Path(__file__), REPO / "scripts/verify_camera_fk_ik.py",
                 REPO / "deploy/robot/control/mink_ik.py", REPO / "abc_minimal/export_mcap.py",
                 real_model, sim_model):
        digest.update(path.read_bytes())
    info = mcap.stat()
    digest.update(str((mcap, info.st_size, info.st_mtime_ns)).encode())
    return digest.hexdigest()


def convert_episode(record, dataset, models, sim_model, output):
    episode = record["episode"]
    started = time.monotonic()
    try:
        mcap = dataset / record["source_relative"]
        calibration = record["calibration"]
        real_model = models / record["model_name"]
        signature = conversion_signature(record, mcap, real_model, sim_model)
        destination = output / f"{record['split']}_real" / episode
        report_path = destination / "joint_conversion_report.json"
        if destination.exists():
            report = json.loads(report_path.read_text())
            if report.get("signature") != signature or report.get("status") != "complete":
                raise ValueError("Existing output uses different conversion inputs; choose a new output directory")
            return {"episode": episode, "split": record["split"], "status": "complete",
                    "resumed": True, "num_steps": report["num_steps"]}
        work_root = output / ".work" / f"{record['split']}_real"
        exported = export_episode((str(mcap), record["task"], str(work_root)))
        if exported != episode:
            raise ValueError(f"Official exporter returned {exported!r}")
        work = work_root / episode
        metadata_path = work / "episode_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        rows, held = aligned_rows(mcap, metadata, calibration["state_time_offset_s"])
        converted, statistics = retarget(rows, calibration, real_model, sim_model)
        temporary = work / "states_actions.bin.tmp"
        converted.tofile(temporary)
        temporary.replace(work / "states_actions.bin")
        metadata["alignment"] = "fixed_clock_30hz_state_interpolated_to_selected_top_frame"
        metadata["joint_conversion"] = {
            "report": "joint_conversion_report.json", "solver": "MinkArmController",
            "state_time_offset_s": calibration["state_time_offset_s"],
            "sampling": {
                "state": "linear interpolation at selected top frame timestamp + calibrated offset",
                "action": "floor sampling at tick + calibrated offset",
            },
            "source_camera_topic": calibration["camera_topic"], "target_camera": "simulation top",
        }
        write_json(metadata_path, metadata)
        report = {
            "status": "complete", "signature": signature, "episode": episode,
            "num_steps": len(converted), "source_mcap": str(mcap),
            "calibration": calibration, "real_model": str(real_model),
            "sim_model": str(sim_model), "calibration_quality": record["calibration_quality"],
            "time_offset_s": calibration["state_time_offset_s"], "endpoint_hold_counts": held,
            "ik_solver": "MinkArmController", "ik_parameters": "repository defaults",
            "ik_dt": 1 / 60, "ik_iterations": 4, "grippers": "time-aligned values preserved",
            "ik_position_tolerance_mm": IK_POSITION_TOL_MM,
            "ik_rotation_tolerance_deg": IK_ROTATION_TOL_DEG,
            "ik_max_solve_calls": IK_MAX_CALLS,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sim-model", type=Path, default=REPO / "abc_sim/models/yam_dishrack_base.xml")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--episodes", nargs="*")
    args = parser.parse_args()
    with gzip.open(args.manifest, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    dataset, models = args.dataset, args.model_dir
    args.output.mkdir(parents=True, exist_ok=True)
    selected = [r for r in manifest["records"] if r["calibration_quality"] == "accepted"]
    skipped = [r for r in manifest["records"] if r["calibration_quality"] != "accepted"]
    if args.episodes:
        selected = [r for r in selected if r["episode"] in args.episodes]
    write_json(args.output / "skipped.json", skipped)
    started = time.monotonic()
    results = []
    def progress(status):
        value = {"status": status, "selected": len(selected), "skipped": len(skipped),
                 "completed": sum(r["status"] == "complete" for r in results),
                 "failed": sum(r["status"] == "failed" for r in results),
                 "elapsed_seconds": time.monotonic() - started}
        write_json(args.output / "progress.json", value)
        return value
    progress("running")
    with (args.output / "results.jsonl").open("w") as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(convert_episode, r, dataset, models, args.sim_model, args.output) for r in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            value = progress("running")
            print(f"[{len(results)}/{len(selected)}] {result['status']} {result['episode']}"
                  f" complete={value['completed']} failed={value['failed']}", flush=True)
    value = progress("complete" if all(r["status"] == "complete" for r in results) else "finished_with_errors")
    write_json(args.output / "summary.json", {**value, "results": results})


if __name__ == "__main__":
    main()
