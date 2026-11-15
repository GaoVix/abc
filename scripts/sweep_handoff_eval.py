"""Independently hand off every saved old-policy control step to the new policy.

Reuse the checkpoint and source trajectory. Reset the environment, policy noise,
and RTC for every start. Each completed point is reusable after interruption.
By default keep trajectories and hashes; --save-video also exports each video.
"""
from scripts import handoff_eval_case as handoff  # Sets headless/deterministic environment before torch imports.

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from abc_minimal.determinism import configure_determinism
from abc_minimal.policy import DiTInferencePolicy
from scripts.search_eval_failure_case import source_hashes, write_json


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(args, source, previous):
    checkpoint = Path(source["new_summary"]["config"]["checkpoint"]).resolve()
    stat = checkpoint.stat()
    checkpoint_stat = dict(path=str(checkpoint), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    old = previous.get("identity", {})
    checkpoint_hash = (old["checkpoint_sha256"] if old.get("checkpoint_stat") == checkpoint_stat
                       else file_hash(checkpoint))
    inputs = ("scene.json", "old/summary.json", "old/initial.npz", "old/trajectory.npz",
              "new/summary.json", "new/initial.npz")
    code = source_hashes()
    for path in (Path(__file__), Path(handoff.__file__)):
        code[path.name] = file_hash(path)
    return dict(
        case_dir=str(args.case_dir), checkpoint_stat=checkpoint_stat, checkpoint_sha256=checkpoint_hash,
        input_hashes={name: file_hash(args.case_dir / name) for name in inputs}, code_hashes=code,
        arm_velocity=args.arm_velocity, ik_attempts=args.ik_attempts,
        start_step=args.start_step, stop_step=args.stop_step, step_stride=args.step_stride,
        torch=torch.__version__, mujoco=mujoco.__version__, numpy=np.__version__,
        gpu_id=args.gpu_id, gpu_name=torch.cuda.get_device_name(args.gpu_id),
    )


def update_progress(output, steps, records, active=None):
    rows = [records[step] for step in steps if step in records]
    runs = [row for row in rows if row["status"] == "complete"]
    successes = sum(row["success"] for row in runs)
    seconds = sum(row["seconds"] for row in runs)
    remaining = len(steps) - len(rows)
    progress = dict(
        status="complete" if not remaining else "partial", total_starts=len(steps),
        completed_starts=len(rows), remaining_starts=remaining, active_step=active,
        policy_runs=len(runs), policy_successes=successes, policy_failures=len(runs) - successes,
        already_success_at_start=sum(row["status"] == "already_success_at_start" for row in rows),
        handoff_failures=sum(row["status"] == "handoff_failed" for row in rows),
        policy_success_percent=100 * successes / len(runs) if runs else None,
        mean_seconds_per_policy_start=seconds / len(runs) if runs else None,
        estimated_remaining_hours=remaining * seconds / len(runs) / 3600 if runs else None,
        eta_note="Based on measured starts; completion time can vary with the starting state.",
    )
    write_json(output / "progress.json", progress)
    csv_path = output / "results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fields = ("source_step", "source_time_s", "status", "success", "initial_reward", "final_reward",
              "max_reward", "rollout_steps", "seconds", "position_max_mm", "rotation_max_deg", "folder", "error")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return progress


def completed_record(step, source, folder, report, result, seconds):
    world = result["world"] if result is not None else None
    return dict(
        source_step=step, source_time_s=float(source["trace"]["time"][step]),
        status="complete" if world is not None else "already_success_at_start",
        success=bool(world["success"]) if world is not None else None,
        initial_reward=report["target_task_eval"]["reward"],
        final_reward=world["reward"] if world is not None else report["target_task_eval"]["reward"],
        max_reward=world["max_reward"] if world is not None else report["target_task_eval"]["reward"],
        rollout_steps=world["steps"] if world is not None else 0, seconds=seconds,
        position_max_mm=max(arm["position_mm"] for arm in report["arms"].values()),
        rotation_max_deg=max(arm["rotation_deg"] for arm in report["arms"].values()), folder=str(folder),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--stop-step", type=int, help="Exclusive; default includes the old final state.")
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-starts", type=int, help="Limit NEW starts this invocation; rerun to continue.")
    parser.add_argument("--arm-velocity", choices=("zero", "match-twist"), default="match-twist")
    parser.add_argument("--ik-attempts", type=int, default=16)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--log-every-chunk", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Show count and settings without loading a model.")
    args = parser.parse_args()
    args.case_dir, args.output = args.case_dir.resolve(), args.output.resolve()
    if min(args.step_stride, args.ik_attempts, args.render_stride) < 1 or (
            args.max_starts is not None and args.max_starts < 1):
        parser.error("Strides, attempts, and max-starts must be positive")
    source = handoff.load_case(args.case_dir)
    count = len(source["trace"]["qpos"])
    args.stop_step = count if args.stop_step is None else args.stop_step
    if not 0 <= args.start_step < args.stop_step <= count:
        parser.error(f"Require 0 <= start-step < stop-step <= {count}")
    steps = list(range(args.start_step, args.stop_step, args.step_stride))
    plan = dict(total_source_states=count, selected_starts=len(steps), first_step=steps[0], last_step=steps[-1],
                source_step_unit="One saved control/action step, not one internal MuJoCo substep",
                control_dt_s=float(np.median(np.diff(source["trace"]["time"]))),
                independent_starts=True, same_noise_stream_at_each_start=True,
                checkpoint=source["new_summary"]["config"]["checkpoint"], videos=args.save_video,
                max_new_starts_this_invocation=args.max_starts)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if not 0 <= args.gpu_id < torch.cuda.device_count():
        parser.error(f"GPU {args.gpu_id} is unavailable; visible GPUs: {torch.cuda.device_count()}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    previous = handoff.load_json(manifest_path) if manifest_path.exists() else {}
    contract = identity(args, source, previous)
    if previous and previous["identity"] != contract:
        raise ValueError("Inputs, code, solver settings, GPU, or selected range changed; use a new output directory")
    write_json(manifest_path, dict(identity=contract, plan=plan))
    records = {}
    for step in steps:
        path = args.output / f"step_{step:06d}/point.json"
        if path.exists():
            records[step] = handoff.load_json(path)
    update_progress(args.output, steps, records)
    policy, added = None, 0
    for step in steps:
        if step in records:
            continue
        if args.max_starts is not None and added >= args.max_starts:
            break
        folder = args.output / f"step_{step:06d}"
        folder.mkdir(exist_ok=True)
        update_progress(args.output, steps, records, active=step)
        print(f"START source_step={step} source_time={source['trace']['time'][step]:.6f}s "
              f"done={len(records)}/{len(steps)}", flush=True)
        trial = SimpleNamespace(**vars(args))
        trial.output, trial.physics_step, trial.video_time = folder, step, 0.
        started = time.perf_counter()
        try:
            cached = folder / "result.json"
            if cached.exists():
                result = handoff.load_json(cached)
                report = result["handoff"]
                record = completed_record(step, source, folder, report, result, result["world"]["wall_s"])
            else:
                initial, report, scene, old_summary, new_summary = handoff.prepare(trial, source)
                np.savez(folder / "handoff_state.npz", **initial)
                write_json(folder / "handoff.json", report)
                result = None
                if not report["target_task_eval"]["success"]:
                    if policy is None:
                        config = handoff.handoff_config(new_summary, args.gpu_id, folder, args.log_every_chunk)
                        configure_determinism(config.policy_seed)
                        torch.cuda.set_device(args.gpu_id)
                        policy = DiTInferencePolicy(Path(config.checkpoint), config, f"cuda:{args.gpu_id}")
                        # Exclude one-time checkpoint loading from the per-start ETA.
                        started = time.perf_counter()
                    result = handoff.run(trial, initial, report, scene, old_summary, new_summary,
                                         policy=policy, save_video=args.save_video,
                                         log_every_chunk=args.log_every_chunk)
                record = completed_record(step, source, folder, report, result, time.perf_counter() - started)
        except handoff.HandoffError as error:
            record = dict(source_step=step, source_time_s=float(source["trace"]["time"][step]),
                          status="handoff_failed", success=None, seconds=time.perf_counter() - started,
                          folder=str(folder), error=str(error))
        except Exception as error:
            write_json(folder / "error.json", dict(source_step=step, type=type(error).__name__, error=str(error)))
            raise
        write_json(folder / "point.json", record)
        records[step] = record
        added += 1
        progress = update_progress(args.output, steps, records)
        print(f"DONE source_step={step} status={record['status']} success={record['success']} "
              f"seconds={record['seconds']:.2f} completed={progress['completed_starts']}/{len(steps)} "
              f"eta_hours={progress['estimated_remaining_hours']}", flush=True)
    print(json.dumps(update_progress(args.output, steps, records), indent=2), flush=True)


if __name__ == "__main__":
    main()
