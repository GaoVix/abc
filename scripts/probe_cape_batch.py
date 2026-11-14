"""Find the largest per-GPU CaPE batch that completes a short real training run.

Each trial starts fresh torchrun workers and uses the normal training loop.
Camera preprocessing, validation, W&B and checkpoint writes are skipped.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RESULT_PREFIX = "BATCH_PROBE_RESULT "


def worker():
    import torch
    import tyro
    from abc_minimal.config import TrainConfig
    from abc_minimal import train_loop

    config = tyro.cli(TrainConfig, args=sys.argv[2:])
    if not torch.cuda.is_available():
        raise RuntimeError("Batch probing requires CUDA")
    config.train_steps = int(os.environ["CAPE_PROBE_STEPS"])
    config.log_every = min(5, config.train_steps)
    config.val_every = config.ckpt_every = config.train_steps + 1
    config.log_wandb = False
    # Only these disposable workers skip the training loop's final checkpoint.
    train_loop.save_checkpoint = lambda *args, **kwargs: None
    train_loop.update_last = lambda *args, **kwargs: None
    with tempfile.TemporaryDirectory(prefix="cape-batch-probe-") as directory:
        config.output_dir = directory
        train_loop.main(config)
    torch.cuda.synchronize()
    result = {"rank": int(os.environ.get("RANK", 0)), "batch_size": config.batch_size,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    print(RESULT_PREFIX + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pretrained-ckpt-name", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--start-batch", type=int, default=64)
    # Batch 90 already ran out of memory on the current four-H100 model.
    parser.add_argument("--max-batch", type=int, default=89)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.start_batch <= args.max_batch or args.steps < 3 or args.num_gpus < 1:
        parser.error("Require 1 <= start-batch <= max-batch, steps >= 3 and num-gpus >= 1")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1",
                       CAPE_PROBE_STEPS=str(args.steps))
    lower, upper, candidate = 0, args.max_batch + 1, args.start_batch
    trials = []
    while upper - lower > 1:
        log_path = args.output_dir / f"batch_{candidate}.log"
        command = [str(ROOT / ".venv/bin/torchrun"), "--standalone",
                   "--nproc-per-node", str(args.num_gpus), "-m", "scripts.probe_cape_batch", "--worker",
                   "--cache-root", str(args.cache_root.resolve()),
                   "--pretrained-ckpt-name", str(args.pretrained_ckpt_name.resolve()),
                   "--load-pretrained", "--model.use-cape",
                   "--mixture-preset", "load_the_plates_into_the_dish_rack",
                   "--batch-size", str(candidate), "--num-workers", str(args.num_workers)]
        print(f"Testing batch={candidate} per GPU for {args.steps} updates; log={log_path}", flush=True)
        with log_path.open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
        output = log_path.read_text(errors="replace")
        if completed.returncode == 0:
            results = [json.JSONDecoder().raw_decode(part)[0]
                       for part in output.split(RESULT_PREFIX)[1:]]
            if len(results) != args.num_gpus:
                raise RuntimeError(f"Missing per-rank results; inspect {log_path}")
            lower = candidate
            peak = max(row["peak_reserved_gib"] for row in results)
            trials.append({"batch": candidate, "status": "passed", "ranks": results})
            print(f"PASS batch={candidate}; maximum reserved GPU memory={peak:.2f} GiB", flush=True)
        elif "cuda out of memory" in output.lower():
            upper = candidate
            trials.append({"batch": candidate, "status": "oom"})
            print(f"OOM batch={candidate}", flush=True)
        else:
            raise RuntimeError(f"Trial failed for a reason other than CUDA OOM; inspect {log_path}")
        candidate = (lower + upper) // 2
    summary = {"num_gpus": args.num_gpus, "steps_per_trial": args.steps,
               "searched_max_batch": args.max_batch, "largest_passing_batch_per_gpu": lower,
               "conservative_batch_per_gpu": max(1, lower * 9 // 10) if lower else None,
               "trials": trials}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}), flush=True)
    if not lower:
        raise SystemExit("No tested batch size completed successfully")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker()
    else:
        main()
