"""Training checkpoint format and I/O.

A checkpoint is a dict with keys ``model``, ``optimizer``, ``scheduler``,
``global_step``, ``norm_stats``, ``batch_size``, and ``data_world``. Legacy
checkpoints may lack the optimizer/scheduler/topology keys and may spell the
step ``step`` or ``training_step``; readers tolerate both.
"""

import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch


def model_state_dict(ckpt) -> dict[str, torch.Tensor]:
    """Return model weights from current, released-VLA, or bare checkpoints.

    ``model`` is the canonical key used by local/DiT checkpoints.  The first
    released VLA checkpoints used ``model_state_dict``; keep reading that
    layout so existing public objects remain usable.  Compile prefixes are an
    implementation detail and are stripped at the checkpoint boundary.
    """
    if not isinstance(ckpt, Mapping):
        raise TypeError(f"checkpoint must be a mapping, got {type(ckpt).__name__}")

    state = None
    for key in ("model", "model_state_dict", "state_dict"):
        candidate = ckpt.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if state is None:
        # A bare state_dict is still a supported model-only checkpoint.
        state = ckpt

    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint does not contain a valid model state_dict")
    non_tensors = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise ValueError(
            "checkpoint does not contain a model state_dict "
            f"(non-tensor entries: {non_tensors[:5]})"
        )
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def checkpoint_step(ckpt) -> int:
    for key in ("global_step", "training_step", "step"):
        if key in ckpt:
            return int(ckpt[key])
    return 0


def load_checkpoint(path):
    """Return (checkpoint dict, saved global step)."""
    ckpt = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    return ckpt, checkpoint_step(ckpt)


def check_topology(ckpt, *, batch_size, data_world, rank) -> None:
    """The step-keyed sampler maps steps to samples via batch size and data
    world, so resuming under a different topology shifts the data stream."""
    if rank != 0:
        return
    for key, current in (("batch_size", batch_size), ("data_world", data_world)):
        saved = ckpt.get(key)
        if saved is not None and int(saved) != int(current):
            print(f"WARNING: resume checkpoint was written with {key}={saved} "
                  f"but this run uses {current}; the resumed data-stream "
                  f"position will not correspond")


def fsdp_state_dicts(model, optimizer):
    """Full, unsharded model and optimizer state on CPU. A collective: every rank calls it."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return (get_model_state_dict(model, options=options),
            get_optimizer_state_dict(model, optimizer, options=options))


def restore_training_state(ckpt, *, optimizer, scheduler, resume_step, rank, source=None,
                           fsdp_model=None) -> None:
    """Restore optimizer/scheduler state, tolerating legacy model-only files.

    ``fsdp_model`` is the sharded model when resuming under FSDP: its optimizer
    state is keyed by parameter name, so DDP and FSDP checkpoints do not
    interchange their optimizer state (the weights load either way)."""
    source = source or "resume checkpoint"
    if "optimizer" in ckpt and fsdp_model is not None:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

        if any(isinstance(key, int) for key in ckpt["optimizer"].get("state", {})):
            raise ValueError(f"{source} holds DDP optimizer state; resume it without --fsdp")
        set_optimizer_state_dict(fsdp_model, optimizer, optim_state_dict=ckpt["optimizer"],
                                 options=StateDictOptions(full_state_dict=True))
    elif "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    elif rank == 0:
        print(f"WARNING: {source} has no optimizer state "
              f"(pre-resume-format checkpoint); Adam moments start fresh")
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        # LambdaLR is a pure function of the step count, so the schedule
        # is recoverable even when the checkpoint predates saved state.
        for _ in range(resume_step):
            scheduler.step()
        if rank == 0 and resume_step:
            print(f"WARNING: no scheduler state in checkpoint; "
                  f"fast-forwarded LR schedule to step {resume_step}")


def save_checkpoint(path, *, module, optimizer, scheduler, global_step, norm_stats,
                    batch_size=None, data_world=None, model_config=None,
                    train_config=None, model_state=None, optimizer_state=None) -> None:
    """Write the checkpoint via tmp file + rename so a crash mid-write can
    never leave a truncated file where resume looks. ``model_state`` and
    ``optimizer_state`` carry the gathered dicts of an FSDP run."""
    payload = {
        "model": module.state_dict() if model_state is None else model_state,
        "optimizer": optimizer.state_dict() if optimizer_state is None else optimizer_state,
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "norm_stats": norm_stats,
    }
    if batch_size is not None:
        payload["batch_size"] = int(batch_size)
    if data_world is not None:
        payload["data_world"] = int(data_world)
    if model_config is not None:
        payload["model_config"] = model_config
    if train_config is not None:
        # Plain dict so eval-side readers (resolve_trained_max_prefix) need
        # no import of this repo's dataclasses to interpret it.
        payload["train_config"] = train_config
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def update_last(output_dir: Path, path: Path) -> None:
    """Point last.pt at the newest checkpoint without re-serializing the
    multi-GB payload: hardlink when the filesystem allows it, else copy."""
    tmp = output_dir / "last.pt.tmp"
    tmp.unlink(missing_ok=True)
    try:
        os.link(path, tmp)
    except OSError:
        shutil.copyfile(path, tmp)
    tmp.replace(output_dir / "last.pt")
