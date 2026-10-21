"""Train an ABC policy on the bottles-in-bin dataset.

A single loop drives both policies, selected by ``config.policy``:
  * ``"dit"`` — the CLIP/DINOv3 ABC-DiT policy (default).
  * ``"vla"`` — ABC-VLA.
Both share this distributed setup, optimizer/scheduler, validation, and
checkpoint format; the policy only changes model construction, how a batch is
conditioned, and the training/sampling forward calls.
"""

import os
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from abc_minimal.checkpointing import (
    check_topology,
    load_checkpoint,
    model_state_dict,
    fsdp_state_dicts,
    restore_training_state,
    save_checkpoint,
    update_last,
)
from abc_minimal.config import (
    TrainConfig,
    validate_train_config,
    validate_vla_checkpoint_config,
)
from abc_minimal.dataloader import (
    build_train_loader,
    build_val_loaders,
    check_shard_consistency,
    data_parallel_scope,
    read_shard_marker,
)
from abc_minimal.dit import (
    CLIPTextEmbedder,
    DiTPolicy,
    load_clip_vision_weights,
    load_pretrained,
    task_name_to_prompt,
)
from abc_minimal.operator import load_operator_label_maps
from abc_minimal.preprocess import load_norm_stats, parse_norm_stats
from abc_minimal.vla import VLAPolicy, stack_camera_batch

# Enable TF32-backed fp32 matmul on NVIDIA GPUs.
torch.set_float32_matmul_precision("high")


def _distributed_context():
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        local_rank, local_world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, world, local_rank, local_world, device


def batch_to_device(batch, device, embedder, camera_keys):
    """Move a batch to the device and attach policy-specific conditioning.

    DiT: images stay a per-camera dict and prompts are CLIP-encoded into
    ``task_vec_clip``. VLA: images are stacked to ``(B, n_cam, C, H, W)`` and the
    raw prompt strings are carried through to the Gemma tokenizer.
    """
    out = {
        "state": batch["state"].to(device, non_blocking=True),
        "actions": batch["actions"].to(device, non_blocking=True),
        "images": {
            cam: v.to(device, non_blocking=True) for cam, v in batch["images"].items()
        },
        "state_is_masked": batch["state_is_masked"].to(device, non_blocking=True),
    }
    if embedder is not None:  # DiT path
        out["task_vec_clip"] = embedder.encode(batch["prompt"]).to(
            device, non_blocking=True
        )
        return out
    # VLA path: keep raw prompts, stack cameras into a single tensor.
    out["prompt"] = batch["prompt"]
    return stack_camera_batch(out, camera_keys)


def warn_bf16_rounding(groups, rank):
    """Report each bf16 param group's rounding floor.

    bf16 parameters have no fp32 master copy and only move when the step exceeds
    half their ulp, |w| / 256; training builds the VLA in fp32 for exactly this
    reason (bf16 parameters at backbone_lr_scale 0.1 froze ~93% of Gemma).
    """
    if rank != 0:
        return
    for group in groups:
        low = [p for p in group["params"] if p.dtype == torch.bfloat16]
        if not low:
            continue
        stuck = sum(int((p.detach().abs() * 2**-8 >= group["lr"]).sum()) for p in low)
        share = stuck / sum(p.numel() for p in low)
        level = "\033[93mWARNING" if share > 0.5 else "note"
        print(
            f"{level}: the {group['name']} group holds bf16 parameters with no fp32 master "
            f"copy; {share:.0%} of them cannot move at lr={group['lr']:.0e} (step below "
            "|w|/256). Keep them in fp32 for training, or keep their lr near 1e-4.\033[0m"
        )


def _shard_vla(model, world, local_world):
    """FSDP2: every transformer block is its own shard group; the root holds the rest.

    Multi-node uses HSDP (shard in-node, replicate across); single node flat FSDP.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    # Broadcast rank-0's parameters and buffers to every rank BEFORE sharding.
    for tensor in (*model.parameters(), *model.buffers()):
        dist.broadcast(tensor.detach(), src=0)

    n_nodes = world // local_world if local_world else 1
    if n_nodes > 1 and n_nodes * local_world == world:
        # HSDP: shard within node (NVLink), replicate across nodes.
        mesh = init_device_mesh(
            "cuda", (n_nodes, local_world), mesh_dim_names=("replicate", "shard")
        )
    else:
        # Full FSDP: shard across the whole world (single node, or ragged layout).
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("shard",))
    gemma = model.vla.gemma_model
    for block in (
        *gemma.model.layers,
        *gemma.siglip_vision_model.encoder_blocks,
        *model.diffusion_head.blocks,
    ):
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)


def _build_optimizer(model, config, is_vla, rank=0):
    """AdamW with a lower LR for the vision/backbone param group.

    DiT: the img_backbone group is scaled by ``optim.vision_lr_scale``.
    VLA: the Gemma+SigLIP backbone group is scaled by ``optim.backbone_lr_scale``.
    """
    if is_vla:
        backbone = [
            p for p in model.vla.gemma_model.parameters() if p.requires_grad
        ]
        scale = config.optim.backbone_lr_scale
    else:
        backbone = list(model.img_backbone.parameters())
        scale = config.optim.vision_lr_scale
    backbone_ids = {id(p) for p in backbone}
    main_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids
    ]
    groups = [{"params": main_params, "lr": config.optim.learning_rate, "name": "head"}]
    if backbone:
        groups.append(
            {"params": backbone, "lr": config.optim.learning_rate * scale, "name": "backbone"}
        )
    warn_bf16_rounding(groups, rank)
    return torch.optim.AdamW(
        groups,
        betas=(config.optim.adam_beta1, config.optim.adam_beta2),
        eps=config.optim.adam_epsilon,
        weight_decay=config.optim.weight_decay,
        # The foreach step materializes a gradient-sized temporary (13.5 GiB for the VLA).
        fused=is_vla,
    )


def _resolve_training_norm_stats(config, cache_root, ckpt_norm_stats, rank):
    """Use embedded stats by default, with the same cache fallback for both policies."""
    if config.inherit_ckpt_norm_stats and ckpt_norm_stats is not None:
        norm_stats = parse_norm_stats(ckpt_norm_stats)
        if rank == 0:
            print("using norm_stats embedded in the checkpoint")
        return norm_stats

    stats_path = cache_root / "norm_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found and the loaded checkpoint embeds no "
            "norm_stats; provide norm_stats.json or finetune from a "
            "checkpoint that embeds its stats."
        )
    return load_norm_stats(stats_path)


def _build_dit_model(config, cache_root, device, resume_ckpt, rank, distributed):
    """Construct DiTPolicy and load its starting weights. Returns (model, norm_stats)."""
    checkpoint_path = cache_root / config.pretrained_ckpt_name
    backbone = config.model.vision_backbone
    model = DiTPolicy(config.model)
    ckpt_norm_stats = None
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ckpt_norm_stats = resume_ckpt.get("norm_stats")
        if rank == 0:
            print(f"resuming from {config.resume_from}")
    elif config.load_pretrained:
        ckpt = load_pretrained(model, checkpoint_path)
        ckpt_norm_stats = ckpt.get("norm_stats") if isinstance(ckpt, dict) else None
        if rank == 0:
            print(f"loaded pretrained checkpoint {checkpoint_path}")
    elif backbone == "clip":
        if distributed and rank != 0:
            dist.barrier()
        missing, unexpected = load_clip_vision_weights(model.img_backbone, config.clip)
        if distributed and rank == 0:
            dist.barrier()
        if rank == 0:
            print(f"loaded CLIP ViT-B/16 vision weights "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")
    else:
        dinov3_ckpt = cache_root / "dinov3_vitb16_pretrain_lvd1689m.pth"
        if dinov3_ckpt.exists():
            sd = torch.load(dinov3_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            missing, unexpected = model.img_backbone.dinov3_model.load_state_dict(
                sd, strict=False
            )
            if rank == 0:
                print(f"loaded DINOv3 from {dinov3_ckpt} "
                      f"(missing={len(missing)} unexpected={len(unexpected)})")
        elif rank == 0:
            print(f"no {dinov3_ckpt}, using random DINOv3")
    if config.dino_bf16:
        model.img_backbone.set_bfloat16(True)
        if rank == 0:
            print(f"{backbone} vision bf16 autocast enabled")

    norm_stats = _resolve_training_norm_stats(
        config, cache_root, ckpt_norm_stats, rank
    )
    return model, norm_stats


def _build_vla_model(config, cache_root, device, resume_ckpt, rank):
    """Construct VLAPolicy and load its starting weights. Returns (model, norm_stats)."""
    del device
    checkpoint_path = cache_root / config.pretrained_ckpt_name
    starting_ckpt = resume_ckpt
    if starting_ckpt is None and config.load_pretrained:
        starting_ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

    if starting_ckpt is not None:
        has_metadata = validate_vla_checkpoint_config(
            config.vla_model,
            starting_ckpt,
            source=str(config.resume_from or checkpoint_path),
        )
        if not has_metadata and rank == 0:
            print(
                "WARNING: VLA checkpoint has no architecture metadata; "
                "only tensor keys and shapes can be validated"
            )

    model_config = deepcopy(config.vla_model)
    if starting_ckpt is not None:
        model_config.backbone.load_base_checkpoint = False
    model = VLAPolicy(model_config, backbone_autocast=config.vla_bf16_autocast)
    if rank == 0:
        backbone_compute = (
            "BF16 backbone autocast" if config.vla_bf16_autocast
            else "FP32 backbone compute"
        )
        print(f"VLA: FP32 parameters/AdamW state; {backbone_compute}; FP32 pool/head/loss")
    ckpt_norm_stats = None
    if starting_ckpt is not None:
        model.load_state_dict(model_state_dict(starting_ckpt), strict=True)
        ckpt_norm_stats = starting_ckpt.get("norm_stats")
        if rank == 0:
            if resume_ckpt is not None:
                print(f"resuming from {config.resume_from}")
            else:
                print(f"loaded VLA pretrained weights from {checkpoint_path} at step 0")
    norm_stats = _resolve_training_norm_stats(
        config, cache_root, ckpt_norm_stats, rank
    )
    return model, norm_stats


def main(config: TrainConfig):
    is_vla = config.policy == "vla"
    cache_root = Path(config.cache_root).expanduser()
    if config.output_dir:
        output_dir = Path(config.output_dir).expanduser()
    else:
        output_dir = cache_root / ("vla_checkpoints" if is_vla else "finetune_checkpoints")

    components = validate_train_config(
        config, cache_root, cache_root / config.pretrained_ckpt_name
    )

    distributed, rank, world, local_rank, local_world, device = _distributed_context()
    data_scope = data_parallel_scope(cache_root, rank, world, local_rank, local_world)
    if distributed and data_scope.placement == "node_sharded":
        check_shard_consistency(read_shard_marker(cache_root), world)

    resume_ckpt, resume_step = (None, 0)
    if config.resume_from:
        resume_ckpt, resume_step = load_checkpoint(config.resume_from)
        check_topology(resume_ckpt, batch_size=config.batch_size,
                       data_world=data_scope.world, rank=rank)

    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    if is_vla:
        model, norm_stats = _build_vla_model(config, cache_root, device, resume_ckpt, rank)
    else:
        model, norm_stats = _build_dit_model(
            config, cache_root, device, resume_ckpt, rank, distributed
        )
    model = model.to(device)
    if is_vla and config.compile_siglip:
        # Compiled backward runs outside autocast.
        import torch._functorch.config as functorch_config

        functorch_config.backward_pass_autocast = "off"
        model.vla.gemma_model.siglip_vision_model.compile(fullgraph=False)

    fsdp = is_vla and config.fsdp
    if fsdp:
        if not distributed:
            raise ValueError("--fsdp shards across ranks; launch with torchrun --nproc-per-node > 1")
        _shard_vla(model, world, local_world)
    optimizer = _build_optimizer(model, config, is_vla, rank)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / config.optim.lr_warmup_steps, 1.0)
    )
    if resume_ckpt is not None:
        restore_training_state(resume_ckpt, optimizer=optimizer, scheduler=scheduler,
                               resume_step=resume_step, rank=rank,
                               source=config.resume_from,
                               fsdp_model=model if fsdp else None)

    # torch.compile: DiT compiles the whole module; VLA only its SigLIP tower
    # (handled above), so skip the whole-module compile for VLA.
    if config.compile and not is_vla:
        model = torch.compile(model, fullgraph=True)
        if rank == 0:
            print("torch.compile(fullgraph=True) enabled")

    if distributed and not fsdp:
        ddp_kwargs = {
            "device_ids": [device.index],
            "output_device": device.index,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        if not is_vla:
            ddp_kwargs.update(static_graph=True, bucket_cap_mb=256)
        model = DDP(model, **ddp_kwargs)
    module = model.module if isinstance(model, DDP) else model
    if hasattr(module, "_orig_mod"):
        module = module._orig_mod

    # Conditioning: DiT uses a CLIP text embedder; VLA feeds raw prompts to Gemma.
    embedder = None
    if not is_vla:
        if distributed and rank != 0:
            dist.barrier()
        embedder = CLIPTextEmbedder(config.clip, device="cpu")
        if distributed and rank == 0:
            dist.barrier()

    camera_keys = config.vla_model.camera_keys if is_vla else config.model.camera_keys

    # Operator prompting (both policies) needs a label-map manifest built a priori.
    operator_label_maps = {}
    if config.prompt.use_operator_id_as_prompt:
        operator_label_maps = load_operator_label_maps(config.prompt.operator_label_map_path)
        if rank == 0:
            if operator_label_maps:
                print(f"[operator] label maps for {len(operator_label_maps)} tasks "
                      f"({config.prompt.operator_label_map_path})")
            else:
                print("\033[93m[operator] WARNING: operator prompting is enabled but "
                      f"{config.prompt.operator_label_map_path} contains no label maps; "
                      "training continues without operator conditioning\033[0m")
    # VLA lets SigLIP own image normalization (raw [0, 1]); DiT normalizes per backbone.
    norm_preset = None if is_vla else "auto"
    data_model_config = config.vla_model if is_vla else config.model
    train_loader, train_components = build_train_loader(
        config, components, norm_stats, data_scope, resume_step,
        model_config=data_model_config,
        operator_label_maps=operator_label_maps,
        norm_preset=norm_preset,
    )
    val_loaders, val_components = build_val_loaders(
        config, components, norm_stats, data_scope,
        model_config=data_model_config,
        operator_label_maps=operator_label_maps,
        norm_preset=norm_preset,
    )

    wandb = None
    if config.log_wandb and rank == 0:
        try:
            import wandb as _wandb

            wandb = _wandb
            wandb.init(project=config.wandb_project, config=asdict(config))
        except Exception as e:  # noqa: BLE001 - optional logging must not stop training
            print(f"wandb disabled: {e}")

    if data_scope.checkpoint_writer:
        output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for c, ds in zip(components, train_components):
            prompts = sorted(
                {p or task_name_to_prompt(t) for *_, t, p in ds.episodes}
            )
            shown = prompts if len(prompts) <= 5 else [*prompts[:5], f"... {len(prompts) - 5} more"]
            print(f"train[{c.train_dir}] weight={c.weight:.4f}: "
                  f"{len(ds.episodes)} episodes, {len(ds)} usable frames, prompts={shown}")
        for name, ds in val_components:
            print(f"val[{name}]: {len(ds.episodes)} episodes")
        print(
            f"policy={config.policy} world={world} local_world={local_world} "
            f"data_world={data_scope.world} data_placement={data_scope.placement}"
        )

    flow = {
        "max_action_prefix": config.flow.max_action_prefix,
        "prefix_conditioning_prob": config.flow.prefix_conditioning_prob,
        "prefix_noise_scale": config.flow.prefix_noise_scale,
    }
    if is_vla:
        flow["num_diffusion_draws"] = config.flow.num_diffusion_draws
    model.train()
    global_step = resume_step
    t_last = time.monotonic()
    for batch in train_loader:
        if global_step >= config.train_steps:
            break
        batch = batch_to_device(batch, device, embedder, camera_keys)
        loss = model(batch, **flow)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.max_grad_norm))
        optimizer.step()
        # Free the gradients now rather than after the next forward: 13.7 GiB less live during it.
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1

        if global_step % config.log_every == 0:
            loss_d = loss.detach()
            if distributed:
                dist.all_reduce(loss_d, op=dist.ReduceOp.AVG)
            if rank == 0:
                dt = time.monotonic() - t_last
                t_last = time.monotonic()
                sps = config.log_every / dt
                lr = scheduler.get_last_lr()[0]
                peak_gib = torch.cuda.max_memory_allocated() / 2**30
                print(f"step {global_step:6d}  loss {loss_d.item():.4f}  "
                      f"lr {lr:.2e}  gnorm {grad_norm:.3f}  {sps:.2f} it/s  peak {peak_gib:.1f} GiB")
                if wandb:
                    wandb.log({"loss": loss_d.item(), "lr": lr,
                               "grad_norm": grad_norm,
                               "steps_per_s": sps}, step=global_step)

        if global_step % config.val_every == 0:
            _run_validation(
                config, module, val_loaders, device, embedder, camera_keys,
                distributed, rank, wandb, global_step,
            )
            model.train()
            t_last = time.monotonic()

        if global_step % config.ckpt_every == 0 and (fsdp or data_scope.checkpoint_writer):
            # Gathering sharded state is a collective, so every FSDP rank takes part.
            model_state, optimizer_state = fsdp_state_dicts(model, optimizer) if fsdp else (None, None)
            if data_scope.checkpoint_writer:
                path = output_dir / f"{global_step}.pt"
                model_config = asdict(config.vla_model) if is_vla else None
                save_checkpoint(path, module=module, optimizer=optimizer, scheduler=scheduler,
                                global_step=global_step, norm_stats=norm_stats,
                                batch_size=config.batch_size, data_world=data_scope.world,
                                model_config=model_config, train_config=asdict(config),
                                model_state=model_state, optimizer_state=optimizer_state)
                update_last(output_dir, path)
                print(f"[rank {rank}] saved {path}")

    if distributed:
        dist.destroy_process_group()


def _run_validation(config, module, val_loaders, device, embedder, camera_keys,
                    distributed, rank, wandb, global_step):
    """Per-component val_recon_error (generation MSE) and val_loss (diffusion loss)."""
    module.eval()
    sharded = hasattr(module, "unshard")  # FSDP2 root: sample_actions bypasses its forward hook
    if sharded:
        module.unshard()
    per_component_recon = {}
    per_component_loss = {}
    skipped_val = []
    for name, vl in val_loaders.items():
        err_sum, elem_count, loss_sum, batch_count = 0.0, 0, 0.0, 0
        for vb in vl:
            vb = batch_to_device(vb, device, embedder, camera_keys)
            with torch.no_grad():
                pred = module.sample_actions(vb, num_steps=config.flow.num_diffusion_steps)
                err_sum += F.mse_loss(pred, vb["actions"], reduction="sum").item()
                elem_count += vb["actions"].numel()
                loss_sum += module(
                    vb, max_action_prefix=0, prefix_conditioning_prob=0.0
                ).item()
                batch_count += 1
        stats = torch.tensor(
            [err_sum, float(elem_count), loss_sum, float(batch_count)], device=device
        )
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() > 0:
            per_component_recon[name] = (stats[0] / stats[1]).item()
            per_component_loss[name] = (stats[2] / stats[3]).item()
        else:
            skipped_val.append(name)
    if sharded:
        module.reshard()
    if rank == 0:
        if per_component_recon:
            parts = "  ".join(f"{n}={v:.4f}" for n, v in per_component_recon.items())
            avg = sum(per_component_recon.values()) / len(per_component_recon)
            avg_loss = sum(per_component_loss.values()) / len(per_component_loss)
            print(f"step {global_step:6d}  val_recon_error {avg:.4f}  "
                  f"val_loss {avg_loss:.4f}  ({parts})")
            if wandb:
                log = {"val_recon_error": avg, "val_loss": avg_loss}
                log.update({f"val_recon_error/{n}": v for n, v in per_component_recon.items()})
                log.update({f"val_loss/{n}": v for n, v in per_component_loss.items()})
                wandb.log(log, step=global_step)
        else:
            print(f"step {global_step:6d}  val skipped (no full validation batches)")
        if skipped_val:
            print(f"step {global_step:6d}  val skipped components: {', '.join(skipped_val)}")
