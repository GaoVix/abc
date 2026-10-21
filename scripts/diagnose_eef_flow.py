"""Read-only checks of an EEF-flow checkpoint; temporary probes never change its weights.

Run with ``uv run python -m scripts.diagnose_eef_flow --help``.
Probe errors are diagnostic measurements, not proven lower bounds on flow loss.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from abc_minimal.config import ClipConfig, DiTConfig, MixtureComponent, PromptConfig, TrainConfig
from abc_minimal.dataloader import EpisodeDataset, collate
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.eef_adapters import EEF_POSE_INDICES, EEFTrainingAdapter
from abc_minimal.operator import load_operator_label_maps
from abc_minimal.preprocess import parse_norm_stats
from scripts.train_eef_adapters import JOINT_STATE_INDICES


GROUPS = {
    "position": [0, 1, 2, 10, 11, 12],
    "rotation6d": [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18],
    "gripper": [9, 19],
}


def mse_groups(prediction, target):
    error = (prediction.float() - target.float()).square()
    return {"all": error.mean().item(), **{
        name: error[..., indices].mean().item() for name, indices in GROUPS.items()
    }}


def fit_decoder_latents(adapter, target, raw_prediction, steps, restarts=2):
    """Best error found by optimizing each token's decoder input; not a lower bound.

    The output MLP stays fixed. Gripper predictions are granted their exact target.
    Start from DiT's actual output and random inputs to reduce local-minimum effects.
    """
    physical = raw_prediction.detach() * adapter.actions_scale + adapter.actions_mean
    start = ((physical[..., JOINT_STATE_INDICES] - adapter.mlps.state_mean)
             / adapter.mlps.state_scale).reshape(-1, 12)
    pose_target = target[..., EEF_POSE_INDICES].detach().reshape(-1, 18)
    with torch.no_grad():
        best = (adapter.mlps.mlp_out(start) - pose_target).square().mean(-1)
        initial = best.mean().item()
    for restart in range(restarts):
        latent = (start.clone() if restart == 0 else torch.randn_like(start)).requires_grad_()
        optimizer = torch.optim.Adam([latent], lr=0.03)
        for _ in range(steps):
            errors = (adapter.mlps.mlp_out(latent) - pose_target).square().mean(-1)
            best = torch.minimum(best, errors.detach())
            optimizer.zero_grad(set_to_none=True)
            latent.grad, = torch.autograd.grad(errors.mean(), latent)
            optimizer.step()
        with torch.no_grad():
            best = torch.minimum(best, (adapter.mlps.mlp_out(latent) - pose_target).square().mean(-1))
    return {"initial_pose_mse": initial, "best_found_pose_mse": best.mean().item(),
            "best_found_all_mse_with_perfect_grippers": best.mean().item() * 18 / 20,
            "steps_per_restart": steps, "restarts": restarts}


def encoder_probe(adapter, steps, seed):
    """Fixed encoder, fresh regressors, held-out Gaussian inputs at t=1.

    With known constant clean action, its velocity differs from noise by a constant.
    Predict noise here to test retained noise information without observation ambiguity.
    Failure of a finite probe alone does not prove an information-theoretic bound.
    """
    torch.manual_seed(seed)
    device = adapter.actions_mean.device
    with torch.no_grad():
        calibration = adapter.eef_to_joint(torch.randn(4096, 20, device=device))
        mean, scale = calibration.mean(0), calibration.std(0).clamp_min(1e-6)
        heldout = torch.randn(4096, 20, device=device)
        heldout_encoded = (adapter.eef_to_joint(heldout) - mean) / scale
    probes = [nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 256),
                            nn.GELU(), nn.Linear(256, 20)).to(device) for dim in (14, 20)]
    optimizer = torch.optim.Adam([p for probe in probes for p in probe.parameters()], lr=1e-3)
    history = []
    for step in range(steps + 1):
        if step in {0, steps // 2, steps}:
            with torch.no_grad():
                row = {"step": step, "encoded_input": mse_groups(probes[0](heldout_encoded), heldout),
                       "full_eef_input": mse_groups(probes[1](heldout), heldout)}
            history.append(row)
            print(f"encoder_probe step={step}: encoded={row['encoded_input']['all']:.5f} "
                  f"full_eef={row['full_eef_input']['all']:.5f}", flush=True)
        if step == steps:
            break
        noise = torch.randn(256, 20, device=device)
        with torch.no_grad():
            encoded = (adapter.eef_to_joint(noise) - mean) / scale
        loss = ((probes[0](encoded) - noise).square().mean()
                + (probes[1](noise) - noise).square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


def optimizer_audit(checkpoint, model):
    """Read saved Adam state for both MLPs using the training optimizer's ordering."""
    saved = checkpoint.get("optimizer", {})
    group = next((g for g in saved.get("param_groups", []) if g.get("name") == "head"), None)
    backbone_ids = {id(p) for p in model.img_backbone.parameters()}
    names = [name for name, p in model.named_parameters() if p.requires_grad and id(p) not in backbone_ids]
    if group is None or len(group["params"]) != len(names):
        return {"status": "unverified: saved optimizer layout does not match"}
    mapping = dict(zip(names, group["params"]))
    result = {}
    for prefix in ("mlp_in", "mlp_out"):
        states = [saved.get("state", {}).get(identifier, {}) for name, identifier in mapping.items()
                  if name.startswith(f"eef_adapters.mlps.{prefix}.")]
        steps = [float(s["step"]) for s in states if "step" in s]
        result[prefix] = {"parameters": len(states), "parameters_with_adam_state": len(steps),
                          "min_step": min(steps) if steps else None,
                          "max_step": max(steps) if steps else None}
    return result


@torch.no_grad()
def fk_reference_checks(adapter, raw, prepared, clean):
    """Separate camera-matmul precision from CPU/Warp FK and normalization errors.

    Preserve the training-precision targets. Only the reference FK call uses
    highest FP32 precision; restore the setting before evaluating the model.
    """
    physical_state = raw["state"] * adapter.state_scale + adapter.state_mean
    physical_actions = raw["actions"] * adapter.actions_scale + adapter.actions_mean
    joints = torch.cat((physical_state[:, None], physical_actions), dim=1)[..., JOINT_STATE_INDICES]
    kinematics = adapter.kinematics
    cpu_pose = kinematics.fk(joints.flatten(0, 1).cpu().numpy())
    cpu_pose = torch.from_numpy(cpu_pose).to(clean.device).reshape(*joints.shape[:2], 18)
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        highest_pose = kinematics.fk_batch(joints.flatten(0, 1)).reshape_as(cpu_pose)
    finally:
        torch.set_float32_matmul_precision(precision)
    original_pose = torch.cat((prepared["eef_state"][:, None], prepared["eef_actions"]), dim=1)
    if not all(torch.isfinite(pose).all().item() for pose in (cpu_pose, highest_pose, original_pose)):
        raise RuntimeError("Nonfinite FK reference values")

    def normalized_actions(poses):
        values = clean.clone()
        values[..., EEF_POSE_INDICES] = (poses[:, 1:] - adapter.mlps.pose_mean) / (adapter.mlps.pose_std + 1e-6)
        values[..., [9, 19]] = raw["actions"][..., [6, 13]]
        return values

    cpu_clean = normalized_actions(cpu_pose)
    highest_clean = normalized_actions(highest_pose)
    checks = {
        "fk_backend": "GPU Warp versus CPU MuJoCo" if clean.is_cuda else "CPU reference only",
        "matmul_precision": precision,
        "fk_physical_max_abs_error": (cpu_pose - original_pose).abs().max().item(),
        "fk_position_max_abs_error": (cpu_pose - original_pose)[..., [0, 1, 2, 9, 10, 11]].abs().max().item(),
        "normalized_target_max_abs_error": (cpu_clean - clean).abs().max().item(),
        "normalized_target_mse": mse_groups(cpu_clean, clean),
        "fk_highest_precision_max_abs_error": (cpu_pose - highest_pose).abs().max().item(),
        "normalized_target_highest_precision_max_abs_error": (cpu_clean - highest_clean).abs().max().item(),
        "normalization_formula_max_abs_error": (normalized_actions(original_pose) - clean).abs().max().item(),
    }
    original_matches = checks["fk_physical_max_abs_error"] <= 5e-5 and checks["normalized_target_max_abs_error"] <= 1e-3
    highest_matches = checks["fk_highest_precision_max_abs_error"] <= 5e-5 and checks["normalized_target_highest_precision_max_abs_error"] <= 1e-3
    checks["fk_reference_status"] = ("consistent" if original_matches else
                                     "camera_matmul_precision_difference" if highest_matches else
                                     "mismatch_after_highest_precision")
    reference_prepared = {**prepared, "eef_state": cpu_pose[:, 0], "eef_actions": cpu_pose[:, 1:]}
    return checks, reference_prepared


@torch.no_grad()
def evaluate_batch(model, raw, times, draws, generator, sample_steps, latent_steps):
    adapter = model.eef_adapters
    prepared = adapter.prepare(raw)
    normalized = adapter(prepared)
    clean, state = normalized["actions"], normalized["state"]
    vision = model.build_vision_tokens(raw["images"])
    result = {"clean_zero_baseline": mse_groups(torch.zeros_like(clean), clean),
              "clean_max_abs": clean.abs().max().item(), "velocity": [], "sampling": []}

    checks, reference_prepared = fk_reference_checks(adapter, raw, prepared, clean)
    result["pipeline_checks"] = checks
    reference_batch = adapter(reference_prepared)
    for draw in range(draws):
        noise = torch.randn(clean.shape, generator=generator, device=clean.device)
        uniform_t = torch.rand(len(clean), 1, 1, generator=generator, device=clean.device)
        uniform_loss = model(prepared, noise=noise, t=uniform_t).item()
        result.setdefault("uniform_t_loss", []).append(uniform_loss)
        for time in times:
            x = (1 - time) * clean + time * noise
            cond = model.compute_cond(state, raw["task_vec_clip"], torch.full((len(clean),), time, device=clean.device))
            captured = []
            handle = model.final_layer.register_forward_hook(lambda module, args, out: captured.append(out.detach()))
            try:
                pred = model.predict_velocity(x, cond, vision)
            finally:
                handle.remove()
            target = noise - clean
            row = {"t": time, "draw": draw, "mse": mse_groups(pred, target),
                   "zero_velocity_baseline": mse_groups(torch.zeros_like(target), target)}
            result["velocity"].append(row)
            if draw == 0 and time == 0.5:
                actual = model(prepared, noise=noise, t=torch.full((len(clean), 1, 1), time, device=clean.device))
                result["pipeline_checks"]["forward_loss_absolute_difference"] = abs(actual.item() - row["mse"]["all"])
                reference_x = (1 - time) * reference_batch["actions"] + time * noise
                reference_cond = model.compute_cond(reference_batch["state"], raw["task_vec_clip"],
                                                    torch.full((len(clean),), time, device=clean.device))
                reference_pred = model.predict_velocity(reference_x, reference_cond, vision)
                reference_target = noise - reference_batch["actions"]
                result["cpu_fk_reference_comparison"] = {
                    "t": time, "original_mse": row["mse"],
                    "cpu_reference_mse": mse_groups(reference_pred, reference_target),
                    "prediction_change_mse": mse_groups(reference_pred, pred),
                }
            if draw == 0 and time in (0.5, 1.0) and latent_steps:
                with torch.enable_grad():
                    row["decoder_oracle"] = fit_decoder_latents(adapter, target, captured[0], latent_steps)
        if draw == 0:
            reference = None
            for steps in sample_steps:
                sample = model.sample_actions(normalized, num_steps=steps, noise=noise)
                entry = {"steps": steps, "mse": mse_groups(sample, clean)}
                if reference is not None:
                    entry["difference_from_first_step_count"] = mse_groups(sample, reference)
                reference = sample if reference is None else reference
                result["sampling"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--latent-steps", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.draws) <= 0 or min(args.latent_steps, args.probe_steps) < 0:
        parser.error("batch size, batches, draws must be positive; probe steps must be non-negative")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = args.output or args.checkpoint.parent / "diagnostics.json"
    if output.resolve() == args.checkpoint.resolve() or (output.exists() and output.samefile(args.checkpoint)):
        parser.error("output must not overwrite or alias the checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if "eef_adapters" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Need an EEF adapter training checkpoint containing eef_adapters and model_config")
    saved = checkpoint.get("train_config", {})
    config = DiTConfig(**checkpoint["model_config"])
    if config.action_dim != 14:
        raise ValueError("This diagnostic expects the unchanged 14D DiT wrapped for EEF flow")
    stats = parse_norm_stats(checkpoint["norm_stats"])
    model = DiTPolicy(config)
    load_pretrained(model, checkpoint)
    model.eef_adapters = EEFTrainingAdapter(checkpoint["eef_adapters"], stats)
    root = Path(__file__).resolve().parents[1]
    report = {"checkpoint": str(args.checkpoint), "global_step": checkpoint.get("global_step"),
              "seed": args.seed, "device": str(device),
              "saved_training_setup": {key: saved.get(key) for key in
                                       ("load_pretrained", "pretrained_ckpt_name", "resume_from", "eef_adapters", "flow")},
              "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                              ("abc_minimal/dit.py", "abc_minimal/eef_adapters.py", "abc_minimal/train_loop.py")},
              "optimizer_adapters": optimizer_audit(checkpoint, model), "splits": {},
              "notes": ["All datasets use eval transforms, no state masking, and no action prefix.",
                        "Per-t losses and small fixed sample budgets do not reproduce the logged aggregate val_loss.",
                        "Original targets and model evaluations retain training matmul precision; highest precision is used only for reference FK.",
                        "CPU/Warp FK discrepancies are reported for investigation and do not suppress the remaining probes.",
                        "Decoder-oracle and finite encoder-probe errors are not theoretical loss lower bounds.",
                        "Checkpoint metadata does not certify the training objective; use this run's EEF-flow checkpoint."]}
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)
    model.img_backbone.set_bfloat16(saved.get("dino_bf16", True) and device.type == "cuda")
    if device.type == "cpu":
        kinematics = model.eef_adapters.kinematics
        kinematics.fk_batch = lambda joints: torch.from_numpy(kinematics.fk(joints.numpy()))
    prompt_config = PromptConfig(**saved.get("prompt", {}))
    labels = load_operator_label_maps(prompt_config.operator_label_map_path) if prompt_config.use_operator_id_as_prompt else {}
    embedder = CLIPTextEmbedder(ClipConfig(**saved.get("clip", {})), device="cpu")
    components = ([MixtureComponent(**c) for c in saved["mixture"]] if saved.get("mixture") else
                  TrainConfig(mixture_preset=saved.get("mixture_preset", "load_the_plates_into_the_dish_rack")).resolve_mixture())
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    print(f"Checkpoint step={report['global_step']}; optimizer={report['optimizer_adapters']}", flush=True)
    for component in components:
        for split in (component.train_dir, component.val_dir):
            if split in report["splits"]:
                continue
            dataset = EpisodeDataset(args.cache_root / split, stats, train=False,
                                     default_task_name=component.task_name, mask_state_ratio=0,
                                     model_config=config, prompt_config=prompt_config, operator_label_maps=labels)
            indices = np.linspace(0, len(dataset) - 1, min(len(dataset), args.batch_size * args.batches), dtype=int)
            batches = []
            report["splits"][split] = batches
            for offset in range(0, len(indices), args.batch_size):
                batch = collate([dataset[int(i)] for i in indices[offset:offset + args.batch_size]], config.camera_keys)
                raw = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
                raw["images"] = {name: value.to(device) for name, value in batch["images"].items()}
                raw["task_vec_clip"] = embedder.encode(batch["prompt"]).to(device)
                first_validation = offset == 0 and split == component.val_dir
                result = evaluate_batch(model, raw, (0.05, 0.25, 0.5, 0.75, 0.95, 1.0), args.draws,
                                        generator, (10, 30) if first_validation else (),
                                        args.latent_steps if first_validation else 0)
                result["dataset_indices"] = indices[offset:offset + args.batch_size].tolist()
                batches.append(result)
                checks = result["pipeline_checks"]
                print(f"{split} batch={len(batches)} uniform_loss={np.mean(result['uniform_t_loss']):.5f} "
                      f"checks={checks}", flush=True)
                if checks["fk_reference_status"] != "consistent":
                    print(f"  FK reference warning: {checks['fk_reference_status']}; continuing with original training targets", flush=True)
                if offset == 0:
                    print(f"  cpu_fk_reference_comparison={result['cpu_fk_reference_comparison']}", flush=True)
                for time in (0.05, 0.5, 1.0):
                    rows = [r["mse"] for r in result["velocity"] if r["t"] == time]
                    print(f"  t={time:.2f}: " + " ".join(f"{g}={np.mean([r[g] for r in rows]):.5f}"
                                                        for g in ("all", *GROUPS)), flush=True)
                if first_validation:
                    for row in result["velocity"]:
                        if "decoder_oracle" in row:
                            print(f"  decoder_oracle t={row['t']}: {row['decoder_oracle']}", flush=True)
                    print(f"  sampling={result['sampling']}", flush=True)
                save()
                if (checks["normalization_formula_max_abs_error"] > 1e-6
                        or checks.get("forward_loss_absolute_difference", 0) > 1e-4 * (1 + np.mean(result["uniform_t_loss"]))):
                    raise RuntimeError(f"Normalization or loss formula mismatch; inspect {output}")
    if args.probe_steps:
        report["encoder_probe"] = encoder_probe(model.eef_adapters, args.probe_steps, args.seed + 1)
    report["summary"] = {}
    for split, batches in report["splits"].items():
        count = sum(len(batch["dataset_indices"]) for batch in batches)
        summary = {"samples": count, "uniform_t_loss": sum(
            len(batch["dataset_indices"]) * float(np.mean(batch["uniform_t_loss"])) for batch in batches
        ) / count, "per_t": {}}
        for time in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0):
            rows = [(len(batch["dataset_indices"]), row["mse"]) for batch in batches
                    for row in batch["velocity"] if row["t"] == time]
            summary["per_t"][str(time)] = {
                group: sum(weight * mse[group] for weight, mse in rows) / sum(weight for weight, _ in rows)
                for group in ("all", *GROUPS)
            }
        report["summary"][split] = summary
        print(f"SUMMARY {split}: {json.dumps(summary)}", flush=True)
    save()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
