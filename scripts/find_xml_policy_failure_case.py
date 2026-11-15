"""Find old-XML successes paired with new-XML failures in saved evaluations.

Standard library only: reads summary.json files, without loading models or GPUs.
"""
import argparse
import json
from pathlib import Path


MODELS = {
    "old": ("dit_joint_retargeted_official_timing_20261022_125331",
            "eval_old_xml_official_timing_direct_strict_100k_highres_pseed"),
    "new": ("dit_xml_default_real_ik_raw_sim_20261103_112255",
            "eval_xml_default_real_ik_raw_sim_direct_strict_100k_highres_pseed"),
}
PROTOCOL_KEYS = (
    "policy", "task", "seed", "policy_seed", "num_worlds", "num_chunks",
    "diffusion_steps", "execute_chunk_dim", "prefix_length", "rtc",
    "rtc_prefix_length", "rtc_inference_lead_steps", "camera_backend",
    "camera_height", "camera_width", "parallel_worlds", "fast_inference",
    "deterministic_inference", "vanilla_physics", "norm_stats_path",
    "disable_arm_control_limits",
)


def read_runs(root, side, step, policy_seed):
    model, prefix = MODELS[side]
    runs = []
    for path in sorted(root.glob(f"{prefix}*/summary.json")):
        data = json.loads(path.read_text())
        config = data["config"]
        checkpoint = Path(data["checkpoint"])
        request = config.get("randomization") or {}
        if isinstance(request, str):
            request = json.loads(request)
        if (checkpoint.parent.name != model or checkpoint.name != f"{step}.pt"
                or config.get("task") != "dishrack"
                or config.get("deterministic_inference") is not True
                or config.get("fast_inference") is not False
                or config.get("parallel_worlds") != 0
                or config.get("compensate_arm_bases") is not False
                or config.get("disable_arm_control_limits", False)
                or request.get("randomize_arm_bases") is not False
                or request.get("arm_base_source") != "xml"
                or request.get("arm_collision_mode") != "eef_only"
                or not data.get("resolved_physics")):
            continue
        if policy_seed is not None and config["policy_seed"] != policy_seed:
            continue
        worlds = data.get("worlds", [])
        if len(worlds) != config["num_worlds"] or not worlds:
            continue
        keyed = {(w["world_index"], w["world_seed"]): w for w in worlds}
        if len(keyed) != len(worlds):
            raise ValueError(f"Duplicate world records: {path}")
        protocol = {key: config.get(key) for key in PROTOCOL_KEYS}
        protocol["disable_arm_control_limits"] = False
        protocol.update(randomization=request, prompt=data.get("prompt"),
                        physics=data["resolved_physics"])
        runs.append(dict(summary=str(path), checkpoint=str(checkpoint),
                         protocol=protocol, worlds={
                             key: {k: v for k, v in world.items() if k != "chunk_metrics"}
                             for key, world in keyed.items()}))
    return runs


def world_result(run, world):
    video = world.get("video_path")
    return dict(summary=run["summary"], checkpoint=run["checkpoint"],
                success=world["success"], max_reward=world.get("max_reward"),
                final_reward=world.get("reward"), steps=world.get("steps"),
                video=video)


def find_cases(old_runs, new_runs):
    cases = []
    rejected_scenes = 0
    for old in old_runs:
        for new in new_runs:
            if old["protocol"] != new["protocol"]:
                continue
            for key in sorted(old["worlds"].keys() & new["worlds"].keys()):
                a, b = old["worlds"][key], new["worlds"][key]
                if a.get("success") is not True or b.get("success") is not False:
                    continue
                # Compare exact saved poses, variants, scales and base metadata.
                state = a.get("randomization")
                if not state or not state.get("object_states") or state != b.get("randomization"):
                    rejected_scenes += 1
                    continue
                cases.append(dict(policy_seed=old["protocol"]["policy_seed"],
                                  world_index=key[0], world_seed=key[1],
                                  initial_randomization_verified_equal=True,
                                  old=world_result(old, a), new=world_result(new, b)))
    cases.sort(key=lambda c: (c["new"]["max_reward"] if c["new"]["max_reward"] is not None else 1,
                             c["old"]["steps"] if c["old"]["steps"] is not None else float("inf"),
                             c["policy_seed"], c["world_index"], c["old"]["summary"],
                             c["new"]["summary"]))
    return cases, rejected_scenes


def build_report(root, step=100000, policy_seed=None, world_index=None):
    old = read_runs(root, "old", step, policy_seed)
    new = read_runs(root, "new", step, policy_seed)
    cases, rejected = find_cases(old, new)
    if world_index is not None:
        cases = [case for case in cases if case["world_index"] == world_index]
    report = dict(status="found" if cases else "no_matching_pair",
                  eligible_runs=dict(old=len(old), new=len(new)),
                  matching_pairs=len(cases), rejected_scene_mismatches=rejected,
                  criteria="Both XML fixed bases, eef_only, no conversion, deterministic; "
                           "same checkpoint step, policy seed, world index, actual environment seed, "
                           "saved initial randomization and evaluation protocol.",
                  selected=cases[0] if cases else None, cases=cases)
    if cases:
        report["selected_configs"] = {}
        report["selected_world_records"] = {}
        report["selected_physics"] = {}
        for side in ("old", "new"):
            data = json.loads(Path(cases[0][side]["summary"]).read_text())
            report["selected_configs"][side] = data["config"]
            report["selected_physics"][side] = data["resolved_physics"]
            report["selected_world_records"][side] = next(
                w for w in data["worlds"] if w["world_index"] == cases[0]["world_index"]
                and w["world_seed"] == cases[0]["world_seed"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/mnt/afs/L202500703/abc_storage/output"))
    parser.add_argument("--step", type=int, default=100000)
    parser.add_argument("--policy-seed", type=int)
    parser.add_argument("--world-index", type=int, help="Restrict selection to this original world index")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Results directory not found: {args.root}")
    report = build_report(args.root, args.step, args.policy_seed, args.world_index)
    output = args.output or args.root / "xml_old_success_new_failure" / "selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("cases", "selected_configs", "selected_world_records")},
                     ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
