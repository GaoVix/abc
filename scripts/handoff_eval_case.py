"""Start the median policy from an old XML-policy video frame.

Only the initial arm joints are retargeted. Preserve object integration state,
gripper joints/controls, and the fixed camera; reset policy noise/RTC as a new
episode. Subsequent inference is direct, without base compensation.
"""
import os

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import copy
import json
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares

import abc_sim
from abc_minimal.config import ClipConfig, DiTConfig, SimEvalConfig
from abc_minimal.determinism import configure_determinism
from abc_minimal.eval_policy import _make_env, build_summary, resolved_physics, resolve_prefix_length, rollout_worlds
from abc_minimal.policy import DiTInferencePolicy
from abc_sim.randomization.arm_bases import set_median_arm_bases
from abc_sim.randomization.core import RandomizationState
from scripts.replay_eval_failure_case import WorldNoise, render_trajectory
from scripts.scipy_ik import pose_error, pose_jacobian
from scripts.search_eval_failure_case import TraceScene, array_hashes, write_json


STATE = mujoco.mjtState.mjSTATE_INTEGRATION
MODEL_FIELDS = ("body_pos", "body_quat", "cam_pos", "cam_quat", "geom_contype", "geom_conaffinity")


def load_json(path):
    return json.loads(path.read_text())


def transform(entity):
    pose = np.eye(4)
    pose[:3, :3] = entity.xmat.reshape(3, 3)
    pose[:3, 3] = entity.xpos
    return pose


def camera_pose(data):
    pose = transform(data.camera("top"))
    pose[:3, :3] @= np.diag([1., -1., -1.])
    return pose


def select_frame(case, trace, step, video_time, stride):
    if step is not None:
        if not 0 <= step < len(trace["qpos"]):
            raise ValueError("physics-step is outside the old trajectory")
        return step, dict(physics_step=step, source_simulation_time_s=float(trace["time"][step]))
    video = case / "comparison_top.mp4"
    if not video.exists():
        video = case / "old/rollout.mp4"
    with imageio.get_reader(str(video)) as reader:
        fps = float(reader.get_meta_data()["fps"])
    frames = list(range(0, len(trace["qpos"]), stride))
    if frames[-1] != len(trace["qpos"]) - 1:
        frames.append(len(trace["qpos"]) - 1)
    frame = int(np.floor(video_time * fps))
    if not 0 <= frame < len(frames):
        raise ValueError("Selected time is beyond the active old video; its final frame is held there")
    step = frames[frame]
    return step, dict(video=str(video), requested_video_time_s=video_time, video_fps=fps,
                      video_frame_zero_based=frame, physics_step=step,
                      source_simulation_time_s=float(trace["time"][step]))


def bounded_ik(model, data, side, target, seed, attempts):
    joints = np.array([model.joint(f"{side}_joint{i}").id for i in range(1, 7)])
    site = model.site(f"{side}_tcp_site").id
    bounds = model.jnt_range[joints].copy()
    limited = model.jnt_limited[joints].astype(bool)
    bounds[~limited] = [-np.inf, np.inf]
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL:
        for index, joint in enumerate(joints):
            actuator = model.actuator(f"{side}_joint{index + 1}").id
            if model.actuator_ctrllimited[actuator]:
                control = model.actuator_ctrlrange[actuator]
                bounds[index] = [max(bounds[index, 0], control[0]), min(bounds[index, 1], control[1])]
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Handoff IK expects finite, nonempty XML position/control bounds")
    scale = np.array([1 / .0001] * 3 + [1 / np.deg2rad(.1)] * 3)
    arguments = (model, data, joints, site, target)
    random = np.random.default_rng(0)
    best = None
    for attempt in range(attempts):
        initial = seed if attempt == 0 else random.uniform(bounds[:, 0], bounds[:, 1])
        initial = np.clip(initial, bounds[:, 0], bounds[:, 1])
        result = least_squares(lambda q: pose_error(q, *arguments) * scale, initial,
                               jac=lambda q: pose_jacobian(q, *arguments) * scale[:, None],
                               bounds=(bounds[:, 0], bounds[:, 1]), max_nfev=500,
                               ftol=1e-12, xtol=1e-12, gtol=1e-10)
        error = pose_error(result.x, *arguments)
        position, rotation = np.linalg.norm(error[:3]) * 1000, np.degrees(np.linalg.norm(error[3:]))
        record = dict(position_mm=float(position), rotation_deg=float(rotation),
                      attempt=attempt + 1, joint_bounds_rad=bounds.tolist())
        if best is None or np.linalg.norm(error * scale) < best[0]:
            best = (np.linalg.norm(error * scale), record)
        if position <= .1 and rotation <= .1:
            return result.x.copy(), record
    raise ValueError(f"{side}: no accepted initial IK after {attempts} attempts; best={best[1]}")


def prepare(args):
    case = args.case_dir.resolve()
    scene = load_json(case / "scene.json")
    old_summary = load_json(case / "old/summary.json")
    new_summary = load_json(case / "new/summary.json")
    with np.load(case / "old/trajectory.npz") as archive:
        trace = {key: archive[key] for key in archive.files}
    with np.load(case / "old/initial.npz") as archive:
        initial = {key: archive[key] for key in archive.files}
    manifest_path = case.parent / "manifest.json"
    stride = load_json(manifest_path)["config"]["render_stride"] if manifest_path.exists() else 2
    step, selection = select_frame(case, trace, args.physics_step, args.video_time, stride)
    env = abc_sim.make_env(task="dishrack", render_cameras=False)
    try:
        state = RandomizationState.from_dict(copy.deepcopy(scene["randomization"]))
        env._task_randomizer.apply(env.model, env.data, state)
        source_model, source = env.model, env.data
        for name in MODEL_FIELDS:
            if getattr(source_model, name).shape != initial[name].shape:
                raise ValueError(f"Saved model shape differs: {name}")
            getattr(source_model, name)[:] = initial[name]
        if trace["integration"].shape[1] != mujoco.mj_stateSize(source_model, STATE):
            raise ValueError("Saved integration state uses a different MuJoCo layout")
        mujoco.mj_setState(source_model, source, trace["integration"][step], STATE)
        mujoco.mj_forward(source_model, source)
        world_from_camera = camera_pose(source)
        targets_camera = {arm: np.linalg.inv(world_from_camera) @ transform(source.site(f"{arm}_tcp_site"))
                          for arm in ("left", "right")}
        model = copy.deepcopy(source_model)
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, trace["integration"][step], STATE)
        mujoco.mj_forward(model, data)
        bases = set_median_arm_bases(model, data)
        np.testing.assert_allclose(camera_pose(data), world_from_camera, atol=1e-12, rtol=0)
        with np.load(case / "new/initial.npz") as median_initial:
            for name in MODEL_FIELDS:
                np.testing.assert_allclose(getattr(model, name), median_initial[name], atol=1e-12, rtol=0,
                                           err_msg=f"Current median model differs from the evaluated model: {name}")
        report = dict(selection=selection, arm_velocity=args.arm_velocity,
                      policy_history="New episode: reset noise stream and RTC; no old action prefix.",
                      ik="TRF with analytic Jacobian; intersection of XML joint and actuator control bounds",
                      camera_fixed=True, bases=bases, arms={}, physics_stepped=False)
        arm_qpos, arm_dofs = [], []
        for arm in ("left", "right"):
            joints = np.array([model.joint(f"{arm}_joint{i}").id for i in range(1, 7)])
            qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
            target = world_from_camera @ targets_camera[arm]
            joint_solution, error = bounded_ik(model, data, arm, target, source.qpos[qpos], args.ik_attempts)
            data.qpos[qpos] = joint_solution
            mujoco.mj_forward(model, data)
            if args.arm_velocity == "zero":
                data.qvel[dofs] = 0
                velocity_report = dict(mode="zero")
            else:
                jacobians = []
                for m, d in ((source_model, source), (model, data)):
                    linear, angular = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                    mujoco.mj_jacSite(m, d, linear, angular, m.site(f"{arm}_tcp_site").id)
                    jacobians.append(np.concatenate((linear[:, dofs], angular[:, dofs])))
                twist = jacobians[0] @ source.qvel[dofs]
                joint_velocity, _, rank, singular_values = np.linalg.lstsq(jacobians[1], twist, rcond=None)
                data.qvel[dofs] = joint_velocity
                actual_twist = jacobians[1] @ joint_velocity
                velocity_error = float(np.linalg.norm(actual_twist - twist))
                if not np.isfinite(joint_velocity).all():
                    raise ValueError(f"{arm}: nonfinite mapped joint velocity")
                if velocity_error > 1e-7:
                    raise ValueError(f"{arm}: cannot match the source EEF twist, residual={velocity_error}")
                rotation = world_from_camera[:3, :3].T
                velocity_report = dict(
                    mode="match-twist", jacobian_rank=int(rank),
                    jacobian_singular_values=singular_values.tolist(),
                    source_linear_camera_m_s=(rotation @ twist[:3]).tolist(),
                    source_angular_camera_rad_s=(rotation @ twist[3:]).tolist(),
                    target_linear_camera_m_s=(rotation @ actual_twist[:3]).tolist(),
                    target_angular_camera_rad_s=(rotation @ actual_twist[3:]).tolist(),
                    linear_residual_mm_s=float(np.linalg.norm(actual_twist[:3] - twist[:3]) * 1000),
                    angular_residual_deg_s=float(np.degrees(np.linalg.norm(actual_twist[3:] - twist[3:]))))
            actuators = [model.actuator(f"{arm}_joint{i}").id for i in range(1, 7)]
            data.ctrl[actuators] = joint_solution
            arm_qpos.extend(qpos)
            arm_dofs.extend(dofs)
            report["arms"][arm] = dict(**error, source_joint_rad=source.qpos[qpos].tolist(),
                                        target_joint_rad=joint_solution.tolist(),
                                        source_velocity_rad_s=source.qvel[dofs].tolist(),
                                        target_velocity_rad_s=data.qvel[dofs].tolist(),
                                        velocity=velocity_report,
                                        target_camera_tcp=targets_camera[arm].tolist())
        qpos_mask, velocity_mask = np.ones(model.nq, bool), np.ones(model.nv, bool)
        qpos_mask[arm_qpos], velocity_mask[arm_dofs] = False, False
        np.testing.assert_array_equal(data.qpos[qpos_mask], source.qpos[qpos_mask])
        np.testing.assert_array_equal(data.qvel[velocity_mask], source.qvel[velocity_mask])
        data.qacc_warmstart[arm_dofs] = 0
        data.time = 0
        mujoco.mj_forward(model, data)
        for arm in ("left", "right"):
            pose = np.linalg.inv(camera_pose(data)) @ transform(data.site(f"{arm}_tcp_site"))
            error = np.linalg.inv(targets_camera[arm]) @ pose
            position_mm = float(np.linalg.norm(error[:3, 3]) * 1000)
            rotation_deg = float(np.degrees(np.arccos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1))))
            if position_mm > .1 or rotation_deg > .1:
                raise ValueError(f"Final paired IK check failed: {arm}, {position_mm}mm, {rotation_deg}deg")
        evaluator = env._task_evaluator
        evaluator.configure_from_randomization(source_model, state)
        report["source_task_eval"] = evaluator.evaluate_model_data(source_model, source).to_info(squeeze=True)
        report["objects_and_gripper_qpos_preserved"] = True
        report["objects_and_gripper_qvel_preserved"] = True
        integration = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, integration, STATE)
        values = dict(integration=integration, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                      **{name: np.array(getattr(model, name)) for name in MODEL_FIELDS})
        return values, report, scene, old_summary, new_summary
    finally:
        env.close()


class HandoffScene(TraceScene):
    def __init__(self, env, scene, initial, collision_mode="eef_only"):
        super().__init__(env, scene, "median", collision_mode)
        self.handoff = initial

    def reset(self, seed, options=None):
        super().reset(seed, options)
        native = self.env.env
        for name in MODEL_FIELDS:
            np.testing.assert_allclose(getattr(native.model, name), self.handoff[name], atol=1e-12, rtol=0)
        mujoco.mj_setState(native.model, native.data, self.handoff["integration"], STATE)
        mujoco.mj_forward(native.model, native.data)
        native.cur_step = 0
        state = self.env.randomization
        for name, pose in state.object_states.items():
            joint = mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                index = native.model.jnt_qposadr[joint]
                pose["pos"] = native.data.qpos[index:index + 3].tolist()
                pose["quat"] = native.data.qpos[index + 3:index + 7].tolist()
        native._task_evaluator.reset(nworld=1)
        native._reset_camera_backend()
        self.qpos, self.qvel, self.times, self.actions, self.integration = [], [], [], [], []
        self.initial_arrays = {name: np.array(getattr(native.model, name)) for name in MODEL_FIELDS}
        self.initial_arrays.update(qpos=native.data.qpos.copy(), qvel=native.data.qvel.copy(),
                                   ctrl=native.data.ctrl.copy(),
                                   top_camera_world=np.r_[native.data.camera("top").xpos,
                                                          native.data.camera("top").xmat])
        self.snapshot()
        return self.env.obs()


def run(args, initial, report, scene, old_summary, new_summary):
    values = new_summary["config"].copy()
    values["model"] = DiTConfig(**values["model"])
    values["clip"] = ClipConfig(**values["clip"])
    allowed = {field.name for field in fields(SimEvalConfig)}
    config = SimEvalConfig(**{key: value for key, value in values.items() if key in allowed})
    config.gpu_id, config.num_worlds, config.parallel_worlds = args.gpu_id, 1, 0
    config.fast_inference, config.deterministic_inference, config.compensate_arm_bases = False, True, False
    config.output_dir, config.log_every_chunk = str(args.output), True
    world_index = old_summary["worlds"][0]["world_index"]
    configure_determinism(config.policy_seed)
    policy = DiTInferencePolicy(Path(config.checkpoint), config, f"cuda:{args.gpu_id}")
    policy = WorldNoise(policy, config.policy_seed, world_index)
    prefix = resolve_prefix_length(config, policy.trained_max_prefix, config.model)
    config.prefix_length = prefix
    collision_mode = json.loads(config.randomization or "{}").get("arm_collision_mode", "full")
    env = HandoffScene(_make_env(config, tuple(config.model.camera_keys)), scene, initial, collision_mode)
    try:
        worlds = rollout_worlds(config, policy, env, prefix, None, args.output, config.model)
        arrays = dict(qpos=np.array(env.qpos), qvel=np.array(env.qvel), time=np.array(env.times),
                      integration=np.array(env.integration), actions=np.array(env.actions),
                      predictions=np.array(policy.predictions), diffusion_noise=np.array(policy.noise))
        np.savez_compressed(args.output / "trajectory.npz", **arrays)
        write_json(args.output / "observations.json", policy.observations)
        build_summary(config=config, ckpt_path=Path(config.checkpoint), device=f"cuda:{args.gpu_id}",
                      worlds=worlds, out_dir=args.output, physics=resolved_physics(env.env))
        write_json(args.output / "result.json", dict(world=worlds[0], handoff=report, trace_hashes=array_hashes(arrays)))
        video, fps = render_trajectory(env, args.output, args.render_stride,
                                       tuple(config.model.camera_keys), lossless=True)
        print(f"Handoff result: success={worlds[0]['success']}; video={video}; fps={fps}; "
              "encoding=lossless RGB H.264, native resolution", flush=True)
    finally:
        env.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--video-time", type=float, default=12.)
    parser.add_argument("--physics-step", type=int)
    parser.add_argument("--arm-velocity", choices=("zero", "match-twist"), default="match-twist")
    parser.add_argument("--ik-attempts", type=int, default=16)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.video_time < 0 or args.ik_attempts < 1 or args.render_stride < 1:
        parser.error("Time must be nonnegative; attempts and stride must be positive")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    args.output = (args.output or args.case_dir / f"handoff_{stamp}").resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "handoff_state.npz").exists():
        parser.error("Output already contains a handoff; use a new directory")
    initial, report, scene, old_summary, new_summary = prepare(args)
    np.savez(args.output / "handoff_state.npz", **initial)
    write_json(args.output / "handoff.json", report)
    print(f"Prepared handoff: {args.output}", flush=True)
    print(f"Selected source frame: {report['selection']}", flush=True)
    for arm, checks in report["arms"].items():
        print(f"{arm}: IK {checks['position_mm']:.6g} mm / {checks['rotation_deg']:.6g} deg; "
              f"velocity check={checks['velocity']}", flush=True)
    if not args.prepare_only:
        run(args, initial, report, scene, old_summary, new_summary)


if __name__ == "__main__":
    main()
