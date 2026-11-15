"""Inspect saved dishrack rollouts without policy inference or physics stepping.

Restore the saved scene/model and each recorded state. Report plate events,
joint tracking, actuator clipping, and camera-frame TCP tracking. Contacts are
recomputed at the saved post-action qpos; the saved live success is authoritative.
"""
import os

os.environ.setdefault("MUJOCO_GL", "disable")

import argparse
import copy
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

import abc_sim
from abc_sim.env import _GRIPPER_CTRL_MAX
from abc_sim.randomization.core import RandomizationState


MODEL_FIELDS = ("body_pos", "body_quat", "cam_pos", "cam_quat", "geom_contype", "geom_conaffinity")
MASKS = ("plate_spatially_in_rack_mask", "plate_rack_contact_mask",
         "plate_gripper_contact_mask", "plate_upright_in_rack_mask")


def load_json(path):
    return json.loads(path.read_text())


def pose_error(actual_pos, actual_rot, target_pos, target_rot):
    position_mm = float(np.linalg.norm(actual_pos - target_pos) * 1000)
    cosine = (np.trace(actual_rot.T @ target_rot) - 1) / 2
    rotation_deg = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return position_mm, rotation_deg


def effective_controls(model, actuator_ids, commands):
    """Mirror the environment's gripper scaling and MuJoCo ctrlrange clamping."""
    controls = np.asarray(commands, dtype=np.float64).copy()
    controls[[6, 13]] *= _GRIPPER_CTRL_MAX
    limited = model.actuator_ctrllimited[actuator_ids].astype(bool)
    if model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL:
        limited[:] = False
    bounds = model.actuator_ctrlrange[actuator_ids]
    effective = controls.copy()
    effective[limited] = np.clip(controls[limited], bounds[limited, 0], bounds[limited, 1])
    return controls, effective


def diagnose(case_dir, side, output):
    folder = case_dir / side
    saved = load_json(folder / "summary.json")
    world = saved["worlds"][0]
    scene = load_json(case_dir / "scene.json")
    state = RandomizationState.from_dict(copy.deepcopy(scene["randomization"]))
    env = abc_sim.make_env(task="dishrack", render_cameras=False)
    rows, events, previous_masks = [], [], {}
    try:
        # apply() restores the recorded variants/count/scales before model fields.
        env._task_randomizer.apply(env.model, env.data, state)
        model, data = env.model, env.data
        with np.load(folder / "initial.npz") as initial:
            for name in MODEL_FIELDS:
                target = getattr(model, name)
                if target.shape != initial[name].shape:
                    raise ValueError(f"{side}: saved {name} shape differs from the current model")
                target[:] = initial[name]
            data.qpos[:] = initial["qpos"]
            data.qvel[:] = initial["qvel"]
            data.ctrl[:] = initial["ctrl"]
            expected_top = initial["top_camera_world"].copy()
        mujoco.mj_forward(model, data)
        camera = data.camera("top")
        np.testing.assert_allclose(np.r_[camera.xpos, camera.xmat], expected_top, atol=1e-12, rtol=0)
        camera_position = camera.xpos.copy()
        camera_from_world = np.diag([1., -1., -1.]) @ camera.xmat.reshape(3, 3).T
        target_data = mujoco.MjData(model)
        evaluator = env._task_evaluator
        evaluator.configure_from_randomization(model, state)
        evaluator.reset(nworld=1)
        actuator_ids = np.asarray(env._ctrl_indices)
        qpos_ids = np.asarray(env._qpos_indices)
        arm_columns = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
        joint_ids = model.actuator_trnid[actuator_ids[arm_columns], 0]
        sites = {arm: model.site(f"{arm}_tcp_site").id for arm in ("left", "right")}
        with np.load(folder / "trajectory.npz") as trace:
            qpos, qvel, times, actions = (trace[name] for name in ("qpos", "qvel", "time", "actions"))
            if len(qpos) != len(actions) + 1 or qpos.shape[1] != model.nq:
                raise ValueError(f"{side}: invalid saved trajectory dimensions")
            np.testing.assert_array_equal(qpos[0], data.qpos)
            integration = trace["integration"] if "integration" in trace else None
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            if integration is not None and integration.shape != (len(qpos), mujoco.mj_stateSize(model, spec)):
                raise ValueError("Saved integration layout differs from this MuJoCo version")
            for frame in range(len(qpos)):
                if integration is not None:
                    mujoco.mj_setState(model, data, integration[frame], spec)
                else:
                    data.qpos[:], data.qvel[:], data.time = qpos[frame], qvel[frame], times[frame]
                mujoco.mj_forward(model, data)
                metrics = evaluator.evaluate_model_data(model, data).to_info(squeeze=True)
                names = metrics["plate_names"]
                row = dict(frame=frame, time_s=float(times[frame] - times[0]),
                           upright_plates=int(metrics["num_upright_plates_in_rack"]),
                           placed_plates=int(metrics["num_plates_in_rack"]))
                plate_positions = []
                for index, name in enumerate(names):
                    body_id = model.body(name).id
                    plate_position = data.xpos[body_id].copy()
                    plate_positions.append(plate_position)
                    position_camera = camera_from_world @ (plate_position - camera_position)
                    for axis, value in zip("xyz", position_camera):
                        row[f"{name}_camera_{axis}_m"] = float(value)
                    row[f"{name}_normal_abs_z"] = float(metrics["plate_normal_abs_z"][index])
                    for key in MASKS:
                        value = bool(metrics[key][index])
                        row[f"{name}_{key}"] = int(value)
                        identity = (name, key)
                        if identity in previous_masks and previous_masks[identity] != value:
                            events.append(dict(frame=frame, time_s=row["time_s"], plate=name,
                                               metric=key, value=value))
                        previous_masks[identity] = value
                plate_positions = np.asarray(plate_positions)
                for arm, site in sites.items():
                    position = data.site_xpos[site].copy()
                    rotation = data.site_xmat[site].reshape(3, 3).copy()
                    local = camera_from_world @ (position - camera_position)
                    for axis, value in zip("xyz", local):
                        row[f"{arm}_tcp_camera_{axis}_m"] = float(value)
                    for axis, value in enumerate((camera_from_world @ rotation).ravel()):
                        row[f"{arm}_tcp_camera_R{axis}"] = float(value)
                    distances = np.linalg.norm(plate_positions - position, axis=1)
                    row[f"{arm}_nearest_plate_center_mm"] = float(distances.min() * 1000)
                    row[f"{arm}_gripper_state"] = float(env.get_state()[6 if arm == "left" else 13])

                # Action i generated the transition qpos[i] -> qpos[i+1].
                if frame:
                    command = actions[frame - 1]
                    raw, effective = effective_controls(model, actuator_ids, command)
                    for arm, offset in (("left", 0), ("right", 7)):
                        cols = np.arange(offset, offset + 6)
                        row[f"{arm}_control_clip_max_deg"] = float(np.degrees(np.abs(raw[cols] - effective[cols])).max())
                        row[f"{arm}_tracking_after_max_deg"] = float(np.degrees(np.abs(command[cols] - data.qpos[qpos_ids[cols]])).max())
                        for kind, controls in (("raw", raw), ("effective", effective)):
                            target_data.qpos[:] = data.qpos
                            target_data.qpos[qpos_ids[cols]] = controls[cols]
                            mujoco.mj_kinematics(model, target_data)
                            error = pose_error(data.site_xpos[sites[arm]], data.site_xmat[sites[arm]].reshape(3, 3),
                                               target_data.site_xpos[sites[arm]], target_data.site_xmat[sites[arm]].reshape(3, 3))
                            row[f"{arm}_{kind}_target_tracking_mm"], row[f"{arm}_{kind}_target_tracking_deg"] = error
                        for col in cols:
                            name = model.actuator(int(actuator_ids[col])).name
                            row[f"{name}_command_rad"] = float(command[col])
                            row[f"{name}_actual_rad"] = float(data.qpos[qpos_ids[col]])
                rows.append(row)

        output.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output / f"{side}_frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        controls_report = {}
        for arm, offset in (("left", 0), ("right", 7)):
            cols = np.arange(offset, offset + 6)
            clipped = [row for row in rows[1:] if row[f"{arm}_control_clip_max_deg"] > 1e-6]
            controls_report[arm] = dict(
                frames_with_control_clipping=len(clipped),
                fraction_with_control_clipping=len(clipped) / max(1, len(actions)),
                max_control_clipping_deg=max(row[f"{arm}_control_clip_max_deg"] for row in rows[1:]),
                first_control_clipping_frame=clipped[0]["frame"] if clipped else None,
                max_tracking_after_deg=max(row[f"{arm}_tracking_after_max_deg"] for row in rows[1:]),
                joint_control_ranges_rad=model.actuator_ctrlrange[actuator_ids[cols]].tolist(),
                joint_control_limits_enabled=model.actuator_ctrllimited[actuator_ids[cols]].astype(bool).tolist())
        transitions = [event for event in events if event["metric"] == "plate_upright_in_rack_mask"]
        result = dict(saved_success=world["success"], saved_steps=world["steps"],
                      saved_max_reward=world["max_reward"], saved_final_reward=world["reward"],
                      controls=controls_report, plate_placement_events=transitions,
                      final_reconstructed_plate_masks={key: rows[-1][key] for key in rows[-1]
                                                       if any(key.endswith(mask) for mask in MASKS)},
                      joint_ranges_rad=model.jnt_range[joint_ids].tolist(),
                      control_clamping_globally_enabled=not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL),
                      all_plate_events=events)
        return result, rows
    finally:
        env.close()


def plot_runs(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, len(runs), figsize=(7 * len(runs), 9), squeeze=False)
    for column, (side, rows) in enumerate(runs.items()):
        times = [row["time_s"] for row in rows]
        axes[0, column].step(times, [row["upright_plates"] for row in rows], where="post")
        axes[0, column].set(title=f"{side}: upright plates in rack", ylabel="Plates")
        for arm in ("left", "right"):
            axes[1, column].plot(times[1:], [r[f"{arm}_control_clip_max_deg"] for r in rows[1:]], label=arm)
            axes[2, column].plot(times[1:], [r[f"{arm}_raw_target_tracking_mm"] for r in rows[1:]], label=arm)
        axes[1, column].set(ylabel="Control clipping (deg)", title="Max clipping across 6 joints")
        axes[2, column].set(ylabel="TCP tracking (mm)", title="Commanded FK vs post-action actual TCP", xlabel="Time (s)")
    for axis in axes.ravel():
        axis.grid(alpha=.25)
    for axis in axes[1:].ravel():
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "diagnosis.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--side", choices=("both", "old", "new"), default="both")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    output = (args.output or case_dir / "diagnostics").resolve()
    report = dict(case=str(case_dir), physics_stepped=False, policy_inference=False,
                  notes=["Contacts are recomputed at recorded post-action qpos; live success remains authoritative.",
                         "Nearest-plate distance uses plate centers, not surface distance.",
                         "Tracking error includes normal dynamics lag; clipping is a candidate mechanism, not proof of causation.",
                         "Both policy and base differ between old/new; this comparison alone cannot isolate their causal effects."],
                  runs={})
    runs = {}
    for side in (("old", "new") if args.side == "both" else (args.side,)):
        report["runs"][side], runs[side] = diagnose(case_dir, side, output)
        print(f"{side}: inspected {len(runs[side]) - 1} recorded actions", flush=True)
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        plot_runs(runs, output)
    compact = {side: {key: value for key, value in result.items()
                      if key not in ("all_plate_events", "final_reconstructed_plate_masks", "joint_ranges_rad")}
               for side, result in report["runs"].items()}
    print(json.dumps(compact, indent=2))
    print(f"Saved diagnostics: {output}")


if __name__ == "__main__":
    main()
