"""Dishrack task evaluator."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec


def _quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    q = q / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body {name!r} not found")
    return int(body_id)


def _body_subtree_ids(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    body_ids = {int(root_body_id)}
    for body_id in range(int(model.nbody)):
        current = int(body_id)
        while current > 0:
            current = int(model.body_parentid[current])
            if current == root_body_id:
                body_ids.add(int(body_id))
                break
    return body_ids


def _geom_ids_for_bodies(model: mujoco.MjModel, body_ids: set[int]) -> set[int]:
    return {
        int(geom_id)
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_ids
    }


def _gripper_geom_ids(model: mujoco.MjModel) -> set[int]:
    gripper_bodies: set[int] = set()
    parts = ("link_left_finger", "link_right_finger", "_lf_rot", "_lf_down", "_rf_rot", "_rf_down")
    for body_id in range(int(model.nbody)):
        body_name = model.body(body_id).name or ""
        if any(part in body_name for part in parts):
            gripper_bodies.update(_body_subtree_ids(model, body_id))
    return _geom_ids_for_bodies(model, gripper_bodies)


def _plate_joint_name(index: int) -> str:
    return "plate_joint" if index == 0 else f"plate_joint_{index}"


def _plate_body_name(index: int) -> str:
    return "plate" if index == 0 else f"plate_{index}"


def _joint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"Joint {name!r} not found")
    return int(model.jnt_qposadr[joint_id])


class DishrackEvaluator:
    """Score dishrack by requiring every active plate to be upright in the rack."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        margin_m: float = 0.05,
        height_min_m: float = -0.06,
        height_max_m: float = 0.45,
        upright_normal_z_max: float = 0.55,
        horizontal_normal_z_min: float = 0.75,
    ) -> None:
        self.model = model
        self.spec = spec
        self.margin_m = float(margin_m)
        self.height_min_m = float(height_min_m)
        self.height_max_m = float(height_max_m)
        self.upright_normal_z_max = float(upright_normal_z_max)
        self.horizontal_normal_z_min = float(horizontal_normal_z_min)
        self.plate_indices: list[int] = []
        self._rack_body_id: int | None = None
        self._rack_geom_ids: set[int] = set()
        self._gripper_geom_ids: set[int] = set()
        self._arm_base_poses: dict[str, Any] = {}
        self._refresh_model(model)
        self.reset(nworld=1)

    def configure_from_randomization(self, model, state) -> None:
        if model is not self.model:
            self._refresh_model(model)
        self._arm_base_poses = state.metadata.get("arm_base_poses", {})

    @staticmethod
    def _resolve_plate_indices(model: mujoco.MjModel) -> list[int]:
        indices: list[int] = []
        for index in range(64):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _plate_body_name(index))
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _plate_joint_name(index))
            if body_id >= 0 and joint_id >= 0:
                indices.append(index)
        return indices

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_plates = np.zeros((self._nworld,), dtype=np.int32)
        self._max_upright = np.zeros((self._nworld,), dtype=np.int32)
        self._max_horizontal = np.zeros((self._nworld,), dtype=np.int32)
        self._max_non_upright = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._ever_placement = np.zeros((self._nworld,), dtype=bool)
        self._ever_upright = np.zeros((self._nworld,), dtype=bool)

    def _refresh_model(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.plate_indices = self._resolve_plate_indices(model)
        rack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dishrack")
        if rack_body_id < 0:
            self._rack_body_id = None
            self._rack_geom_ids = set()
        else:
            self._rack_body_id = int(rack_body_id)
            self._rack_geom_ids = _geom_ids_for_bodies(
                model, _body_subtree_ids(model, self._rack_body_id)
            )
        self._gripper_geom_ids = _gripper_geom_ids(model)

    def _rack_half_extents(self, data: mujoco.MjData) -> tuple[float, float]:
        if self._rack_body_id is None:
            raise ValueError("Dishrack body not found in current model")
        rack_pos = np.asarray(data.xpos[self._rack_body_id], dtype=np.float64)
        rack_from_world = _quat_to_rotmat(np.asarray(data.xquat[self._rack_body_id], dtype=np.float64)).T
        local_points = []
        for geom_id in self._rack_geom_ids:
            geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            local = rack_from_world @ (geom_pos - rack_pos)
            rbound = float(self.model.geom_rbound[geom_id])
            local_points.extend(
                [
                    [local[0] - rbound, local[1] - rbound],
                    [local[0] + rbound, local[1] + rbound],
                ]
            )
        if not local_points:
            return 0.12, 0.12
        points = np.asarray(local_points, dtype=np.float64)
        return float(np.max(np.abs(points[:, 0]))), float(np.max(np.abs(points[:, 1])))

    def _contact_flags_for_plate(self, data: mujoco.MjData, plate_body_id: int) -> tuple[bool, bool]:
        plate_geom_ids = _geom_ids_for_bodies(self.model, _body_subtree_ids(self.model, plate_body_id))
        has_rack_contact = False
        has_gripper_contact = False
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 not in plate_geom_ids and geom2 not in plate_geom_ids:
                continue
            other_geom = geom2 if geom1 in plate_geom_ids else geom1
            has_rack_contact = has_rack_contact or other_geom in self._rack_geom_ids
            has_gripper_contact = has_gripper_contact or other_geom in self._gripper_geom_ids
        return has_rack_contact, has_gripper_contact

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        if model is not self.model:
            self._refresh_model(model)
        result = self._evaluate_one(data, world_index=0)
        return TaskEvalResult(
            reward=np.asarray([result["reward"]], dtype=np.float32),
            success=np.asarray([result["success"]], dtype=bool),
            metrics={
                key: [value] if isinstance(value, list) else np.asarray([value])
                for key, value in result.items()
                if key not in {"reward", "success"}
            },
        )

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}")
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        rewards = np.zeros((qpos_batch.shape[0],), dtype=np.float32)
        successes = np.zeros((qpos_batch.shape[0],), dtype=bool)
        metric_rows: list[dict[str, Any]] = []
        for world_index, qpos in enumerate(qpos_batch):
            data = mujoco.MjData(self.model)
            data.qpos[:] = qpos
            original_poses = {
                name: (self.model.body(name).pos.copy(), self.model.body(name).quat.copy())
                for name in self._arm_base_poses
            }
            try:
                for name, pose in self._arm_base_poses.items():
                    self.model.body(name).pos[:] = pose["pos"]
                    self.model.body(name).quat[:] = pose["quat"]
                mujoco.mj_forward(self.model, data)
                row = self._evaluate_one(data, world_index=world_index)
            finally:
                for name, (pos, quat) in original_poses.items():
                    self.model.body(name).pos[:] = pos
                    self.model.body(name).quat[:] = quat
            rewards[world_index] = float(row["reward"])
            successes[world_index] = bool(row["success"])
            metric_rows.append(row)
        if metric_rows:
            metrics = {}
            for key in metric_rows[0]:
                if key in {"reward", "success"}:
                    continue
                values = [row[key] for row in metric_rows]
                metrics[key] = values if isinstance(values[0], list) else np.asarray(values)
        else:
            metrics = {}
        return TaskEvalResult(reward=rewards, success=successes, metrics=metrics)

    def _evaluate_one(self, data: mujoco.MjData, *, world_index: int) -> dict[str, Any]:
        if self._rack_body_id is None:
            raise ValueError("Dishrack body not found in current model")
        rack_pos = np.asarray(data.xpos[self._rack_body_id], dtype=np.float64)
        rack_from_world = _quat_to_rotmat(np.asarray(data.xquat[self._rack_body_id], dtype=np.float64)).T
        rack_half_x, rack_half_y = self._rack_half_extents(data)

        in_rack_mask: list[bool] = []
        upright_mask: list[bool] = []
        horizontal_mask: list[bool] = []
        upright_in_rack_mask: list[bool] = []
        horizontal_in_rack_mask: list[bool] = []
        non_upright_in_rack_mask: list[bool] = []
        spatial_mask: list[bool] = []
        rack_contact_mask: list[bool] = []
        gripper_contact_mask: list[bool] = []
        plate_names: list[str] = []
        normal_abs_z: list[float] = []

        for index in self.plate_indices:
            body_name = _plate_body_name(index)
            body_id = _body_id(self.model, body_name)
            qpos_adr = _joint_qpos_addr(self.model, _plate_joint_name(index))
            pos = np.asarray(data.qpos[qpos_adr : qpos_adr + 3], dtype=np.float64)
            quat = np.asarray(data.qpos[qpos_adr + 3 : qpos_adr + 7], dtype=np.float64)
            local = rack_from_world @ (pos - rack_pos)
            spatially_in_rack = (
                rack_half_x + self.margin_m - abs(float(local[0])) >= 0.0
                and rack_half_y + self.margin_m - abs(float(local[1])) >= 0.0
                and float(local[2]) - self.height_min_m >= 0.0
                and self.height_max_m - float(local[2]) >= 0.0
            )
            has_rack_contact, has_gripper_contact = self._contact_flags_for_plate(data, body_id)
            in_rack = spatially_in_rack and has_rack_contact and not has_gripper_contact
            plate_normal_abs_z = abs(float((_quat_to_rotmat(quat)[:, 2])[2]))
            upright = plate_normal_abs_z <= self.upright_normal_z_max
            horizontal = plate_normal_abs_z >= self.horizontal_normal_z_min

            plate_names.append(body_name)
            spatial_mask.append(bool(spatially_in_rack))
            rack_contact_mask.append(bool(has_rack_contact))
            gripper_contact_mask.append(bool(has_gripper_contact))
            in_rack_mask.append(bool(in_rack))
            upright_mask.append(bool(upright))
            horizontal_mask.append(bool(horizontal))
            upright_in_rack_mask.append(bool(in_rack and upright))
            horizontal_in_rack_mask.append(bool(in_rack and horizontal))
            non_upright_in_rack_mask.append(bool(in_rack and not upright))
            normal_abs_z.append(plate_normal_abs_z)

        plate_count = len(self.plate_indices)
        num_in_rack = int(np.asarray(in_rack_mask, dtype=bool).sum())
        num_upright = int(np.asarray(upright_in_rack_mask, dtype=bool).sum())
        num_horizontal = int(np.asarray(horizontal_in_rack_mask, dtype=bool).sum())
        num_non_upright = int(np.asarray(non_upright_in_rack_mask, dtype=bool).sum())

        self._max_plates[world_index] = max(self._max_plates[world_index], num_in_rack)
        self._max_upright[world_index] = max(self._max_upright[world_index], num_upright)
        self._max_horizontal[world_index] = max(self._max_horizontal[world_index], num_horizontal)
        self._max_non_upright[world_index] = max(self._max_non_upright[world_index], num_non_upright)
        placement_success = num_in_rack >= plate_count
        upright_success = num_upright >= plate_count
        self._ever_placement[world_index] |= placement_success
        self._ever_upright[world_index] |= upright_success
        self._ever_success[world_index] |= upright_success

        return {
            "reward": float(num_upright / max(1, plate_count)),
            "placement_reward": float(num_in_rack / max(1, plate_count)),
            "upright_reward": float(num_upright / max(1, plate_count)),
            "success": bool(upright_success),
            "ever_success": bool(self._ever_success[world_index]),
            "placement_success": bool(placement_success),
            "ever_placement_success": bool(self._ever_placement[world_index]),
            "upright_success": bool(upright_success),
            "ever_upright_success": bool(self._ever_upright[world_index]),
            "num_plates_in_rack": num_in_rack,
            "num_upright_plates_in_rack": num_upright,
            "num_horizontal_plates_in_rack": num_horizontal,
            "num_non_upright_plates_in_rack": num_non_upright,
            "max_plates_in_rack_so_far": int(self._max_plates[world_index]),
            "max_upright_plates_in_rack_so_far": int(self._max_upright[world_index]),
            "max_horizontal_plates_in_rack_so_far": int(self._max_horizontal[world_index]),
            "max_non_upright_plates_in_rack_so_far": int(self._max_non_upright[world_index]),
            "plate_count": plate_count,
            "plate_names": plate_names,
            "plate_spatially_in_rack_mask": spatial_mask,
            "plate_rack_contact_mask": rack_contact_mask,
            "plate_gripper_contact_mask": gripper_contact_mask,
            "plate_in_rack_mask": in_rack_mask,
            "plate_upright_mask": upright_mask,
            "plate_horizontal_mask": horizontal_mask,
            "plate_upright_in_rack_mask": upright_in_rack_mask,
            "plate_horizontal_in_rack_mask": horizontal_in_rack_mask,
            "plate_non_upright_in_rack_mask": non_upright_in_rack_mask,
            "plate_normal_abs_z": normal_abs_z,
            "rack_half_extents_xy": [float(rack_half_x), float(rack_half_y)],
            "rack_margin_m": self.margin_m,
            "rack_height_band_m": [self.height_min_m, self.height_max_m],
            "upright_normal_z_max": self.upright_normal_z_max,
            "horizontal_normal_z_min": self.horizontal_normal_z_min,
        }
