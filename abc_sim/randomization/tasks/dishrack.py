from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import mujoco
import numpy as np

from ..assets.dishrack import (
    _DISHRACK_BASE_SCENE_XML,
    _DISHRACK_DEFAULT_VARIANTS,
    _DISHRACK_MAX_PLATE_COUNT,
    _build_dishrack_scene_xml,
    _dishrack_canonical_variant_name,
    _dishrack_compiled_metadata,
    _dishrack_normalize_plate_variants,
    _dishrack_plate_joint_name,
    _dishrack_sample_variant_name,
    _dishrack_variant_names,
)
from ..core import (
    PerturbRange,
    RandomizationState,
    SceneRandomizer,
    _quat_from_yaw,
    _quat_mul,
    _sample_orientation_delta,
    _yaw_from_quat,
)
from ..requests import DishRackResetRequest

# World-frame [dx mm, dy mm, yaw deg] from 1239 training episodes.
_ARM_BASE_STATS_MM_DEG = {
    "left_arm": (
        [20.090777, -25.850766, -0.308589],
        [
            [140.354628, -122.732350, -2.807340],
            [-122.732350, 442.281744, -9.292575],
            [-2.807340, -9.292575, 1.775417],
        ],
        (61.007573727, 2.212487484),
    ),
    "right_arm": (
        [24.275754, -14.639903, 1.515973],
        [
            [101.036659, -108.047273, 0.362798],
            [-108.047273, 513.822078, 4.252998],
            [0.362798, 4.252998, 0.911754],
        ],
        (54.960767602, 2.940367934),
    ),
}
# The mounting plate can bridge the 5 mm seam between the platform and table.
_ARM_BASE_SUPPORT_CENTER_M = np.array([0.30, 0.0])
_ARM_BASE_SUPPORT_HALF_SIZE_M = np.array([0.5975, 0.65])
_ARM_BASE_PLATE_HALF_SIZE_M = np.array([0.035, 0.10])


class DishRackRandomizer(SceneRandomizer):
    min_clearance_m = 0.1
    rack_plate_margin_m = 0.02
    plate_plate_margin_m = 0.02
    rack_table_margin_m = 0.02
    plate_table_margin_m = 0.02
    # Physical tabletop edges, before the center-position margin in table_bounds.
    rack_table_edge_bounds: tuple[float, float, float, float] = (
        0.3025,
        0.8975,
        -0.65,
        0.65,
    )
    max_plate_count = _DISHRACK_MAX_PLATE_COUNT
    _scene_model_cache_size = 16
    _plate_sample_tries = 64
    perturbations = [
        PerturbRange("dishrack",    delta_x=(-0.10, 0.09), delta_y=(-0.15, 0.15), delta_yaw=(-0.25, 0.25)),
        PerturbRange("plate_joint", delta_x=(-0.35, 0.25), delta_y=(-0.35, 0.35)),
    ]

    def prepare_env(self) -> None:
        self._current_scale_states = {}
        self._current_plate_variants = [_DISHRACK_DEFAULT_VARIANTS["plate"]]
        self._current_variant_names = {
            "dish_rack": _DISHRACK_DEFAULT_VARIANTS["dish_rack"],
            "plate": _DISHRACK_DEFAULT_VARIANTS["plate"],
        }
        self._current_plate_collision_radii: list[float] = [0.0]
        self._current_rack_half_extents_xy: tuple[float, float] = (0.0, 0.0)
        self._compiled_scene_model_cache: OrderedDict[tuple[Any, ...], mujoco.MjModel] = OrderedDict()
        self._set_active_plate_count(1)
        self._refresh_collision_metadata()
        self._reload_variant_scene(
            _DISHRACK_DEFAULT_VARIANTS["dish_rack"],
            [_DISHRACK_DEFAULT_VARIANTS["plate"]],
            {},
        )

    def randomize(
        self,
        model: Any,
        data: Any,
        seed: int | None = None,
        request: Any | None = None,
    ) -> RandomizationState:
        rng = np.random.default_rng(seed)
        reset_request = DishRackResetRequest.from_value(request)
        dish_rack_variant, plate_variants = self._resolve_variant_selection(rng, reset_request)
        self._set_active_plate_count(len(plate_variants))
        should_randomize_scales = (
            True if reset_request.randomize_scales is None else bool(reset_request.randomize_scales)
        )
        scale_states = self._sample_scale_states(rng) if should_randomize_scales else {}
        self._current_scale_states = dict(scale_states)
        self._reload_variant_scene(dish_rack_variant, plate_variants, scale_states)

        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data

        arm_base_poses = self._randomize_arm_bases(model, data, reset_request, seed)
        nominals = self._read_nominals(model, data)
        for _ in range(self.max_tries):
            states = self._sample_once(nominals, rng)
            if not self._bounds_ok(states):
                continue
            if not self._pairwise_ok(states):
                continue
            self._apply_states(model, data, states)
            mujoco.mj_forward(model, data)
            if not self._contacts_ok(model, data):
                continue
            return RandomizationState(
                seed=seed or 0,
                object_states=states,
                scale_states=scale_states,
                metadata={
                    "dish_rack_variant": dish_rack_variant,
                    "plate_variant": plate_variants[0],
                    "plate_variants": list(plate_variants),
                    "plate_count": len(plate_variants),
                    **({"arm_base_poses": arm_base_poses} if arm_base_poses else {}),
                },
            )

        self._raise_sampling_failure(seed)

    @staticmethod
    def _randomize_arm_bases(model, data, request, seed):
        if not request.randomize_arm_bases:
            return {}
        rng = np.random.default_rng(None if seed is None else [seed, 1])
        poses = {}
        for name, (mean, covariance, p95) in _ARM_BASE_STATS_MM_DEG.items():
            body = model.body(name)
            while True:
                offset = rng.multivariate_normal(mean, covariance)
                if np.linalg.norm(offset[:2]) > p95[0] or abs(offset[2]) > p95[1]:
                    continue
                position = body.pos[:2] + offset[:2] / 1000
                yaw = np.deg2rad(offset[2])
                angle = _yaw_from_quat(body.quat) + yaw
                c, s = np.abs([np.cos(angle), np.sin(angle)])
                half_size = np.array([[c, s], [s, c]]) @ _ARM_BASE_PLATE_HALF_SIZE_M
                if np.all(np.abs(position - _ARM_BASE_SUPPORT_CENTER_M) + half_size <= _ARM_BASE_SUPPORT_HALF_SIZE_M):
                    break
            body.pos[:2] = position
            body.quat[:] = _quat_mul(_quat_from_yaw(yaw), body.quat)
            poses[name] = {"pos": body.pos.tolist(), "quat": body.quat.tolist()}
        mujoco.mj_forward(model, data)
        return poses

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        dish_rack_variant = _dishrack_canonical_variant_name(
            "dish_rack",
            str(state.metadata.get("dish_rack_variant", _DISHRACK_DEFAULT_VARIANTS["dish_rack"])),
        )
        raw_plate_variants = state.metadata.get("plate_variants")
        if isinstance(raw_plate_variants, (list, tuple)) and raw_plate_variants:
            plate_variants = [_dishrack_canonical_variant_name("plate", str(value)) for value in raw_plate_variants]
        else:
            plate_variants = [
                _dishrack_canonical_variant_name(
                    "plate",
                    str(state.metadata.get("plate_variant", _DISHRACK_DEFAULT_VARIANTS["plate"])),
                )
            ]
        self._current_scale_states = dict(state.scale_states)
        self._current_plate_variants = list(plate_variants)
        self._current_variant_names = {
            "dish_rack": dish_rack_variant,
            "plate": plate_variants[0],
        }
        self._set_active_plate_count(len(plate_variants))
        self._reload_variant_scene(dish_rack_variant, plate_variants, state.scale_states)
        if self._env_ref is not None:
            model = self._env_ref.model
            data = self._env_ref.data
        self._apply_states(model, data, state.object_states)
        for name, pose in state.metadata.get("arm_base_poses", {}).items():
            model.body(name).pos[:] = pose["pos"]
            model.body(name).quat[:] = pose["quat"]
        mujoco.mj_forward(model, data)

    def _refresh_collision_metadata(self) -> None:
        plate_radii: list[float] = []
        for index, plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            plate_half_x, plate_half_y, _, _, _ = _dishrack_compiled_metadata("plate", plate_variant)
            plate_scale = float(
                self._current_scale_states.get(
                    _dishrack_plate_joint_name(index),
                    self._current_scale_states.get("plate_joint", 1.0),
                )
            )
            plate_radii.append(max(plate_half_x, plate_half_y) * plate_scale)

        rack_variant = getattr(self, "_current_variant_names", {}).get(
            "dish_rack", _DISHRACK_DEFAULT_VARIANTS["dish_rack"]
        )
        rack_half_x, rack_half_y, _, _, _ = _dishrack_compiled_metadata("dish_rack", rack_variant)
        rack_scale = float(self._current_scale_states.get("dishrack", 1.0))

        self._current_plate_collision_radii = plate_radii
        self._current_rack_half_extents_xy = (
            rack_half_x * rack_scale,
            rack_half_y * rack_scale,
        )

    def _scene_model_cache_key(
        self,
        dish_rack_variant: str,
        plate_variants: list[str],
        scale_states: dict[str, float],
    ) -> tuple[Any, ...]:
        scale_key = tuple(
            sorted((str(name), round(float(value), 8)) for name, value in scale_states.items())
        )
        return (dish_rack_variant, tuple(plate_variants), scale_key)

    def _try_reload_cached_scene_model(self, cache_key: tuple[Any, ...]) -> bool:
        if self._env_ref is None:
            return False
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if not cache:
            return False
        cached_model = cache.get(cache_key)
        if cached_model is None:
            return False
        cache.move_to_end(cache_key)
        self._env_ref.reload_from_model(cached_model)
        return True

    def _store_compiled_scene_model(self, cache_key: tuple[Any, ...]) -> None:
        if self._env_ref is None:
            return
        cache = getattr(self, "_compiled_scene_model_cache", None)
        if cache is None:
            self._compiled_scene_model_cache = OrderedDict()
            cache = self._compiled_scene_model_cache
        cache[cache_key] = copy.deepcopy(self._env_ref.model)
        cache.move_to_end(cache_key)
        while len(cache) > self._scene_model_cache_size:
            cache.popitem(last=False)

    def _resolve_variant_selection(
        self,
        rng: np.random.Generator,
        request: DishRackResetRequest,
    ) -> tuple[str, list[str]]:
        randomize_variants = request.randomize_variants
        if randomize_variants is None:
            randomize_variants = not any(
                (
                    request.plate_variant is not None,
                    request.dish_rack_variant is not None,
                    request.cycle_plate != 0,
                    request.cycle_dish_rack != 0,
                )
            )

        plate_count = self._resolve_plate_count(
            rng=rng,
            request=request,
            randomize_variants=randomize_variants,
        )
        dish_rack_variant = self._resolve_variant_name(
            kind="dish_rack",
            explicit_variant=request.dish_rack_variant,
            cycle_step=request.cycle_dish_rack,
            randomize_variants=randomize_variants,
            rng=rng,
        )
        repeated_variant = self._resolve_variant_name(
            kind="plate",
            explicit_variant=request.plate_variant,
            cycle_step=request.cycle_plate,
            randomize_variants=randomize_variants,
            rng=rng,
        )
        if request.plate_variant is not None or request.cycle_plate != 0 or not randomize_variants:
            plate_variants = [repeated_variant] * plate_count
        else:
            plate_variants = [
                _dishrack_sample_variant_name("plate", rng)
                for _ in range(plate_count)
            ]
        return dish_rack_variant, plate_variants

    def _resolve_plate_count(
        self,
        *,
        rng: np.random.Generator,
        request: DishRackResetRequest,
        randomize_variants: bool,
    ) -> int:
        if request.plate_count is not None:
            plate_count = int(request.plate_count)
        elif request.plate_variant is not None or request.cycle_plate != 0 or not randomize_variants:
            plate_count = len(getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]]))
        else:
            plate_count = int(rng.integers(1, self.max_plate_count + 1))

        if not 1 <= plate_count <= self.max_plate_count:
            raise ValueError(
                f"DishRack plate_count must be in [1, {self.max_plate_count}], got {plate_count}"
            )
        return plate_count

    def _set_active_plate_count(self, plate_count: int) -> None:
        if not 1 <= plate_count <= self.max_plate_count:
            raise ValueError(
                f"DishRack plate_count must be in [1, {self.max_plate_count}], got {plate_count}"
            )
        self.perturbations = [
            PerturbRange(
                "dishrack",
                delta_x=(-0.10, 0.09),
                delta_y=(-0.15, 0.15),
                delta_yaw=(-0.25, 0.25),
            ),
            *[
                PerturbRange(
                    _dishrack_plate_joint_name(index),
                    delta_x=(-0.35, 0.25),
                    delta_y=(-0.35, 0.35),
                )
                for index in range(plate_count)
            ],
        ]

    def _resolve_variant_name(
        self,
        *,
        kind: str,
        explicit_variant: str | None,
        cycle_step: int,
        randomize_variants: bool,
        rng: np.random.Generator,
    ) -> str:
        variants = _dishrack_variant_names(kind)
        current_variant = getattr(self, "_current_variant_names", {}).get(kind, _DISHRACK_DEFAULT_VARIANTS[kind])
        if current_variant not in variants:
            current_variant = variants[0]

        if explicit_variant is not None:
            explicit_variant = _dishrack_canonical_variant_name(kind, explicit_variant)
            if explicit_variant not in variants:
                raise ValueError(
                    f"Unknown {kind} variant {explicit_variant!r}. Available: {', '.join(variants)}"
                )
            return explicit_variant

        if cycle_step:
            current_index = variants.index(current_variant)
            return variants[(current_index + cycle_step) % len(variants)]

        if not randomize_variants:
            return current_variant

        return _dishrack_sample_variant_name(kind, rng)

    def _sample_once(
        self,
        nominals: dict[str, tuple[np.ndarray, np.ndarray]],
        rng: np.random.Generator,
    ) -> dict[str, dict[str, list[float]]]:
        x_min, x_max, y_min, y_max = self.table_bounds
        perturbations_by_name = {p.joint_name: p for p in self.perturbations}
        states: dict[str, dict[str, list[float]]] = {}

        def sample_state(p: PerturbRange) -> dict[str, list[float]]:
            nom_pos, nom_quat = nominals[p.joint_name]
            eff_dx = (
                max(p.delta_x[0], x_min - nom_pos[0]),
                min(p.delta_x[1], x_max - nom_pos[0]),
            )
            eff_dy = (
                max(p.delta_y[0], y_min - nom_pos[1]),
                min(p.delta_y[1], y_max - nom_pos[1]),
            )
            if eff_dx[0] > eff_dx[1]:
                eff_dx = (0.0, 0.0)
            if eff_dy[0] > eff_dy[1]:
                eff_dy = (0.0, 0.0)

            new_pos = nom_pos + np.array([
                rng.uniform(*eff_dx),
                rng.uniform(*eff_dy),
                rng.uniform(*p.delta_z),
            ])
            new_quat = _quat_mul(_sample_orientation_delta(p, rng), nom_quat)
            return {
                "pos": new_pos.tolist(),
                "quat": new_quat.tolist(),
            }

        rack_perturb = perturbations_by_name.get("dishrack")
        if rack_perturb is not None and "dishrack" in nominals:
            states["dishrack"] = sample_state(rack_perturb)

        for index, _plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            joint_name = _dishrack_plate_joint_name(index)
            perturb = perturbations_by_name.get(joint_name)
            if perturb is None or joint_name not in nominals:
                continue

            last_candidate: dict[str, list[float]] | None = None
            for _ in range(self._plate_sample_tries):
                candidate = sample_state(perturb)
                trial_states = dict(states)
                trial_states[joint_name] = candidate
                last_candidate = candidate
                if self._pairwise_ok(trial_states):
                    break
            if last_candidate is not None:
                states[joint_name] = last_candidate

        # 50% chance: reflect the scene about the XZ plane (negate all Y coords)
        # and rotate the rack 180 about Z so it faces the opposite direction.
        if rng.integers(2) == 1:
            q_180 = _quat_from_yaw(np.pi)
            for key, s in states.items():
                s["pos"][1] = -s["pos"][1]
                if key == "dishrack":
                    s["quat"] = _quat_mul(q_180, np.array(s["quat"])).tolist()
        return states

    def _bounds_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        if not super()._bounds_ok(states):
            return False

        x_min, x_max, y_min, y_max = self.rack_table_edge_bounds
        plate_radii = getattr(self, "_current_plate_collision_radii", [])
        for index, _plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            joint_name = _dishrack_plate_joint_name(index)
            plate_state = states.get(joint_name)
            if plate_state is None:
                continue
            if index < len(plate_radii):
                plate_radius = plate_radii[index]
            else:
                plate_half_x, plate_half_y, _, _, _ = _dishrack_compiled_metadata("plate", _plate_variant)
                plate_scale = float(
                    self._current_scale_states.get(joint_name, self._current_scale_states.get("plate_joint", 1.0))
                )
                plate_radius = max(plate_half_x, plate_half_y) * plate_scale

            margin = plate_radius + self.plate_table_margin_m
            x, y = float(plate_state["pos"][0]), float(plate_state["pos"][1])
            if not (x_min + margin <= x <= x_max - margin and y_min + margin <= y <= y_max - margin):
                return False

        rack_state = states.get("dishrack")
        if rack_state is None:
            return True

        rack_half_x, rack_half_y = getattr(self, "_current_rack_half_extents_xy", (0.0, 0.0))
        rack_yaw = _yaw_from_quat(np.asarray(rack_state["quat"], dtype=np.float64))
        c = abs(float(np.cos(rack_yaw)))
        s = abs(float(np.sin(rack_yaw)))
        world_half_x = c * rack_half_x + s * rack_half_y
        world_half_y = s * rack_half_x + c * rack_half_y

        margin = self.rack_table_margin_m
        x, y = float(rack_state["pos"][0]), float(rack_state["pos"][1])
        return (
            x_min + world_half_x + margin <= x <= x_max - world_half_x - margin
            and y_min + world_half_y + margin <= y <= y_max - world_half_y - margin
        )

    def _pairwise_ok(self, states: dict[str, dict[str, list[float]]]) -> bool:
        rack_state = states.get("dishrack")
        plate_entries: list[tuple[np.ndarray, float]] = []
        plate_radii = getattr(self, "_current_plate_collision_radii", [])
        for index, _plate_variant in enumerate(
            getattr(self, "_current_plate_variants", [_DISHRACK_DEFAULT_VARIANTS["plate"]])
        ):
            joint_name = _dishrack_plate_joint_name(index)
            plate_state = states.get(joint_name)
            if plate_state is None:
                continue
            if index < len(plate_radii):
                plate_radius = plate_radii[index]
            else:
                plate_half_x, plate_half_y, _, _, _ = _dishrack_compiled_metadata("plate", _plate_variant)
                plate_scale = float(
                    self._current_scale_states.get(joint_name, self._current_scale_states.get("plate_joint", 1.0))
                )
                plate_radius = max(plate_half_x, plate_half_y) * plate_scale
            plate_entries.append(
                (
                    np.asarray(plate_state["pos"][:2], dtype=np.float64),
                    plate_radius,
                )
            )

        for i in range(len(plate_entries)):
            pos_i, radius_i = plate_entries[i]
            for j in range(i + 1, len(plate_entries)):
                pos_j, radius_j = plate_entries[j]
                min_dist = radius_i + radius_j + self.plate_plate_margin_m
                if np.linalg.norm(pos_i - pos_j) < min_dist:
                    return False

        if rack_state is None:
            return True

        rack_half_x, rack_half_y = getattr(self, "_current_rack_half_extents_xy", (0.0, 0.0))
        rack_pos_xy = np.asarray(rack_state["pos"][:2], dtype=np.float64)
        rack_yaw = _yaw_from_quat(np.asarray(rack_state["quat"], dtype=np.float64))
        c = float(np.cos(rack_yaw))
        s = float(np.sin(rack_yaw))
        for plate_pos_xy, plate_radius in plate_entries:
            rel_xy = plate_pos_xy - rack_pos_xy
            rack_local_xy = np.array(
                [c * rel_xy[0] + s * rel_xy[1], -s * rel_xy[0] + c * rel_xy[1]],
                dtype=np.float64,
            )
            exclude_x = rack_half_x + plate_radius + self.rack_plate_margin_m
            exclude_y = rack_half_y + plate_radius + self.rack_plate_margin_m
            if abs(rack_local_xy[0]) < exclude_x and abs(rack_local_xy[1]) < exclude_y:
                return False
        return True

    def _reload_variant_scene(
        self,
        dish_rack_variant: str,
        plate_variants: str | list[str] | tuple[str, ...],
        scale_states: dict[str, float],
    ) -> None:
        if self._env_ref is None:
            raise RuntimeError("DishRackRandomizer requires a bound env before scene reload")

        dish_rack_variant = _dishrack_canonical_variant_name("dish_rack", dish_rack_variant)
        preserved_arm_state = self._env_ref._get_reset_arm_state()
        normalized_plate_variants = _dishrack_normalize_plate_variants(plate_variants)
        self._set_active_plate_count(len(normalized_plate_variants))
        scene_cache_key = self._scene_model_cache_key(
            dish_rack_variant,
            normalized_plate_variants,
            scale_states,
        )

        base_scene_xml = self._base_scene_xml_string
        base_scene_dir = self._base_scene_xml_dir
        base_scene_transformed = self._base_scene_xml_transformed
        if not self._try_reload_cached_scene_model(scene_cache_key):
            if (
                base_scene_xml is None
                or "<!-- TASK_ASSETS_PLACEHOLDER -->" not in base_scene_xml
                or "<!-- TASK_BODY_PLACEHOLDER -->" not in base_scene_xml
            ):
                base_scene_xml = _DISHRACK_BASE_SCENE_XML.read_text()
                base_scene_dir = _DISHRACK_BASE_SCENE_XML.parent
                base_scene_transformed = False

            xml = _build_dishrack_scene_xml(
                dish_rack_variant=dish_rack_variant,
                plate_variants=normalized_plate_variants,
                scale_states=scale_states,
                base_scene_xml=base_scene_xml,
                base_scene_dir=base_scene_dir,
            )
            if self._scene_xml_transform_options is not None and not base_scene_transformed:
                from abc_sim.scene_xml import transform_scene_xml

                xml, _ = transform_scene_xml(xml, options=self._scene_xml_transform_options)

            self._env_ref.reload_from_xml(xml)
            self._store_compiled_scene_model(scene_cache_key)

        self._current_plate_variants = list(normalized_plate_variants)
        self._current_variant_names = {
            "dish_rack": dish_rack_variant,
            "plate": normalized_plate_variants[0],
        }
        self._refresh_collision_metadata()
        mujoco.mj_resetData(self._env_ref.model, self._env_ref.data)
        self._env_ref._set_qpos_from_state(preserved_arm_state)
        mujoco.mj_forward(self._env_ref.model, self._env_ref.data)
        self._fixed_body_nominals = None




__all__ = ["DishRackRandomizer"]
