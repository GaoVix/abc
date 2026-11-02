"""Replay render runtime for MJWarp and Madrona backends."""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

import numpy as np

# Model fields the task randomizers edit in place per episode. MJWarp indexes
# them per world, so a runtime keeps one copy per world and refreshes them from
# the MuJoCo model on reset.
PER_EPISODE_MODEL_FIELDS = ("body_pos", "body_quat", "mat_rgba")


def _sanitize_missing_int_fields(device_struct, cls) -> None:
    for field in dataclasses.fields(cls):
        if field.type is int and getattr(device_struct, field.name, None) is None:
            setattr(device_struct, field.name, 0)


def _put_model_and_data_safe(
    mjw,
    mjm,
    mjd,
    *,
    nworld: int,
    nconmax: int | None = None,
    njmax: int | None = None,
):
    import mujoco_warp._src.io as mjw_io
    from mujoco_warp._src import types as mjw_types

    original_create_array = mjw_io._create_array

    def _create_array_with_safe_sizes(data, spec, sizes, *args, **kwargs):
        safe_sizes = {key: (0 if value is None else value) for key, value in sizes.items()}
        return original_create_array(data, spec, safe_sizes, *args, **kwargs)

    resolved_nconmax = int(nconmax) if nconmax is not None else max(int(getattr(mjm, "nconmax", 0) or 0), int(getattr(mjd, "ncon", 0) or 0), 4096)
    resolved_njmax = int(njmax) if njmax is not None else max(int(getattr(mjm, "njmax", 0) or 0), int(getattr(mjd, "nefc", 0) or 0), 4096)

    put_data_kwargs = {
        "nworld": nworld,
        "nconmax": resolved_nconmax,
        "njmax": resolved_njmax,
    }

    mjw_io._create_array = _create_array_with_safe_sizes
    try:
        m_warp = mjw.put_model(mjm)
        d_warp = mjw.put_data(mjm, mjd, **put_data_kwargs)
    finally:
        mjw_io._create_array = original_create_array

    _sanitize_missing_int_fields(m_warp, mjw_types.Model)
    _sanitize_missing_int_fields(d_warp, mjw_types.Data)
    return m_warp, d_warp


class WarpReplayRuntime:
    """Owns persistent MuJoCo-Warp model/data for a single replay world."""

    def __init__(
        self,
        mjm,
        mjd,
        *,
        nworld: int = 1,
        gpu_id: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
    ):
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise RuntimeError(
                "Sim camera rendering requires optional dependencies: mujoco_warp and warp."
            ) from exc

        self._mjw = mjw
        self._wp = wp
        self.mjm = mjm
        self.mjd = mjd
        self.nworld = nworld
        self.device = str(wp.get_device())
        if gpu_id is not None:
            desired_device = f"cuda:{gpu_id}"
            try:
                wp.set_device(desired_device)
            except Exception as exc:
                raise RuntimeError(f"Failed to select Warp device {desired_device!r}") from exc
            self.device = desired_device
        self.m_warp, self.d_warp = _put_model_and_data_safe(
            mjw,
            mjm,
            mjd,
            nworld=nworld,
            nconmax=nconmax,
            njmax=njmax,
        )
        for name in PER_EPISODE_MODEL_FIELDS:
            field = getattr(self.m_warp, name)
            if field.size:
                expanded = np.repeat(field.numpy(), nworld, axis=0)
                setattr(self.m_warp, name, wp.array(expanded, dtype=field.dtype))
        self._step_graphs: dict[int, object] = {}
        self._closed = False

    def close(self) -> None:
        """Drop Warp model/data references owned by this runtime."""
        if self._closed:
            return
        try:
            self._wp.synchronize()
        except Exception:
            pass
        self._step_graphs = {}
        self.m_warp = None
        self.d_warp = None
        self.mjm = None
        self.mjd = None
        try:
            self._wp.synchronize()
        except Exception:
            pass
        self._closed = True

    def _copy_field(self, target, values: np.ndarray, *, dtype=None) -> None:
        self._wp.copy(target, self._wp.from_numpy(values, dtype=dtype))

    def upload_qpos_frames(self, qpos_frames: np.ndarray):
        """Upload a replay qpos sequence to the same device as Warp data."""
        qpos_frames = np.asarray(qpos_frames, dtype=np.float32)
        if qpos_frames.ndim != 2 or qpos_frames.shape[1] != self.mjm.nq:
            raise ValueError(
                f"Expected qpos frame shape (T, {self.mjm.nq}), got {qpos_frames.shape}"
            )
        return self._wp.array(
            qpos_frames,
            dtype=self._wp.float32,
            device=self.d_warp.qpos.device,
        )

    def sync_model_from_mujoco(self, per_world: dict[str, np.ndarray] | None = None) -> None:
        """Copy the per-episode model fields into every world of the Warp model.

        ``per_world`` maps a field name to a ``(nworld, ...)`` array holding one
        value per world; other fields broadcast the MuJoCo model's value.
        """
        for name in PER_EPISODE_MODEL_FIELDS:
            target = getattr(self.m_warp, name)
            if not target.size:
                continue
            values = None if per_world is None else per_world.get(name)
            if values is None:
                values = np.repeat(np.asarray(getattr(self.mjm, name))[None], self.nworld, axis=0)
            self._copy_field(target, np.ascontiguousarray(values, dtype=np.float32), dtype=target.dtype)

    def sync_from_mujoco(self) -> None:
        """Copy the current MuJoCo state into the persistent Warp data."""
        self.sync_model_from_mujoco()
        self._copy_field(
            self.d_warp.qpos,
            np.broadcast_to(
                np.asarray(self.mjd.qpos, dtype=np.float32),
                (self.nworld, self.mjm.nq),
            ),
            dtype=self._wp.float32,
        )

        if self.mjm.nv > 0:
            self._copy_field(
                self.d_warp.qvel,
                np.broadcast_to(
                    np.asarray(self.mjd.qvel, dtype=np.float32),
                    (self.nworld, self.mjm.nv),
                ),
                dtype=self._wp.float32,
            )

        if self.mjm.nu > 0:
            self._copy_field(
                self.d_warp.ctrl,
                np.broadcast_to(
                    np.asarray(self.mjd.ctrl, dtype=np.float32),
                    (self.nworld, self.mjm.nu),
                ),
                dtype=self._wp.float32,
            )

        if self.mjm.na > 0 and hasattr(self.d_warp, "act"):
            self._copy_field(
                self.d_warp.act,
                np.broadcast_to(
                    np.asarray(self.mjd.act, dtype=np.float32),
                    (self.nworld, self.mjm.na),
                ),
                dtype=self._wp.float32,
            )

        if self.mjm.nmocap > 0:
            self._copy_field(
                self.d_warp.mocap_pos,
                np.broadcast_to(
                    np.asarray(self.mjd.mocap_pos, dtype=np.float32),
                    (self.nworld, self.mjm.nmocap, 3),
                ),
                dtype=self._wp.vec3f,
            )
            self._copy_field(
                self.d_warp.mocap_quat,
                np.broadcast_to(
                    np.asarray(self.mjd.mocap_quat, dtype=np.float32),
                    (self.nworld, self.mjm.nmocap, 4),
                ),
                dtype=self._wp.quatf,
            )

        if hasattr(self.d_warp, "time"):
            self._copy_field(
                self.d_warp.time,
                np.full((self.nworld,), self.mjd.time, dtype=np.float32),
                dtype=self._wp.float32,
            )

    def load_qpos_batch(self, qpos_batch: np.ndarray) -> None:
        """Copy a batch of qpos frames into the first worlds of Warp data."""
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.mjm.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.mjm.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] > self.nworld:
            raise ValueError(
                f"Batch size {qpos_batch.shape[0]} exceeds runtime world count {self.nworld}"
            )

        self._wp.copy(
            self.d_warp.qpos[: qpos_batch.shape[0]],
            self._wp.from_numpy(qpos_batch, dtype=self._wp.float32),
        )

    def set_ctrl_batch(self, ctrl_batch: np.ndarray) -> None:
        """Copy a control batch into Warp data."""
        ctrl_batch = np.asarray(ctrl_batch, dtype=np.float32)
        if ctrl_batch.ndim != 2 or ctrl_batch.shape != (self.nworld, self.mjm.nu):
            raise ValueError(
                f"Expected ctrl batch shape {(self.nworld, self.mjm.nu)}, got {ctrl_batch.shape}"
            )
        self._copy_field(self.d_warp.ctrl, ctrl_batch, dtype=self._wp.float32)

    def load_qpos_batch_from_device(self, qpos_frames_device, *, start: int, stop: int) -> None:
        """Copy a qpos window from a preloaded device array into Warp data."""
        if start < 0 or stop < start:
            raise ValueError(f"Invalid qpos batch window: start={start}, stop={stop}")

        actual = stop - start
        if actual > self.nworld:
            raise ValueError(f"Batch size {actual} exceeds runtime world count {self.nworld}")
        if len(qpos_frames_device.shape) != 2 or qpos_frames_device.shape[1] != self.mjm.nq:
            raise ValueError(
                f"Expected device qpos frame shape (T, {self.mjm.nq}), got {qpos_frames_device.shape}"
            )

        self._wp.copy(
            self.d_warp.qpos[:actual],
            qpos_frames_device[start:stop],
        )

    def load_state_batch(
        self,
        *,
        qpos: np.ndarray,
        qvel: np.ndarray | None = None,
        ctrl: np.ndarray | None = None,
        act: np.ndarray | None = None,
        mocap_pos: np.ndarray | None = None,
        mocap_quat: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ) -> None:
        """Copy a full batch of simulation state into Warp data."""
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape != (self.nworld, self.mjm.nq):
            raise ValueError(
                f"Expected qpos shape {(self.nworld, self.mjm.nq)}, got {qpos.shape}"
            )
        self._copy_field(self.d_warp.qpos, qpos, dtype=self._wp.float32)

        if qvel is not None and self.mjm.nv > 0:
            qvel = np.asarray(qvel, dtype=np.float32)
            if qvel.shape != (self.nworld, self.mjm.nv):
                raise ValueError(
                    f"Expected qvel shape {(self.nworld, self.mjm.nv)}, got {qvel.shape}"
                )
            self._copy_field(self.d_warp.qvel, qvel, dtype=self._wp.float32)

        if ctrl is not None and self.mjm.nu > 0:
            ctrl = np.asarray(ctrl, dtype=np.float32)
            if ctrl.shape != (self.nworld, self.mjm.nu):
                raise ValueError(
                    f"Expected ctrl shape {(self.nworld, self.mjm.nu)}, got {ctrl.shape}"
                )
            self._copy_field(self.d_warp.ctrl, ctrl, dtype=self._wp.float32)

        if act is not None and self.mjm.na > 0 and hasattr(self.d_warp, "act"):
            act = np.asarray(act, dtype=np.float32)
            if act.shape != (self.nworld, self.mjm.na):
                raise ValueError(
                    f"Expected act shape {(self.nworld, self.mjm.na)}, got {act.shape}"
                )
            self._copy_field(self.d_warp.act, act, dtype=self._wp.float32)

        if mocap_pos is not None and self.mjm.nmocap > 0:
            mocap_pos = np.asarray(mocap_pos, dtype=np.float32)
            expected = (self.nworld, self.mjm.nmocap, 3)
            if mocap_pos.shape != expected:
                raise ValueError(f"Expected mocap_pos shape {expected}, got {mocap_pos.shape}")
            self._copy_field(self.d_warp.mocap_pos, mocap_pos, dtype=self._wp.vec3f)

        if mocap_quat is not None and self.mjm.nmocap > 0:
            mocap_quat = np.asarray(mocap_quat, dtype=np.float32)
            expected = (self.nworld, self.mjm.nmocap, 4)
            if mocap_quat.shape != expected:
                raise ValueError(f"Expected mocap_quat shape {expected}, got {mocap_quat.shape}")
            self._copy_field(self.d_warp.mocap_quat, mocap_quat, dtype=self._wp.quatf)

        if time is not None and hasattr(self.d_warp, "time"):
            time = np.asarray(time, dtype=np.float32)
            if time.shape != (self.nworld,):
                raise ValueError(f"Expected time shape {(self.nworld,)}, got {time.shape}")
            self._copy_field(self.d_warp.time, time, dtype=self._wp.float32)

    def reset_from_mujoco(self) -> None:
        """Reset Warp-side solver state and reload the current MuJoCo state."""
        self._mjw.reset_data(self.m_warp, self.d_warp)
        self.sync_from_mujoco()

    def forward(self) -> None:
        """Run Warp forward kinematics for the current state."""
        self._mjw.forward(self.m_warp, self.d_warp)

    def step(self, *, nstep: int = 1) -> None:
        """Advance every world by ``nstep`` substeps, replaying a captured CUDA graph."""
        if nstep < 1:
            raise ValueError(f"nstep must be >= 1, got {nstep}")
        graph = self._step_graphs.get(nstep)
        if graph is None:
            with self._wp.ScopedCapture() as capture:
                for _ in range(nstep):
                    self._mjw.step(self.m_warp, self.d_warp)
            graph = self._step_graphs[nstep] = capture.graph
        self._wp.capture_launch(graph)


class RendererWrapper:
    """Unified renderer abstraction for MJWarp and Madrona backends."""

    _madrona_created = False

    def __init__(
        self,
        *,
        backend: Literal["mjwarp", "madrona"],
        runtime: WarpReplayRuntime,
        cam_res: tuple[int, int],
        gpu_id: int | None = None,
        camera_names: tuple[str, ...] | None = None,
    ):
        try:
            import mujoco as mj
            import mujoco_warp as mjw
            import torch
            import warp as wp
        except ImportError as exc:
            raise RuntimeError(
                "Sim camera rendering requires optional dependencies: mujoco_warp, warp, and torch."
            ) from exc

        self._mj = mj
        self._mjw = mjw
        self._torch = torch
        self._wp = wp
        self._runtime = runtime
        self.backend_name = backend
        self.mjm = runtime.mjm
        self.m_warp = runtime.m_warp
        self.d_warp = runtime.d_warp
        self.nworld = runtime.nworld
        available_names = tuple(self.mjm.cam(i).name for i in range(self.mjm.ncam))
        self.camera_names = available_names
        camera_options = {}
        if camera_names is not None:
            if backend != "mjwarp":
                raise ValueError("Selecting rendered cameras requires the MJWarp backend")
            missing = set(camera_names) - set(available_names)
            if missing or not camera_names:
                raise ValueError(
                    f"Invalid render camera selection: {camera_names}; missing: {sorted(missing)}"
                )
            self.camera_names = tuple(name for name in available_names if name in camera_names)
            camera_options["cam_active"] = [name in camera_names for name in available_names]
        self.ncam = len(self.camera_names)
        self.width, self.height = cam_res
        self._render_width, self._render_height = cam_res
        self._madrona_crop: tuple[slice, slice] | None = None
        self._madrona_model = self.mjm

        if backend == "madrona":
            if RendererWrapper._madrona_created:
                raise RuntimeError(
                    "Madrona BatchRenderer cannot be created twice in the same process. "
                    "Reuse the existing renderer or switch to MJWarp."
                )
            try:
                from madrona_mjwarp import BatchRenderer
            except ImportError as exc:
                raise RuntimeError("Madrona rendering requires the optional package `madrona_mjwarp`.") from exc

            rank = os.environ.get("LOCAL_RANK", "0")
            os.environ.setdefault("MADRONA_BVH_KERNEL_CACHE", f"/tmp/madrona_bvh_cache_{rank}.bin")
            os.environ.setdefault("MADRONA_MWGPU_KERNEL_CACHE", f"/tmp/madrona_kernel_cache_{rank}.bin")

            if gpu_id is None:
                gpu_id = int(os.environ.get("LOCAL_RANK", 0))

            if hasattr(self._torch, "cuda") and self._torch.cuda.is_available():
                self._torch.cuda.set_device(gpu_id)

            self._configure_madrona_render_model()
            self._madrona = BatchRenderer(
                self._madrona_model,
                self.m_warp,
                gpu_id,
                self.nworld,
                self._render_width,
                self._render_height,
            )
            self._madrona.reset(self.d_warp)
            self._rc = None
            self._bg = self._capture_mjw_background()
            RendererWrapper._madrona_created = True
        elif backend == "mjwarp":
            self._madrona = None
            self._rc = mjw.create_render_context(
                mjm=self.mjm,
                nworld=self.nworld,
                cam_res=cam_res,
                render_rgb=[True] * self.ncam,
                render_depth=[False] * self.ncam,
                use_textures=True,
                use_shadows=True,
                **camera_options,
            )
            self._bg = None
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def _configure_madrona_render_model(self) -> None:
        """Madrona's CUDA raytracer is square-only; adapt rectangular requests."""
        if self.width == self.height:
            self._render_width = self.width
            self._render_height = self.height
            self._madrona_crop = None
            self._madrona_model = self.mjm
            return

        side = max(self.width, self.height)
        self._render_width = side
        self._render_height = side
        self._madrona_model = self.mjm.__copy__()

        if self.width > self.height:
            aspect = self.width / self.height
            vfov = np.deg2rad(np.asarray(self._madrona_model.cam_fovy, dtype=np.float32))
            hfov = 2.0 * np.arctan(np.tan(vfov * 0.5) * aspect)
            self._madrona_model.cam_fovy[:] = np.rad2deg(hfov).astype(np.float32)
            pad = side - self.height
            top = pad // 2
            self._madrona_crop = (slice(top, top + self.height), slice(0, self.width))
        else:
            pad = side - self.width
            left = pad // 2
            self._madrona_crop = (slice(0, self.height), slice(left, left + self.width))

    def _capture_mjw_background(self):
        mjm_bg = self._madrona_model.__copy__() if self._madrona is not None else self.mjm.__copy__()
        mjm_bg.geom_size[:] = 0
        mjm_bg.mesh_vert[:] = 0
        m_bg, d_bg = _put_model_and_data_safe(
            self._mjw,
            mjm_bg,
            self._mj.MjData(mjm_bg),
            nworld=1,
        )
        self._mjw.forward(m_bg, d_bg)
        self._wp.synchronize()
        rc = self._mjw.create_render_context(
            mjm=mjm_bg,
            nworld=1,
            cam_res=(self._render_width, self._render_height),
            render_rgb=[True] * mjm_bg.ncam,
            render_depth=[False] * mjm_bg.ncam,
            use_textures=True,
            use_shadows=True,
        )
        self._mjw.refit_bvh(m_bg, d_bg, rc)
        self._mjw.render(m_bg, d_bg, rc)
        img = (
            self._wp.to_torch(rc.rgb_data)
            .view(self._torch.uint8)
            .reshape(1, self.ncam, self._render_height, self._render_width, 4)
        )
        return img[:, :, :, :, [2, 1, 0]].clone()

    def _crop_madrona_output(self, img):
        if self._madrona_crop is None:
            return img
        y_slice, x_slice = self._madrona_crop
        return img[:, :, y_slice, x_slice, :]

    def _maybe_sync_cuda(self) -> None:
        if hasattr(self._torch, "cuda") and self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def reset(self, *, actual_batch: int = 1):
        """Reset the renderer state for a fresh replay sequence."""
        if self._madrona is not None:
            self._wp.synchronize()
            raw = self._madrona.reset(self.d_warp)
            self._maybe_sync_cuda()
            rgb = raw[:actual_batch, :, :, :, :3]
            depth = self._madrona.madrona.depth_tensor().to_torch()[:actual_batch]
            return self._crop_madrona_output(self._torch.where(depth == 0, self._bg, rgb))
        return self.render(actual_batch=actual_batch)

    def render(self, *, actual_batch: int = 1):
        """Render and return (B, ncam, H, W, 3) uint8 RGB torch tensor."""
        if self._madrona is not None:
            self._wp.synchronize()
            raw = self._madrona.render(self.d_warp)
            self._maybe_sync_cuda()
            rgb = raw[:actual_batch, :, :, :, :3]
            depth = self._madrona.madrona.depth_tensor().to_torch()[:actual_batch]
            return self._crop_madrona_output(self._torch.where(depth == 0, self._bg, rgb))

        self._mjw.refit_bvh(self.m_warp, self.d_warp, self._rc)
        self._mjw.render(self.m_warp, self.d_warp, self._rc)
        self._wp.synchronize_stream()
        img = (
            self._wp.to_torch(self._rc.rgb_data)
            .view(self._torch.uint8)
            .reshape(self.nworld, self.ncam, self.height, self.width, 4)
        )
        return img[:actual_batch, :, :, :, [2, 1, 0]]

    def reset_numpy(self, *, actual_batch: int = 1) -> np.ndarray:
        if self._madrona is None:
            return self._render_numpy_mjwarp(actual_batch=actual_batch)
        return self.reset(actual_batch=actual_batch).cpu().numpy()

    def render_numpy(self, *, actual_batch: int = 1) -> np.ndarray:
        if self._madrona is None:
            return self._render_numpy_mjwarp(actual_batch=actual_batch)
        return self.render(actual_batch=actual_batch).cpu().numpy()

    def _render_numpy_mjwarp(self, *, actual_batch: int) -> np.ndarray:
        self._mjw.refit_bvh(self.m_warp, self.d_warp, self._rc)
        self._mjw.render(self.m_warp, self.d_warp, self._rc)
        rgba = self._rc.rgb_data.numpy().view(np.uint8).reshape(
            self.nworld,
            self.ncam,
            self.height,
            self.width,
            4,
        )
        return rgba[:actual_batch, :, :, :, [2, 1, 0]].copy()

    def close(self) -> None:
        """Release renderer-owned state so another renderer can be created later."""
        if self._madrona is not None:
            self._madrona = None
            RendererWrapper._madrona_created = False
        self._rc = None
        self._bg = None
        self.m_warp = None
        self.d_warp = None
        self.mjm = None
        try:
            self._wp.synchronize()
        except Exception:
            pass
