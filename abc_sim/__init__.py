"""abc-sim: Standalone MuJoCo simulation for the YAM bimanual robot."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from abc_sim.env import MuJoCoYAMEnv


def __getattr__(name: str):
    if name == "MuJoCoYAMEnv":
        from abc_sim.env import MuJoCoYAMEnv

        return MuJoCoYAMEnv

    if name in {"get_i2rt_sim_config", "get_i2rt_config", "RobotSystemConfig"}:
        from abc_sim.config import RobotSystemConfig, get_i2rt_config, get_i2rt_sim_config

        return {
            "get_i2rt_sim_config": get_i2rt_sim_config,
            "get_i2rt_config": get_i2rt_config,
            "RobotSystemConfig": RobotSystemConfig,
        }[name]

    if name in {"RandomizationSamplingError", "RandomizationState", "TASK_RANDOMIZERS"}:
        from abc_sim.randomization import (
            RandomizationSamplingError,
            RandomizationState,
            TASK_RANDOMIZERS,
        )

        return {
            "RandomizationSamplingError": RandomizationSamplingError,
            "RandomizationState": RandomizationState,
            "TASK_RANDOMIZERS": TASK_RANDOMIZERS,
        }[name]

    if name in {"SimTaskSpec", "get_task_spec", "list_task_specs", "maybe_get_task_spec"}:
        from abc_sim.task_specs import (
            SimTaskSpec,
            get_task_spec,
            list_task_specs,
            maybe_get_task_spec,
        )

        return {
            "SimTaskSpec": SimTaskSpec,
            "get_task_spec": get_task_spec,
            "list_task_specs": list_task_specs,
            "maybe_get_task_spec": maybe_get_task_spec,
        }[name]

    if name in {"TaskEvalResult", "TaskEvaluator", "make_task_evaluator"}:
        from abc_sim.task_eval import TaskEvalResult, TaskEvaluator, make_task_evaluator

        return {
            "TaskEvalResult": TaskEvalResult,
            "TaskEvaluator": TaskEvaluator,
            "make_task_evaluator": make_task_evaluator,
        }[name]

    if name in {"ConveyorPickRuntime", "TaskRuntime", "make_task_runtime"}:
        from abc_sim.task_runtime import ConveyorPickRuntime, TaskRuntime, make_task_runtime

        return {
            "ConveyorPickRuntime": ConveyorPickRuntime,
            "TaskRuntime": TaskRuntime,
            "make_task_runtime": make_task_runtime,
        }[name]

    if name in {
        "DEFAULT_SCENE_XML",
        "ResolvedTask",
        "SCENE_XMLS",
        "SceneTaskSpec",
        "get_scene_task_spec",
        "get_task_physics_defaults",
        "get_task_randomizer",
        "get_task_scene_xml",
        "list_scene_task_names",
        "list_scene_tasks",
        "maybe_get_scene_task_spec",
        "resolve_env_task_name",
        "resolve_task",
    }:
        from abc_sim.task_registry import (
            DEFAULT_SCENE_XML,
            ResolvedTask,
            SCENE_XMLS,
            SceneTaskSpec,
            get_scene_task_spec,
            get_task_physics_defaults,
            get_task_randomizer,
            get_task_scene_xml,
            list_scene_task_names,
            list_scene_tasks,
            maybe_get_scene_task_spec,
            resolve_env_task_name,
            resolve_task,
        )

        return {
            "DEFAULT_SCENE_XML": DEFAULT_SCENE_XML,
            "ResolvedTask": ResolvedTask,
            "SCENE_XMLS": SCENE_XMLS,
            "SceneTaskSpec": SceneTaskSpec,
            "get_scene_task_spec": get_scene_task_spec,
            "get_task_physics_defaults": get_task_physics_defaults,
            "get_task_randomizer": get_task_randomizer,
            "get_task_scene_xml": get_task_scene_xml,
            "list_scene_task_names": list_scene_task_names,
            "list_scene_tasks": list_scene_tasks,
            "maybe_get_scene_task_spec": maybe_get_scene_task_spec,
            "resolve_env_task_name": resolve_env_task_name,
            "resolve_task": resolve_task,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_env(
    task: str = "bottles",
    render_cameras: bool = True,
    camera_backend: str = "mujoco",
    camera_gpu_id: int | None = None,
    prompt: str | None = None,
    chunk_dim: int = 30,
    wrist_fov: float = 58.0,
    scene_xml: str | Path | None = None,
    scene_xml_string: str | None = None,
    scene_xml_transform_options=None,
    enable_task_randomizer: bool = True,
    max_episode_steps: int | None = None,
    terminate_on_success: bool = False,
    **kwargs,
) -> "MuJoCoYAMEnv":
    """Create a MuJoCo YAM bimanual environment.

    Args:
        task: Task scene — "bottles" (default), "inhand_transfer", etc.
        render_cameras: Whether to render camera images in observations.
        prompt: Task prompt string included in observations. Defaults to the
            resolved task prompt when available.
        chunk_dim: Number of timesteps per action chunk.
        wrist_fov: Vertical field-of-view in degrees for the wrist cameras
            ("left" and "right"). Default 58 matches the real D405 mounting.
        scene_xml: Optional explicit XML file path for standard scene tasks.
        scene_xml_string: Optional in-memory XML override for standard scene
            tasks. Mutually exclusive with ``scene_xml``.
        scene_xml_transform_options: Optional runtime XML transform options for
            model-swapping tasks such as ``inhand_transfer``.
        enable_task_randomizer: Whether to attach and prepare the task
            randomizer. Replay/export callers with a fully assembled scene XML
            should disable this so the randomizer cannot reload the model.
        max_episode_steps: Optional single-action step horizon. When reached,
            ``step`` returns ``truncated=True``.
        terminate_on_success: Whether ``step`` should return ``terminated=True``
            when the task evaluator reports success.
        **kwargs: Additional kwargs passed to MuJoCoYAMEnv.

    Returns:
        Configured MuJoCoYAMEnv instance.
    """
    import mujoco

    from abc_sim.config import get_i2rt_sim_config
    from abc_sim.env import MuJoCoYAMEnv
    from abc_sim.task_registry import (
        get_task_physics_defaults,
        get_task_randomizer,
        get_task_scene_xml,
        list_scene_task_names,
        resolve_task,
    )

    if scene_xml is not None and scene_xml_string is not None:
        raise ValueError("scene_xml and scene_xml_string are mutually exclusive")

    resolved_task = resolve_task(task)
    env_task = resolved_task.env_task or task
    resolved_prompt = prompt
    if resolved_prompt is None:
        if resolved_task.task_spec is not None:
            resolved_prompt = resolved_task.task_spec.prompt
        elif isinstance(task, str) and task:
            resolved_prompt = task.replace("_", " ")
        else:
            resolved_prompt = "fold the towel"

    if env_task == "inhand_transfer":
        import numpy as _np

        from abc_sim.randomization import (
            InHandTransferRandomizer,
            _INHAND_CATEGORIES,
            _OBJ_Z,
            _X_MAX,
            _X_MIN,
            _Y_LEFT_MAX,
            _Y_LEFT_MIN,
            _inhand_apply_scene_transforms,
            _inhand_build_xml,
            _inhand_get_variants,
        )

        if scene_xml is not None or scene_xml_string is not None:
            raise ValueError("scene_xml overrides are not supported for task='inhand_transfer'")

        config = get_i2rt_sim_config()
        seed = kwargs.pop("seed", None)
        rng = _np.random.default_rng(seed)

        categories = _INHAND_CATEGORIES
        category = categories[int(rng.integers(0, len(categories)))]
        variants = _inhand_get_variants(category)
        variant_dir = variants[int(rng.integers(0, len(variants)))]
        x = float(rng.uniform(_X_MIN, _X_MAX))
        y = float(rng.uniform(_Y_LEFT_MIN, _Y_LEFT_MAX))
        yaw = float(rng.uniform(-_np.pi, _np.pi))
        xml = _inhand_build_xml(category, variant_dir, x, y, _OBJ_Z, yaw)
        xml = _inhand_apply_scene_transforms(xml, scene_xml_transform_options)

        physics_kwargs = {
            **kwargs,
            **get_task_physics_defaults(env_task),
            "max_episode_steps": max_episode_steps,
            "terminate_on_success": terminate_on_success,
        }
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            camera_backend=camera_backend,
            camera_gpu_id=camera_gpu_id,
            prompt=resolved_prompt,
            chunk_dim=chunk_dim,
            scene_xml=None,
            **physics_kwargs,
        )
        env.reload_from_xml(xml)
        env.set_task(task)
        env._inhand_category = category
        env._inhand_variant = variant_dir.name

        # Bind a randomizer so XdofSimNode can call it on every reset.
        randomizer = InHandTransferRandomizer(
            scene_xml_transform_options=scene_xml_transform_options,
        )
        randomizer.bind_env(env)
        randomizer._rng = rng
        env._task_randomizer = randomizer
        return env

    scene_xml_path = Path(scene_xml) if scene_xml is not None else get_task_scene_xml(env_task)
    if scene_xml_path is None:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(list_scene_task_names())}"
        )

    tmp_scene_xml_path: Path | None = None
    if scene_xml_string is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=f".{env_task}_",
            dir=scene_xml_path.parent,
            delete=False,
        ) as handle:
            handle.write(scene_xml_string)
            tmp_scene_xml_path = Path(handle.name)
        scene_xml_path = tmp_scene_xml_path

    config = get_i2rt_sim_config()
    physics_kwargs = {
        **kwargs,
        **get_task_physics_defaults(env_task),
        "max_episode_steps": max_episode_steps,
        "terminate_on_success": terminate_on_success,
    }
    try:
        env = MuJoCoYAMEnv(
            config=config,
            render_cameras=render_cameras,
            camera_backend=camera_backend,
            camera_gpu_id=camera_gpu_id,
            prompt=resolved_prompt,
            chunk_dim=chunk_dim,
            scene_xml=scene_xml_path,
            **physics_kwargs,
        )
    finally:
        if tmp_scene_xml_path is not None:
            tmp_scene_xml_path.unlink(missing_ok=True)
    env.set_task(task)
    env._scene_xml_transform_options = scene_xml_transform_options
    if scene_xml_string is not None:
        env._scene_xml_string = scene_xml_string

    if wrist_fov != 58.0:
        import math

        # Wider FOV causes the frustum edges to clip into the gripper geometry.
        tan_baseline = math.tan(math.radians(58.0 / 2))
        tan_new = math.tan(math.radians(wrist_fov / 2))
        clearance_offset = max(0.0, (tan_new - tan_baseline) / tan_baseline) * 0.025
        for cam_name in ("left", "right"):
            cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                env.model.cam_fovy[cam_id] = wrist_fov
                env.model.cam_pos[cam_id][1] -= clearance_offset

    base_randomizer = get_task_randomizer(env_task) if enable_task_randomizer else None
    if base_randomizer is not None:
        randomizer = base_randomizer.clone()
        randomizer.bind_env(env)
        env._task_randomizer = randomizer
        prepare_env = getattr(randomizer, "prepare_env", None)
        if callable(prepare_env):
            prepare_env()

    return env


def make_batched_env(
    task: str,
    *,
    num_worlds: int,
    prompt: str | None = None,
    camera_height: int = 480,
    camera_width: int = 640,
    camera_gpu_id: int | None = None,
    render_camera_names: tuple[str, ...] | None = None,
    **kwargs,
):
    """Create ``num_worlds`` copies of a task scene stepped together in MJWarp.

    The CPU env built by :func:`make_env` still owns the task's randomizer and
    resets each world; physics and camera rendering run batched on the GPU.
    """
    from abc_sim.batched_env import BatchedWarpYAMEnv

    base_env = make_env(task=task, prompt=prompt, render_cameras=False, **kwargs)
    return BatchedWarpYAMEnv(
        base_env,
        num_worlds=num_worlds,
        camera_height=camera_height,
        camera_width=camera_width,
        gpu_id=camera_gpu_id,
        render_camera_names=render_camera_names,
    )


__all__ = [
    "MuJoCoYAMEnv",
    "make_env",
    "make_batched_env",
    "get_i2rt_sim_config",
    "get_i2rt_config",
    "RobotSystemConfig",
    "RandomizationSamplingError",
    "RandomizationState",
    "TASK_RANDOMIZERS",
    "SimTaskSpec",
    "get_task_spec",
    "list_task_specs",
    "maybe_get_task_spec",
    "TaskEvalResult",
    "TaskEvaluator",
    "make_task_evaluator",
    "ConveyorPickRuntime",
    "TaskRuntime",
    "make_task_runtime",
    "DEFAULT_SCENE_XML",
    "ResolvedTask",
    "SCENE_XMLS",
    "SceneTaskSpec",
    "get_scene_task_spec",
    "get_task_physics_defaults",
    "get_task_randomizer",
    "get_task_scene_xml",
    "list_scene_task_names",
    "list_scene_tasks",
    "maybe_get_scene_task_spec",
    "resolve_env_task_name",
    "resolve_task",
]
