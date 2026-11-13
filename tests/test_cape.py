"""Geometry, image alignment, and pretrained-checkpoint regression tests for CaPE."""

import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from abc_minimal.cape import (
    FrozenBatchNorm2d,
    PluckerLateFusion,
    mujoco_camera_geometry,
    observation_plucker,
    plucker_raymap,
    validate_camera_matrices,
)
from abc_minimal.config import DiTConfig
from abc_minimal.camera_geometry import CameraGeometry
from abc_minimal.camera_parameters import (
    clear_camera_parameter_caches,
    episode_plucker,
    exported_intrinsics,
    load_camera_directions,
    load_camera_parameters,
    prepare_camera_parameters,
)
from abc_minimal.camera_rays import (
    cached_camera_directions,
    camera_calibration,
    camera_direction_map,
    plucker_from_directions,
)
from abc_minimal.dataloader import EpisodeDataset, collate
from abc_minimal.dit import DiTPolicy, load_pretrained
from abc_minimal.preprocess import augment_and_normalize
from abc_minimal.train_loop import batch_to_device


def tiny_config(use_cape=False):
    return DiTConfig(hidden_size=32, depth=1, num_heads=4, mlp_ratio=2,
                     vit_embed_dim=32, vit_depth=1, vit_num_heads=4,
                     vision_pool_num_queries=2, vision_pool_num_heads=4,
                     chunk_length=4, task_embed_dim=8, use_cape=use_cape)


def calibration():
    k = torch.tensor([[24., 0, 16], [0, 24, 16], [0, 0, 1]])
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([.3, -.2, 1.])
    return k, pose


def batch(config):
    k, pose = calibration()
    return {"state": torch.randn(1, 14), "actions": torch.randn(1, 4, 14),
            "task_vec_clip": torch.randn(1, 8),
            "images": {cam: torch.randn(1, 3, 32, 32) for cam in config.camera_keys},
            "plucker": {cam: plucker_raymap(k, pose, 32, 32)[None] for cam in config.camera_keys}}


class GeometryTests(unittest.TestCase):
    def test_direction_and_moment_match_analytic_world_line(self):
        k = torch.tensor([[2., 0, .5], [0, 3., .5], [0, 0, 1.]])
        pose = torch.eye(4)
        pose[:3, 3] = torch.tensor([2., 3., 4.])
        rays = plucker_raymap(k, pose, 1, 1)[:, 0, 0]
        torch.testing.assert_close(rays[3:], torch.tensor([0., 0., 1.]))
        torch.testing.assert_close(rays[:3], torch.tensor([3., -2., 0.]))
        validate_camera_matrices(k, pose)

    def test_batched_rotated_cameras_match_individual_rays(self):
        k, pose = calibration()
        other = pose.clone()
        other[:3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        poses = torch.stack([pose, other])
        rays = plucker_raymap(k, poses, 11, 15)
        expected = torch.stack([plucker_raymap(k, p, 11, 15) for p in poses])
        torch.testing.assert_close(rays, expected)
        torch.testing.assert_close(rays[:, 3:].norm(dim=1), torch.ones(2, 11, 15))
        torch.testing.assert_close((rays[:, :3] * rays[:, 3:]).sum(1), torch.zeros(2, 11, 15), atol=1e-6, rtol=0)

    def test_rays_match_official_opencv_snippet(self):
        # Golden values generated with the unmodified CamPose README snippet,
        # reordered from its [d,m] HWC output to ACT's [m,d] CHW layout.
        k = torch.tensor([[2., 0, 1.], [0, 3., .5], [0, 0, 1.]])
        pose = torch.tensor([[0., -1, 0, .3], [1, 0, 0, -.2],
                             [0, 0, 1, 1], [0, 0, 0, 1]])
        expected = torch.tensor([
            [.048507124, -.291042775, -.072760694, 0., -.242535636, .970142543],
            [-.436564147, -.291042775, .072760694, 0., .242535636, .970142543],
            [-.760000050, -.240000010, .180000007, 0., .600000024, .800000012],
            [.046153840, -.584615350, -.130769223, -.307692289, -.230769217, .923076868],
            [-.415384591, -.584615350, .007692310, -.307692289, .230769217, .923076868],
            [-.734338522, -.489558995, .122389764, -.257662624, .579740942, .772987902],
        ]).T.reshape(6, 2, 3)
        torch.testing.assert_close(plucker_raymap(k, pose, 2, 3), expected)

    def test_invalid_calibration_is_rejected(self):
        k, pose = calibration()
        pose[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "right-handed"):
            validate_camera_matrices(k, pose)
        pose = torch.eye(4)
        k[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_camera_matrices(k, pose)

    def test_rotation_crop_and_padding_keep_rgb_and_rays_aligned(self):
        k, pose = calibration()
        k[0, 2] = k[1, 2] = .5
        pose[:3, 3] = 0
        rays = plucker_raymap(k, pose, 40, 52)
        # RGB contains the direction field itself, so identical spatial sampling
        # must preserve it up to homogeneous ray renormalization.
        image = rays[3:].clone()
        with patch("torch.rand", side_effect=[torch.tensor([.75]), torch.tensor([.5]),
                                               torch.tensor([.5]), torch.tensor([.5])]):
            images, maps = augment_and_normalize({"top": image}, True, norm_preset=None,
                                                 image_size=32, plucker_maps={"top": rays})
        torch.testing.assert_close(F.normalize(images["top"], dim=0), maps["top"][3:], atol=1e-6, rtol=1e-5)

    def test_validation_and_inference_spatial_preprocessing_agree(self):
        k, pose = calibration()
        image = torch.rand(3, 24, 32)
        rays = plucker_raymap(k, pose, 24, 32)
        _, expected = augment_and_normalize({"top": image}, False, plucker_maps={"top": rays})
        obs = {"state": np.zeros((2, 14)), "images": {"top": image[None].expand(2, -1, -1, -1)},
               "camera_intrinsics": {"top": k}, "camera_to_world": {"top": pose}}
        actual = observation_plucker(obs, ("top",), "cpu")["top"]
        torch.testing.assert_close(actual, expected["top"][None].expand(2, -1, -1, -1))

    def test_mujoco_axes_and_moving_camera_pose(self):
        import mujoco

        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
          <joint name="slide" type="slide" axis="1 0 0"/>
          <geom type="sphere" size=".1" mass="1"/>
          <camera name="top" pos="0 0 1" fovy="60"/>
        </body></worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        first = mujoco_camera_geometry(model, data, ("top",), 3, 3)
        ray = plucker_raymap(first["camera_intrinsics"]["top"], first["camera_to_world"]["top"], 3, 3)
        torch.testing.assert_close(ray[3:, 1, 1], torch.tensor([0., 0., -1.]))
        data.qpos[0] = .7
        mujoco.mj_forward(model, data)
        second = mujoco_camera_geometry(model, data, ("top",), 3, 3)
        self.assertAlmostEqual(float(second["camera_to_world"]["top"][0, 3]), .7, places=6)
        self.assertEqual(float(first["camera_to_world"]["top"][0, 3]), 0)


class FrozenBatchNormTests(unittest.TestCase):
    def test_loads_standard_batchnorm_checkpoint_with_prefix_and_custom_epsilon(self):
        for eps in (1e-5, 1e-3):
            with self.subTest(eps=eps):
                reference = torch.nn.Sequential(torch.nn.BatchNorm2d(3, eps=eps)).eval()
                with torch.no_grad():
                    reference[0].weight.copy_(torch.tensor([.7, 1.1, .9]))
                    reference[0].bias.copy_(torch.tensor([.1, 0., -.2]))
                    reference[0].running_mean.copy_(torch.tensor([.2, -.4, .7]))
                    reference[0].running_var.copy_(torch.tensor([.5, 1.5, 2.]))
                    reference[0].num_batches_tracked.fill_(7)
                frozen = torch.nn.Sequential(FrozenBatchNorm2d(3, eps=eps))
                frozen.load_state_dict(reference.state_dict(), strict=True)
                x = torch.randn(2, 3, 4, 5)
                torch.testing.assert_close(frozen(x), reference(x))


class DistortionTests(unittest.TestCase):
    def camera(self, model="inverse_brown_conrady"):
        return {"width": 4, "height": 3, "K": [[2.3, 0, 1.4], [0, 2.2, .9], [0, 0, 1]],
                "distortion_model": model, "D": [-.05, .01, .001, -.002, -.001], "pixel_center": 0.}

    def test_inverse_brown_matches_sdk_scalar_golden_directions(self):
        # Golden directions from the scalar librealsense rs.cpp 10-iteration
        # INVERSE_BROWN_CONRADY update at native integer pixel centers.
        directions = camera_direction_map(self.camera())
        expected = torch.tensor([[-.497687637917, -.335566385485, .799813863344],
                                 [-.171244463851, .044741872159, .984212120671],
                                 [.540247376584, .386809843947, .747335879454]])
        actual = torch.stack([directions[:, v, u] for u, v in ((0, 0), (1, 1), (3, 2))])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_corrected_directions_project_back_to_original_distorted_pixels(self):
        # Check inverse results with the independent forward projection model,
        # including tangential distortion, rather than duplicating the solver.
        for model in ("brown_conrady", "inverse_brown_conrady", "plumb_bob"):
            with self.subTest(model=model):
                camera = self.camera(model)
                directions = camera_direction_map(camera).double().numpy()
                x, y = directions[:2] / directions[2:]
                k1, k2, p1, p2, k3 = camera["D"]
                r2 = x*x + y*y
                radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
                xf, yf = x*radial, y*radial
                tx, ty = (xf, yf) if model == "inverse_brown_conrady" else (x, y)
                px = xf + 2*p1*tx*ty + p2*(r2 + 2*tx*tx)
                py = yf + 2*p2*tx*ty + p1*(r2 + 2*ty*ty)
                u, v = np.meshgrid(np.arange(4), np.arange(3))
                np.testing.assert_allclose(px*2.3 + 1.4, u, atol=1e-6)
                np.testing.assert_allclose(py*2.2 + .9, v, atol=1e-6)

    def test_zero_distortion_preserves_existing_pinhole_convention(self):
        k, pose = calibration()
        camera = {"height": 32, "width": 32, "K": k.tolist(),
                  "distortion_model": "inverse_brown_conrady", "D": [0.] * 5,
                  "pixel_center": .5}
        actual = plucker_from_directions(camera_direction_map(camera), pose)
        torch.testing.assert_close(actual, plucker_raymap(k, pose, 32, 32))

    def test_invalid_distortion_is_rejected_instead_of_ignored(self):
        for change in ({"distortion_model": "fisheye"}, {"D": [0.]},
                       {"D": [float("nan")] * 5}, {"distortion_model": "none"},
                       {"pixel_center": .25}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                camera_direction_map({**self.camera(), **change})
        camera = self.camera()
        del camera["D"]
        with self.assertRaisesRegex(ValueError, "requires five"):
            camera_calibration(camera)

    def test_table_cache_reuses_calibration_but_separates_coefficients_and_grid(self):
        camera = self.camera()
        first = cached_camera_directions(camera, 24, 32)
        self.assertIs(first, cached_camera_directions(camera, 24, 32))
        changed = {**camera, "D": [-.02, .01, .001, -.002, -.001]}
        self.assertFalse(torch.allclose(first, cached_camera_directions(changed, 24, 32)))
        self.assertEqual(cached_camera_directions(camera, 12, 16).shape, (3, 12, 16))

    def test_export_bicubic_matches_pillow_and_preserves_line_geometry(self):
        _, pose = calibration()
        pose[:3, :3] = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        camera = {**self.camera(), "height": 48, "width": 64,
                  "K": [[50, 0, 32], [0, 50, 24], [0, 0, 1]]}
        raw = plucker_from_directions(camera_direction_map(camera), pose)
        # Independent Pillow reference, including all six line coordinates.
        resized = np.stack([np.asarray(Image.fromarray(channel.numpy()).resize(
            (32, 24), Image.Resampling.BICUBIC)) for channel in raw])
        expected = F.pad(torch.from_numpy(resized), (0, 0, 4, 4))
        expected = expected / expected[3:].norm(dim=0, keepdim=True).clamp_min(1e-8)
        actual = plucker_from_directions(cached_camera_directions(camera, 32, 32), pose)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-5)
        torch.testing.assert_close(actual[3:, 4:28].norm(dim=0), torch.ones(24, 32))
        torch.testing.assert_close((actual[:3] * actual[3:]).sum(0), torch.zeros(32, 32),
                                   atol=1e-7, rtol=0)
        self.assertEqual(float(actual[:, :4].abs().sum()), 0.)
        self.assertEqual(float(actual[:, 28:].abs().sum()), 0.)

    def test_distorted_inference_uses_same_table_and_handles_batched_intrinsics(self):
        from abc_minimal.cape import resize_pad_plucker

        k, pose = calibration()
        camera = {**self.camera(), "height": 24, "width": 32, "K": k.tolist()}
        expected = resize_pad_plucker(plucker_from_directions(camera_direction_map(camera), pose)[None])[0]
        obs = {"state": np.zeros((2, 14)), "images": {"top": torch.rand(2, 3, 24, 32)},
               "camera_intrinsics": {"top": k[None].expand(2, -1, -1)},
               "camera_to_world": {"top": pose},
               "camera_distortion": {"top": {"distortion_model": camera["distortion_model"], "D": camera["D"],
                                              "pixel_center": camera["pixel_center"]}}}
        actual = observation_plucker(obs, ("top",), "cpu")["top"]
        torch.testing.assert_close(actual, expected[None].expand(2, -1, -1, -1))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = DiTPolicy(tiny_config())
        self.cape = DiTPolicy(tiny_config(True))
        self.temp = tempfile.TemporaryDirectory()
        self.parent = Path(self.temp.name) / "parent.pt"
        torch.save({"model": self.base.state_dict()}, self.parent)
        load_pretrained(self.cape, self.parent, initialize_cape=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_identity_fusion_preserves_tokens_sampling_and_rtc(self):
        self.base.eval()
        self.cape.eval()
        data = batch(self.cape.config)
        torch.testing.assert_close(self.base.build_vision_tokens(data["images"]),
                                   self.cape.build_vision_tokens(data["images"], data["plucker"]))
        noise = torch.randn(1, 4, 14)
        torch.testing.assert_close(self.base.sample_actions(data, num_steps=2, noise=noise),
                                   self.cape.sample_actions(data, num_steps=2, noise=noise))
        prefix = torch.randn_like(noise)
        a = self.base.sample_actions_rtc(data, prefix, 2, num_steps=2, noise=noise)
        b = self.cape.sample_actions_rtc(data, prefix, 2, num_steps=2, noise=noise)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(b[:, :2], prefix[:, :2], atol=0, rtol=0)
        self.assertEqual(self.cape.build_vision_tokens(data["images"], data["plucker"]).shape, (1, 6, 32))

    def test_geometry_branch_receives_gradients_during_finetuning(self):
        data = batch(self.cape.config)
        noise = torch.randn_like(data["actions"])
        t = torch.full((1, 1, 1), .5)
        optimizer = torch.optim.AdamW(self.cape.parameters(), lr=1e-3)
        loss = self.cape(data, noise=noise, t=t)
        loss.backward()
        self.assertGreater(self.cape.cape.fusion.weight.grad[:, 32:].abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        self.cape(data, noise=noise, t=t).backward()
        self.assertGreater(self.cape.cape.plucker_encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.cape.vision_camera_embed.weight.grad.abs().sum().item(), 0)
        zero_rays = {cam: torch.zeros_like(value) for cam, value in data["plucker"].items()}
        a = self.cape.build_vision_tokens(data["images"], data["plucker"])
        b = self.cape.build_vision_tokens(data["images"], zero_rays)
        self.assertGreater((a - b).abs().max().item(), 1e-6)

    def test_geometry_grid_matches_abc_patch_resolution_directly(self):
        # Exercise ABC's real input/width as well as the smaller test model.
        with torch.no_grad():
            for embed_dim, height, width in ((DiTConfig().vit_embed_dim, 224, 224), (32, 32, 48)):
                with self.subTest(embed_dim=embed_dim, image_size=(height, width)):
                    fusion = PluckerLateFusion(embed_dim)
                    rays = torch.randn(1, 6, height, width)
                    grid = (height // 16, width // 16)
                    features = fusion.plucker_encoder(rays)
                    self.assertEqual(features.shape, (1, embed_dim, *grid))
                    tokens = torch.randn(1, 1 + grid[0] * grid[1], embed_dim)
                    fused = fusion(tokens, rays, grid)
                    torch.testing.assert_close(fused, tokens)

    def test_mismatched_geometry_grid_is_rejected(self):
        rays = torch.randn(1, 6, 32, 32)
        tokens = torch.randn(1, 2, 32)
        with self.assertRaisesRegex(ValueError, "geometry grid.*does not match image patch grid"):
            self.cape.cape(tokens, rays, (1, 1))

    def test_checkpoint_initialization_is_explicit_and_strict(self):
        with self.assertRaisesRegex(RuntimeError, "missing model keys"):
            load_pretrained(self.cape, self.parent)
        broken = self.base.state_dict()
        del broken["y_embedder.weight"]
        path = Path(self.temp.name) / "broken.pt"
        torch.save({"model": broken}, path)
        with self.assertRaisesRegex(RuntimeError, "missing model keys"):
            load_pretrained(self.cape, path, initialize_cape=True)
        partial = self.base.state_dict()
        partial["cape.fusion.bias"] = self.cape.cape.fusion.bias.detach().clone()
        torch.save({"model": partial}, path)
        with self.assertRaisesRegex(RuntimeError, "missing model keys"):
            load_pretrained(self.cape, path, initialize_cape=True)
        torch.save({"model": self.cape.state_dict()}, path)
        load_pretrained(self.cape, path)
        with self.assertRaisesRegex(RuntimeError, "unexpected checkpoint keys"):
            load_pretrained(self.base, path)

    def test_cape_never_silently_omits_geometry(self):
        data = batch(self.cape.config)
        with self.assertRaisesRegex(ValueError, "requires image-aligned"):
            self.cape.build_vision_tokens(data["images"])
        with self.assertRaisesRegex(ValueError, "requires per-camera"):
            observation_plucker({"state": np.zeros(14), "images": data["images"]},
                                self.cape.camera_keys, "cpu")


class FixedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        models = root / "abc_sim/models"
        models.mkdir(parents=True)
        arms = []
        for side in ("left", "right"):
            arm = f'<body name="{side}_arm">'
            for i in range(1, 7):
                arm += (f'<body name="{side}_link_{i}" pos=".1 0 0">'
                        f'<joint name="{side}_joint{i}" axis="0 0 1"/>'
                        '<geom type="sphere" size=".02" mass="1"/>')
            arms.append(arm + f'<camera name="{side}" pos=".02 .03 .04"/>' + '</body>' * 7)
        xml = ('<mujoco><worldbody><camera name="top" pos="0 -1 1"/>'
               + ''.join(arms) + '</worldbody></mujoco>')
        for name in ("yam_dishrack_base.xml", "original.xml"):
            (models / name).write_text(xml)
        top_mount = np.eye(4)
        top_mount[:3, 3] = [0, -.4, .8]
        wrist_mount = np.eye(4)
        wrist_mount[:3, 3] = [.02, .03, .04]
        cameras = {}
        for name, body, mount in (("top", "left_arm", top_mount),
                                  ("left-wrist", "left_link_6", wrist_mount),
                                  ("right-wrist", "right_link_6", wrist_mount)):
            cameras[name] = {"reference_body": body, "T_reference_from_camera": mount.tolist(),
                             "recorded_intrinsics": {"width": 640, "height": 480,
                                                      "K": [400, 0, 320, 0, 400, 240, 0, 0, 1]}}
        left_base, right_base = np.eye(4), np.eye(4)
        left_base[:3, 3] = [.2, .1, 0]
        right_base[:3, 3] = [-.2, -.1, 0]
        record = {"episode": "episode_test", "model_name": "original.xml",
                  "calibration_quality": "accepted", "cameras": cameras,
                  "base_pose": {"base_poses": {"left": left_base.tolist(), "right": right_base.tolist()}},
                  "time_offset": {"state_time_offset_s": 1000, "path": "must_not_be_opened.npz"}}
        path = root / "calibrations.json.gz"
        with gzip.open(path, "wt") as stream:
            json.dump({"records": [record]}, stream)
        with patch("abc_minimal.camera_geometry.ROOT", root):
            self.geometry = CameraGeometry(path, models)
        self.episode = root / "episode_test"

    def tearDown(self):
        clear_camera_parameter_caches()
        self.temp.cleanup()

    def test_real_intrinsics_reproject_to_rgb_pixels_before_and_after_resize(self):
        cameras, _ = self.geometry.camera_parameters(self.episode, np.zeros(14), {})
        camera = cameras["top"]
        raw_k = torch.tensor([[400., 0, 320], [0, 400., 240], [0, 0, 1.]], dtype=torch.float64)
        expected_k = raw_k.numpy().copy()
        expected_k[:2, 2] += .5
        np.testing.assert_array_equal(camera["K"], expected_k)
        # Repeated access must not accumulate shifts or modify the source file.
        again, _ = self.geometry.camera_parameters(self.episode, np.zeros(14), {})
        np.testing.assert_array_equal(again["top"]["K"], expected_k)
        with gzip.open(self.episode.parent / "calibrations.json.gz", "rt") as stream:
            original = json.load(stream)["records"][0]["cameras"]["top"]["recorded_intrinsics"]
        np.testing.assert_array_equal(np.asarray(original["K"]).reshape(3, 3), raw_k.numpy())

        for height, width in ((480, 640), (224, 224)):
            k, _ = exported_intrinsics(camera, height, width)
            rays = plucker_raymap(k, torch.eye(4), height, width)
            scale = width / 640
            pad_top = (height - round(480 * scale)) // 2
            for row, col in ((pad_top, 0), (height // 2, width // 2),
                             (height - pad_top - 1, width - 1)):
                direction = rays[3:, row, col].double()
                pixel = raw_k @ (direction / direction[2])
                # Inverse of RGB's pixel-center resize and vertical letterbox.
                expected = torch.tensor([(col + .5) / scale - .5,
                                         (row + .5 - pad_top) / scale - .5], dtype=torch.float64)
                torch.testing.assert_close(pixel[:2], expected, atol=5e-5, rtol=1e-6)


    def test_old_pixel_convention_cache_requires_explicit_regeneration(self):
        self.episode.mkdir()
        np.zeros((1, 28)).tofile(self.episode / "states_actions.bin")
        prepare_camera_parameters(self.episode, self.geometry)
        path = self.episode / "camera_parameters.json"
        metadata = json.loads(path.read_text())
        self.assertEqual(metadata["version"], 4)
        metadata["version"] = 3  # Bilinear direction tables must not be reused.
        path.write_text(json.dumps(metadata))
        load_camera_parameters.cache_clear()
        with self.assertRaisesRegex(ValueError, "regenerate with --overwrite"):
            load_camera_parameters(str(self.episode.resolve()))
        with self.assertRaisesRegex(ValueError, "regenerate with --overwrite"):
            prepare_camera_parameters(self.episode, self.geometry)
        self.assertTrue(prepare_camera_parameters(self.episode, self.geometry, overwrite=True))
        self.assertEqual(load_camera_parameters(str(self.episode.resolve()))[0]["version"], 4)


    def test_fixed_mounts_use_current_state_and_ignore_temporal_fields(self):
        state = np.zeros(14)
        record, first = self.geometry.camera_poses(self.episode, state, {})
        self.assertNotIn("time_offset", record)
        metadata = {"state_time_offset_s": -500, "t0_ns": 9999999,
                    "top_camera_timestamps_ns": [100, 200], "source_frame_range": [900, 1000]}
        _, same = self.geometry.camera_poses(self.episode, state, metadata)
        for cam in first:
            np.testing.assert_array_equal(first[cam], same[cam])
        changed = state.copy()
        changed[0] = np.pi / 2
        _, moved = self.geometry.camera_poses(self.episode, changed, metadata)
        np.testing.assert_array_equal(first["top"], moved["top"])
        np.testing.assert_array_equal(first["right"], moved["right"])
        self.assertFalse(np.allclose(first["left"], moved["left"]))
        # First joint is at x=.1; its remaining five links and fixed mount
        # rotate with that joint, independently of every temporal field.
        expected = np.array([[0., -1, 0, .1], [1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        expected[:3, 3] += expected[:3, :3] @ np.array([.52, .03, .04])
        anchor = self.geometry.world_from_top @ np.linalg.inv(
            np.asarray(record["cameras"]["top"]["T_reference_from_camera"]))
        np.testing.assert_allclose(moved["left"], anchor @ expected, atol=1e-12)

    def test_retargeted_state_uses_target_bases_without_time_lookup(self):
        bases = np.stack([np.eye(4), np.eye(4)])
        bases[:, 0, 3] = [.3, -.3]
        conversion = {"source_episode": "episode_test", "camera_from_base":
                      (np.linalg.inv(self.geometry.world_from_top) @ bases).tolist()}
        metadata = {"sequence_ik": conversion}
        state = np.zeros(14)
        state[7] = .4
        _, poses = self.geometry.camera_poses(self.episode, state, metadata)
        metadata["sequence_ik"]["state_time_offset_s"] = 1000
        _, same = self.geometry.camera_poses(self.episode, state, metadata)
        for cam in poses:
            np.testing.assert_array_equal(poses[cam], same[cam])
        # At zero left joint angles the six .1m links plus fixed mount
        # are added to the target left base, rather than the original base.
        np.testing.assert_allclose(poses["left"][:3, 3], [.92, .03, .04], atol=1e-12)

    def test_sim_fixed_mounts_use_current_state_and_export_letterbox(self):
        episode = self.episode.parent / "episode_sim"
        episode.mkdir()
        images = {cam: torch.zeros(3, 224, 224) for cam in ("top", "left", "right")}
        rows = np.zeros((2, 28))
        rows[1, 0] = .4
        rows.tofile(episode / "states_actions.bin")
        (episode / "episode_metadata.json").write_text(json.dumps({
            "task_name": "sim_test",
            "scene_xml_file": None,
            "source_camera_resolutions": {cam: [640, 480] for cam in images}}))
        # A mixed real/sim preprocessing job may also have real-camera D loaded.
        for camera in self.geometry.calibrations["episode_test"]["cameras"].values():
            camera["recorded_intrinsics"].update(
                distortion_model="inverse_brown_conrady", D=[-.05, .01, .001, -.002, -.001])
        prepare_camera_parameters(episode, self.geometry)
        metadata, _ = load_camera_parameters(str(episode.resolve()))
        for cam in images:
            # Sim K remains in the renderer's original half-pixel convention.
            np.testing.assert_array_equal(np.asarray(metadata["cameras"][cam]["K"])[:2, 2], [320., 240.])
            self.assertEqual(metadata["cameras"][cam]["distortion_model"], "none")
            self.assertEqual(metadata["cameras"][cam]["D"], [0.] * 5)
            camera = metadata["cameras"][cam]
            expected = plucker_raymap(torch.tensor(camera["K"]), torch.eye(4), 480, 640)[3:]
            torch.testing.assert_close(camera_direction_map(camera), expected)
        with patch("abc_minimal.camera_geometry.StateFK.update", side_effect=AssertionError("Training must not run FK")):
            first = episode_plucker(episode, 0, rows[0, :14], 2, images)
            moved = episode_plucker(episode, 1, rows[1, :14], 2, images)
        for cam in ("top", "right"):
            torch.testing.assert_close(first[cam], moved[cam], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["left"], moved["left"]))
        self.assertEqual(float(first["top"][:, :28].abs().sum()), 0.)
        torch.testing.assert_close(first["top"][3:, 28:196].norm(dim=0), torch.ones(168, 224))

    def test_sim_uses_each_actual_source_resolution(self):
        episode = self.episode.parent / "episode_sim_sizes"
        episode.mkdir()
        sizes = {"top": [1280, 960], "left": [640, 480], "right": [1024, 768]}
        cameras, _ = self.geometry.sim_camera_parameters(episode, np.zeros(14), {
            "source_camera_resolutions": sizes})
        for cam, (width, height) in sizes.items():
            camera = cameras[cam]
            self.assertEqual((camera["width"], camera["height"]), (width, height))
            self.assertEqual(camera["K"][0][2], width / 2)
            self.assertEqual(camera["K"][1][2], height / 2)
            focal = height / (2 * np.tan(np.deg2rad(45.) / 2))
            self.assertAlmostEqual(camera["K"][1][1], focal)
        with self.assertRaisesRegex(ValueError, "sim source image size missing"):
            self.geometry.sim_camera_parameters(episode, np.zeros(14), {})

    def test_sim_non_square_export_grid_is_preserved_until_joint_model_padding(self):
        episode = self.episode.parent / "episode_sim_224x168"
        episode.mkdir()
        rows = np.zeros((2, 28))
        rows.tofile(episode / "states_actions.bin")
        # The published sim format can carry image_width/image_height without
        # camera_resolutions; neither source nor exported grid is 224-square.
        (episode / "episode_metadata.json").write_text(json.dumps({
            "task_name": "sim_test", "image_width": 224, "image_height": 168}))
        prepare_camera_parameters(episode, self.geometry)
        metadata, _ = load_camera_parameters(str(episode.resolve()))
        self.assertEqual(metadata["ray_image_size"], [168, 224])
        self.assertEqual(metadata["cameras"]["top"]["height"], 168)
        images = {cam: torch.zeros(3, 168, 224) for cam in ("top", "left", "right")}
        rays = episode_plucker(episode, 0, rows[0, :14], 2, images)
        self.assertEqual(rays["top"].shape, (6, 168, 224))
        _, processed = augment_and_normalize(images, False, plucker_maps=rays)
        self.assertEqual(processed["top"].shape, (6, 224, 224))
        self.assertEqual(float(processed["top"][:, :28].abs().sum()), 0.)
        self.assertEqual(float(processed["top"][:, 196:].abs().sum()), 0.)
        torch.testing.assert_close(processed["top"][3:, 28:196].norm(dim=0), torch.ones(168, 224))

    def test_real_source_rgb_must_match_calibration_resolution(self):
        self.episode.mkdir()
        np.zeros((2, 28)).tofile(self.episode / "states_actions.bin")
        (self.episode / "episode_metadata.json").write_text(json.dumps({
            "source_camera_resolutions": {"top": [1280, 960]}}))
        with self.assertRaisesRegex(ValueError, "source RGB size.*differs from calibration"):
            prepare_camera_parameters(self.episode, self.geometry)

    def test_offline_real_parameters_match_fk_and_are_saved_beside_states(self):
        self.episode.mkdir()
        rows = np.zeros((3, 28))
        rows[:, 0] = [0, .2, .4]
        rows.tofile(self.episode / "states_actions.bin")
        (self.episode / "episode_metadata.json").write_text(json.dumps({"state_time_offset_s": 1000}))
        expected = [self.geometry.camera_parameters(self.episode, state, {})[1] for state in rows[:, :14]]
        self.assertTrue(prepare_camera_parameters(self.episode, self.geometry))
        metadata, frames = load_camera_parameters(str(self.episode.resolve()))
        self.assertIsInstance(frames, np.memmap)
        self.assertEqual(metadata["camera_keys"], ["top", "left", "right"])
        self.assertTrue((self.episode / "camera_parameters.json").is_file())
        np.testing.assert_array_equal(frames["state"], rows[:, :14])
        for i, poses in enumerate(expected):
            for j, cam in enumerate(metadata["camera_keys"]):
                np.testing.assert_allclose(frames["camera_to_world"][i, j], poses[cam], atol=1e-7)
        with patch.object(self.geometry, "camera_parameters", side_effect=AssertionError("Do not repeat FK")):
            self.assertFalse(prepare_camera_parameters(self.episode, self.geometry))

    def test_changed_source_states_require_regeneration(self):
        self.episode.mkdir()
        rows = np.zeros((2, 28))
        source = self.episode / "states_actions.bin"
        rows.tofile(source)
        prepare_camera_parameters(self.episode, self.geometry)
        rows[0, 0] = .4
        rows.tofile(source)
        images = {cam: torch.zeros(3, 224, 224) for cam in ("top", "left", "right")}
        with self.assertRaisesRegex(ValueError, "cached state differs"):
            episode_plucker(self.episode, 0, rows[0, :14], 2, images)
        with self.assertRaisesRegex(ValueError, "source state changed"):
            prepare_camera_parameters(self.episode, self.geometry)
        self.assertTrue(prepare_camera_parameters(self.episode, self.geometry, overwrite=True))
        rays = episode_plucker(self.episode, 0, rows[0, :14], 2, images)
        self.assertEqual(rays["top"].shape, (6, 224, 224))
        with self.assertRaisesRegex(ValueError, "frame count differs"):
            episode_plucker(self.episode, 0, rows[0, :14], 3, images)

    def test_interrupted_write_is_not_treated_as_complete(self):
        self.episode.mkdir()
        np.zeros((2, 28)).tofile(self.episode / "states_actions.bin")
        prepare_camera_parameters(self.episode, self.geometry)
        replace = Path.replace

        def interrupt_manifest(path, target):
            if path.name == "camera_parameters.json.tmp":
                raise OSError("Interrupted manifest write")
            return replace(path, target)

        with patch.object(Path, "replace", interrupt_manifest):
            with self.assertRaisesRegex(OSError, "Interrupted"):
                prepare_camera_parameters(self.episode, self.geometry, overwrite=True)
        self.assertFalse((self.episode / "camera_parameters.json").exists())
        self.assertTrue(prepare_camera_parameters(self.episode, self.geometry))

    def test_distortion_is_saved_offline_and_current_pose_updates_rays(self):
        self.episode.mkdir()
        rows = np.zeros((2, 28))
        rows[1, 0] = .4
        rows.tofile(self.episode / "states_actions.bin")
        models = {"top": ("inverse_brown_conrady", [-.05, .01, .001, -.002, -.001]),
                  "left-wrist": ("none", [0.] * 5),
                  "right-wrist": ("brown_conrady", [.02, -.01, -.001, .002, .001])}
        for view, (model, coefficients) in models.items():
            self.geometry.calibrations["episode_test"]["cameras"][view]["recorded_intrinsics"].update(
                distortion_model=model, D=coefficients)
        prepare_camera_parameters(self.episode, self.geometry)
        metadata, _ = load_camera_parameters(str(self.episode.resolve()))
        self.assertEqual(metadata["version"], 4)
        self.assertEqual(metadata["cameras"]["top"]["D"], [-.05, .01, .001, -.002, -.001])
        self.assertEqual(metadata["cameras"]["top"]["pixel_center"], .5)
        for cam, view in (("top", "top"), ("left", "left-wrist"), ("right", "right-wrist")):
            self.assertEqual(metadata["cameras"][cam]["distortion_model"], models[view][0])
            self.assertEqual(metadata["cameras"][cam]["D"], models[view][1])
        table = np.load(self.episode / "camera_directions.npy")
        self.assertEqual(table.shape, (3, 3, 224, 224))
        self.assertEqual(table.dtype, np.float32)
        self.assertFalse(np.allclose(table[0], table[1]))
        self.assertFalse(np.allclose(table[1], table[2]))
        images = {cam: torch.zeros(3, 224, 224) for cam in ("top", "left", "right")}
        clear_camera_parameter_caches()
        first = episode_plucker(self.episode, 0, rows[0, :14], 2, images)
        moved = episode_plucker(self.episode, 1, rows[1, :14], 2, images)
        torch.testing.assert_close(first["top"], moved["top"], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["left"], moved["left"]))
        with self.assertRaisesRegex(ValueError, "image size does not match"):
            episode_plucker(self.episode, 0, rows[0, :14], 2, {"top": torch.zeros(3, 32, 32)})

    def test_old_format_or_missing_direction_table_requires_regeneration(self):
        self.episode.mkdir()
        np.zeros((2, 28)).tofile(self.episode / "states_actions.bin")
        prepare_camera_parameters(self.episode, self.geometry)
        path = self.episode / "camera_parameters.json"
        metadata = json.loads(path.read_text())
        metadata["version"] = 2
        path.write_text(json.dumps(metadata))
        load_camera_parameters.cache_clear()
        with self.assertRaisesRegex(ValueError, "regenerate with --overwrite"):
            load_camera_parameters(str(self.episode.resolve()))
        prepare_camera_parameters(self.episode, self.geometry, overwrite=True)
        (self.episode / "camera_directions.npy").unlink()
        load_camera_parameters.cache_clear()
        with self.assertRaisesRegex(ValueError, "camera parameters missing.*camera_directions"):
            load_camera_parameters(str(self.episode.resolve()))

    def test_episodes_with_same_calibration_share_one_loaded_direction_table(self):
        self.episode.mkdir()
        np.zeros((2, 28)).tofile(self.episode / "states_actions.bin")
        prepare_camera_parameters(self.episode, self.geometry)
        other = self.episode.with_name("episode_other")
        shutil.copytree(self.episode, other)
        clear_camera_parameter_caches()
        with patch("numpy.load", wraps=np.load) as reader:
            first = load_camera_directions(str(self.episode.resolve()))
            second = load_camera_directions(str(other.resolve()))
        self.assertIs(first, second)
        direction_reads = [call for call in reader.call_args_list
                           if Path(call.args[0]).name == "camera_directions.npy"]
        self.assertEqual(len(direction_reads), 1)


class DatasetTests(unittest.TestCase):
    def test_cached_parameters_match_raw_state_without_training_fk(self):
        with tempfile.TemporaryDirectory() as folder:
            episode = Path(folder) / "episode_test"
            episode.mkdir()
            config = tiny_config(True)
            rows = np.zeros((8, 28), dtype=np.float64)
            rows[:, :14] = np.arange(8)[:, None] * .1 + 1.
            rows.tofile(episode / "states_actions.bin")
            (episode / "episode_metadata.json").write_text(json.dumps({
                "task_name": "test", "cameras": config.camera_keys, "state_time_offset_s": 1000,
                "camera_resolutions": {cam: [32, 32] for cam in config.camera_keys}}))
            stats = {key: {"mean": np.full(14, .5), "std": np.full(14, 2.)} for key in ("state", "actions")}
            k, pose = calibration()
            fk_states = []

            def parameters(ep_dir, state, metadata):
                fk_states.append(state.copy())
                current_pose = pose.clone()
                current_pose[0, 3] = float(state[0])
                cameras = {cam: {"K": k.numpy(), "height": 32, "width": 32} for cam in config.camera_keys}
                return cameras, {cam: current_pose.numpy() for cam in config.camera_keys}

            with self.assertRaisesRegex(ValueError, "camera parameters missing"):
                EpisodeDataset(folder, stats, True, "test", 1., config)
            geometry = Mock()
            geometry.camera_parameters.side_effect = parameters
            prepare_camera_parameters(episode, geometry)
            np.testing.assert_array_equal(fk_states, rows[:, :14])
            dataset = EpisodeDataset(folder, stats, True, "test", 1., config)
            # Deliberately perturb and mask model inputs; neither augmentation
            # changes the physical camera pose encoded from the dataset row.
            dataset.state_noise_std = np.full((1, 14), .1)
            images = {cam: torch.rand(3, 32, 32) for cam in config.camera_keys}
            with patch("abc_minimal.dataloader.decode_frame", return_value=images), \
                 patch("abc_minimal.camera_geometry.StateFK.update", side_effect=AssertionError("Training must not run FK")):
                samples = [dataset[0], dataset[1]]
            self.assertEqual(geometry.camera_parameters.call_count, len(rows))
            self.assertTrue(all(sample["state_is_masked"] for sample in samples))
            self.assertEqual(float(samples[0]["state"].abs().sum()), 0.)
            self.assertFalse(torch.equal(samples[0]["plucker"]["top"], samples[1]["plucker"]["top"]))
            data = collate(samples, config.camera_keys)
            self.assertEqual(data["plucker"]["top"].shape, (2, 6, 224, 224))
            class Embedder:
                def encode(self, prompts):
                    return torch.zeros(len(prompts), 8)
            moved = batch_to_device(data, "cpu", Embedder(), config.camera_keys)
            self.assertIn("plucker", moved)


if __name__ == "__main__":
    unittest.main()
