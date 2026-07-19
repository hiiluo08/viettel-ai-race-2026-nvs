import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_taming_utils():
    sys.path.insert(0, str(PROJECT_ROOT))

    renderer_module = types.ModuleType("gaussian_renderer")
    renderer_module.render = None
    sys.modules["gaussian_renderer"] = renderer_module

    fused_ssim_module = types.ModuleType("fused_ssim")
    fused_ssim_module.fused_ssim = None
    sys.modules["fused_ssim"] = fused_ssim_module

    torchvision_module = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.ToPILImage = None
    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module

    sys.modules.pop("utils.taming_utils", None)
    return importlib.import_module("utils.taming_utils")


class DummyGaussians:
    def __init__(self):
        self.get_xyz = torch.zeros((4, 3), dtype=torch.float32)
        self.xyz_gradient_accum = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        self.denom = torch.ones((4, 1), dtype=torch.float32)
        self.get_opacity = torch.ones((4, 1), dtype=torch.float32)
        self.get_scaling = torch.ones((4, 3), dtype=torch.float32)


class DummyScene:
    def __init__(self):
        self.gaussians = DummyGaussians()


class DummyImage:
    def cuda(self):
        return torch.ones((3, 2, 2), dtype=torch.float32)


class DummyCamera:
    def __init__(self, index, weight):
        self.index = index
        self.weight = weight
        self.original_image = DummyImage()


class ComputeGaussianScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taming_utils = load_taming_utils()

    def test_accumulates_scores_without_per_camera_matrix(self):
        scene = DummyScene()
        cameras = [DummyCamera(0, 1.0), DummyCamera(1, 2.0)]
        visibility_masks = [
            torch.tensor([[0], [2]], dtype=torch.long),
            torch.tensor([[1], [2]], dtype=torch.long),
        ]

        def fake_render(camera, _gaussians, _pipe, _bg, pixel_weights=None):
            if pixel_weights is None:
                return {"render": torch.ones((3, 2, 2), dtype=torch.float32)}

            return {
                "accum_weights": torch.ones(4, dtype=torch.float32),
                "accum_dist": torch.ones(4, dtype=torch.float32),
                "accum_blend": torch.ones(4, dtype=torch.float32),
                "accum_count": torch.ones(4, dtype=torch.float32),
                "visibility_filter": visibility_masks[camera.index],
                "gaussian_depths": torch.ones(4, dtype=torch.float32),
                "gaussian_radii": torch.ones(4, dtype=torch.float32),
            }

        importance_values = {
            "grad_importance": 1.0,
            "opac_importance": 0.0,
            "dept_importance": 0.0,
            "radii_importance": 0.0,
            "scale_importance": 0.0,
            "dist_importance": 0.0,
            "loss_importance": 0.0,
            "count_importance": 0.0,
            "blend_importance": 0.0,
            "view_importance": 1.0,
        }
        expected_legacy_matrix = torch.tensor(
            [[0.5, 0.0, 1.5, 0.0], [0.0, 2.0, 3.0, 0.0]],
            dtype=torch.float32,
        )
        expected_score = expected_legacy_matrix.sum(dim=0)

        allocation_shapes = []
        allocation_devices = []
        original_zeros = self.taming_utils.torch.zeros

        def cpu_zeros(*shape, **kwargs):
            allocation_shapes.append(shape[0] if len(shape) == 1 else shape)
            allocation_devices.append(kwargs["device"])
            kwargs["device"] = "cpu"
            return original_zeros(*shape, **kwargs)

        with mock.patch.object(self.taming_utils, "render", side_effect=fake_render), mock.patch.object(
            self.taming_utils,
            "compute_photometric_loss",
            side_effect=lambda camera, _image: torch.tensor(camera.weight),
        ), mock.patch.object(
            self.taming_utils,
            "get_loss_map",
            return_value=torch.ones((2, 2), dtype=torch.float32),
        ), mock.patch.object(self.taming_utils.torch, "zeros", side_effect=cpu_zeros):
            score = self.taming_utils.compute_gaussian_score(
                scene,
                cameras,
                [DummyImage(), DummyImage()],
                scene.gaussians,
                pipe=None,
                bg=None,
                importance_values=importance_values,
                opt=None,
            )

        torch.testing.assert_close(score, expected_score)
        self.assertEqual(allocation_shapes, [4])
        self.assertEqual(allocation_devices, ["cuda"])


if __name__ == "__main__":
    unittest.main()
