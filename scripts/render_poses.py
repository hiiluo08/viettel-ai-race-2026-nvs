#!/usr/bin/env python
"""Render Taming-3DGS outputs for camera poses supplied in a COLMAP-style CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from decimal import Decimal, InvalidOperation
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from nvs_utils import REQUIRED_TEST_POSE_COLUMNS, require_directory, require_file, safe_relative_output_path


@dataclass(frozen=True)
class TestPose:
    image_name: str
    qvec: Any
    tvec: Any
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class TamingModules:
    render: Any
    qvec2rotmat: Any
    GaussianModel: Any
    focal2fov: Any
    getWorld2View2: Any


def find_checkpoint_ply(model_path: Path, requested_iteration: int) -> tuple[int, Path]:
    """Select the requested checkpoint or the highest complete numeric iteration."""
    if requested_iteration < -1:
        raise ValueError("requested_iteration must be -1 or nonnegative")

    point_cloud_root = model_path / "point_cloud"
    if requested_iteration >= 0:
        checkpoint = point_cloud_root / f"iteration_{requested_iteration}" / "point_cloud.ply"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint point cloud does not exist: {checkpoint}")
        return requested_iteration, checkpoint

    candidates: list[tuple[int, Path]] = []
    if point_cloud_root.is_dir():
        for iteration_dir in point_cloud_root.glob("iteration_*"):
            try:
                iteration = int(iteration_dir.name.rsplit("_", 1)[1])
            except ValueError:
                continue
            checkpoint = iteration_dir / "point_cloud.ply"
            if checkpoint.is_file():
                candidates.append((iteration, checkpoint))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint point_cloud.ply files found in: {point_cloud_root}"
        )
    return max(candidates, key=lambda candidate: candidate[0])


def output_path_for_pose(output_dir: Path, image_name: str) -> Path:
    """Resolve a pose image name below the output directory without traversal."""
    return output_dir / safe_relative_output_path(image_name)


class TestPoseCamera:
    """Minimal camera object with the attributes required by Taming-3DGS."""

    def __init__(
        self,
        pose: TestPose,
        uid: int,
        modules: TamingModules,
        data_device: str = "cuda",
    ) -> None:
        import numpy as np
        import torch

        self.uid = uid
        self.colmap_id = uid
        self.image_name = pose.image_name
        self.image_width = pose.width
        self.image_height = pose.height
        self.FoVx = modules.focal2fov(pose.fx, pose.width)
        self.FoVy = modules.focal2fov(pose.fy, pose.height)

        # Match the COLMAP convention used by Taming's camera loaders.
        self.R = np.transpose(modules.qvec2rotmat(pose.qvec))
        self.T = pose.tvec
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = torch.tensor(
            modules.getWorld2View2(self.R, self.T), dtype=torch.float32, device=data_device
        ).transpose(0, 1)
        self.projection_matrix = build_projection_matrix(
            width=pose.width,
            height=pose.height,
            fx=pose.fx,
            fy=pose.fy,
            cx=pose.cx,
            cy=pose.cy,
            znear=self.znear,
            zfar=self.zfar,
            device=data_device,
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def build_projection_matrix(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    znear: float,
    zfar: float,
    device: str,
) -> Any:
    """Build a projection matrix using the CSV's direct camera intrinsics."""
    import torch

    left = -cx / fx * znear
    right = (width - cx) / fx * znear
    bottom = -(height - cy) / fy * znear
    top = cy / fy * znear

    matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Taming-3DGS images for every pose in a test-poses CSV."
    )
    parser.add_argument("--taming-root", required=True, help="Taming-3DGS checkout root.")
    parser.add_argument("--model-path", required=True, help="Taming-3DGS model directory.")
    parser.add_argument("--poses-csv", required=True, help="CSV file containing test camera poses.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered images.")
    parser.add_argument(
        "--iteration",
        type=int,
        default=-1,
        help="Checkpoint iteration to load; -1 selects the largest complete iteration.",
    )
    parser.add_argument("--sh-degree", type=int, default=3, help="Gaussian model SH degree.")
    background = parser.add_mutually_exclusive_group()
    background.add_argument("--white-background", action="store_true")
    background.add_argument("--black-background", action="store_true")
    parser.add_argument("--device", default="cuda", help="CUDA device used for rendering.")
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N poses.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def require_taming_root(raw_path: str) -> Path:
    """Require a Taming checkout with the renderer and scene packages."""
    taming_root = require_directory(raw_path, "--taming-root")
    missing = [
        name
        for name in ("scene", "gaussian_renderer")
        if not (taming_root / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "--taming-root is not a Taming-3DGS checkout; missing directories: "
            + ", ".join(missing)
        )
    return taming_root


def resolve_output_dir(raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise FileNotFoundError("--output-dir must name an output directory; received an empty path")
    output_dir = Path(raw_path).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileNotFoundError(f"--output-dir must name a directory: {output_dir}")
    return output_dir


def read_test_poses(csv_path: Path) -> list[TestPose]:
    import numpy as np

    def parse_finite_float(row: dict[str, str], field: str) -> float:
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite; non-finite value")
        return value

    def parse_positive_integer(row: dict[str, str | None], field: str) -> int:
        raw_value = row.get(field)
        if not isinstance(raw_value, str):
            raise ValueError(f"{field} must be a positive integer")
        try:
            value = Decimal(raw_value.strip())
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive integer") from exc
        if not value.is_finite() or value <= 0 or value != value.to_integral_value():
            raise ValueError(f"{field} must be a positive integer")
        return int(value)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED_TEST_POSE_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        poses: list[TestPose] = []
        for row_idx, row in enumerate(reader, start=2):
            try:
                qvec = np.array(
                    [parse_finite_float(row, field) for field in ("qw", "qx", "qy", "qz")],
                    dtype=np.float64,
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    quaternion_norm = np.linalg.norm(qvec)
                if not math.isfinite(float(quaternion_norm)) or quaternion_norm <= 0.0:
                    raise ValueError("quaternion norm must be finite and nonzero")

                fx = parse_finite_float(row, "fx")
                fy = parse_finite_float(row, "fy")
                if fx <= 0:
                    raise ValueError("fx must be positive")
                if fy <= 0:
                    raise ValueError("fy must be positive")

                poses.append(
                    TestPose(
                        image_name=row["image_name"],
                        qvec=qvec,
                        tvec=np.array(
                            [parse_finite_float(row, field) for field in ("tx", "ty", "tz")],
                            dtype=np.float64,
                        ),
                        fx=fx,
                        fy=fy,
                        cx=parse_finite_float(row, "cx"),
                        cy=parse_finite_float(row, "cy"),
                        width=parse_positive_integer(row, "width"),
                        height=parse_positive_integer(row, "height"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Failed to parse CSV row {row_idx}: {exc}") from exc

    if not poses:
        raise ValueError(f"No test poses found in: {csv_path}")
    return poses


def warn_if_principal_point_is_unusual(poses: Iterable[TestPose]) -> None:
    unusual = [
        pose
        for pose in poses
        if not math.isclose(pose.cx, pose.width / 2.0, abs_tol=1e-3)
        or not math.isclose(pose.cy, pose.height / 2.0, abs_tol=1e-3)
    ]
    if unusual:
        print(f"Warning: {len(unusual)} poses use an off-center principal point from the CSV.")


def tensor_to_image(rendered: Any) -> Any:
    from PIL import Image

    array = (
        rendered.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def save_image(image: Any, path: Path, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=jpeg_quality, subsampling=0)
    else:
        image.save(path)


def import_taming_modules(taming_root: Path) -> TamingModules:
    taming_root_string = str(taming_root)
    if taming_root_string not in sys.path:
        sys.path.insert(0, taming_root_string)

    try:
        from gaussian_renderer import render
        from scene.colmap_loader import qvec2rotmat
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import focal2fov, getWorld2View2
    except ImportError as exc:
        raise RuntimeError(f"Could not import Taming-3DGS from --taming-root: {exc}") from exc

    return TamingModules(render, qvec2rotmat, GaussianModel, focal2fov, getWorld2View2)


def build_render_configuration(
    args: argparse.Namespace, modules: Any
) -> tuple[Namespace, Any]:
    pipeline = Namespace(
        separate_sh=True,
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    gaussians = modules.GaussianModel(
        args.sh_degree,
        optimizer_type="default",
        rendering_mode="abs",
    )
    return pipeline, gaussians


def background_color_for_flags(
    white_background: bool, black_background: bool
) -> tuple[float, float, float]:
    if white_background and black_background:
        raise ValueError("Choose only one background override")
    return (1.0, 1.0, 1.0) if white_background else (0.0, 0.0, 0.0)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")
    if not args.device.startswith("cuda"):
        raise ValueError("Taming-3DGS rendering requires a CUDA device.")

    taming_root = require_taming_root(args.taming_root)
    model_path = require_directory(args.model_path, "--model-path")
    poses_csv = require_file(args.poses_csv, "--poses-csv")
    output_dir = resolve_output_dir(args.output_dir)
    iteration, ply_path = find_checkpoint_ply(model_path, args.iteration)
    poses = read_test_poses(poses_csv)
    if args.limit is not None:
        poses = poses[: args.limit]
    output_paths = [output_path_for_pose(output_dir, pose.image_name) for pose in poses]
    warn_if_principal_point_is_unusual(poses)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Taming-3DGS rendering requires an NVIDIA GPU.")
    try:
        render_device = torch.device(args.device)
    except RuntimeError as exc:
        raise ValueError(f"Invalid CUDA device: {args.device}") from exc
    if render_device.type != "cuda":
        raise ValueError("Taming-3DGS rendering requires a CUDA device.")
    if render_device.index is not None:
        torch.cuda.set_device(render_device)

    modules = import_taming_modules(taming_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline, gaussians = build_render_configuration(args, modules)
    gaussians.load_ply(str(ply_path))
    background = torch.tensor(
        background_color_for_flags(args.white_background, args.black_background),
        dtype=torch.float32,
        device=str(render_device),
    )

    from PIL import Image
    from tqdm import tqdm

    rendered_count = 0
    skipped_count = 0
    with torch.no_grad():
        iterator = tqdm(poses, desc="Rendering Taming test poses", disable=args.quiet)
        for idx, (pose, output_path) in enumerate(zip(poses, output_paths)):
            if args.skip_existing and output_path.exists():
                skipped_count += 1
                continue

            camera = TestPoseCamera(pose, uid=idx, modules=modules, data_device=str(render_device))
            rendered = modules.render(camera, gaussians, pipeline, background)["render"]
            image = tensor_to_image(rendered)
            if not isinstance(image, Image.Image) or image.size != (pose.width, pose.height):
                actual_size = getattr(image, "size", None)
                raise RuntimeError(
                    f"Rendered size mismatch for {pose.image_name}: "
                    f"got {actual_size}, expected {(pose.width, pose.height)}"
                )
            save_image(image, output_path, args.jpeg_quality)
            rendered_count += 1

    print(f"Taming-3DGS root: {taming_root}")
    print(f"Model path: {model_path}")
    print(f"Checkpoint iteration: {iteration}")
    print(f"Checkpoint: {ply_path}")
    print(f"Output directory: {output_dir}")
    print(f"Rendered images: {rendered_count}")
    print(f"Skipped existing images: {skipped_count}")
    print("Taming test-pose rendering completed successfully.")


if __name__ == "__main__":
    main()
