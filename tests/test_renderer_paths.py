from __future__ import annotations

import csv
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from render_poses import (
    background_color_for_flags,
    build_projection_matrix,
    build_render_configuration,
    find_checkpoint_ply,
    main,
    output_path_for_pose,
    parse_args,
    read_test_poses,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RENDERER = PROJECT_ROOT / "scripts" / "render_poses.py"


def write_checkpoint(model_path: Path, iteration: int) -> Path:
    checkpoint = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"ply")
    return checkpoint


def test_find_checkpoint_ply_uses_largest_complete_iteration(tmp_path: Path) -> None:
    older = write_checkpoint(tmp_path, 7000)
    write_checkpoint(tmp_path, 30000)
    (tmp_path / "point_cloud" / "iteration_40000").mkdir()

    iteration, checkpoint = find_checkpoint_ply(tmp_path, -1)

    assert iteration == 30000
    assert checkpoint.name == "point_cloud.ply"
    assert older.exists()


def test_find_checkpoint_ply_requires_requested_ply(tmp_path: Path) -> None:
    (tmp_path / "point_cloud" / "iteration_7000").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="point_cloud.ply"):
        find_checkpoint_ply(tmp_path, 7000)


def test_find_checkpoint_ply_rejects_iteration_below_minus_one(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, 7000)

    with pytest.raises(ValueError, match="-1 or nonnegative"):
        find_checkpoint_ply(tmp_path, -2)


def test_output_path_for_pose_rejects_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe relative"):
        output_path_for_pose(tmp_path, "../outside.jpg")


def test_output_path_for_pose_keeps_nested_relative_name(tmp_path: Path) -> None:
    assert output_path_for_pose(tmp_path, "nested/result.jpg") == tmp_path / "nested/result.jpg"


def test_renderer_requires_explicit_paths_and_lists_taming_root() -> None:
    result = subprocess.run(
        [sys.executable, str(RENDERER)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    for option in ("--taming-root", "--model-path", "--poses-csv", "--output-dir"):
        assert option in result.stderr

    help_result = subprocess.run(
        [sys.executable, str(RENDERER), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--taming-root" in help_result.stdout


POSE_FIELDS = [
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
]
DEFAULT_POSE = {
    "image_name": "view.png",
    "qw": "1",
    "qx": "0",
    "qy": "0",
    "qz": "0",
    "tx": "0",
    "ty": "0",
    "tz": "0",
    "fx": "400",
    "fy": "300",
    "cx": "320",
    "cy": "240",
    "width": "640",
    "height": "480",
}


def write_pose_csv(path: Path, **updates: str) -> None:
    row = {**DEFAULT_POSE, **updates}
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=POSE_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_build_projection_matrix_uses_off_center_csv_intrinsics_on_cpu() -> None:
    matrix = build_projection_matrix(
        width=640,
        height=480,
        fx=400,
        fy=300,
        cx=300,
        cy=200,
        znear=0.01,
        zfar=100.0,
        device="cpu",
    )

    assert tuple(matrix.shape) == (4, 4)
    assert matrix[0, 0].item() == pytest.approx(1.25)
    assert matrix[1, 1].item() == pytest.approx(1.25)
    assert matrix[0, 2].item() == pytest.approx(0.0625)
    assert matrix[1, 2].item() == pytest.approx(-1.0 / 6.0)
    assert matrix[3, 2].item() == pytest.approx(1.0)


def test_background_color_for_flags_is_cpu_only() -> None:
    assert background_color_for_flags(False, False) == (0.0, 0.0, 0.0)
    assert background_color_for_flags(False, True) == (0.0, 0.0, 0.0)
    assert background_color_for_flags(True, False) == (1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="only one background"):
        background_color_for_flags(True, True)


def test_read_test_poses_accepts_valid_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "poses.csv"
    write_pose_csv(csv_path)

    poses = read_test_poses(csv_path)

    assert len(poses) == 1
    assert poses[0].image_name == "view.png"
    assert poses[0].width == 640
    assert poses[0].height == 480
    assert poses[0].fx == pytest.approx(400.0)


def test_read_test_poses_rejects_missing_trailing_dimension(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing-height.csv"
    csv_path.write_text(
        ",".join(POSE_FIELDS)
        + "\nview.png,1,0,0,0,0,0,0,400,300,320,240,640\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="CSV row 2.*height must be a positive integer"):
        read_test_poses(csv_path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"tx": "nan"}, "non-finite"),
        ({"fx": "0"}, "fx must be positive"),
        ({"fy": "-1"}, "fy must be positive"),
        ({"width": "640.5"}, "width must be a positive integer"),
        ({"width": "640.00000000000001"}, "width must be a positive integer"),
        ({"height": "0"}, "height must be a positive integer"),
        ({"qw": "0", "qx": "0", "qy": "0", "qz": "0"}, "quaternion norm"),
        ({"qw": "1e308"}, "quaternion norm"),
    ],
)
def test_read_test_poses_rejects_invalid_row_with_row_number(
    tmp_path: Path, updates: dict[str, str], message: str
) -> None:
    csv_path = tmp_path / "invalid-poses.csv"
    write_pose_csv(csv_path, **updates)

    with pytest.raises(ValueError, match=rf"CSV row 2.*{message}"):
        read_test_poses(csv_path)


def test_main_rejects_nonpositive_limit_before_files_or_cuda() -> None:
    with pytest.raises(ValueError, match="--limit must be positive"):
        main(
            [
                "--taming-root",
                "missing-taming",
                "--model-path",
                "missing-model",
                "--poses-csv",
                "missing.csv",
                "--output-dir",
                "output",
                "--limit",
                "0",
            ]
        )


def test_main_rejects_invalid_jpeg_quality_before_files_or_cuda() -> None:
    with pytest.raises(ValueError, match="--jpeg-quality must be between 1 and 100"):
        main(
            [
                "--taming-root",
                "missing-taming",
                "--model-path",
                "missing-model",
                "--poses-csv",
                "missing.csv",
                "--output-dir",
                "output",
                "--jpeg-quality",
                "101",
            ]
        )


def test_parse_args_rejects_conflicting_backgrounds() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(
            [
                "--taming-root",
                "taming",
                "--model-path",
                "model",
                "--poses-csv",
                "poses.csv",
                "--output-dir",
                "output",
                "--white-background",
                "--black-background",
            ]
        )

    assert exc_info.value.code == 2


def test_build_render_configuration_uses_abs_mode_and_separate_sh() -> None:
    class FakeGaussianModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    fake_modules = SimpleNamespace(GaussianModel=FakeGaussianModel)
    args = Namespace(sh_degree=7)

    pipeline, gaussians = build_render_configuration(args, fake_modules)

    assert vars(pipeline) == {
        "separate_sh": True,
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
    }
    assert gaussians.args == (7,)
    assert gaussians.kwargs == {
        "optimizer_type": "default",
        "rendering_mode": "abs",
    }
