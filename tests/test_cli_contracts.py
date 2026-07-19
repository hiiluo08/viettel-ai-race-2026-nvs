from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_inspector_requires_explicit_paths() -> None:
    result = run_script("inspect_dataset.py")

    assert result.returncode == 2
    assert "--data-root" in result.stderr
    assert "--output-dir" in result.stderr


def test_colmap_verifier_requires_explicit_paths() -> None:
    result = run_script("verify_colmap_scene.py")

    assert result.returncode == 2
    assert "--data-root" in result.stderr
    assert "--output-file" in result.stderr
    assert "--taming-root" in result.stderr


def test_inspector_help_uses_hyphenated_options() -> None:
    result = run_script("inspect_dataset.py", "--help")

    assert result.returncode == 0
    assert "--data-root" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--data_root" not in result.stdout


def test_inspector_reports_invalid_scene_path_types(tmp_path: Path) -> None:
    scene_dir = tmp_path / "set" / "scene"
    (scene_dir / "train").mkdir(parents=True)
    (scene_dir / "train" / "images").write_text("not a directory", encoding="utf-8")
    (scene_dir / "train" / "sparse").mkdir()
    (scene_dir / "train" / "sparse" / "0").write_text("not a directory", encoding="utf-8")
    (scene_dir / "test").mkdir()
    (scene_dir / "test" / "images").write_text("not a directory", encoding="utf-8")
    (scene_dir / "test" / "test_poses.csv").mkdir()

    output_dir = tmp_path / "output"
    result = run_script(
        "inspect_dataset.py",
        "--data-root",
        str(tmp_path),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0
    assert (output_dir / "dataset_manifest_phase1.json").is_file()
    assert "Missing train/images" in result.stdout
    assert "Missing test/images" in result.stdout
    assert "Missing test_poses.csv" in result.stdout


def test_verifier_reports_train_images_file(tmp_path: Path) -> None:
    taming_root = tmp_path / "taming"
    (taming_root / "scene").mkdir(parents=True)
    (taming_root / "gaussian_renderer").mkdir()
    (taming_root / "scene" / "__init__.py").write_text("", encoding="utf-8")
    (taming_root / "scene" / "colmap_loader.py").write_text(
        "def read_intrinsics_binary(path): return {}\n"
        "def read_extrinsics_binary(path): return {}\n"
        "def read_points3D_binary(path): return ([], [], [])\n",
        encoding="utf-8",
    )

    scene_dir = tmp_path / "data" / "set" / "scene"
    (scene_dir / "train").mkdir(parents=True)
    (scene_dir / "train" / "images").write_text("not a directory", encoding="utf-8")
    sparse_dir = scene_dir / "train" / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse_dir / name).write_bytes(b"")

    output_file = tmp_path / "report.md"
    result = run_script(
        "verify_colmap_scene.py",
        "--data-root",
        str(tmp_path / "data"),
        "--output-file",
        str(output_file),
        "--taming-root",
        str(taming_root),
    )

    assert result.returncode == 0
    assert output_file.is_file()
    assert "train/images is missing or not a directory" in output_file.read_text(
        encoding="utf-8"
    )


def test_verifier_reports_malformed_pose_numbers(tmp_path: Path) -> None:
    taming_root = tmp_path / "taming"
    (taming_root / "scene").mkdir(parents=True)
    (taming_root / "gaussian_renderer").mkdir()
    (taming_root / "scene" / "__init__.py").write_text("", encoding="utf-8")
    (taming_root / "scene" / "colmap_loader.py").write_text(
        "def read_intrinsics_binary(path): return {}\n"
        "def read_extrinsics_binary(path): return {}\n"
        "def read_points3D_binary(path): return ([], [], [])\n",
        encoding="utf-8",
    )

    scene_dir = tmp_path / "data" / "set" / "scene"
    sparse_dir = scene_dir / "train" / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse_dir / name).write_bytes(b"")
    (scene_dir / "test").mkdir()
    (scene_dir / "test" / "test_poses.csv").write_text(
        "image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n"
        "view.png,1,0,0,0,0,0,0,not-a-number,2,3,4,640,480\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "report.md"
    result = run_script(
        "verify_colmap_scene.py",
        "--data-root",
        str(tmp_path / "data"),
        "--output-file",
        str(output_file),
        "--taming-root",
        str(taming_root),
    )

    assert result.returncode == 0
    assert output_file.is_file()
    assert "invalid numeric" in output_file.read_text(encoding="utf-8")


def test_verifier_rejects_output_directory_before_import(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "set" / "scene").mkdir(parents=True)
    taming_root = tmp_path / "taming"
    (taming_root / "scene").mkdir(parents=True)
    (taming_root / "gaussian_renderer").mkdir()
    output_directory = tmp_path / "report-directory"
    output_directory.mkdir()

    result = run_script(
        "verify_colmap_scene.py",
        "--data-root",
        str(data_root),
        "--output-file",
        str(output_directory),
        "--taming-root",
        str(taming_root),
    )

    assert result.returncode == 2
    assert "--output-file" in result.stderr


def test_evaluator_requires_taming_and_output_paths() -> None:
    result = run_script("evaluate_predictions.py")

    assert result.returncode == 2
    for option in ("--taming-root", "--pred-images", "--gt-images", "--output-dir"):
        assert option in result.stderr


def test_display_script_requires_images_dir() -> None:
    result = run_script("show_rendered_images.py")

    assert result.returncode == 2
    assert "--images-dir" in result.stderr


def test_taming_train_requires_source_and_model_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "external" / "taming-3dgs" / "train.py")],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    required_checks = ["-s/--source_path is required", "-m/--model_path is required"]
    assert any(check in result.stderr for check in required_checks)


def test_verifier_reports_missing_colmap_loader(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "set" / "scene").mkdir(parents=True)
    taming_root = tmp_path / "taming"
    (taming_root / "scene").mkdir(parents=True)
    (taming_root / "gaussian_renderer").mkdir()
    output_file = tmp_path / "report.md"

    result = run_script(
        "verify_colmap_scene.py",
        "--data-root",
        str(data_root),
        "--output-file",
        str(output_file),
        "--taming-root",
        str(taming_root),
    )

    assert result.returncode == 2
    assert "colmap_loader" in result.stderr
