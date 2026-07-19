import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from nvs_utils import (
    IMAGE_SUFFIXES,
    REQUIRED_TEST_POSE_COLUMNS,
    iter_dataset_scenes,
    require_directory,
)


IMAGE_EXTENSIONS = IMAGE_SUFFIXES
EXPECTED_EXTRA_SPARSE_FILES = {"frames.bin", "rigs.bin", "points3D.ply"}


@dataclass
class CameraSummary:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]
    fx: Optional[float]
    fy: Optional[float]
    cx: Optional[float]
    cy: Optional[float]


@dataclass
class SceneReport:
    set_name: str
    scene_name: str
    scene_dir: Path
    status: str = "PASS"
    cameras_count: int = 0
    registered_images_count: int = 0
    sparse_points_count: int = 0
    train_images_count: int = 0
    registered_images_with_files: int = 0
    registered_images_missing_files: int = 0
    train_images_missing_registration: int = 0
    camera_summaries: List[CameraSummary] = field(default_factory=list)
    sparse_files: List[str] = field(default_factory=list)
    tolerated_extra_files: List[str] = field(default_factory=list)
    unexpected_extra_files: List[str] = field(default_factory=list)
    test_pose_count: int = 0
    test_pose_resolution: str = "N/A"
    test_pose_intrinsics: str = "N/A"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        if self.status == "PASS":
            self.status = "WARN"

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "FAIL"


def require_taming_root(raw_path: str) -> Path:
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


def import_colmap_loader(taming_root: Path):
    taming_root_string = str(taming_root)
    if taming_root_string not in sys.path:
        sys.path.insert(0, taming_root_string)
    from scene import colmap_loader

    return colmap_loader


def image_files(images_dir: Path) -> Dict[str, Path]:
    if not images_dir.is_dir():
        return {}
    return {
        item.name: item
        for item in images_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    }


def camera_to_pinhole_values(camera) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    params = list(map(float, camera.params))
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = params[:3]
        return focal, focal, cx, cy
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    if camera.model in {"SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}:
        focal, cx, cy = params[:3]
        return focal, focal, cx, cy
    if camera.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        return fx, fy, cx, cy
    return None, None, None, None


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def format_params(params: List[float]) -> str:
    return "[" + ", ".join(format_float(value, 6) for value in params) + "]"


def read_test_pose_summary(test_csv_path: Path, report: SceneReport) -> Optional[Dict[str, float]]:
    if not test_csv_path.is_file():
        report.add_warning("test/test_poses.csv is missing; intrinsics comparison was skipped.")
        return None

    with test_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            report.add_warning("test/test_poses.csv is empty; intrinsics comparison was skipped.")
            return None

        missing_columns = REQUIRED_TEST_POSE_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            report.add_warning(
                "test/test_poses.csv is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )
            return None

        rows = list(reader)

    report.test_pose_count = len(rows)
    if not rows:
        report.add_warning("test/test_poses.csv has no pose rows.")
        return None

    try:
        widths = [int(float(row["width"])) for row in rows]
        heights = [int(float(row["height"])) for row in rows]
        fx_values = [float(row["fx"]) for row in rows]
        fy_values = [float(row["fy"]) for row in rows]
        cx_values = [float(row["cx"]) for row in rows]
        cy_values = [float(row["cy"]) for row in rows]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        report.add_warning(
            "test/test_poses.csv has invalid numeric values; intrinsics comparison "
            f"was skipped: {exc}"
        )
        return None

    unique_resolutions = sorted(set(zip(widths, heights)))
    if len(unique_resolutions) == 1:
        width, height = unique_resolutions[0]
        report.test_pose_resolution = f"{width}x{height}"
    else:
        report.test_pose_resolution = ", ".join(f"{w}x{h}" for w, h in unique_resolutions)
        report.add_warning(f"test/test_poses.csv has multiple resolutions: {report.test_pose_resolution}")

    summary = {
        "width": float(median(widths)),
        "height": float(median(heights)),
        "fx": median(fx_values),
        "fy": median(fy_values),
        "cx": median(cx_values),
        "cy": median(cy_values),
    }
    report.test_pose_intrinsics = (
        f"fx={format_float(summary['fx'])}, fy={format_float(summary['fy'])}, "
        f"cx={format_float(summary['cx'])}, cy={format_float(summary['cy'])}"
    )
    return summary


def relative_delta(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def check_camera_intrinsics(report: SceneReport, test_pose_intrinsics: Optional[Dict[str, float]]) -> None:
    for camera in report.camera_summaries:
        if camera.fx is None or camera.fy is None or camera.cx is None or camera.cy is None:
            report.add_warning(f"Camera {camera.camera_id} uses unsupported model {camera.model}.")
            continue
        if camera.width <= 0 or camera.height <= 0:
            report.add_warning(f"Camera {camera.camera_id} has non-positive image size.")
        if camera.fx <= 0 or camera.fy <= 0:
            report.add_warning(f"Camera {camera.camera_id} has non-positive focal length.")
        if not (0 <= camera.cx <= camera.width):
            report.add_warning(f"Camera {camera.camera_id} has cx outside the image bounds.")
        if not (0 <= camera.cy <= camera.height):
            report.add_warning(f"Camera {camera.camera_id} has cy outside the image bounds.")

    if not test_pose_intrinsics:
        return

    for camera in report.camera_summaries:
        if camera.fx is None or camera.fy is None or camera.cx is None or camera.cy is None:
            continue
        if int(camera.width) != int(test_pose_intrinsics["width"]) or int(camera.height) != int(test_pose_intrinsics["height"]):
            report.add_warning(
                f"Camera {camera.camera_id} size {camera.width}x{camera.height} differs from test pose median "
                f"{int(test_pose_intrinsics['width'])}x{int(test_pose_intrinsics['height'])}."
            )
        comparisons = {
            "fx": (camera.fx, test_pose_intrinsics["fx"]),
            "fy": (camera.fy, test_pose_intrinsics["fy"]),
            "cx": (camera.cx, test_pose_intrinsics["cx"]),
            "cy": (camera.cy, test_pose_intrinsics["cy"]),
        }
        for name, (camera_value, test_value) in comparisons.items():
            if relative_delta(camera_value, test_value) > 0.01:
                report.add_warning(
                    f"Camera {camera.camera_id} {name}={format_float(camera_value)} differs from "
                    f"test pose median {name}={format_float(test_value)} by more than 1%."
                )


def inspect_scene(set_name: str, scene_dir: Path, colmap_loader) -> SceneReport:
    report = SceneReport(set_name=set_name, scene_name=scene_dir.name, scene_dir=scene_dir)
    train_dir = scene_dir / "train"
    images_dir = train_dir / "images"
    sparse_dir = train_dir / "sparse" / "0"
    test_csv_path = scene_dir / "test" / "test_poses.csv"

    required_files = {
        "cameras.bin": sparse_dir / "cameras.bin",
        "images.bin": sparse_dir / "images.bin",
        "points3D.bin": sparse_dir / "points3D.bin",
    }
    for label, path in required_files.items():
        if not path.exists():
            report.add_error(f"{label} is missing at {path}.")

    if report.errors:
        return report

    try:
        cameras = colmap_loader.read_intrinsics_binary(required_files["cameras.bin"])
    except Exception as exc:
        report.add_error(f"Failed to read cameras.bin: {exc}")
        cameras = {}

    try:
        registered_images = colmap_loader.read_extrinsics_binary(required_files["images.bin"])
    except Exception as exc:
        report.add_error(f"Failed to read images.bin: {exc}")
        registered_images = {}

    try:
        xyz, _, _ = colmap_loader.read_points3D_binary(required_files["points3D.bin"])
        report.sparse_points_count = int(len(xyz))
    except Exception as exc:
        report.add_error(f"Failed to read points3D.bin: {exc}")

    report.cameras_count = len(cameras)
    report.registered_images_count = len(registered_images)

    for camera_id in sorted(cameras):
        camera = cameras[camera_id]
        fx, fy, cx, cy = camera_to_pinhole_values(camera)
        report.camera_summaries.append(
            CameraSummary(
                camera_id=int(camera.id),
                model=str(camera.model),
                width=int(camera.width),
                height=int(camera.height),
                params=list(map(float, camera.params)),
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        )

    if not images_dir.is_dir():
        report.add_warning("train/images is missing or not a directory.")
    train_images = image_files(images_dir)
    report.train_images_count = len(train_images)
    registered_names = {image.name for image in registered_images.values()}
    train_names = set(train_images.keys())
    report.registered_images_with_files = len(registered_names & train_names)
    report.registered_images_missing_files = len(registered_names - train_names)
    report.train_images_missing_registration = len(train_names - registered_names)

    if report.registered_images_count != report.train_images_count:
        report.add_warning(
            f"Registered image count ({report.registered_images_count}) differs from train image count "
            f"({report.train_images_count})."
        )
    if report.registered_images_missing_files:
        report.add_warning(
            f"{report.registered_images_missing_files} registered image entries do not have matching files in train/images."
        )
    if report.train_images_missing_registration:
        report.add_warning(
            f"{report.train_images_missing_registration} train image files do not have matching COLMAP registrations."
        )

    if sparse_dir.exists():
        report.sparse_files = sorted(item.name for item in sparse_dir.iterdir() if item.is_file())
        report.tolerated_extra_files = sorted(EXPECTED_EXTRA_SPARSE_FILES & set(report.sparse_files))
        expected_core = set(required_files.keys())
        allowed_files = expected_core | EXPECTED_EXTRA_SPARSE_FILES
        report.unexpected_extra_files = sorted(set(report.sparse_files) - allowed_files)
        if report.unexpected_extra_files:
            report.add_warning(
                "Unexpected sparse files were found: " + ", ".join(report.unexpected_extra_files)
            )

    test_pose_intrinsics = read_test_pose_summary(test_csv_path, report)
    check_camera_intrinsics(report, test_pose_intrinsics)
    return report


def markdown_table_row(values: List[object]) -> str:
    safe_values = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in values]
    return "| " + " | ".join(safe_values) + " |"


def write_markdown_report(reports: List[SceneReport], output_path: Path, data_root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    readable_count = sum(1 for report in reports if not report.errors)
    pass_count = sum(1 for report in reports if report.status == "PASS")
    warn_count = sum(1 for report in reports if report.status == "WARN")
    fail_count = sum(1 for report in reports if report.status == "FAIL")

    lines: List[str] = []
    lines.append("# Phase 1 COLMAP Compatibility Report")
    lines.append("")
    lines.append(f"- Data root: `{data_root}`")
    lines.append(f"- Scenes checked: {len(reports)}")
    lines.append(f"- Readable COLMAP sparse reconstructions: {readable_count}/{len(reports)}")
    lines.append(f"- Status counts: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count}")
    lines.append("- Tolerated extra sparse files: `frames.bin`, `rigs.bin`, `points3D.ply`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Scene | Set | Status | Cameras | Registered Images | Train Images | Matched Images | Sparse Points | Camera Models | Test Pose Intrinsics | Notes |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for report in reports:
        camera_models = ", ".join(sorted({camera.model for camera in report.camera_summaries})) or "N/A"
        notes = []
        if report.errors:
            notes.extend(report.errors)
        if report.warnings:
            notes.extend(report.warnings)
        if not notes:
            notes.append("No issues found.")
        lines.append(
            markdown_table_row(
                [
                    report.scene_name,
                    report.set_name,
                    report.status,
                    report.cameras_count,
                    report.registered_images_count,
                    report.train_images_count,
                    report.registered_images_with_files,
                    report.sparse_points_count,
                    camera_models,
                    report.test_pose_intrinsics,
                    "<br>".join(notes),
                ]
            )
        )

    for report in reports:
        lines.append("")
        lines.append(f"## {report.scene_name}")
        lines.append("")
        lines.append(f"- Set: `{report.set_name}`")
        lines.append(f"- Scene directory: `{report.scene_dir}`")
        lines.append(f"- Status: **{report.status}**")
        lines.append(f"- Cameras: {report.cameras_count}")
        lines.append(f"- Registered images in `images.bin`: {report.registered_images_count}")
        lines.append(f"- Train image files: {report.train_images_count}")
        lines.append(f"- Registered images with matching files: {report.registered_images_with_files}")
        lines.append(f"- Registered entries missing image files: {report.registered_images_missing_files}")
        lines.append(f"- Train image files missing COLMAP registration: {report.train_images_missing_registration}")
        lines.append(f"- Sparse points: {report.sparse_points_count}")
        lines.append(f"- Test poses: {report.test_pose_count}")
        lines.append(f"- Test pose resolution: {report.test_pose_resolution}")
        lines.append(f"- Test pose intrinsics: {report.test_pose_intrinsics}")
        lines.append(f"- Sparse files: {', '.join(report.sparse_files) if report.sparse_files else 'N/A'}")
        lines.append(
            f"- Tolerated extra sparse files present: {', '.join(report.tolerated_extra_files) if report.tolerated_extra_files else 'None'}"
        )
        lines.append("")
        lines.append("### Cameras")
        lines.append("")
        lines.append("| ID | Model | Size | fx | fy | cx | cy | Raw Params |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---|")
        for camera in report.camera_summaries:
            lines.append(
                markdown_table_row(
                    [
                        camera.camera_id,
                        camera.model,
                        f"{camera.width}x{camera.height}",
                        format_float(camera.fx),
                        format_float(camera.fy),
                        format_float(camera.cx),
                        format_float(camera.cy),
                        format_params(camera.params),
                    ]
                )
            )

        if report.errors:
            lines.append("")
            lines.append("### Errors")
            lines.append("")
            for error in report.errors:
                lines.append(f"- {error}")

        if report.warnings:
            lines.append("")
            lines.append("### Warnings")
            lines.append("")
            for warning in report.warnings:
                lines.append(f"- {warning}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(reports: List[SceneReport], output_path: Path) -> None:
    print("COLMAP compatibility check")
    print(f"Scenes checked: {len(reports)}")
    for report in reports:
        print(
            f"[{report.status}] {report.set_name}/{report.scene_name}: "
            f"cameras={report.cameras_count}, registered_images={report.registered_images_count}, "
            f"train_images={report.train_images_count}, matched_images={report.registered_images_with_files}, "
            f"sparse_points={report.sparse_points_count}"
        )
        for error in report.errors:
            print(f"  ERROR: {error}")
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
    print(f"Report written to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify COLMAP binary compatibility for Phase 1 NVS scenes."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--taming-root", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when warnings are found.",
    )
    args = parser.parse_args()

    try:
        data_root = require_directory(args.data_root, "--data-root")
        taming_root = require_taming_root(args.taming_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    scenes = list(iter_dataset_scenes(data_root))
    if not scenes:
        parser.error(f"No scene directories were found below --data-root: {data_root}")

    output_path = Path(args.output_file).expanduser().resolve()
    if output_path.exists() and not output_path.is_file():
        parser.error(f"--output-file must name a file, not a directory: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"--output-file parent could not be created: {exc}")

    try:
        colmap_loader = import_colmap_loader(taming_root)
    except ImportError as exc:
        parser.error(f"Could not import scene.colmap_loader from --taming-root: {exc}")
    reports = [
        inspect_scene(set_name, scene_dir, colmap_loader)
        for set_name, scene_dir in scenes
    ]
    write_markdown_report(reports, output_path, data_root)
    print_console_summary(reports, output_path)

    if any(report.errors for report in reports):
        return 1
    if args.strict and any(report.warnings for report in reports):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
