from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from nvs_utils import IMAGE_SUFFIXES, REQUIRED_TEST_POSE_COLUMNS, iter_dataset_scenes, require_directory


IMAGE_EXTS = IMAGE_SUFFIXES


def scan_scene(scene_dir: Path, set_name: str, warnings: List[str]) -> dict[str, object]:
    """Scan one scene using the expected train, sparse, and test layout."""
    scene_name = scene_dir.name

    train_dir = scene_dir / "train"
    train_img_dir = train_dir / "images"
    sparse_dir = train_dir / "sparse" / "0"

    test_dir = scene_dir / "test"
    test_img_dir = test_dir / "images"
    test_csv_path = test_dir / "test_poses.csv"

    if not train_img_dir.is_dir():
        warnings.append(f"[{scene_name}] Missing train/images")
    if not (sparse_dir / "cameras.bin").is_file():
        warnings.append(f"[{scene_name}] Missing train/sparse/0/cameras.bin")
    if not (sparse_dir / "images.bin").is_file():
        warnings.append(f"[{scene_name}] Missing train/sparse/0/images.bin")
    if not (sparse_dir / "points3D.bin").is_file():
        warnings.append(f"[{scene_name}] Missing train/sparse/0/points3D.bin")
    if not test_img_dir.is_dir():
        warnings.append(f"[{scene_name}] Missing test/images")
    if not test_csv_path.is_file():
        warnings.append(f"[{scene_name}] Missing test_poses.csv")

    num_train = (
        sum(
            1
            for item in train_img_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        )
        if train_img_dir.is_dir()
        else 0
    )
    if num_train == 0:
        warnings.append(f"[{scene_name}] No train images!")

    num_test_gt = (
        sum(
            1
            for item in test_img_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        )
        if test_img_dir.is_dir()
        else 0
    )

    num_test = 0
    resolution = "N/A"
    if test_csv_path.is_file():
        with test_csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                warnings.append(f"[{scene_name}] test_poses.csv has a missing or empty header")
            else:
                rows = list(reader)
                num_test = len(rows)
                missing_columns = REQUIRED_TEST_POSE_COLUMNS - set(reader.fieldnames)
                if missing_columns:
                    warnings.append(
                        f"[{scene_name}] test_poses.csv missing columns: "
                        f"{sorted(missing_columns)}"
                    )
                else:
                    image_names = [row["image_name"] for row in rows]
                    if len(image_names) != len(set(image_names)):
                        warnings.append(
                            f"[{scene_name}] Duplicate image name in test_poses.csv"
                        )

                    resolutions = {(row["width"], row["height"]) for row in rows}
                    if len(resolutions) > 1:
                        warnings.append(
                            f"[{scene_name}] Multiple resolutions in test_poses.csv: "
                            f"{resolutions}"
                        )
                    elif resolutions:
                        width, height = next(iter(resolutions))
                        resolution = f"{width}x{height}"

    sparse_files = (
        sorted(item.name for item in sparse_dir.iterdir() if item.is_file())
        if sparse_dir.is_dir()
        else []
    )

    notes = ""
    if 0 < num_train < 240:
        notes = "Train images are less than standard."

    return {
        "scene_name": scene_name,
        "set_name": set_name,
        "num_train_images": num_train,
        "num_test_poses": num_test,
        "num_test_gt_images": num_test_gt,
        "resolution": resolution,
        "sparse_files": sparse_files,
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect every VAI NVS scene below an explicit dataset root."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory whose child directories are dataset sets.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for dataset manifest JSON and Markdown.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        data_root = require_directory(args.data_root, "--data-root")
    except FileNotFoundError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(f"--output-dir could not be created: {exc}")

    warnings: List[str] = []
    manifest = [
        scan_scene(scene_dir, set_name, warnings)
        for set_name, scene_dir in iter_dataset_scenes(data_root)
    ]
    if not manifest:
        parser.error(f"No scene directories were found below --data-root: {data_root}")

    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No warnings found.")

    json_out = output_dir / "dataset_manifest_phase1.json"
    json_out.write_text(
        json.dumps(manifest, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {json_out}")

    md_out = output_dir / "dataset_manifest_phase1.md"
    with md_out.open("w", encoding="utf-8") as markdown_file:
        markdown_file.write("# Phase 1 Dataset Manifest\n\n")
        markdown_file.write(
            "| Scene | Set | Train Images | Test Poses | Test GT Images | "
            "Resolution | Sparse Files | Notes |\n"
        )
        markdown_file.write("|---|---|---:|---:|---:|---|---|---|\n")
        for scene in manifest:
            sparse_str = ", ".join(scene["sparse_files"])
            markdown_file.write(
                f"| {scene['scene_name']} | {scene['set_name']} | "
                f"{scene['num_train_images']} | {scene['num_test_poses']} | "
                f"{scene['num_test_gt_images']} | {scene['resolution']} | "
                f"{sparse_str} | {scene['notes']} |\n"
            )
    print(f"Saved {md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
