#!/usr/bin/env python
"""Display rendered Phase-1 images as public pairs or private galleries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

from PIL import Image

PUBLIC_SCENES = frozenset({"hcm0031", "hcm0034", "hcm0181", "hcm0193", "hcm0204"})
PRIVATE_SCENES = frozenset({
    "hcm0249",
    "hcm0254",
    "hcm0276",
    "hcm1439",
    "hni0131",
    "hni0265",
    "hni0366",
    "hni0437",
})
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def infer_scene_kind(images_dir: Path) -> tuple[str, str]:
    """Infer a known scene ID and its public/private classification from the path."""
    path_str = images_dir.expanduser().resolve().as_posix().casefold()
    for scene_id in PUBLIC_SCENES:
        if scene_id in path_str:
            return scene_id, "public"
    for scene_id in PRIVATE_SCENES:
        if scene_id in path_str:
            return scene_id, "private"
    raise ValueError(
        "could not infer a known Phase 1 scene from --images-dir path; "
        "include the public/private scene ID in the path."
    )


def _image_paths(directory: Path, *, allow_empty: bool = False) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
    }
    if not paths and not allow_empty:
        raise ValueError(f"No supported images were found in: {directory}")
    return dict(sorted(paths.items(), key=lambda item: item[0].casefold()))


def build_public_pairs(images_dir: Path, gt_images: Path) -> list[str]:
    """Validate matching public render/GT names and dimensions, returning relative names."""
    rendered = _image_paths(images_dir)
    ground_truth = _image_paths(gt_images, allow_empty=True)
    missing_gt = sorted(set(rendered) - set(ground_truth))
    extra_gt = sorted(set(ground_truth) - set(rendered))
    if missing_gt or extra_gt:
        details = []
        if missing_gt:
            details.append(f"Missing ground-truth images: {missing_gt[:10]}")
        if extra_gt:
            details.append(f"Extra ground-truth images: {extra_gt[:10]}")
        raise FileNotFoundError("; ".join(details))

    mismatches: list[str] = []
    for image_name in rendered:
        with Image.open(rendered[image_name]) as render_image:
            rendered_size = render_image.size
        with Image.open(ground_truth[image_name]) as gt_image:
            gt_size = gt_image.size
        if rendered_size != gt_size:
            mismatches.append(
                f"{image_name}: render={rendered_size}, ground_truth={gt_size}"
            )
    if mismatches:
        raise ValueError("Rendered/ground-truth dimension mismatch: " + "; ".join(mismatches[:10]))
    return list(rendered)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display Phase-1 rendered images with automatic public/private scene handling."
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing rendered images.")
    parser.add_argument(
        "--gt-images",
        help="Explicit ground-truth directory required for known public scenes.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=12,
        help="Images per page; private galleries use four columns.",
    )
    return parser.parse_args(argv)


def _show_public_page(
    images_dir: Path, gt_images: Path, image_names: list[str],
    page_number: int, page_size: int,
) -> None:
    import matplotlib.pyplot as plt

    page_names = image_names[page_number * page_size : (page_number + 1) * page_size]
    figure, axes = plt.subplots(len(page_names), 2, squeeze=False, figsize=(12, 4 * len(page_names)))
    for row, image_name in enumerate(page_names):
        with Image.open(images_dir / image_name) as render_image:
            axes[row][0].imshow(render_image.convert("RGB"))
        with Image.open(gt_images / image_name) as gt_image:
            axes[row][1].imshow(gt_image.convert("RGB"))
        axes[row][0].set_title(f"Render | {image_name}")
        axes[row][1].set_title(f"Ground truth | {image_name}")
        axes[row][0].axis("off")
        axes[row][1].axis("off")
    figure.suptitle(f"Public scene page {page_number + 1}")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def _show_private_page(
    images_dir: Path, image_names: list[str],
    page_number: int, page_size: int,
) -> None:
    import matplotlib.pyplot as plt

    page_names = image_names[page_number * page_size : (page_number + 1) * page_size]
    columns = 4
    rows = max(1, math.ceil(len(page_names) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(16, 4 * rows))
    for index, image_name in enumerate(page_names):
        axis = axes[index // columns][index % columns]
        with Image.open(images_dir / image_name) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(image_name)
        axis.axis("off")
    for index in range(len(page_names), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(f"Private scene page {page_number + 1}")
    figure.tight_layout()
    plt.show()
    plt.close(figure)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.page_size <= 0:
        raise ValueError("--page-size must be positive.")
    images_dir = Path(args.images_dir).expanduser().resolve()
    scene_id, scene_kind = infer_scene_kind(images_dir)
    if scene_kind == "public":
        if not args.gt_images:
            raise ValueError(
                f"Public scene {scene_id} requires explicit --gt-images; "
                "the ground-truth location is never inferred."
            )
        gt_images = Path(args.gt_images).expanduser().resolve()
        image_names = build_public_pairs(images_dir, gt_images)
        for page_number in range(math.ceil(len(image_names) / args.page_size)):
            _show_public_page(images_dir, gt_images, image_names, page_number, args.page_size)
    else:
        image_names = list(_image_paths(images_dir))
        for page_number in range(math.ceil(len(image_names) / args.page_size)):
            _show_private_page(images_dir, image_names, page_number, args.page_size)


if __name__ == "__main__":
    main()
