#!/usr/bin/env python
"""Evaluate prediction images against ground-truth images with LPIPS, SSIM, and PSNR."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from nvs_utils import require_directory


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImagePair:
    image_name: str


@dataclass(frozen=True)
class ImageMetrics:
    image_name: str
    lpips: float
    ssim: float
    psnr: float
    psnr_norm: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LPIPS, SSIM, PSNR, and VAI NVS local score from paired image directories."
    )
    parser.add_argument(
        "--taming-root",
        required=True,
        help="Taming-3DGS checkout containing lpipsPyTorch and utils.",
    )
    parser.add_argument(
        "--pred-images",
        required=True,
        help="Directory containing rendered prediction images.",
    )
    parser.add_argument(
        "--gt-images",
        required=True,
        help="Directory containing ground-truth images with the same file names as predictions.",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Name used for report files, for example baseline_7000.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV and Markdown reports are written.",
    )
    parser.add_argument(
        "--psnr-max",
        type=float,
        default=40.0,
        help="PSNR normalization threshold. Score uses clamp(psnr / psnr_max, 0, 1).",
    )
    parser.add_argument(
        "--lpips-net",
        default="vgg",
        choices=["alex", "squeeze", "vgg"],
        help="LPIPS backbone. The original 3DGS metrics script uses vgg.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional smoke-test image limit. Do not use for final reporting.",
    )
    return parser.parse_args()


def import_metric_modules(taming_root: Path) -> tuple[Any, Any]:
    """Validate and lazily import LPIPS and SSIM from the supplied Taming checkout."""
    taming_root = taming_root.expanduser().resolve()
    if not taming_root.is_dir():
        raise FileNotFoundError(f"--taming-root must name an existing directory: {taming_root}")
    missing = [
        name
        for name in ("lpipsPyTorch", "utils")
        if not (taming_root / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "--taming-root is not a Taming-3DGS checkout; missing directories: "
            + ", ".join(missing)
        )
    root_string = str(taming_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        from lpipsPyTorch.modules.lpips import LPIPS
        from utils.loss_utils import ssim
    except ImportError as exc:
        raise RuntimeError(f"Could not import evaluator metrics from --taming-root: {exc}") from exc
    return LPIPS, ssim


def find_image_names(directory: Path) -> set[str]:
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def collect_image_pairs(pred_dir: Path, gt_dir: Path) -> list[ImagePair]:
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"Prediction image directory does not exist: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth image directory does not exist: {gt_dir}")

    pred_names = find_image_names(pred_dir)
    gt_names = find_image_names(gt_dir)
    if not gt_names:
        raise ValueError(f"No supported ground-truth images were found in: {gt_dir}")

    missing_predictions = sorted(gt_names.difference(pred_names))
    if missing_predictions:
        raise FileNotFoundError(
            f"Missing {len(missing_predictions)} prediction images. "
            f"First missing files: {missing_predictions[:10]}"
        )

    extra_predictions = sorted(pred_names.difference(gt_names))
    if extra_predictions:
        print(
            f"Warning: ignoring {len(extra_predictions)} prediction images without matching ground truth. "
            f"First ignored files: {extra_predictions[:10]}"
        )

    mismatched_sizes = []
    pairs = []
    for image_name in sorted(gt_names, key=str.lower):
        with Image.open(pred_dir / image_name) as pred_image:
            pred_size = pred_image.size
        with Image.open(gt_dir / image_name) as gt_image:
            gt_size = gt_image.size
        if pred_size != gt_size:
            mismatched_sizes.append((image_name, pred_size, gt_size))
        pairs.append(ImagePair(image_name=image_name))

    if mismatched_sizes:
        preview = "; ".join(
            f"{name} prediction={pred_size} ground_truth={gt_size}"
            for name, pred_size, gt_size in mismatched_sizes[:10]
        )
        raise ValueError(f"Found {len(mismatched_sizes)} image-size mismatches. {preview}")
    return pairs


def image_to_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return tf.to_tensor(rgb).unsqueeze(0).to(device)


def compute_psnr(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    mse = torch.mean((prediction - ground_truth) ** 2)
    if mse.item() == 0:
        return float("inf")
    return float(20.0 * torch.log10(1.0 / torch.sqrt(mse)).item())


def normalize_psnr(psnr_value: float, psnr_max: float) -> float:
    if math.isinf(psnr_value):
        return 1.0
    return max(0.0, min(psnr_value / psnr_max, 1.0))


def compute_score(lpips_value: float, ssim_value: float, psnr_norm: float) -> float:
    return 0.4 * (1.0 - lpips_value) + 0.3 * ssim_value + 0.3 * psnr_norm


def evaluate_images(
    pred_dir: Path,
    gt_dir: Path,
    pairs: list[ImagePair],
    psnr_max: float,
    device: torch.device,
    lpips_model: Any,
    ssim_fn: Any,
    max_images: int | None,
) -> list[ImageMetrics]:
    if max_images is not None:
        pairs = pairs[:max_images]

    metrics = []
    for item in tqdm(pairs, desc="Evaluating images", leave=False):
        prediction = image_to_tensor(pred_dir / item.image_name, device)
        ground_truth = image_to_tensor(gt_dir / item.image_name, device)

        with torch.no_grad():
            ssim_value = float(ssim_fn(prediction, ground_truth).item())
            psnr_value = compute_psnr(prediction, ground_truth)
            psnr_norm = normalize_psnr(psnr_value, psnr_max)
            lpips_value = float(lpips_model(prediction, ground_truth).item())
            score = compute_score(lpips_value, ssim_value, psnr_norm)

        metrics.append(
            ImageMetrics(
                image_name=item.image_name,
                lpips=lpips_value,
                ssim=ssim_value,
                psnr=psnr_value,
                psnr_norm=psnr_norm,
                score=score,
            )
        )
    return metrics


def mean(values: Iterable[float]) -> float:
    clean = list(values)
    if not clean:
        raise ValueError("Cannot compute an average from no values.")
    return sum(clean) / len(clean)


def format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.8f}"


def write_scores_csv(path: Path, rows: list[ImageMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_name", "lpips", "ssim", "psnr", "psnr_norm", "score"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_name": row.image_name,
                    "lpips": format_float(row.lpips),
                    "ssim": format_float(row.ssim),
                    "psnr": format_float(row.psnr),
                    "psnr_norm": format_float(row.psnr_norm),
                    "score": format_float(row.score),
                }
            )


def write_summary_csv(path: Path, rows: list[ImageMetrics]) -> None:
    summary = {
        "images": str(len(rows)),
        "lpips": format_float(mean(row.lpips for row in rows)),
        "ssim": format_float(mean(row.ssim for row in rows)),
        "psnr": format_float(mean(row.psnr for row in rows)),
        "psnr_norm": format_float(mean(row.psnr_norm for row in rows)),
        "score": format_float(mean(row.score for row in rows)),
    }
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["images", "lpips", "ssim", "psnr", "psnr_norm", "score"],
        )
        writer.writeheader()
        writer.writerow(summary)


def write_summary_md(
    path: Path,
    args: Namespace,
    rows: list[ImageMetrics],
    score_csv: Path,
    summary_csv: Path,
) -> None:
    public_lpips = mean(row.lpips for row in rows)
    public_ssim = mean(row.ssim for row in rows)
    public_psnr = mean(row.psnr for row in rows)
    public_psnr_norm = mean(row.psnr_norm for row in rows)
    public_score = mean(row.score for row in rows)

    lines = [
        f"# Public Evaluation Summary: {args.experiment_name}",
        "",
        "## Configuration",
        "",
        f"- Prediction images: `{Path(args.pred_images).resolve()}`",
        f"- Ground-truth images: `{Path(args.gt_images).resolve()}`",
        f"- PSNR max: `{args.psnr_max}`",
        f"- LPIPS net: `{args.lpips_net}`",
        f"- Device: `{args.device}`",
        f"- Image rows: `{len(rows)}`",
        f"- Per-image CSV: `{score_csv}`",
        f"- Summary CSV: `{summary_csv}`",
        "",
        "## Average",
        "",
        "| LPIPS | SSIM | PSNR | PSNR norm | Score |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {format_float(public_lpips)} | {format_float(public_ssim)} | "
            f"{format_float(public_psnr)} | {format_float(public_psnr_norm)} | "
            f"{format_float(public_score)} |"
        ),
        "",
    ]
    lines.append("")
    if args.max_images is not None:
        lines.extend(
            [
                "## Warning",
                "",
                f"This report used `--max-images {args.max_images}` and may not represent the full image set.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_images).expanduser().resolve()
    gt_dir = Path(args.gt_images).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    taming_root = require_directory(args.taming_root, "--taming-root")

    if args.psnr_max <= 0:
        raise ValueError("--psnr-max must be positive.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive.")
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in {"", ".", ".."}:
        raise ValueError("--experiment-name must be a single safe file-name component.")
    print(f"Experiment: {args.experiment_name}")
    print(f"Prediction images: {pred_dir}")
    print(f"Ground-truth images: {gt_dir}")
    print(f"PSNR max: {args.psnr_max}")

    image_pairs = collect_image_pairs(pred_dir, gt_dir)
    print("Input validation passed.")

    device = torch.device(args.device)
    lpips_class, ssim_fn = import_metric_modules(taming_root)
    print(f"Loading LPIPS model: {args.lpips_net}")
    lpips_model = lpips_class(args.lpips_net).to(device).eval()

    all_metrics = evaluate_images(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        pairs=image_pairs,
        psnr_max=args.psnr_max,
        device=device,
        lpips_model=lpips_model,
        ssim_fn=ssim_fn,
        max_images=args.max_images,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    scores_csv = output_dir / f"{args.experiment_name}_public_scores.csv"
    summary_csv = output_dir / f"{args.experiment_name}_summary_scores.csv"
    summary_md = output_dir / f"{args.experiment_name}_summary.md"
    write_scores_csv(scores_csv, all_metrics)
    write_summary_csv(summary_csv, all_metrics)
    write_summary_md(summary_md, args, all_metrics, scores_csv, summary_csv)

    public_score = mean(row.score for row in all_metrics)
    print(f"Wrote per-image scores: {scores_csv}")
    print(f"Wrote summary scores: {summary_csv}")
    print(f"Wrote summary: {summary_md}")
    print(f"Public score: {format_float(public_score)}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
