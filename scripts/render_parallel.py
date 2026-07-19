#!/usr/bin/env python
"""Render test poses in parallel across two GPUs by splitting the pose list.

Each GPU renders roughly half the test poses simultaneously, then images are
merged into the final output directory.

Example:
    python scripts/render_parallel.py \
        --taming-root /kaggle/working/vai_nvs_code/external/taming-absgs \
        --model-path /kaggle/working/vai_nvs_outputs/checkpoints/hcm0031 \
        --poses-csv /kaggle/input/.../test/test_poses.csv \
        --output-dir /kaggle/working/vai_nvs_outputs/renders/hcm0031
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render test poses on two GPUs in parallel."
    )
    parser.add_argument("--taming-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--poses-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--black-background", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _write_split_csv(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_cmd(args: argparse.Namespace, gpu: int,
               tmp_csv: str, tmp_out: str) -> list[str]:
    render_script = str(Path(__file__).resolve().parent / "render_poses.py")
    cmd = [
        sys.executable, "-u", render_script,
        "--taming-root", args.taming_root,
        "--model-path", args.model_path,
        "--poses-csv", tmp_csv,
        "--output-dir", tmp_out,
        "--iteration", str(args.iteration),
        "--sh-degree", str(args.sh_degree),
        "--jpeg-quality", str(args.jpeg_quality),
        "--device", f"cuda:{gpu}",
    ]
    if args.white_background:
        cmd.append("--white-background")
    if args.black_background:
        cmd.append("--black-background")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.quiet:
        cmd.append("--quiet")
    return cmd


def main(argv: Sequence[str] | None = None) -> None:
    import os

    args = parse_args(argv)
    poses_csv = str(Path(args.poses_csv).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()

    with open(poses_csv, "r", encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))

    if not all_rows:
        raise ValueError(f"No poses found in: {poses_csv}")

    # Split poses: first half -> GPU 0, second half -> GPU 1
    midpoint = max(1, len(all_rows) // 2)
    rows_gpu0, rows_gpu1 = all_rows[:midpoint], all_rows[midpoint:]

    tmp_dir = Path(tempfile.mkdtemp(prefix="render_parallel_"))
    tmp_csv_0 = str(tmp_dir / "poses_gpu0.csv")
    tmp_csv_1 = str(tmp_dir / "poses_gpu1.csv")
    tmp_out_0 = str(tmp_dir / "out_gpu0")
    tmp_out_1 = str(tmp_dir / "out_gpu1")

    _write_split_csv(rows_gpu0, tmp_csv_0)
    _write_split_csv(rows_gpu1, tmp_csv_1)

    cmd0 = _build_cmd(args, 0, tmp_csv_0, tmp_out_0)
    cmd1 = _build_cmd(args, 1, tmp_csv_1, tmp_out_1)

    print(f"Total poses: {len(all_rows)}")
    print(f"GPU 0: {len(rows_gpu0)} poses | GPU 1: {len(rows_gpu1)} poses")
    print("-" * 60)

    env0 = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    env1 = dict(os.environ, CUDA_VISIBLE_DEVICES="1")

    proc0 = subprocess.Popen(cmd0, env=env0)
    proc1 = subprocess.Popen(cmd1, env=env1)
    print("Both render jobs launched. Waiting for completion...\n")

    rc0 = proc0.wait()
    rc1 = proc1.wait()

    print(f"GPU 0: {'OK' if rc0 == 0 else f'FAILED (exit={rc0})'}")
    print(f"GPU 1: {'OK' if rc1 == 0 else f'FAILED (exit={rc1})'}")
    if rc0 != 0 or rc1 != 0:
        raise SystemExit(1)

    # Merge into final output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for tmp_out in (tmp_out_0, tmp_out_1):
        tmp_path = Path(tmp_out)
        if tmp_path.is_dir():
            for img in tmp_path.iterdir():
                if img.is_file():
                    shutil.copy2(img, output_dir / img.name)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    num_out = sum(1 for p in output_dir.iterdir() if p.is_file())
    print(f"\nRendered {num_out} images -> {output_dir}")
    print("Parallel render completed successfully.")


if __name__ == "__main__":
    main()
