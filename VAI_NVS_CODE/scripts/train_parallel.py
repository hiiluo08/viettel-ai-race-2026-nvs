#!/usr/bin/env python
"""Train multiple Taming-AbsGS scenes in parallel on separate GPUs.

Supports both fresh training and resume-from-checkpoint across 2 GPUs.

Scene spec format (":" separated):
    LABEL:TRAIN_DIR:MODEL_SUFFIX[:CHECKPOINT[:TARGET_ITERATIONS[:DENSIFY_UNTIL_OVERRIDE]]]

Examples:

  # Fresh: 2 scenes on GPU 0,1
  python scripts/train_parallel.py \
      --hybrid-root ... --model-dir ... \
      --scenes "hcm0031:/data/hcm0031/train:hcm0031_model" \
               "hcm0034:/data/hcm0034/train:hcm0034_model"

  # Resume: each scene from its own checkpoint, target 60k total iterations
  python scripts/train_parallel.py \
      --hybrid-root ... --model-dir ... \
      --scenes "hcm0031:/data/hcm0031/train:hcm0031_model:/out/ckpt/chkpnt30000.pth:60000" \
               "hcm0034:/data/hcm0034/train:hcm0034_model:/out/ckpt/chkpnt30000.pth:60000"

  # Resume with custom densify_until_iter for extended densification
  python scripts/train_parallel.py \
      --hybrid-root ... --model-dir ... \
      --scenes "hcm0031:/data/hcm0031/train:hcm0031_model:/out/ckpt/chkpnt30000.pth:60000:25000"

  # A/B test thresholds
  python scripts/train_parallel.py \
      --hybrid-root ... --model-dir ... \
      --scenes "abs04:/data/train:model_04" "abs06:/data/train:model_06" \
      --densify-grad-abs-threshold 0.0004 0.0006
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JobConfig:
    label: str
    train_dir: str
    model_dir: str
    gpu: int
    iterations: int = 30_000
    budget: int = 3_000_000
    mode: str = "final_count"
    cams: int = 20
    lambda_dssim: float = 0.2
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    densify_grad_threshold: float = 0.0002
    densify_grad_abs_threshold: float = 0.0004
    ho_iteration: int = 15_000
    save_iterations: int = 30_000
    checkpoint_iterations: int = 30_000
    start_checkpoint: str | None = None
    position_lr_max_steps: int | None = None
    sh_degree: int = 3
    seed: int | None = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Taming-AbsGS scenes in parallel on multiple GPUs.",
    )
    parser.add_argument("--hybrid-root", required=True,
                        help="Taming-AbsGS hybrid checkout root.")
    parser.add_argument(
        "--scenes", required=True, nargs="+",
        help=(
            "Each: LABEL:TRAIN_DIR:MODEL_SUFFIX"
            "[:CHECKPOINT[:TARGET_ITERATIONS[:DENSIFY_UNTIL_OVERRIDE]]]"
        ),
    )
    parser.add_argument("--model-dir", required=True,
                        help="Base directory for per-scene model checkpoints.")
    parser.add_argument("--gpus", default="0,1",
                        help="Comma-separated GPU ids (default: 0,1).")

    # Shared hyperparameters (per-job nth-value overrides via nargs lists).
    parser.add_argument("--iterations", type=int, default=30_000, nargs="+")
    parser.add_argument("--budget", type=int, default=3_000_000)
    parser.add_argument("--mode", default="final_count")
    parser.add_argument("--cams", type=int, default=20)
    parser.add_argument("--lambda-dssim", type=float, default=0.2,
                        dest="lambda_dssim")
    parser.add_argument("--densify-from-iter", type=int, default=500)
    parser.add_argument("--densify-until-iter", type=int, default=15_000)
    parser.add_argument("--densification-interval", type=int, default=100)
    parser.add_argument("--densify-grad-threshold", type=float, default=0.0002,
                        nargs="+")
    parser.add_argument("--densify-grad-abs-threshold", type=float,
                        default=0.0004, nargs="+")
    parser.add_argument("--save-iterations", type=int, nargs="+",
                        default=[30_000])
    parser.add_argument("--checkpoint-iterations", type=int, nargs="+",
                        default=[30_000])
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--compact", action="store_true",
                        help="Only show per-job progress lines (loss + tqdm); "
                             "full output still saved to log files.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(values: list, idx: int) -> float:
    """values[idx] or the last element."""
    return values[min(idx, len(values) - 1)]


# ---------------------------------------------------------------------------
# Scene-spec parser
# ---------------------------------------------------------------------------

def resolve_scene_jobs(
    scenes: list[str],
    model_dir: str,
    gpus: list[int],
    args: argparse.Namespace,
) -> list[JobConfig]:
    """Parse each scene spec into a JobConfig.

    Spec fields (positional, ":" separated):
      0  LABEL                    (required)
      1  TRAIN_DIR                (required)
      2  MODEL_SUFFIX             (default: same as LABEL)
      3  CHECKPOINT               (start_checkpoint .pth path; omit for fresh)
      4  TARGET_ITERATIONS        (per-job iterations; overrides --iterations global)
      5  DENSIFY_UNTIL_OVERRIDE   (override --densify-until-iter for this job)

    Fields 3-5 are optional and positional — skip with empty string to reach a
    later field (e.g. "label:dir:suffix::60000" sets TARGET_ITERATIONS without
    a checkpoint).

    When a checkpoint (field 3) is supplied:
      - --start_checkpoint is set
      - --iterations defaults to TARGET_ITERATIONS if given, else the global value
      - --position_lr_max_steps matches --iterations
      - --densify_until_iter can be overridden per-job (field 5)

    TARGET_ITERATIONS works for both fresh and resume runs — it always
    overrides the per-job iteration count.
    """
    jobs: list[JobConfig] = []
    for idx, spec in enumerate(scenes):
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid scene spec '{spec}': "
                f"expected at least LABEL:TRAIN_DIR"
            )

        label = parts[0]
        train_dir = parts[1]
        model_suffix = parts[2] if len(parts) > 2 else label
        checkpoint = parts[3] if len(parts) > 3 and parts[3] else None
        target_iters_str = parts[4] if len(parts) > 4 and parts[4] else None
        densify_override_str = parts[5] if len(parts) > 5 and parts[5] else None

        gpu = gpus[idx % len(gpus)]

        # Per-job parameter picking
        g_thresh = _pick(args.densify_grad_threshold, idx)
        abs_thresh = _pick(args.densify_grad_abs_threshold, idx)

        # --- Per-job iterations ---
        iterations = int(_pick(args.iterations, idx))
        save_iters = int(_pick(args.save_iterations, idx))
        ckpt_iters = int(_pick(args.checkpoint_iterations, idx))
        ho_iter = int(args.densify_until_iter)

        if target_iters_str is not None:
            iterations = int(target_iters_str)
            save_iters = int(target_iters_str)
            ckpt_iters = int(target_iters_str)

        position_lr_max_steps = iterations

        if densify_override_str is not None:
            ho_iter = int(densify_override_str)

        job = JobConfig(
            label=label,
            train_dir=train_dir,
            model_dir=str(Path(model_dir) / model_suffix),
            gpu=gpu,
            iterations=iterations,
            budget=args.budget,
            mode=args.mode,
            cams=args.cams,
            lambda_dssim=args.lambda_dssim,
            densify_from_iter=args.densify_from_iter,
            densify_until_iter=args.densify_until_iter,
            densification_interval=args.densification_interval,
            densify_grad_threshold=g_thresh,
            densify_grad_abs_threshold=abs_thresh,
            ho_iteration=ho_iter,
            save_iterations=save_iters,
            checkpoint_iterations=ckpt_iters,
            start_checkpoint=checkpoint,
            position_lr_max_steps=position_lr_max_steps,
            sh_degree=args.sh_degree,
            seed=args.seed,
        )
        jobs.append(job)
    return jobs


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(job: JobConfig, hybrid_root: str) -> list[str]:
    cmd = [
        sys.executable, "-u", "train.py",
        "-s", job.train_dir,
        "-m", job.model_dir,
        "--iterations", str(job.iterations),
        "--budget", str(job.budget),
        "--mode", job.mode,
        "--cams", str(job.cams),
        "--lambda_dssim", str(job.lambda_dssim),
        "--densify_from_iter", str(job.densify_from_iter),
        "--densify_until_iter", str(job.densify_until_iter),
        "--densification_interval", str(job.densification_interval),
        "--densify_grad_threshold", str(job.densify_grad_threshold),
        "--densify_grad_abs_threshold", str(job.densify_grad_abs_threshold),
        "--ho_iteration", str(job.ho_iteration),
        "--save_iterations", str(job.save_iterations),
        "--checkpoint_iterations", str(job.checkpoint_iterations),
        "--test_iterations", "-1",
        "--sh_degree", str(job.sh_degree),
        "--position_lr_max_steps", str(job.position_lr_max_steps),
    ]
    if job.start_checkpoint:
        cmd.extend(["--start_checkpoint", job.start_checkpoint])
    if job.seed is not None:
        cmd.extend(["--seed", str(job.seed)])
    return cmd


# ---------------------------------------------------------------------------
# Process launcher
# ---------------------------------------------------------------------------

def _is_progress_line(line: str) -> bool:
    """In compact mode, only show training milestones — skip per-iteration progress."""
    lower = line.lower()

    # Errors always pass through
    if any(kw in lower for kw in ("error", "traceback")):
        return True

    # Milestones — check BEFORE tqdm filter because densif/checkpoint
    # messages often share the same line as the tqdm progress bar
    if any(kw in lower for kw in (
        "output folder",            # training start
        "gaussian count schedule",   # budget schedule
        "number of points",          # initial point count
        "densif",                   # densification events (incl. embedded in tqdm line)
        "peak allocated",           # VRAM info
        "saving", "saved",          # checkpoint saves
        "checkpoint",               # checkpoint info
        "training complete",        # done
        "optimizing",               # scene path at start
    )):
        return True

    # Skip all tqdm progress bars
    if "it/s" in lower or ("|" in line and "/" in line and "%" in line):
        return False

    return False


def run_job(job: JobConfig, hybrid_root: str, log_dir: str,
            compact: bool = False) -> subprocess.Popen:
    import os
    import threading

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log_path = Path(log_dir) / f"train_{job.label}_gpu{job.gpu}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")

    cmd = build_command(job, hybrid_root)
    print(f"[{job.label}] Log: {log_path}")

    proc = subprocess.Popen(
        cmd, cwd=hybrid_root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True,
    )

    prefix = f"[{job.label}] "

    def _tee():
        for line in proc.stdout:
            if not compact or _is_progress_line(line):
                print(prefix + line, end="")
            log_file.write(line)
        log_file.flush()
        log_file.close()

    threading.Thread(target=_tee, daemon=True).start()
    return proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    import os

    args = parse_args(argv)
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be specified via --gpus.")

    hybrid_root = str(Path(args.hybrid_root).expanduser().resolve())
    if not Path(hybrid_root).is_dir():
        raise FileNotFoundError(f"--hybrid-root not found: {hybrid_root}")
    if not (Path(hybrid_root) / "train.py").is_file():
        raise FileNotFoundError(
            f"train.py not found in --hybrid-root: {hybrid_root}")

    jobs = resolve_scene_jobs(args.scenes, args.model_dir, gpus, args)

    # --- Print summary ---
    print(f"Launching {len(jobs)} training job(s) on GPU(s): {gpus}")
    print(f"Hybrid root: {hybrid_root}")
    print("-" * 60)
    for job in jobs:
        mode = "RESUME" if job.start_checkpoint else "FRESH"
        print(f"[{job.label}] GPU {job.gpu}  |  {mode}")
        print(f"           model_dir : {job.model_dir}")
        print(f"           train_dir : {job.train_dir}")
        print(f"           iterations: {job.iterations:,}")
        print(f"           budget    : {job.budget:,}")
        print(f"           grad_thr  : {job.densify_grad_threshold}")
        print(f"           abs_thr   : {job.densify_grad_abs_threshold}")
        if job.start_checkpoint:
            print(f"           checkpoint: {job.start_checkpoint}")
            print(f"           ho_iter   : {job.ho_iteration}")
            print(f"           pos_lr_max: {job.position_lr_max_steps}")
    print("-" * 60)

    # --- Launch ---
    log_dir = Path(args.model_dir) / "parallel_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: dict[str, subprocess.Popen] = {}
    for job in jobs:
        procs[job.label] = run_job(job, hybrid_root, str(log_dir),
                                   compact=args.compact)

    print(f"\nAll {len(procs)} job(s) launched. Waiting for completion...\n")

    # --- Wait ---
    results: dict[str, int] = {}
    try:
        for label, proc in procs.items():
            rc = proc.wait()
            results[label] = rc
            status = "OK" if rc == 0 else f"FAILED (exit={rc})"
            print(f"[{label}] Finished: {status}")
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating all jobs...")
        for _label, proc in procs.items():
            proc.terminate()
        for _label, proc in procs.items():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise

    failures = [label for label, rc in results.items() if rc != 0]
    if failures:
        print(f"\nFAILED jobs: {', '.join(failures)}")
        raise SystemExit(1)

    print("\nAll training jobs completed successfully.")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
