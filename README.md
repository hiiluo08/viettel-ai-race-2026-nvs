# Viettel AI Race 2026 — Novel View Synthesis (BTS Digital Twin)

> **Viettel AI Race — Problem 1: BTS Digital Twin**
>
> Organized by **Viettel** · [competition.viettel.vn/contests/var-2026](https://competition.viettel.vn/contests/var-2026)

---

## Overview

The goal is to build an AI model that reconstructs a 3D spatial structure of a scene from multi-view images and synthesizes photorealistic RGB images at **novel camera viewpoints** never seen during training.

| Key | Description |
|---|---|
| **Input** | Multi-view images + camera poses + COLMAP sparse reconstruction |
| **Output** | RGB images rendered at specified test poses |
| **Target objects** | BTS (cell tower) stations, telecom infrastructure, real-world scenes |
| **Domains** | Computer Vision · 3D Vision · Neural Rendering · Digital Twin |

---

## Method: Taming-AbsGS Hybrid

Our approach combines two complementary improvements over the original 3D Gaussian Splatting (3DGS) into a single, patched training pipeline.

### Background: 3D Gaussian Splatting

[3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (Kerbl et al., SIGGRAPH 2023) represents a scene as a collection of anisotropic 3D Gaussians. During training, Gaussians are periodically **densified** (cloned or split) based on the magnitude of their view-space positional gradients, then **pruned** to control the total count. While effective, the original heuristic treats all Gaussians uniformly: the same gradient threshold gates both cloning small Gaussians and splitting large ones.

### Taming-3DGS: Score-Based Budget Control

[Taming-3DGS](https://github.com/humansensinglab/taming-3dgs) (Mallick et al., ECCV 2024) addresses the exponential growth and training instability of 3DGS by introducing two key mechanisms:

**1. Gaussian Importance Score.** Each Gaussian is ranked by a multi-factor score that aggregates geometry and photometric signals across all training views:

- **Geometry factors** (*G*): normalized view-space gradient magnitude, opacity, median depth, projected radii, and scale
- **Photometric factors** (*P*): per-pixel L1 loss, accumulated blending weight, transmittance-weighted distance, and inverse ray count

The importance of Gaussian *i* for view *v* is:

```
S_i(v) = w_v · L_photo(v) · (P_i(v) + G_i(v))
```

Scores are summed across all validation views, and only the top-*k* Gaussians (controlled by a budget schedule) survive densification and pruning. This ensures training stays within a predictable memory envelope.

**2. Budget Scheduling.** The target Gaussian count *B(t)* follows a quadratic ramp from the initial count *N₀* to the final budget *B_final* over *T* densification steps (Eq. 2 in the paper):

```
B(t) = a·t² + b·t + N₀
```

where *a, b* are derived from the slope lower bound *(B_final − N₀) / T*. This prevents both early overshoot and late-stage starvation.

**3. Loss-Guided Rendering.** An edge-aware loss map weights the photometric loss spatially, emphasizing high-detail regions (edges detected via PIL's `FIND_EDGES` filter). The training loss combines L1 and a DSSIM term with the learned loss map.

**4. Post-HO Opacity Mode.** After a harmonic-oscillator (HO) schedule, opacity activations switch to absolute-value mode (`torch.abs`), improving convergence on fine structures.

### AbsGS: Absolute Gradient for Split Qualification

[AbsGS](https://github.com/Zhenyu-Yang22/AbsGS) (Yang et al., 2024) identifies a subtle failure mode in 3DGS densification: the **homodirectional gradient cancellation** problem. When neighboring pixels push a large Gaussian in opposite directions, the signed gradient components cancel out during atomic accumulation, so the Gaussian's gradient norm falls below the densification threshold — even though the Gaussian is clearly under-representing the scene.

AbsGS fixes this by accumulating **per-pixel absolute gradients** (`Σ|∂L/∂x|`, `Σ|∂L/∂y|`) in the CUDA backward pass, alongside the standard signed gradients. Since absolute values never cancel, large, under-resolved Gaussians reliably exceed the split threshold.

### Hybrid Architecture

Our `build_taming_absgs.py` script produces a **patched Taming-3DGS workspace** that integrates AbsGS gradient statistics without replacing Taming's core logic. The hybrid is generated deterministically — no source files in either original repo are modified. The script copies Taming-3DGS into a new writable directory and applies surgical patches across 7 files:

| File patched | Change |
|---|---|
| `arguments/__init__.py` | Adds `densify_grad_abs_threshold` (default: `0.0004`) |
| `gaussian_model.py` | Adds `xyz_gradient_accum_abs` buffer, 4-channel gradient accumulation, hybrid `densify_with_score`, backward-compatible checkpoint restore (supports both 13-tuple Taming-only and 14-tuple hybrid checkpoints) |
| `gaussian_renderer/__init__.py` | Expands `screenspace_points` from 3 to 4 channels (channels 0–1: signed gradient; channels 2–3: absolute gradient) |
| `train.py` | Post-HO opacity-mode restore, passes dual thresholds to densification |
| `submodules/diff-gaussian-rasterization/rasterize_points.cu` | 4-channel `dL_dmeans2D` tensor |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h` | `float4*` declarations |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu` | Absolute gradient accumulators (`fabs`), atomic-add to channels 2–3 |

**The key design principle:** Taming's score-based sampling and budget schedule remain fully intact. AbsGS only changes **which large Gaussians are eligible** for splitting:

```
Clone candidates  →  norm(signed_gradient) >= densify_grad_threshold     (Taming default)
Split candidates  →  norm(abs_gradient)    >= densify_grad_abs_threshold  (AbsGS contribution)
```

Small Gaussians (`max(scale) ≤ percent_dense × extent`) are cloned using Taming's normal gradient — preserving the ability to populate empty regions. Large Gaussians (`max(scale) > percent_dense × extent`) are split using AbsGS's absolute gradient — preventing the cancellation problem on under-resolved structures. Taming scores still **rank** all candidates, so the budget schedule and quality-driven pruning remain in effect.

### Scoring Formula

The competition evaluates rendered images against hidden ground truth using a weighted combination:

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
```

| Metric | What it measures | Target |
|---|---|---|
| **LPIPS** (Zhang et al., CVPR 2018) | Perceptual similarity via deep features (AlexNet) | Lower |
| **SSIM** | Structural similarity (luminance, contrast, structure) | Higher |
| **PSNR** (normalized) | Pixel-level error, clamped to `[0, 1]` | Higher |

The leaderboard score is the **average across all scenes**.

---

## Project Structure

```
ViettelAIRace2026/
├── VAI_NVS_CODE/                       # Deployable source bundle (Kaggle-ready)
│   ├── scripts/                        # Pipeline automation
│   │   ├── build_taming_absgs.py       #   Generate hybrid Taming-AbsGS workspace
│   │   ├── verify_taming_absgs.py      #   Validate hybrid workspace integrity
│   │   ├── train_parallel.py           #   Multi-GPU training launcher
│   │   ├── render_parallel.py          #   Parallel test-pose rendering (2 GPUs)
│   │   ├── render_poses.py             #   Single-GPU test-pose renderer
│   │   ├── evaluate_predictions.py     #   Local scoring (LPIPS/SSIM/PSNR)
│   │   ├── inspect_dataset.py          #   Dataset structure auditor
│   │   ├── verify_colmap_scene.py      #   COLMAP model validator
│   │   ├── show_rendered_images.py     #   Side-by-side render vs. GT viewer
│   │   ├── nvs_utils.py               #   Shared utilities (path/pose/COLMAP I/O)
│   │   ├── kaggle_taming_absgs.md      #   Kaggle runbook (single GPU)
│   │   └── kaggle_taming_absgs_dual_gpu.md  # Kaggle runbook (dual GPU)
│   └── external/                       # 3DGS variant implementations
│       ├── taming-3dgs/                #   Taming-3DGS (score-based budget control)
│       └── absgs/                      #   AbsGS (absolute gradient split)
├── external/                           # Local copies of external repositories
│   ├── gaussian-splatting/             #   Original 3DGS (Inria, 2023)
│   ├── taming-3dgs/                    #   Taming-3DGS
│   └── absgs/                          #   AbsGS
├── scripts/                            # Local mirror of VAI_NVS_CODE/scripts/
├── tests/                              # Pytest test suite
│   ├── conftest.py                     #   Shared fixtures
│   ├── test_nvs_utils.py              #   Utility function tests
│   ├── test_renderer_paths.py          #   Renderer path contract tests
│   ├── test_cli_contracts.py           #   CLI contract tests
│   └── test_display_scenes.py          #   Display utility tests
├── docs/                               # Documentation
│   ├── topic.md                        #   Competition problem description
│   ├── environment_setup.md            #   Local environment setup guide
│   ├── phase1_nvs_high_score_plan.md   #   Phase 1 execution plan
│   └── superpowers/                    #   Agentic development artifacts
├── configs/                            # Training configuration files
├── .gitignore
├── README.md
└── LICENSE
```

> **Note:** `VAI_NVS_DATA/` (competition dataset, ~5 GB), `outputs/` (checkpoints & renders), `.conda-envs/`, and large binary assets (`.pdf`, `.zip`) are excluded via `.gitignore`. Dataset must be obtained from the competition organizers.

---

## Setup

### Requirements

- **OS:** Linux (Kaggle) or Windows (local dev)
- **GPU:** NVIDIA GPU with CUDA 12.1+
- **Python:** 3.10+
- **Build tools:** Visual Studio Build Tools (Windows) or GCC (Linux)

### Environment

```bash
# Create conda environment
conda create -n gaussian_splatting python=3.10
conda activate gaussian_splatting

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Base dependencies
pip install plyfile tqdm opencv-python pillow numpy scipy matplotlib lpips

# Build CUDA extensions from the hybrid workspace
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
```

See [docs/environment_setup.md](docs/environment_setup.md) for detailed Windows-specific build notes.

---

## Usage

### 1. Generate the Hybrid Workspace

```bash
python scripts/build_taming_absgs.py \
    --taming-root VAI_NVS_CODE/external/taming-3dgs \
    --absgs-root VAI_NVS_CODE/external/absgs \
    --output-root outputs/taming-absgs \
    --overwrite

python scripts/verify_taming_absgs.py \
    --hybrid-root outputs/taming-absgs
```

### 2. Inspect Dataset

```bash
python scripts/inspect_dataset.py --data_dir VAI_NVS_DATA/phase1
```

### 3. Training

```bash
# Single scene
python outputs/taming-absgs/train.py \
    -s VAI_NVS_DATA/phase1/public_set/hcm0031/train \
    -m outputs/checkpoints/hcm0031 \
    --iterations 30000

# Multiple scenes across 2 GPUs
python scripts/train_parallel.py \
    --hybrid-root outputs/taming-absgs \
    --model-dir outputs/checkpoints \
    --scenes "hcm0031:/data/hcm0031/train:hcm0031" \
             "hcm0034:/data/hcm0034/train:hcm0034"
```

### 4. Render Test Poses

```bash
# Single GPU
python scripts/render_poses.py \
    --taming-root outputs/taming-absgs \
    --model-path outputs/checkpoints/hcm0031 \
    --poses-csv VAI_NVS_DATA/phase1/public_set/hcm0031/test/test_poses.csv \
    --output-dir outputs/renders/hcm0031

# Parallel across 2 GPUs
python scripts/render_parallel.py \
    --taming-root outputs/taming-absgs \
    --model-path outputs/checkpoints/hcm0031 \
    --poses-csv VAI_NVS_DATA/phase1/public_set/hcm0031/test/test_poses.csv \
    --output-dir outputs/renders/hcm0031
```

### 5. Local Evaluation (public set only)

```bash
python scripts/evaluate_predictions.py \
    --pred_dir outputs/renders \
    --gt_dir VAI_NVS_DATA/phase1/public_set
```

### 6. Package Submission

```bash
python scripts/package_submission.py \
    --render_dir outputs/renders \
    --output submission.zip
```

---

## Submission Format

```
submission.zip
├── hcm0031/
│   ├── DJI_20241227155343_0023_V.JPG
│   └── ...
├── hcm0034/
│   └── ...
└── ...
```

**Requirements:**
- Correct scene count and scene IDs
- Correct image filenames (as specified in `test_poses.csv`)
- Correct image dimensions (`width × height` from `test_poses.csv`)
- Correct number of images per scene

> ⚠️ Missing or extra scenes/images will cause the submission to be **rejected**.

---

## Competition Schedule

| Round | Period | Format |
|---|---|---|
| Round 1 — Qualification | Jul 1 – Jul 30, 2026 | ZIP upload (GPU) |
| Round 2 — Semi-finals | Aug 16 – Aug 19, 2026 | ZIP upload (GPU) |
| Round 3 — Finals | Sep 8 – Sep 10, 2026 | ZIP upload (GPU) |

---

## Rules

- Only competition-provided data may be used for training
- No external data related to the competition scenes or objects
- No access or inference of private test ground truth
- No manual editing of output images — all renders must be fully automated
- Results must be reproducible (source code, configs, dependencies, checkpoints, logs)

---

## References

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl, Kopanas, Leimkühler & Drettakis, SIGGRAPH 2023
- [Taming 3DGS: High-Quality Radiance Fields with Controlled Gaussian Count](https://github.com/humansensinglab/taming-3dgs) — Mallick et al., ECCV 2024
- [AbsGS: Recovering Fine Details in 3D Gaussian Splatting](https://github.com/Zhenyu-Yang22/AbsGS) — Yang et al., ACM MM 2024
- [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) — Zhang, Isola, Efros, Shechtman & Wang, CVPR 2018

---

## License

This project is developed for the Viettel AI Race 2026 competition. See [LICENSE](LICENSE) for details. External repositories under `external/` and `VAI_NVS_CODE/external/` retain their original licenses.
