# Environment Setup

This project uses a local Conda environment for the Gaussian Splatting baseline:

```powershell
F:\Projects\ViettelAIRace2026\.conda-envs\gaussian_splatting_cuda12
```

The upstream `external/gaussian-splatting/environment.yml` was tested first, but its Python 3.7 + PyTorch 1.12 CUDA 11.6 stack fails on this Windows machine with:

```text
OSError: [WinError 182] Error loading torch\lib\shm.dll
```

The working environment uses Python 3.10 and PyTorch CUDA 12.1, which matches the installed CUDA 12.x toolkit closely enough for local CUDA extension builds.

## Activate

From PowerShell:

```powershell
& C:\Users\huylu\miniconda3\shell\condabin\conda-hook.ps1
conda activate F:\Projects\ViettelAIRace2026\.conda-envs\gaussian_splatting_cuda12
cd F:\Projects\ViettelAIRace2026\external\gaussian-splatting
```

Or without activating:

```powershell
C:\Users\huylu\miniconda3\Scripts\conda.exe run --prefix F:\Projects\ViettelAIRace2026\.conda-envs\gaussian_splatting_cuda12 python train.py --help
```

## Verified Versions

```text
Python: 3.10.20
PyTorch: 2.5.1+cu121
PyTorch CUDA runtime: 12.1
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
nvcc: 12.8
```

Installed local CUDA extensions:

```text
diff_gaussian_rasterization
simple_knn
fused_ssim
```

## Build Notes

CUDA extensions were built from:

```text
external/gaussian-splatting/submodules/diff-gaussian-rasterization
external/gaussian-splatting/submodules/simple-knn
external/gaussian-splatting/submodules/fused-ssim
```

On Windows, build them from a Visual Studio Build Tools environment:

```powershell
cmd.exe /d /s /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set DISTUTILS_USE_SDK=1 && C:\Users\huylu\miniconda3\Scripts\conda.exe run --prefix F:\Projects\ViettelAIRace2026\.conda-envs\gaussian_splatting_cuda12 python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization"
cmd.exe /d /s /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set DISTUTILS_USE_SDK=1 && C:\Users\huylu\miniconda3\Scripts\conda.exe run --prefix F:\Projects\ViettelAIRace2026\.conda-envs\gaussian_splatting_cuda12 python -m pip install --no-build-isolation submodules/simple-knn"
cmd.exe /d /s /c "`"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set DISTUTILS_USE_SDK=1 && C:\Users\huylu\miniconda3\Scripts\conda.exe run --prefix F:\Projects\ViettelAIRace2026\.conda-envs\gaussian_splatting_cuda12 python -m pip install --no-build-isolation submodules/fused-ssim"
```

## Dataset Reader Patch

The Viettel Phase 1 COLMAP sparse model for `hcm0031` uses `SIMPLE_RADIAL`, while upstream Gaussian Splatting only accepted `PINHOLE` and `SIMPLE_PINHOLE`. The local loader now treats radial/OPENCV COLMAP cameras as pinhole cameras for training FOV calculation.

The sparse model also contains camera entries whose image files are not present under `train/images`. The loader skips those missing images. For `hcm0031`, training uses 200 real images and skips 188 missing COLMAP entries.

## Smoke Test

This command completed 1000 iterations:

```powershell
python train.py -s ../../VAI_NVS_DATA/phase1/public_set/hcm0031/train -m ../../outputs/checkpoints/smoke_hcm0031_1000iter --iterations 1000 --save_iterations 1000 --disable_viewer
```

Output:

```text
outputs/checkpoints/smoke_hcm0031_1000iter/
  cfg_args
  input.ply
  cameras.json
  exposure.json
  point_cloud/iteration_1000/point_cloud.ply
```

