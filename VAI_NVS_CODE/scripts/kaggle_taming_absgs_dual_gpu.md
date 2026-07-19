# Kaggle pipeline: Taming-AbsGS (Dual GPU – T4 x2)

Pipeline này tạo workspace **Taming-AbsGS** trong `/kaggle/working`, tận dụng cả **2 GPU T4** để train 2 scene song song, render song song, tiết kiệm ~40-50% thời gian so với single GPU.

- Taming-3DGS giữ score-based sampling, final Gaussian `BUDGET`, loss và renderer mở rộng.
- AbsGS bổ sung absolute view-space gradient theo pixel để chọn Gaussian **lớn** cho `split`.
- Gaussian **nhỏ** vẫn được chọn `clone` bằng gradient thường của Taming.

---

## 1. Kaggle settings

- **Accelerator:** GPU T4 x2 (bắt buộc).
- **Internet:** bật nếu cần tải weights LPIPS/VGG16 hoặc upload Hugging Face.
- Attach hai Kaggle datasets: code (`VAI_NVS_CODE`) và data (`VAI_NVS_DATA`).

---

## 2. Khai báo path và copy code sang thư mục writable

```python
CODE_ROOT = "/kaggle/input/<CODE_DATASET>/VAI_NVS_CODE"
WORK_ROOT = "/kaggle/working/vai_nvs_code"

TAMING_SOURCE_ROOT = f"{WORK_ROOT}/external/taming-3dgs"
ABSGS_ROOT = f"{WORK_ROOT}/external/absgs"
HYBRID_ROOT = f"{WORK_ROOT}/external/taming-absgs"

DATA_ROOT = "/kaggle/input/<DATA_DATASET>/VAI_NVS_DATA/phase1"
OUTPUT_ROOT = "/kaggle/working/vai_nvs_outputs"

import os
import shutil

if os.path.exists(WORK_ROOT):
    shutil.rmtree(WORK_ROOT)
os.makedirs(WORK_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Dataset mount là read-only; tất cả build và training đều diễn ra trong WORK_ROOT.
!cp -a "$CODE_ROOT"/. "$WORK_ROOT"/
```

---

## 3. Tạo source Taming-AbsGS hybrid

```python
!python "$WORK_ROOT/scripts/build_taming_absgs.py" \
    --taming-root "$TAMING_SOURCE_ROOT" \
    --absgs-root "$ABSGS_ROOT" \
    --output-root "$HYBRID_ROOT" \
    --overwrite

!python "$WORK_ROOT/scripts/verify_taming_absgs.py" \
    --hybrid-root "$HYBRID_ROOT"
```

---

## 4. Cài packages và build CUDA extensions

```bash
%%bash
set -euo pipefail

WORK_ROOT=/kaggle/working/vai_nvs_code
HYBRID_ROOT="$WORK_ROOT/external/taming-absgs"

pip install -q plyfile tqdm Pillow matplotlib lpips torchvision huggingface_hub
pip install -q --upgrade setuptools wheel ninja

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.5"  # Kaggle T4

cd "$HYBRID_ROOT"
pip install -v --no-build-isolation submodules/simple-knn 2>&1 | tail -60
pip install -v --no-build-isolation submodules/diff-gaussian-rasterization 2>&1 | tail -80
pip install -v --no-build-isolation submodules/fused-ssim 2>&1 | tail -60

python - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizer

print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print("Patched Taming rasterizer import: OK")
PY
```

---

## 5. Khai báo scene, output paths và hyperparameters

```python
SET_NAME = "public_set"       # hoac "private_set1"
SCENE_IDS = ["hcm0031", "hcm0034"]

# ---------- Hyperparameters ----------
ITERATIONS = 30_000
BUDGET = 3_000_000
MODE = "final_count"

CAMS = 20
LAMBDA_DSSIM = 0.2

DENSIFY_FROM_ITER = 500
DENSIFY_UNTIL_ITER = 15_000
DENSIFICATION_INTERVAL = 100

DENSIFY_GRAD_THRESHOLD = 0.0002
DENSIFY_GRAD_ABS_THRESHOLD = 0.0004

HO_ITERATION = DENSIFY_UNTIL_ITER
SAVE_ITERATIONS = 30_000
CHECKPOINT_ITERATIONS = 30_000

# ---------- Experiment tag (ma hoa hyperparams vao path) ----------
EXP_TAG = (
    f"it-{ITERATIONS}_bud-{BUDGET}_cam-{CAMS}_ldssim-{LAMBDA_DSSIM}"
    f"_dens-interval-{DENSIFICATION_INTERVAL}"
    f"_from-{DENSIFY_FROM_ITER}_to-{DENSIFY_UNTIL_ITER}"
    f"_grad-{DENSIFY_GRAD_THRESHOLD}_agrad-{DENSIFY_GRAD_ABS_THRESHOLD}"
)

# ---------- Build paths cho tung scene ----------
SCENES = {}
for sid in SCENE_IDS:
    scene_root = f"{OUTPUT_ROOT}/{sid}"
    SCENES[sid] = {
        "train_dir":   f"{DATA_ROOT}/{SET_NAME}/{sid}/train",
        "poses_csv":   f"{DATA_ROOT}/{SET_NAME}/{sid}/test/test_poses.csv",
        "gt_images":   f"{DATA_ROOT}/{SET_NAME}/{sid}/test/images",
        "model_dir":   f"{scene_root}/checkpoints/{sid}_{EXP_TAG}",
        "render_dir":  f"{scene_root}/renders/{sid}_{EXP_TAG}",
        "report_dir":  f"{scene_root}/reports/{sid}_{EXP_TAG}",
    }

MANIFEST_DIR = f"{OUTPUT_ROOT}/manifests"

import os
os.makedirs(MANIFEST_DIR, exist_ok=True)
for sid, cfg in SCENES.items():
    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["render_dir"], exist_ok=True)
    os.makedirs(cfg["report_dir"], exist_ok=True)
```

---

## 6. Audit dataset va COLMAP verification

```python
%cd "$WORK_ROOT"
!python scripts/inspect_dataset.py --data-root "$DATA_ROOT" --output-dir "$MANIFEST_DIR"
!python scripts/verify_colmap_scene.py \
    --data-root "$DATA_ROOT" \
    --output-file "$MANIFEST_DIR/colmap_check.md" \
    --taming-root "$HYBRID_ROOT"
```

---

## 7. Train 2 scenes song song voi dual GPU

`train_parallel.py` tu dong phan phoi moi scene vao 1 GPU rieng biet qua `CUDA_VISIBLE_DEVICES`.

```python
%cd "$WORK_ROOT"

# MODEL_SUFFIX = subpath tu OUTPUT_ROOT toi model_dir
# Vi --model-dir = OUTPUT_ROOT nen MODEL_SUFFIX = {sid}/checkpoints/{sid}_{EXP_TAG}

scene_spec_0 = f"{SCENE_IDS[0]}:{SCENES[SCENE_IDS[0]]['train_dir']}:{SCENE_IDS[0]}/checkpoints/{SCENE_IDS[0]}_{EXP_TAG}"
scene_spec_1 = f"{SCENE_IDS[1]}:{SCENES[SCENE_IDS[1]]['train_dir']}:{SCENE_IDS[1]}/checkpoints/{SCENE_IDS[1]}_{EXP_TAG}"

!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_parallel.py \
    --hybrid-root "$HYBRID_ROOT" \
    --scenes "{scene_spec_0}" "{scene_spec_1}" \
    --model-dir "$OUTPUT_ROOT" \
    --gpus 0,1 \
    --iterations "$ITERATIONS" \
    --budget "$BUDGET" \
    --mode "$MODE" \
    --cams "$CAMS" \
    --lambda-dssim "$LAMBDA_DSSIM" \
    --densify-from-iter "$DENSIFY_FROM_ITER" \
    --densify-until-iter "$DENSIFY_UNTIL_ITER" \
    --densification-interval "$DENSIFICATION_INTERVAL" \
    --densify-grad-threshold "$DENSIFY_GRAD_THRESHOLD" \
    --densify-grad-abs-threshold "$DENSIFY_GRAD_ABS_THRESHOLD" \
    --save-iterations "$SAVE_ITERATIONS" \
    --checkpoint-iterations "$CHECKPOINT_ITERATIONS"
```

**Logs** moi scene: `$OUTPUT_ROOT/parallel_logs/train_<label>_gpu<N>.log`

### 7.1. Train tung scene rieng le (fallback single GPU)

```python
%cd "$HYBRID_ROOT"
cfg = SCENES[SCENE_IDS[0]]
!CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    -s "{cfg['train_dir']}" \
    -m "{cfg['model_dir']}" \
    --iterations "$ITERATIONS" \
    --budget "$BUDGET" \
    --mode "$MODE" \
    --cams "$CAMS" \
    --lambda_dssim "$LAMBDA_DSSIM" \
    --densify_from_iter "$DENSIFY_FROM_ITER" \
    --densify_until_iter "$DENSIFY_UNTIL_ITER" \
    --densification_interval "$DENSIFICATION_INTERVAL" \
    --densify_grad_threshold "$DENSIFY_GRAD_THRESHOLD" \
    --densify_grad_abs_threshold "$DENSIFY_GRAD_ABS_THRESHOLD" \
    --ho_iteration "$HO_ITERATION" \
    --save_iterations "$SAVE_ITERATIONS" \
    --checkpoint_iterations "$CHECKPOINT_ITERATIONS" \
    --test_iterations -1
```

---

### 7.2. A/B test 2 threshold AbsGS cung luc (cung 1 scene, 2 GPU)

```python
cfg = SCENES["hcm0031"]
SID = "hcm0031"

!python scripts/train_parallel.py \
    --hybrid-root "$HYBRID_ROOT" \
    --scenes "abs04:{cfg['train_dir']}:{SID}/checkpoints/{SID}_{EXP_TAG}_abs04" \
             "abs06:{cfg['train_dir']}:{SID}/checkpoints/{SID}_{EXP_TAG}_abs06" \
    --model-dir "$OUTPUT_ROOT" \
    --gpus 0,1 \
    --iterations "$ITERATIONS" \
    --budget "$BUDGET" \
    --mode "$MODE" \
    --cams "$CAMS" \
    --lambda-dssim "$LAMBDA_DSSIM" \
    --densify-from-iter "$DENSIFY_FROM_ITER" \
    --densify-until-iter "$DENSIFY_UNTIL_ITER" \
    --densify-grad-abs-threshold 0.0004 0.0006 \
    --save-iterations "$SAVE_ITERATIONS" \
    --checkpoint-iterations "$CHECKPOINT_ITERATIONS"
```

---

### 7.3. Resume 2 scenes tu checkpoint (dual GPU)

Moi scene co checkpoint rieng. Them `CHECKPOINT:TARGET_ITERATIONS` vao scene spec:

```python
%cd "$WORK_ROOT"

cfg0 = SCENES[SCENE_IDS[0]]
cfg1 = SCENES[SCENE_IDS[1]]
CKPT_0 = f"{cfg0['model_dir']}/chkpnt30000.pth"
CKPT_1 = f"{cfg1['model_dir']}/chkpnt30000.pth"
TARGET_ITER = 60_000  # tong iteration sau resume

!python scripts/train_parallel.py \
    --hybrid-root "$HYBRID_ROOT" \
    --scenes "{SCENE_IDS[0]}:{cfg0['train_dir']}:{SCENE_IDS[0]}/checkpoints/{SCENE_IDS[0]}_{EXP_TAG}:{CKPT_0}:{TARGET_ITER}" \
             "{SCENE_IDS[1]}:{cfg1['train_dir']}:{SCENE_IDS[1]}/checkpoints/{SCENE_IDS[1]}_{EXP_TAG}:{CKPT_1}:{TARGET_ITER}" \
    --model-dir "$OUTPUT_ROOT" \
    --gpus 0,1 \
    --budget "$BUDGET" \
    --mode "$MODE" \
    --cams "$CAMS" \
    --lambda-dssim "$LAMBDA_DSSIM" \
    --densify-from-iter "$DENSIFY_FROM_ITER" \
    --densify-until-iter "$DENSIFY_UNTIL_ITER" \
    --densification-interval "$DENSIFICATION_INTERVAL" \
    --densify-grad-threshold "$DENSIFY_GRAD_THRESHOLD" \
    --densify-grad-abs-threshold "$DENSIFY_GRAD_ABS_THRESHOLD"
```

Muon mo rong `DENSIFY_UNTIL_ITER` khi resume, them field thu 6 vao scene spec:

```
LABEL:TRAIN_DIR:MODEL_SUFFIX:CHECKPOINT:TARGET_ITERATIONS:DENSIFY_UNTIL_OVERRIDE
```

---

## 8. Render test poses song song (dual GPU)

`render_parallel.py` chia deu test poses cho 2 GPU, render dong thoi, roi gop ket qua.

```python
%cd "$WORK_ROOT"

for sid in SCENE_IDS:
    cfg = SCENES[sid]
    print(f"\n=== Rendering {sid} (parallel dual GPU) ===")
    !python scripts/render_parallel.py \
        --taming-root "$HYBRID_ROOT" \
        --model-path "{cfg['model_dir']}" \
        --poses-csv "{cfg['poses_csv']}" \
        --output-dir "{cfg['render_dir']}" \
        --iteration -1
```

Fallback single GPU render: dung `render_poses.py` nhu single-GPU guide.

---

## 9. Xem anh render

```python
%matplotlib inline
import importlib
import sys

sys.path.insert(0, WORK_ROOT)
import scripts.show_rendered_images as show
importlib.reload(show)

sid = SCENE_IDS[0]
cfg = SCENES[sid]
# Public set: them --gt-images
show.main(["--images-dir", cfg["render_dir"], "--gt-images", cfg["gt_images"]])
```

---

## 10. Evaluate public set

```python
%cd "$WORK_ROOT"

for sid in SCENE_IDS:
    cfg = SCENES[sid]
    if not cfg["gt_images"]:
        print(f"Skipping {sid}: no GT images (private set)")
        continue
    print(f"\n=== Evaluating {sid} ===")
    !python scripts/evaluate_predictions.py \
        --taming-root "$HYBRID_ROOT" \
        --pred-images "{cfg['render_dir']}" \
        --gt-images "{cfg['gt_images']}" \
        --experiment-name "{sid}_{EXP_TAG}" \
        --output-dir "{cfg['report_dir']}"
```

---

## 11. Upload output len Hugging Face (tuy chon)

---

## 12. Upload output lên Hugging Face (tùy chọn)

```python
from huggingface_hub import HfApi
import os

HF_REPO_ID = "your-username/your-repo-name"
HF_REPO_TYPE = "model"
HF_PATH_IN_REPO = "phase1_outputs"

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path=OUTPUT_ROOT,
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
    path_in_repo=HF_PATH_IN_REPO,
)
```

---

## Tóm tắt: Khác biệt so với single-GPU pipeline

| Task | Single GPU | Dual GPU (T4 x2) |
|---|---|---|
| Train 2 scenes | Tuần tự (~2x thời gian) | Song song `train_parallel.py` |
| Render test poses | 1 GPU render toàn bộ | `render_parallel.py` chia đôi poses |
| A/B test thresholds | 2 lần train tuần tự | 1 lần train song song |
| Logs | stdout | `parallel_logs/train_<label>_gpu<N>.log` |
| Build hybrid | Không đổi | Không đổi (không cần GPU) |
| Dataset inspect | Không đổi | Không đổi (không cần GPU) |
| Evaluate | Không đổi (nhanh) | Không đổi (tuần tự, evaluation nhẹ) |
