# Kaggle pipeline: Taming-AbsGS

Pipeline nay tao mot workspace **Taming-AbsGS** trong `/kaggle/working` ma khong sua hai source repository duoc mount tu Kaggle Dataset.

- Taming-3DGS giu score-based sampling, final Gaussian `BUDGET`, loss va renderer mo rong cua repo.
- AbsGS bo sung absolute view-space gradient theo pixel chi de chon Gaussian **lon** cho nhanh `split`.
- Gaussian **nho** van duoc chon `clone` bang gradient thuong cua Taming.

Khong cai dat hoac chay `external/absgs/train.py`: repo nay la reference cho logic AbsGS. CUDA extension can build la extension cua **hybrid workspace**, da giu cac output statistics cua Taming va bo sung gradient AbsGS.

---

## 1. Kaggle settings

- Accelerator: GPU T4 hoac manh hon.
- Internet: bat neu can tai weights LPIPS/VGG16 lan dau hoac upload Hugging Face.
- Attach hai Kaggle datasets: code (`VAI_NVS_CODE`) va data (`VAI_NVS_DATA`).

---

## 2. Khai bao path va copy code sang thu muc writable

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

# Dataset mount la read-only; tat ca build va training deu dien ra trong WORK_ROOT.
!cp -a "$CODE_ROOT"/. "$WORK_ROOT"/
```

---

## 3. Tao source Taming-AbsGS hybrid

```python
!python "$WORK_ROOT/scripts/build_taming_absgs.py" \
    --taming-root "$TAMING_SOURCE_ROOT" \
    --absgs-root "$ABSGS_ROOT" \
    --output-root "$HYBRID_ROOT" \
    --overwrite

!python "$WORK_ROOT/scripts/verify_taming_absgs.py" \
    --hybrid-root "$HYBRID_ROOT"
```

Sau cell nay, `HYBRID_ROOT` la Taming copy da duoc patch. Hai folder `external/taming-3dgs` va `external/absgs` van khong thay doi.

---

## 4. Cai packages va build CUDA extensions cua hybrid

```bash
%%bash
set -euo pipefail

WORK_ROOT=/kaggle/working/vai_nvs_code
HYBRID_ROOT="$WORK_ROOT/external/taming-absgs"

pip install -q plyfile tqdm Pillow matplotlib lpips torchvision huggingface_hub
pip install -q --upgrade setuptools wheel ninja

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="7.5"  # Kaggle T4; doi neu dung GPU khac

cd "$HYBRID_ROOT"
pip install -v --no-build-isolation submodules/simple-knn 2>&1 | tail -60
pip install -v --no-build-isolation submodules/diff-gaussian-rasterization 2>&1 | tail -80
pip install -v --no-build-isolation submodules/fused-ssim 2>&1 | tail -60

python - <<'PY'
import torch
from diff_gaussian_rasterization import GaussianRasterizer

print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Patched Taming rasterizer import: OK")
PY
```

Neu Kaggle khong dung T4, thay `TORCH_CUDA_ARCH_LIST` bang compute capability phu hop; co the bo dong nay neu khong chac.

---

## 5. Khai bao scene va output paths

```python
SET_NAME = "public_set"       # hoac "private_set1"
SCENE_ID = "hcm0031"

TRAIN_DIR = f"{DATA_ROOT}/{SET_NAME}/{SCENE_ID}/train"
POSES_CSV = f"{DATA_ROOT}/{SET_NAME}/{SCENE_ID}/test/test_poses.csv"
GT_IMAGES_DIR = f"{DATA_ROOT}/{SET_NAME}/{SCENE_ID}/test/images"  # chi public set

MODEL_DIR = f"{OUTPUT_ROOT}/checkpoints/{SCENE_ID}_taming_absgs"
RENDER_DIR = f"{OUTPUT_ROOT}/renders/{SCENE_ID}_taming_absgs"
REPORT_DIR = f"{OUTPUT_ROOT}/eval_reports"
MANIFEST_DIR = f"{OUTPUT_ROOT}/manifests"

import os
for path in (MODEL_DIR, RENDER_DIR, REPORT_DIR, MANIFEST_DIR):
    os.makedirs(path, exist_ok=True)
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

## 7. Khai bao hyperparameters

```python
ITERATIONS = 30_000
BUDGET = 3_000_000
MODE = "final_count"

CAMS = 20                         # phai <= so train cameras; -1 dung tat ca
LAMBDA_DSSIM = 0.2

DENSIFY_FROM_ITER = 500
DENSIFY_UNTIL_ITER = 15_000
DENSIFICATION_INTERVAL = 100

# Normal gradient: clone Gaussian nho (Taming).
DENSIFY_GRAD_THRESHOLD = 0.0002
# Absolute per-pixel gradient: split Gaussian lon (AbsGS).
DENSIFY_GRAD_ABS_THRESHOLD = 0.0004

# Dat bang densify-until de mo opacity limit sau khi densification ket thuc.
HO_ITERATION = DENSIFY_UNTIL_ITER

SAVE_ITERATIONS = 30_000
CHECKPOINT_ITERATIONS = 30_000
```

`DENSIFY_GRAD_ABS_THRESHOLD=0.0004` la baseline AbsGS. A/B test cac gia tri `0.0004`, `0.0006`, `0.0008`; giu nguyen seed, budget, iterations va cac tham so khac.

---

## 8. Train Taming-AbsGS

```python
%cd "$HYBRID_ROOT"
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    -s "$TRAIN_DIR" \
    -m "$MODEL_DIR" \
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

`--test_iterations -1` bo evaluation dinh ky; render/evaluate sau khi train.

---

### 8.1. Resume tu checkpoint

Dat `START_CHECKPOINT` toi `chkpnt<iteration>.pth`. `TARGET_ITERATIONS` la
**tong** so iteration sau khi resume, khong phai so iteration can train them;
vi du checkpoint `chkpnt30000.pth` va muon train them 30k thi dat 60k.
`HO_ITERATION` bat buoc phai trung voi luc train checkpoint goc. Hybrid tu dong
khoi phuc absolute-opacity mode neu checkpoint nam sau moc nay.

```python
START_CHECKPOINT = f"{MODEL_DIR}/chkpnt30000.pth"  # sua theo checkpoint thuc te
TARGET_ITERATIONS = 60_000

# Giu nguyen cac tham so da tao checkpoint, tru khi chu dong mo mot experiment moi.
%cd "$HYBRID_ROOT"
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py \
    -s "$TRAIN_DIR" \
    -m "$MODEL_DIR" \
    --start_checkpoint "$START_CHECKPOINT" \
    --iterations "$TARGET_ITERATIONS" \
    --position_lr_max_steps "$TARGET_ITERATIONS" \
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
    --save_iterations "$TARGET_ITERATIONS" \
    --checkpoint_iterations "$TARGET_ITERATIONS" \
    --test_iterations -1
```

Neu checkpoint da o sau `DENSIFY_UNTIL_ITER`, densification se khong tu chay
lai: day la continuation dung cua lich train cu. Chi mo rong
`DENSIFY_UNTIL_ITER` khi ban co chu dich thu nghiem densification moi.

---

## 9. Render test poses

```python
%cd "$WORK_ROOT"
!python scripts/render_poses.py \
    --taming-root "$HYBRID_ROOT" \
    --model-path "$MODEL_DIR" \
    --poses-csv "$POSES_CSV" \
    --output-dir "$RENDER_DIR" \
    --iteration -1
```

---

## 10. Xem anh render

```python
%matplotlib inline
import importlib
import sys

sys.path.insert(0, WORK_ROOT)
import scripts.show_rendered_images as show
importlib.reload(show)

# Public set: them --gt-images=GT_IMAGES_DIR.
show.main(["--images-dir", RENDER_DIR, "--gt-images", GT_IMAGES_DIR])
```

Voi private set, go bo hai tham so `--gt-images`, `GT_IMAGES_DIR`.

---

## 11. Evaluate public set

```python
%cd "$WORK_ROOT"
!python scripts/evaluate_predictions.py \
    --taming-root "$HYBRID_ROOT" \
    --pred-images "$RENDER_DIR" \
    --gt-images "$GT_IMAGES_DIR" \
    --experiment-name "${SCENE_ID}_taming_absgs" \
    --output-dir "$REPORT_DIR"
```

So sanh file CSV cua run nay voi Taming baseline. Chi ket luan AbsGS co loi khi cung scene, `BUDGET`, `ITERATIONS`, seed va evaluation pipeline.

---

## 12. Upload output len Hugging Face (tuy chon)

```python
from huggingface_hub import HfApi
import os

HF_REPO_ID = "your-username/your-repo-name"
HF_REPO_TYPE = "model"  # hoac "dataset"
HF_PATH_IN_REPO = "phase1_outputs"

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    folder_path=OUTPUT_ROOT,
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
    path_in_repo=HF_PATH_IN_REPO,
)
```

Dat `HF_TOKEN` trong Kaggle Secrets, khong ghi token truc tiep vao notebook.
