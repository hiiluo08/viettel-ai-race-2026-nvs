# Viettel AI Race 2026 — Novel View Synthesis (BTS Digital Twin)

> **Giải đấu AI cho kỹ sư Việt Nam — Bài 1: BTS Digital Twin**
>
> Tổ chức bởi: **Viettel**
>
> Trang chủ: [competition.viettel.vn/contests/var-2026](https://competition.viettel.vn/contests/var-2026)

---

## Tổng Quan / Overview

Bài toán yêu cầu xây dựng mô hình AI có khả năng **tái dựng cấu trúc không gian 3D** của một scene từ tập ảnh đa góc nhìn và **sinh ra ảnh tại các góc nhìn mới** (Novel View Synthesis) chưa từng xuất hiện trong dữ liệu đầu vào.

**Input:** Tập ảnh đa góc nhìn + camera poses + sparse reconstruction từ COLMAP

**Output:** Ảnh RGB tại các test poses được chỉ định

**Đối tượng:** Trạm BTS, công trình hạ tầng viễn thông, các đối tượng thực tế khác

**Lĩnh vực:** Computer Vision · 3D Vision · Neural Rendering · Digital Twin

---

## Kiến Trúc Dự Án / Project Structure

```
ViettelAIRace2026/
├── VAI_NVS_CODE/                  # Mã nguồn chính
│   ├── scripts/                   # Pipeline scripts
│   │   ├── train_parallel.py      # Training đa GPU / đa scene
│   │   ├── render_parallel.py     # Render đa GPU / đa scene
│   │   ├── evaluate_predictions.py # Đánh giá local (LPIPS/SSIM/PSNR)
│   │   ├── render_poses.py        # Render từ checkpoint
│   │   ├── nvs_utils.py           # Tiện ích dùng chung
│   │   ├── inspect_dataset.py     # Kiểm tra dataset
│   │   ├── verify_colmap_scene.py # Xác minh COLMAP scene
│   │   ├── build_taming_absgs.py  # Build pipeline Taming-3DGS + AbsGS
│   │   └── verify_taming_absgs.py # Verify pipeline
│   └── external/                  # Các phương pháp 3DGS
│       ├── taming-3dgs/           # Taming-3DGS
│       └── absgs/                 # AbsGS (anti-aliased 3DGS)
├── external/                      # Bản sao các external repos
│   ├── gaussian-splatting/        # 3D Gaussian Splatting gốc
│   ├── taming-3dgs/               # Taming-3DGS
│   └── absgs/                     # AbsGS
├── configs/                       # File cấu hình training
├── scripts/                       # Scripts tiện ích bổ trợ
├── tests/                         # Unit tests
│   ├── test_nvs_utils.py
│   ├── test_renderer_paths.py
│   ├── test_cli_contracts.py
│   └── test_display_scenes.py
├── docs/                          # Tài liệu
│   ├── topic.md                   # Chi tiết bài toán
│   ├── environment_setup.md       # Hướng dẫn cài đặt môi trường
│   └── phase1_nvs_high_score_plan.md  # Kế hoạch đạt điểm cao
├── VAI_NVS_DATA/                  # Dữ liệu (KHÔNG lên GitHub)
│   └── phase1/
│       ├── public_set/            # 5 scenes (có GT test images)
│       └── private_set1/          # 8 scenes (không có GT)
├── outputs/                       # Output training (KHÔNG lên GitHub)
└── .conda-envs/                   # Conda environment (KHÔNG lên GitHub)
```

---

## Phương Pháp

Pipeline sử dụng các biến thể của **3D Gaussian Splatting (3DGS)** làm baseline:

| Phương pháp | Đặc điểm |
|---|---|
| **3D Gaussian Splatting** | Baseline gốc từ [Kerbl et al. 2023](https://github.com/graphdeco-inria/gaussian-splatting) |
| **Taming-3DGS** | Phiên bản tinh chỉnh với cải thiện về chất lượng rendering và ổn định training |
| **AbsGS** | Biến thể anti-aliased, giảm artifacts ở các góc nhìn xa |

Chiến lược tổng thể:
1. **Train** mô hình 3DGS trên mỗi scene với COLMAP sparse model
2. **Render** ảnh tại test poses
3. **Evaluate** local score trên public set (có ground truth)
4. **Package** submission ZIP đúng format BTC yêu cầu

---

## Công Thức Tính Điểm

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
```

| Metric | Mô tả | Hướng |
|---|---|---|
| **LPIPS** | Perceptual similarity (deep features) | Càng thấp càng tốt |
| **SSIM** | Structural similarity | Càng cao càng tốt |
| **PSNR** | Pixel-level error (normalized) | Càng cao càng tốt |

Điểm trên bảng xếp hạng là **trung bình của toàn bộ các scene**.

---

## Cài Đặt / Setup

### Yêu cầu hệ thống

- **OS:** Windows (hỗ trợ Linux thông qua Kaggle notebook)
- **GPU:** NVIDIA GPU with CUDA 12.x
- **Python:** 3.10+
- **CUDA Toolkit:** 12.1+

### Cài đặt môi trường

```powershell
# Tạo conda environment
conda create -n gaussian_splatting python=3.10
conda activate gaussian_splatting

# Cài PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài các dependencies
pip install plyfile tqdm opencv-python pillow numpy scipy

# Build CUDA extensions (yêu cầu Visual Studio Build Tools)
# Từ thư mục VAI_NVS_CODE/external/taming-3dgs:
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim
```

Chi tiết xem [docs/environment_setup.md](docs/environment_setup.md).

---

## Sử Dụng / Usage

### 1. Kiểm tra dataset

```bash
python VAI_NVS_CODE/scripts/inspect_dataset.py --data_dir VAI_NVS_DATA/phase1
```

### 2. Training

```bash
# Train một scene
python VAI_NVS_CODE/external/taming-3dgs/train.py \
  -s VAI_NVS_DATA/phase1/public_set/hcm0031/train \
  -m outputs/checkpoints/hcm0031 \
  --iterations 30000

# Train song song nhiều scene
python VAI_NVS_CODE/scripts/train_parallel.py
```

### 3. Render test poses

```bash
python VAI_NVS_CODE/scripts/render_parallel.py
```

### 4. Đánh giá local (chỉ trên public set)

```bash
python VAI_NVS_CODE/scripts/evaluate_predictions.py \
  --pred_dir outputs/renders \
  --gt_dir VAI_NVS_DATA/phase1/public_set
```

### 5. Đóng gói submission

```bash
python VAI_NVS_CODE/scripts/package_submission.py \
  --render_dir outputs/renders \
  --output submission.zip
```

---

## Cấu Trúc Submission

```
submission.zip
├── hcm0031/
│   ├── DJI_20241227155343_0023_V.JPG
│   └── ...
├── hcm0034/
│   └── ...
└── ...
```

**Yêu cầu bắt buộc:**
- Đúng số lượng và tên scene
- Đúng tên file ảnh
- Đúng kích thước ảnh (theo `width x height` trong `test_poses.csv`)
- Đúng số lượng ảnh mỗi scene

---

## Lịch Thi

| Vòng | Thời gian | Hình thức |
|---|---|---|
| Vòng 1 — Sơ loại | 01/07/2026 → 30/07/2026 | File ZIP (GPU) |
| Vòng 2 — Sơ khảo | 16/08/2026 → 19/08/2026 | File ZIP (GPU) |
| Vòng 3 — Chung kết | 08/09/2026 → 10/09/2026 | File ZIP (GPU) |

---

## Quy Định

- Chỉ sử dụng dữ liệu do BTC cung cấp
- Không dùng dữ liệu ngoài liên quan đến scene/đối tượng trong đề
- Không truy xuất hoặc suy đoán ground-truth private test
- Không chỉnh sửa thủ công ảnh đầu ra
- Kết quả phải có khả năng tái lập (mã nguồn, config, dependencies, checkpoints, logs)

---

## Tài Liệu Tham Khảo

- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al., SIGGRAPH 2023
- [Taming-3DGS](https://github.com/humansensinglab/taming-3dgs) — Mallick et al., ECCV 2024
- [AbsGS](https://github.com/Zhenyu-Yang22/AbsGS) — Yang et al., 2024
- [LPIPS — The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924) — Zhang et al., CVPR 2018

---

## Tác Giả / Author

**huylu** (hopeSo00810)

---

## License

Dự án nghiên cứu phục vụ cuộc thi Viettel AI Race 2026. Các external repos (`external/`, `VAI_NVS_CODE/external/`) giữ license gốc của từng dự án.

---

*README prepared with Claude Code*
