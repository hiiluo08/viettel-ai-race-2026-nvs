# Viettel AI Race 2026 Phase 1 NVS High-Score Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` hoặc `superpowers:executing-plans` nếu triển khai kế hoạch này theo từng task. Các bước dùng checkbox (`- [ ]`) để theo dõi tiến độ.

**Goal:** Xây dựng pipeline tự động để train, render, đánh giá và đóng gói submission cho bài toán **BTS Digital Twin / Novel View Synthesis** Phase 1, với mục tiêu tối đa hóa điểm theo công thức của BTC.

**Architecture:** Pipeline nên xoay quanh một baseline mạnh là **3D Gaussian Splatting**, sau đó xây thêm các lớp kiểm soát dữ liệu, render test poses, local evaluation trên public set, experiment tracking, tuning và submission verification. Mọi cải tiến phải được đo bằng validation offline trước khi áp dụng đại trà cho private scenes.

**Tech Stack:** Python, PyTorch, CUDA, COLMAP format, 3D Gaussian Splatting, LPIPS/SSIM/PSNR evaluator, shell/Python automation, image I/O bằng Pillow/OpenCV.

## Global Constraints

- Phase 1 deadline: **30/07/2026**.
- Submission là file ZIP chứa ảnh RGB cho toàn bộ test poses.
- Phải giữ đúng:
  - tên scene,
  - tên ảnh output,
  - số lượng ảnh,
  - resolution từng ảnh,
  - cấu trúc thư mục trong ZIP.
- Dataset mới theo từng vòng; pipeline phải tái chạy được trên dataset vòng sau.
- Không dùng dữ liệu ngoài liên quan trực tiếp đến scene/object trong đề.
- Không truy xuất hoặc suy đoán ground-truth private test.
- Không chỉnh sửa ảnh đầu ra thủ công.
- Kết quả phải có khả năng tái lập: source code, config, dependency versions, checkpoints, logs.

---

## 1. Tóm Tắt Yêu Cầu Bài Toán

Bài toán yêu cầu sinh ảnh RGB ở các góc nhìn mới từ một tập ảnh train đa góc nhìn đã có camera poses và sparse reconstruction từ COLMAP.

### Input mỗi scene

```text
<scene_id>/
├── train/
│   ├── images/
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── test/
    └── test_poses.csv
```

Trong dataset thực tế, `sparse/0/` còn có thêm:

```text
frames.bin
rigs.bin
points3D.ply    # chỉ có ở một số scene
```

Ngoài ra dataset có thư mục `__MACOSX/` do giải nén từ macOS, cần **bỏ qua hoàn toàn** khi scan dữ liệu.

### Output cần nộp

```text
submission.zip
├── <scene_id_1>/
│   ├── <image_name_1>.JPG hoặc .png
│   └── ...
├── <scene_id_2>/
│   └── ...
└── ...
```

### Metrics

Điểm cuối:

```text
Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
```

Hàm mục tiêu thực tế:

1. **Giảm LPIPS** vì trọng số cao nhất.
2. **Giữ SSIM cao** để bảo toàn cấu trúc BTS, dây, khung thép.
3. **Tăng PSNR** bằng màu sắc, exposure và background ổn định.
4. Không overfit chỉ vào public scenes; private scenes mới là phần quyết định leaderboard.

---

## 2. Khảo Sát Dataset Phase 1

Dữ liệu đang nằm trong:

```text
VAI_NVS_DATA/phase1/
├── public_set/
└── private_set1/
```

### 2.1. Public set

| Scene | Train images | Test poses | Test GT images | Output resolution |
|---|---:|---:|---:|---|
| `hcm0031` | 200 | 50 | 50 | 1320 × 989 |
| `hcm0034` | 240 | 60 | 60 | 1320 × 989 |
| `HCM0181` | 240 | 60 | 60 | 1320 × 989 |
| `HCM0193` | 240 | 60 | 60 | 1320 × 989 |
| `HCM0204` | 240 | 60 | 60 | 1320 × 989 |

Public set có ground truth test images, nên dùng để:

- kiểm tra renderer,
- tính local score,
- tuning hyperparameters,
- so sánh các biến thể model.

### 2.2. Private set

| Scene | Train images | Test poses | Test GT images | Output resolution | Ghi chú |
|---|---:|---:|---:|---|---|
| `HCM0249` | 240 | 60 | 0 | 1320 × 989 | Private |
| `HCM0254` | 240 | 60 | 0 | 1320 × 989 | Private |
| `HCM0276` | 240 | 60 | 0 | 1320 × 989 | Private |
| `HCM1439` | 103 | 26 | 0 | 1320 × 989 | Ít ảnh train, cần xử lý riêng |
| `HNI0131` | 240 | 60 | 0 | 1320 × 989 | Private |
| `HNI0265` | 205 | 52 | 0 | 1320 × 989 | Ít ảnh hơn chuẩn |
| `HNI0366` | 240 | 60 | 0 | 1320 × 989 | Private |
| `HNI0437` | 224 | 56 | 0 | 1320 × 989 | Ít ảnh hơn chuẩn |

### 2.3. Tổng khối lượng render

| Nhóm | Scenes | Train images | Test images cần render |
|---|---:|---:|---:|
| Public | 5 | 1160 | 290 |
| Private | 8 | 1732 | 434 |
| Tổng | 13 | 2892 | 724 |

### 2.4. Format `test_poses.csv`

Header thực tế:

```csv
image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height
```

Ví dụ:

```csv
DJI_20241227155343_0023_V.JPG,0.2613255563,-0.0010596123,0.5014572668,0.8247717505,-0.4055412371,-0.6586624884,3.3187491928,925.6781996712,925.6781996712,660.0,494.5,1320,989
```

Ý nghĩa quan trọng:

- `image_name`: tên file output phải khớp chính xác.
- `qw, qx, qy, qz`: quaternion theo convention COLMAP, thứ tự scalar-first.
- `tx, ty, tz`: translation world-to-camera theo COLMAP.
- `fx, fy, cx, cy`: intrinsics.
- `width, height`: phải render đúng kích thước.

---

## 3. Chiến Lược Tổng Thể Để Đạt Điểm Cao

### 3.1. Nguyên tắc chính

Không nên bắt đầu bằng việc thử quá nhiều method phức tạp. Thứ tự đúng là:

1. **Có baseline render đúng pose.**
2. **Có local evaluator đáng tin.**
3. **Có submission checker chống lỗi format.**
4. **Có experiment log để biết thay đổi nào thật sự tăng điểm.**
5. **Sau đó mới tuning model/loss/hyperparameters.**

Sai pose, sai tên ảnh hoặc sai ZIP format sẽ làm mất điểm nhiều hơn mọi cải tiến model.

### 3.2. Baseline nên chọn

Baseline khuyến nghị:

```text
3D Gaussian Splatting
```

Lý do:

- BTC gợi ý trực tiếp.
- Dataset đã có COLMAP sparse reconstruction.
- Bài toán là per-scene novel view synthesis, rất phù hợp với 3DGS.
- Test poses chủ yếu là interpolation giữa train views nên 3DGS có lợi thế mạnh.

### 3.3. Trọng tâm scoring

Vì LPIPS có trọng số 40%, pipeline nên ưu tiên:

- texture nhìn tự nhiên,
- hạn chế floaters,
- giữ chi tiết thanh BTS/dây/cạnh mảnh,
- tránh blur quá mức,
- tránh màu trời/background sai lệch lớn.

Tuy nhiên không được hy sinh geometry: nếu pose hoặc cấu trúc sai, cả LPIPS, SSIM, PSNR đều giảm.

---

## 4. File/Module Nên Tạo Trong Dự Án

> Đây là cấu trúc đề xuất để triển khai sau khi kế hoạch được duyệt.

```text
docs/
└── phase1_nvs_high_score_plan.md

scripts/
├── inspect_dataset.py
├── verify_colmap_scene.py
├── make_train_val_split.py
├── render_test_poses.py
├── evaluate_predictions.py
├── batch_train.py
├── batch_render.py
├── verify_submission.py
└── pack_submission.py

configs/
├── baseline_3dgs.yaml
├── high_quality_3dgs.yaml
└── small_scene_3dgs.yaml

experiments/
├── README.md
└── phase1_results.csv

outputs/
├── checkpoints/
├── renders/
├── eval_reports/
└── submissions/
```

### Trách nhiệm từng file

| File | Trách nhiệm |
|---|---|
| `scripts/inspect_dataset.py` | Scan dataset, đếm ảnh, kiểm tra CSV, xuất manifest |
| `scripts/verify_colmap_scene.py` | Kiểm tra `cameras.bin`, `images.bin`, `points3D.bin` đọc được |
| `scripts/make_train_val_split.py` | Tạo split validation nội bộ từ train nếu cần |
| `scripts/render_test_poses.py` | Load checkpoint và render theo `test_poses.csv` |
| `scripts/evaluate_predictions.py` | Tính LPIPS, SSIM, PSNR, score trên public GT |
| `scripts/batch_train.py` | Train hàng loạt scene theo config |
| `scripts/batch_render.py` | Render toàn bộ public/private test poses |
| `scripts/verify_submission.py` | Kiểm tra đủ scene, đủ ảnh, đúng tên, đúng resolution |
| `scripts/pack_submission.py` | Đóng gói ZIP đúng format |
| `configs/*.yaml` | Lưu hyperparameters có thể tái lập |
| `experiments/phase1_results.csv` | Ghi kết quả từng experiment |

---

## 5. Task 1 — Dataset Audit Và Manifest

**Mục tiêu:** Có một file manifest đáng tin mô tả toàn bộ dataset, tránh sai scene, sai số ảnh hoặc bỏ sót private scene.

**Files:**

- Create: `scripts/inspect_dataset.py`
- Create: `outputs/eval_reports/dataset_manifest_phase1.json`
- Create: `outputs/eval_reports/dataset_manifest_phase1.md`

**Công việc:**

- [ ] Scan `VAI_NVS_DATA/phase1/public_set`.
- [ ] Scan `VAI_NVS_DATA/phase1/private_set1`.
- [ ] Bỏ qua toàn bộ `__MACOSX`.
- [ ] Với mỗi scene, ghi:
  - scene name,
  - set name,
  - số ảnh train,
  - số dòng trong `test_poses.csv`,
  - số ảnh GT test nếu có,
  - resolution yêu cầu,
  - danh sách sparse files.
- [ ] Kiểm tra mọi scene đều có:
  - `train/images`,
  - `train/sparse/0/cameras.bin`,
  - `train/sparse/0/images.bin`,
  - `train/sparse/0/points3D.bin`,
  - `test/test_poses.csv`.
- [ ] Cảnh báo nếu:
  - thiếu file COLMAP,
  - CSV thiếu cột,
  - nhiều hơn một resolution trong cùng scene,
  - tên ảnh test duplicate,
  - scene không có ảnh train.

**Acceptance criteria:**

- Manifest liệt kê đúng 13 scenes.
- Tổng test poses = 724.
- Public GT images = 290.
- Tất cả output resolution = 1320 × 989.
- `HCM1439`, `HNI0265`, `HNI0437` được đánh dấu là scene có ít ảnh hơn chuẩn.

---

## 6. Task 2 — Environment Setup Và Baseline Repo

**Mục tiêu:** Có môi trường chạy được 3DGS trên GPU và train thử 1 scene không lỗi.

**Files:**

- Create: `docs/environment_setup.md`
- Create: `configs/baseline_3dgs.yaml`

**Công việc:**

- [ ] Chọn môi trường Python riêng, ví dụ Conda env:

```bash
conda create -n vai-nvs python=3.10 -y
conda activate vai-nvs
```

- [ ] Cài PyTorch theo CUDA driver thực tế của máy.
- [ ] Clone baseline 3DGS được BTC gợi ý:

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git external/gaussian-splatting
```

- [ ] Cài dependencies và build CUDA extension.
- [ ] Ghi lại version:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

- [ ] Train thử scene nhỏ nhất hoặc public scene đầu tiên trong 500–1000 iterations để kiểm tra pipeline:

```bash
python train.py \
  -s ../../VAI_NVS_DATA/phase1/public_set/hcm0031/train \
  -m ../../outputs/checkpoints/smoke_hcm0031 \
  --iterations 1000
```

**Acceptance criteria:**

- CUDA available.
- 3DGS train được ít nhất 1000 iterations.
- Checkpoint/output được ghi ra ổ đĩa.
- Không crash vì COLMAP files hoặc CUDA extension.

---

## 7. Task 3 — COLMAP Compatibility Check

**Mục tiêu:** Đảm bảo code đọc đúng binary COLMAP do BTC cung cấp.

**Files:**

- Create: `scripts/verify_colmap_scene.py`
- Output: `outputs/eval_reports/colmap_check_phase1.md`

**Công việc:**

- [ ] Viết script đọc:
  - `cameras.bin`,
  - `images.bin`,
  - `points3D.bin`.
- [ ] In ra:
  - số camera,
  - số registered images,
  - số sparse points,
  - camera model,
  - width/height,
  - intrinsics.
- [ ] Chạy trên 13 scenes.
- [ ] Xác nhận code bỏ qua được `frames.bin`, `rigs.bin`, `points3D.ply`.
- [ ] Nếu 3DGS gốc không đọc được format hiện tại, chuyển sang một trong hai hướng:
  - cập nhật COLMAP loader mới,
  - convert binary model sang text rồi load text.

**Acceptance criteria:**

- 13/13 scenes đọc được COLMAP sparse.
- Số registered images khớp số ảnh train.
- Camera intrinsics hợp lý và gần với `test_poses.csv`.
- Có report riêng cho từng scene.

---

## 8. Task 4 — Render Test Poses Chính Xác

**Mục tiêu:** Có renderer nhận checkpoint 3DGS và `test_poses.csv`, sinh ảnh đúng tên và đúng kích thước.

**Files:**

- Create: `scripts/render_test_poses.py`

**Công việc:**

- [ ] Parse `test_poses.csv`.
- [ ] Chuyển quaternion COLMAP `(qw, qx, qy, qz)` sang rotation matrix đúng convention.
- [ ] Giữ translation theo world-to-camera convention của COLMAP.
- [ ] Tạo camera object tương thích renderer.
- [ ] Render từng row trong CSV.
- [ ] Save ảnh theo đúng `image_name`.
- [ ] Nếu renderer lưu `.png` nhưng BTC yêu cầu tên `.JPG`, cần xác nhận:
  - hoặc lưu đúng extension trong CSV,
  - hoặc lưu ảnh với tên y hệt `image_name`.

**Rủi ro lớn nhất:**

Pose convention sai sẽ gây ảnh:

- xoay sai,
- lật trục,
- nhìn vào hướng ngược,
- render đen/trắng,
- object nằm ngoài khung hình.

**Test bắt buộc:**

- Render thử một camera train pose và so với ảnh train gốc.
- Render public test poses và so với public GT.
- Kiểm tra tên output khớp 100% với CSV.

**Acceptance criteria:**

- Render được 50 ảnh cho `hcm0031`.
- Ảnh không đen/trắng hoàn toàn.
- Object chính nằm đúng khung hình.
- Output names match `test_poses.csv`.
- Output resolution = 1320 × 989.

---

## 9. Task 5 — Local Evaluator Theo Công Thức BTC

**Mục tiêu:** Tự tính điểm offline trên public set để biết thay đổi nào tốt hơn.

**Files:**

- Create: `scripts/evaluate_predictions.py`
- Output: `outputs/eval_reports/<experiment_name>_public_scores.csv`
- Output: `outputs/eval_reports/<experiment_name>_summary.md`

**Công việc:**

- [ ] Load ảnh prediction và GT theo filename.
- [ ] Kiểm tra đủ ảnh trước khi tính metric.
- [ ] Tính:
  - LPIPS,
  - SSIM,
  - PSNR,
  - score tổng.
- [ ] Report theo:
  - từng ảnh,
  - từng scene,
  - trung bình public set.
- [ ] Cho phép cấu hình `psnr_max`, vì đề chỉ nói PSNR được normalize bằng ngưỡng chọn trước nhưng không nêu rõ giá trị.

**Acceptance criteria:**

- Tính được score cho 5 public scenes.
- Nếu thiếu ảnh hoặc sai resolution, evaluator fail sớm.
- Có CSV để so sánh nhiều experiment.
- Local score có thể dùng để chọn config tốt hơn baseline.

---

## 10. Task 6 — Baseline Training Toàn Bộ Public Set

**Mục tiêu:** Có điểm baseline đáng tin trước khi tuning.

**Files:**

- Create: `configs/baseline_3dgs.yaml`
- Create: `scripts/batch_train.py`
- Create: `scripts/batch_render.py`
- Update: `experiments/phase1_results.csv`

**Công việc:**

- [ ] Train 5 public scenes bằng config mặc định.
- [ ] Render public test poses.
- [ ] Evaluate bằng public GT.
- [ ] Ghi kết quả vào experiment log:

```csv
experiment_id,scene,config,iterations,lpips,ssim,psnr,score,notes
baseline_001,hcm0031,baseline_3dgs,30000,,,,,
```

- [ ] Quan sát ảnh render bằng mắt, ghi lỗi:
  - blur,
  - floaters,
  - sky artifacts,
  - missing thin structures,
  - exposure mismatch,
  - aliasing.

**Acceptance criteria:**

- Có baseline score cho 5 public scenes.
- Có ảnh render để so sánh trực quan.
- Có log đủ để biết scene nào yếu nhất.

---

## 11. Task 7 — Validation Protocol Và Experiment Discipline

**Mục tiêu:** Không tuning mù; mọi cải tiến phải có A/B test rõ ràng.

**Files:**

- Create: `experiments/README.md`
- Update: `experiments/phase1_results.csv`

**Quy tắc experiment:**

- Chỉ thay đổi 1–2 yếu tố mỗi lần.
- Mỗi experiment phải lưu:
  - commit/hash nếu có git,
  - config file,
  - scene list,
  - command,
  - runtime,
  - GPU,
  - metric result,
  - nhận xét ảnh.
- Public set không nên bị overfit quá mức. Nên giữ ít nhất 1 public scene làm “holdout mental check”, không dùng để quyết định mọi tuning.

**Bảng thí nghiệm tối thiểu:**

| Experiment | Thay đổi | Scene test trước | Lý do |
|---|---|---|---|
| `baseline_001` | 3DGS default | all public | Mốc so sánh |
| `iter_40k_001` | tăng iterations | hcm0031, HCM0181 | Kiểm tra còn hội tụ không |
| `densify_low_001` | giảm densify threshold | scene nhiều chi tiết | Giữ thanh BTS/dây |
| `small_scene_001` | config riêng cho HCM1439 | HCM1439-like split | Scene ít ảnh |
| `aa_001` | anti-aliasing/filter | scene bị răng cưa | Cải thiện LPIPS/SSIM |
| `bg_001` | background/sky strategy | scene nhiều trời | Cải thiện PSNR/LPIPS |

**Acceptance criteria:**

- Không có experiment “không biết đã đổi gì”.
- Có thể rerun config tốt nhất từ command + YAML.
- Có ranking config theo public score.

---

## 12. Task 8 — High-Score Model Tuning

**Mục tiêu:** Cải thiện score sau baseline, ưu tiên ROI cao và rủi ro thấp.

### 12.1. Tăng iterations

**Lý do:** 3DGS thường còn cải thiện sau baseline iterations nếu scene phức tạp.

Thử:

```text
30000 → 40000 → 50000
```

Chỉ giữ nếu public score tăng hoặc visual quality tốt hơn.

### 12.2. Densification tuning

BTS có cấu trúc mảnh: thanh kim loại, dây, anten. Cần đủ Gaussians nhỏ để tái tạo.

Thử các hướng:

```text
densify_grad_threshold: giảm nhẹ
densify_until_iter: tăng
opacity reset schedule: kiểm tra lại
```

Cần theo dõi:

- số Gaussians cuối,
- VRAM,
- floaters,
- thin structures.

### 12.3. Loss tuning

3DGS mặc định tối ưu L1 + D-SSIM. Vì metric có LPIPS trọng số cao, có thể thử thêm perceptual loss.

Nhưng cần lưu ý compliance:

- Đề cấm dùng dữ liệu ngoài liên quan trực tiếp tới scene.
- LPIPS thường dùng pretrained network. Đây không phải scene-specific data, nhưng vẫn nên xem kỹ quy định/BTC FAQ trước khi dùng cho training.
- Nếu không chắc, dùng LPIPS chỉ để evaluate offline, không dùng làm training loss cho submission chính thức.

Nếu được phép, thử:

```text
L = alpha * L1 + beta * (1 - SSIM) + gamma * LPIPS
```

Chiến lược an toàn hơn:

- Không bật LPIPS từ đầu.
- Bật sau khi geometry đã ổn, ví dụ nửa sau training.
- So sánh kỹ PSNR/SSIM vì LPIPS tốt hơn không phải lúc nào score tổng cũng tốt hơn.

### 12.4. Anti-aliasing

Nếu ảnh có răng cưa ở thanh BTS/dây:

- thử rasterizer/filter tốt hơn,
- thử variant 3DGS có anti-aliasing,
- kiểm tra ảnh xa/gần riêng.

### 12.5. Background/sky handling

Bầu trời chiếm nhiều pixel. Artifacts ở trời có thể làm PSNR/LPIPS giảm.

Các hướng nên thử:

1. Background color hợp lý.
2. Regularization để giảm floaters.
3. Không dùng external segmentation model nếu chưa rõ có được phép không.
4. Nếu cần mask, ưu tiên mask tạo từ chính dữ liệu hoặc phương pháp không dùng pretrained external model, trừ khi BTC xác nhận được phép.

### 12.6. Config riêng cho scene ít ảnh

Các scene cần chú ý:

- `HCM1439`: 103 train / 26 test.
- `HNI0265`: 205 train / 52 test.
- `HNI0437`: 224 train / 56 test.

Với scene ít ảnh, rủi ro:

- overfit views train,
- geometry thiếu,
- floaters,
- blur ở view mới.

Chiến lược:

- train lâu hơn nhưng theo dõi overfit,
- regularization mạnh hơn,
- không densify quá mức nếu sinh floaters,
- đánh giá visual bằng interpolation public scenes có số ảnh thấp tương tự.

**Acceptance criteria:**

- Có ít nhất 1 config tốt hơn baseline trên public average.
- Không có scene public nào tụt nghiêm trọng vì config mới.
- Config cuối cùng được lưu rõ ràng.

---

## 13. Task 9 — Batch Training Toàn Bộ 13 Scenes

**Mục tiêu:** Train toàn bộ scenes bằng config đã chọn, có checkpoint cho mỗi scene.

**Files:**

- Create: `scripts/batch_train.py`
- Create: `configs/high_quality_3dgs.yaml`
- Create: `configs/small_scene_3dgs.yaml`

**Công việc:**

- [ ] Tạo scene list từ manifest thay vì hard-code thủ công.
- [ ] Với mỗi scene:
  - chọn config default hoặc small-scene config,
  - train,
  - lưu log,
  - lưu checkpoint,
  - ghi trạng thái pass/fail.
- [ ] Nếu scene fail:
  - ghi lỗi vào report,
  - tiếp tục scene khác,
  - retry sau.

**Acceptance criteria:**

- 13/13 scenes có checkpoint.
- Log train được lưu.
- Không cần thao tác tay giữa chừng.
- Có thể resume nếu crash.

---

## 14. Task 10 — Batch Render Và Submission Verification

**Mục tiêu:** Render toàn bộ 724 test images và kiểm tra format trước khi ZIP.

**Files:**

- Create: `scripts/batch_render.py`
- Create: `scripts/verify_submission.py`
- Create: `scripts/pack_submission.py`

**Công việc:**

- [ ] Render theo từng scene từ `test_poses.csv`.
- [ ] Save vào:

```text
outputs/submissions/submission_candidate_001/<scene_id>/<image_name>
```

- [ ] Verify:
  - đủ 13 scenes,
  - mỗi scene đúng số ảnh,
  - mỗi ảnh đúng tên trong CSV,
  - mỗi ảnh đúng width/height,
  - không có file thừa,
  - không có `__MACOSX`,
  - không có hidden file,
  - không có nested folder sai.
- [ ] Pack ZIP:

```text
outputs/submissions/submission_candidate_001.zip
```

**Acceptance criteria:**

- Total images = 724.
- Public scenes:
  - `hcm0031`: 50
  - `hcm0034`: 60
  - `HCM0181`: 60
  - `HCM0193`: 60
  - `HCM0204`: 60
- Private scenes:
  - `HCM0249`: 60
  - `HCM0254`: 60
  - `HCM0276`: 60
  - `HCM1439`: 26
  - `HNI0131`: 60
  - `HNI0265`: 52
  - `HNI0366`: 60
  - `HNI0437`: 56
- ZIP pass verification trước khi submit.

---

## 15. Task 11 — Submit Sớm Và Calibrate

**Mục tiêu:** Không để đến cuối Phase 1 mới phát hiện lỗi format hoặc lỗi pipeline.

**Công việc:**

- [ ] Submit baseline hợp lệ càng sớm càng tốt.
- [ ] Ghi lại server score.
- [ ] So sánh server score với local public score.
- [ ] Nếu server báo lỗi:
  - kiểm tra ZIP structure,
  - kiểm tra tên scene,
  - kiểm tra tên ảnh,
  - kiểm tra extension,
  - kiểm tra resolution.
- [ ] Nếu server score thấp bất thường:
  - kiểm tra pose convention,
  - kiểm tra color range,
  - kiểm tra output bị black/white,
  - kiểm tra ảnh private có render bằng checkpoint đúng scene không.

**Acceptance criteria:**

- Có ít nhất 1 submission hợp lệ trước giai đoạn tuning cuối.
- Biết rõ server chấp nhận format nào.
- Có baseline leaderboard score để đo tiến bộ.

---

## 16. Timeline Đề Xuất Từ 06/07/2026 Đến 30/07/2026

Hiện còn khoảng **24 ngày** đến deadline Phase 1.

### Giai đoạn 1 — 06/07 đến 08/07: Hạ tầng tối thiểu

- Dataset audit.
- Environment setup.
- COLMAP compatibility.
- Train smoke test.
- Render thử 1 scene.

**Deliverable:** 1 scene public render được test poses.

### Giai đoạn 2 — 09/07 đến 12/07: Baseline + evaluator

- Train 5 public scenes.
- Render public test.
- Build local evaluator.
- Có baseline score.

**Deliverable:** bảng điểm baseline public.

### Giai đoạn 3 — 13/07 đến 18/07: Tuning có kiểm soát

- Iteration tuning.
- Densification tuning.
- Anti-aliasing nếu cần.
- Background/sky handling nếu cần.
- Config riêng scene ít ảnh.

**Deliverable:** chọn được config tốt hơn baseline.

### Giai đoạn 4 — 19/07 đến 23/07: Full private/public run

- Batch train 13 scenes.
- Batch render 724 images.
- Verify submission.
- Submit candidate đầu tiên.

**Deliverable:** submission hợp lệ.

### Giai đoạn 5 — 24/07 đến 28/07: Improve và rerun

- Phân tích scene yếu.
- Tuning từng nhóm scene.
- Submit candidate 2/3.

**Deliverable:** score cải thiện ổn định.

### Giai đoạn 6 — 29/07 đến 30/07: Freeze

- Không đổi pipeline lớn.
- Chỉ sửa lỗi format hoặc rerun config đã chứng minh tốt.
- Lưu code/config/log/checkpoint.
- Submit bản cuối.

**Deliverable:** final ZIP + reproducibility package.

---

## 17. Kiến Thức Cần Có Để Đạt Điểm Cao

### 17.1. Camera Geometry Và Coordinate Systems

Cần hiểu:

- pinhole camera model,
- camera intrinsics: `fx`, `fy`, `cx`, `cy`,
- extrinsics: rotation + translation,
- quaternion,
- world-to-camera vs camera-to-world,
- COLMAP coordinate convention.

Điểm dễ sai nhất:

```text
Camera center không phải tx,ty,tz.
Camera center = -R^T t
```

Nếu nhầm convention, render sẽ sai hoàn toàn.

### 17.2. COLMAP Và Structure-from-Motion

Cần biết:

- `cameras.bin` chứa camera intrinsics/model.
- `images.bin` chứa registered images và poses.
- `points3D.bin` chứa sparse point cloud.
- Sparse reconstruction là khởi tạo ban đầu cho 3DGS.
- COLMAP binary format có thể khác theo version.

### 17.3. Novel View Synthesis

Cần hiểu:

- interpolation vs extrapolation views,
- multi-view consistency,
- occlusion/disocclusion,
- view-dependent effects,
- artifacts thường gặp khi render góc nhìn mới.

### 17.4. 3D Gaussian Splatting

Cần nắm:

- mỗi Gaussian gồm position, scale, rotation, opacity, SH color,
- differentiable rasterization,
- alpha blending,
- densification,
- pruning,
- opacity reset,
- spherical harmonics,
- trade-off giữa sharpness và floaters.

### 17.5. Image Quality Metrics

Cần hiểu tác động của từng metric:

| Metric | Muốn tối ưu | Cần chú ý |
|---|---|---|
| LPIPS | thấp | ảnh tự nhiên, texture đúng, ít artifacts |
| SSIM | cao | cấu trúc BTS, cạnh, khung thép |
| PSNR | cao | màu/exposure/pixel alignment |

Không nên chỉ nhìn loss train; phải tính đúng metric trên validation/public GT.

### 17.6. PyTorch/CUDA Engineering

Cần biết:

- quản lý VRAM,
- mixed precision nếu cần,
- build CUDA extension,
- debug OOM,
- batch script chạy dài,
- checkpoint/resume.

### 17.7. Experiment Management

Cần có kỷ luật:

- mỗi experiment có config riêng,
- không thay nhiều thứ cùng lúc,
- ghi score và nhận xét ảnh,
- giữ lại config tốt nhất,
- có thể rerun.

### 17.8. Reproducibility Và Competition Compliance

Cần lưu:

- code,
- config,
- dependency versions,
- training logs,
- commands,
- checkpoints,
- submission ZIP.

Cần tránh:

- data ngoài liên quan scene,
- chỉnh ảnh thủ công,
- đoán private ground truth,
- sửa riêng từng test pose bằng tay,
- pipeline không tái lập.

---

## 18. Rủi Ro Và Phương Án Dự Phòng

| Rủi ro | Ảnh hưởng | Cách phát hiện | Phương án |
|---|---|---|---|
| Sai pose convention | Mất gần toàn bộ điểm | ảnh lệch/xoay/đen | render train pose để sanity check |
| Sai ZIP format | Submission không được tính | server reject / score 0 | `verify_submission.py` bắt buộc |
| COLMAP loader không tương thích | Không train được | crash khi đọc sparse | update loader hoặc convert model |
| OOM GPU | Không train/render được | CUDA OOM | giảm resolution, giảm config, train từng scene |
| Overfit public set | Private score không tăng | public tăng nhưng server không | giữ config đơn giản, tránh tuning quá hẹp |
| Floaters | LPIPS/PSNR giảm | đốm mờ trên trời | opacity/prune/regularization |
| Thin structures bị mất | SSIM/LPIPS giảm | thanh BTS/dây mờ | densification tuning |
| Scene ít ảnh kém | private tụt | scene HCM1439 render mờ | config riêng, regularization |
| Dùng external model không rõ luật | nguy cơ vi phạm | không chắc quy định | hỏi BTC/xác nhận trước |

---

## 19. Definition of Done

Kế hoạch Phase 1 được xem là hoàn thành khi có đủ:

- [ ] Manifest dataset 13 scenes.
- [ ] 3DGS environment chạy được.
- [ ] COLMAP reader verified.
- [ ] Test pose renderer verified.
- [ ] Local evaluator chạy được trên public GT.
- [ ] Baseline public score.
- [ ] Ít nhất một config cải thiện public score.
- [ ] Batch train 13 scenes.
- [ ] Batch render 724 ảnh.
- [ ] Submission ZIP pass checker.
- [ ] Ít nhất một submission hợp lệ trên server.
- [ ] Final package có code/config/log để tái lập.

---

## 20. Ưu Tiên Thực Thi Ngay

Nếu chỉ chọn 5 việc quan trọng nhất để bắt đầu:

1. `inspect_dataset.py`
2. environment + 3DGS smoke test
3. `render_test_poses.py`
4. `evaluate_predictions.py`
5. `verify_submission.py`

Sau 5 việc này, team sẽ có pipeline khép kín:

```text
data → train → render → evaluate → package → submit
```

Khi pipeline khép kín đã chạy được, mọi cải tiến model sau đó mới đáng làm.

---

## 21. Ghi Chú Độc Lập

Kế hoạch này được lập dựa trên:

- yêu cầu trong `docs/topic.md`,
- cấu trúc dataset thực tế trong `VAI_NVS_DATA/phase1`,
- khảo sát trực tiếp các scene public/private,
- format `test_poses.csv`,
- metrics và submission format do đề bài nêu.

Kế hoạch này **không dựa vào** các file kế hoạch cũ:

```text
docs/competition_plan.md
docs/competition_plan_v2.md
```
