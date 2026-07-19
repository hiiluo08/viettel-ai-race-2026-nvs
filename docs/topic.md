# AI Race 2026 — Bài 1: BTS Digital Twin (Novel View Synthesis)

> **Cuộc đua AI cho kỹ sư Việt Nam**  
> Tổ chức bởi: Viettel  
> Trang chủ: https://competition.viettel.vn/contests/var-2026

---

## Lịch thi

| Vòng | Tên | Thời gian | Hình thức nộp |
|------|-----|-----------|---------------|
| 1 | Vòng 1 — Sơ loại | 01/07/2026 → 30/07/2026 | File ZIP (GPU) |
| 2 | Vòng 2 — Sơ khảo | 16/08/2026 → 19/08/2026 | File ZIP (GPU) |
| 3 | Vòng 3 — Chung kết | 08/09/2026 → 10/09/2026 | File ZIP (GPU) |

---

## 1. Tổng quan bài toán

Mục tiêu của bài toán là xây dựng mô hình AI có khả năng **tái dựng cấu trúc không gian 3D** của một scene từ tập ảnh đa góc nhìn và **sinh ra ảnh tại các góc nhìn mới** chưa từng xuất hiện trong dữ liệu đầu vào.

**Dữ liệu có thể được thu thập từ:**
- Drone bay quanh đối tượng
- Camera cầm tay (hand-held camera)

**Đối tượng trong scene có thể là:**
- Trạm BTS
- Công trình hạ tầng
- Các đối tượng thực tế khác

**Bài toán thuộc các lĩnh vực:**
- Computer Vision
- 3D Vision
- Neural Rendering
- Novel View Synthesis
- Digital Twin

---

## 2. Cấu trúc dữ liệu

Mỗi scene dữ liệu có cấu trúc như sau:

```text
<scene_id>/
├── train/
│   ├── images/          : Ảnh training
│   ├── sparse/0/        : Sparse reconstruction từ COLMAP
│   │                       ├── cameras.bin
│   │                       ├── images.bin
│   │                       └── points3D.bin
└── test/
    └── test_poses.csv   : Camera poses cho test images
```

---

## 3. Thông tin dữ liệu

- **Train images:** ~80%
- **Test images:** ~20%
- Camera poses và sparse reconstruction đã được dựng sẵn bằng **COLMAP** và cung cấp cho thí sinh

---

## 4. Format `test_poses.csv`

```text
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
```

| Trường | Ý nghĩa |
|--------|---------|
| `image_name` | Tên ảnh đầu ra cần sinh |
| `qw, qx, qy, qz` | Quaternion rotation theo format COLMAP |
| `tx, ty, tz` | Camera translation |
| `fx, fy` | Focal length |
| `cx, cy` | Principal point |
| `width, height` | Kích thước ảnh cần sinh |

---

## 5. Đầu vào bài toán

Đầu vào bao gồm:
- Tập ảnh train đa góc nhìn
- Camera intrinsics
- Camera poses
- Sparse reconstruction từ COLMAP
- Danh sách test poses

---

## 6. Đầu ra bài toán

Thí sinh cần sinh:
- Ảnh RGB tương ứng với toàn bộ test poses được cung cấp

Ảnh đầu ra cần:
- Đúng cấu trúc hình học
- Đúng vị trí các vật thể
- Đảm bảo chất lượng hình ảnh chân thực và nhất quán

---

## 7. Format Submission

Submission là file ZIP chứa toàn bộ ảnh kết quả:

```text
submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
```

**Yêu cầu:**
- Đúng số lượng và tên scene
- Đúng tên file ảnh
- Đúng kích thước ảnh
- Đúng số lượng ảnh mỗi scene

> Nếu thiếu scene hoặc thừa scene so với ground-truth, kết quả sẽ **không được tính**.

---

## 8. Metrics đánh giá

Kết quả được đánh giá bằng cách so sánh ảnh sinh ra với ảnh ground-truth bằng ba metrics:

### 8.1. LPIPS

Đánh giá độ tương đồng cảm quan giữa hai ảnh bằng đặc trưng deep learning.

- **Giá trị càng thấp càng tốt.**

> Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.  
> *"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."*  
> CVPR 2018. https://arxiv.org/abs/1801.03924

### 8.2. SSIM

Đánh giá độ tương đồng về cấu trúc hình ảnh.

- **Giá trị càng cao càng tốt.**

> Zhou Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.  
> *"Image quality assessment: from error visibility to structural similarity."*  
> IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, April 2004.  
> doi: 10.1109/TIP.2003.819861

### 8.3. PSNR

Đánh giá sai số mức pixel giữa ảnh dự đoán và ground-truth.

- **Giá trị càng cao càng tốt.**

Để kết hợp với các metrics khác, giá trị PSNR được chuẩn hóa về khoảng `[0, 1]` theo công thức:

```python
psnr_norm = torch.clamp(psnr_val / psnr_max, 0.0, 1.0)
```

Trong đó:
- `psnr_max`: ngưỡng PSNR tối đa được lựa chọn trước
- `clamp`: giới hạn giá trị trong khoảng từ 0 đến 1

### 8.4. Công thức tính điểm cuối cùng

$$
\text{Score} = 0.4 \times (1 - \text{LPIPS}) + 0.3 \times \text{SSIM} + 0.3 \times \text{PSNR}_\text{norm}
$$

Điểm trên bảng xếp hạng là **điểm trung bình của toàn bộ các scene**.

---

## 9. Hình thức thi

Dữ liệu và scene hoàn toàn **mới được cung cấp cho mỗi vòng thi**; cách thức tính điểm được giữ nguyên qua các vòng.

---

## 10. Quy định chống gian lận và đảm bảo tính công bằng

### 10.1. Cấm sử dụng dữ liệu ngoài

Thí sinh chỉ được phép sử dụng dữ liệu do Ban Tổ Chức cung cấp trong từng vòng thi.

**Nghiêm cấm:**
- Sử dụng ảnh, video hoặc dữ liệu 3D bên ngoài có chứa cùng đối tượng hoặc cùng scene của bộ dữ liệu thi
- Thu thập bổ sung dữ liệu thực địa hoặc từ Internet liên quan trực tiếp đến các scene được cung cấp
- Sử dụng bất kỳ nguồn dữ liệu nào nhằm tái tạo hoặc suy luận ground-truth của tập test

### 10.2. Cấm truy xuất hoặc suy đoán dữ liệu kiểm thử

Nghiêm cấm mọi hành vi nhằm:
- Truy cập trái phép vào dữ liệu ground-truth
- Khai thác lỗ hổng hệ thống để thu thập thông tin về ảnh kiểm thử

### 10.3. Yêu cầu khả năng tái lập kết quả

Ban Tổ Chức có quyền yêu cầu các đội đạt thứ hạng cao cung cấp:
- Mã nguồn huấn luyện và suy luận
- File cấu hình (config)
- Danh sách thư viện và phiên bản sử dụng
- Checkpoint mô hình
- Nhật ký huấn luyện (training logs)

Đội thi phải chứng minh rằng kết quả nộp bài có thể được tái tạo từ pipeline đã công bố.

### 10.4. Cấm chỉnh sửa thủ công ảnh đầu ra

Toàn bộ ảnh kết quả phải được **sinh tự động** bởi thuật toán hoặc mô hình AI.

**Nghiêm cấm:**
- Chỉnh sửa thủ công từng ảnh bằng các phần mềm đồ họa
- Ghép ảnh, vẽ thêm hoặc xóa vật thể bằng thao tác thủ công
- Can thiệp thủ công vào từng test pose

Ban Tổ Chức có quyền yêu cầu chứng minh quy trình sinh ảnh hoàn toàn tự động.

---

## 11. Baseline tham khảo

Thí sinh có thể tham khảo baseline sau đây:

> **3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
> https://github.com/graphdeco-inria/gaussian-splatting
