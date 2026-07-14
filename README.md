# DSKNetTransMMFI3D — Bản code trọng tâm (ver2)

Ước lượng **pose 3D người từ tín hiệu WiFi-CSI** trên dataset MMFi.

**Khác ver1 (`hpe_li_3d`):** bỏ chuẩn hóa z-score — huấn luyện trực tiếp trên
tọa độ **mét thô** với loss **SmoothL1(beta=0.1) + 0.2 × bone-length** (thay cho
MSE trên z-score), và báo cáo thêm **RC-MPJPE / g_PCK** để so sánh được với
WiFlow. Lý do: MSE gây mean collapse (tay/chân "lười" cử động), còn z-score làm
loss phạt lệch trọng số giữa các trục. Kiến trúc model và data pipeline giữ nguyên.

Đây là bản **rút gọn để đọc và trình bày** của project `HPE-Li-3D`: giữ đầy đủ
đường đi từ dữ liệu → model → huấn luyện → đánh giá, nhưng lược bỏ mọi phần thừa
(code tương thích checkpoint cũ, các nhánh cấu hình không dùng, tham số chết,
index dữ liệu không cần thiết…). Comment bằng tiếng Việt, chú thích shape tensor.

```text
Input : (B, 3, 114, 10)   3 antenna × 114 sub-carrier × 10 bước thời gian
Output: (B, 17, 3)        tọa độ (x, y, z) của 17 khớp cơ thể
```

## Cấu trúc thư mục

```text
code clean/
  model/                     ★ KIẾN TRÚC (nơi chứa đóng góp)
    dsknet3d.py                Model đầy đủ + DSKConv + DSKUnit
    channel_transformer.py     Channel-wise Transformer (attention C×C)
    regression.py              Đầu hồi quy MLP -> 51 giá trị
  data/                      DỮ LIỆU
    mmfi.py                    Đọc CSI (.mat) + pose (.npy), chia subject
    splits.py                  Chia val/test theo sequence (không rò rỉ)
  losses.py                  LOSS: SmoothL1 (mét thô) + bone-length loss
  metrics.py                 ĐÁNH GIÁ: MPJPE, RC-MPJPE, PA-MPJPE, PCK, g_PCK
  config.py                  Cấu hình thí nghiệm P1-S1 (một chỗ duy nhất)
  train.py                   ĐIỂM VÀO: vòng train / val / test
  demo.py                    Chạy thử model với CSI giả, in shape từng tầng
  demo_web.py                Demo 3 panel trên trình duyệt (RGB | GT 3D | Pred 3D)
  demo_web.html              Giao diện web cho demo_web.py
  README.md
```

## Luồng code toàn dự án

```text
train.py  (main)
   │
   ├─ config.py            cấu hình P1-S1 (protocol, ratio, epochs, lr…)
   │
   ├─ data/mmfi.py         make_datasets()  → đọc .mat + .npy, chia subject train/eval
   ├─ data/splits.py       split_by_sequence() → eval thành val/test theo sequence
   │
   ├─ losses.py            PoseLoss = SmoothL1 (mét thô) + 0.2 × bone-length loss
   │
   ├─ model/               DSKNetTransMMFI3D  → dự đoán pose (đơn vị mét)
   │      dsknet3d.py           SKUnit → DSKConv → ChannelTransformer
   │      channel_transformer.py    attention (C×C)
   │      regression.py             MLP → 51 tọa độ
   │
   └─ metrics.py           compute_metrics() → MPJPE (chọn best), RC-MPJPE,
                                               PA-MPJPE, PCK@50/100, g_PCK@20/50
```

## Ba đóng góp (trong `model/`)

1. **Multi-scale** — 3 nhánh convolution dilation 1/2/3 quan sát CSI ở nhiều
   phạm vi (cục bộ → rộng).
2. **Dual selection** — chọn nhánh theo **từng channel** (channel attention) và
   theo **từng hàng sub-carrier** (frequency attention).
3. **Channel Transformer** — học quan hệ **giữa các channel**. Điểm mấu chốt:
   ma trận attention có shape `(C, C)` chứ không phải `(N, N)` như ViT chuẩn.

## Luồng dữ liệu bên trong model

```text
CSI (B,3,114,10)
  │
  ├─ SKUnit1 ─┐   conv1×1 → AvgPool → DSKConv → BN → conv1×1
  │           └─ DSKConv                         ← trái tim mô hình
  │                ├─ 3 nhánh conv dilation 1/2/3      → (B, 3, C, H, W)
  │                ├─ channel_attention   (chọn nhánh / channel)
  │                ├─ frequency_attention (chọn nhánh / hàng tần số)
  │                ├─ cat theo chiều W                 → (B, C, H, 2W)
  │                └─ ChannelTransformer → attention Qᵀ·K → (B, T, C, C)  ★
  ├─ BatchNorm
  ├─ SKUnit2  (giống trên, C: 128 → 256)
  ├─ final_pool                                        → (B, 256, 14, 1)
  └─ RegressionHead → (B, 51) → reshape                → (B, 17, 3)
```

| Tầng            | Output shape        | Ghi chú                          |
|-----------------|---------------------|----------------------------------|
| Input CSI       | `(B, 3, 114, 10)`   | 3 antenna × 114 sub-carrier × 10 |
| SKUnit1         | `(B, 128, 57, 5)`   | tăng channel, giảm H, W          |
| SKUnit2         | `(B, 256, 28, 2)`   | channel ×2, không gian giảm      |
| final_pool      | `(B, 256, 14, 1)`   | AvgPool 2×2                      |
| RegressionHead  | `(B, 51)`           | MLP 3584 → 32 → 64 → 51          |
| reshape         | `(B, 17, 3)`        | 17 khớp × (x, y, z)              |

## Cách chạy

Yêu cầu: `torch`, `numpy`, `scipy`, `tqdm`.

**1. Chạy thử model (không cần dataset)** — kiểm tra luồng và in shape:

```bash
python demo.py
```

**2. Huấn luyện đầy đủ (cần dataset MMFi)**:

```bash
# đặt đường dẫn gốc dataset
python train.py --data-root /đường/dẫn/tới/mmfi/dataset
# hoặc export MMFI_DATASET_ROOT rồi chạy: python train.py
```

Kết quả in ra MPJPE/RC-MPJPE/PA-MPJPE/PCK/g_PCK mỗi epoch trên val, chọn
checkpoint tốt nhất theo val MPJPE, lưu `best_model.pt` (định dạng mà
`demo_web.py` đọc trực tiếp được), và đánh giá lần cuối trên test.

> **Đọc số cho đúng:** `mpjpe_mm` là sai số TUYỆT ĐỐI (gồm cả lỗi định vị toàn
> cục); `rc_mpjpe_mm` đã trừ khớp gốc — bảng MMFi của WiFlow dùng quy ước
> root-aligned này, nên khi so với họ phải dùng `rc_mpjpe_mm`/`g_pck_20`.

**3. Demo trực quan trên trình duyệt (cần dataset + checkpoint đã train)**:

```bash
python demo_web.py `
  --checkpoint "D:\...\results\A3\A3_l1_h6_seed0\checkpoints\best.pt" `
  --dataset-root "D:\MMFi-dataset\MMFi_Dataset" --device cpu
```

Mở app rồi **chọn subject + action ngay trên header và bấm "Chạy"** — model chỉ
nạp một lần, mỗi lần chọn chỉ chạy inference lại, không cần khởi động lại server.
Danh sách subject/action được tự quét từ dataset. (Muốn nạp sẵn một sequence khi
mở, thêm `--subject S21 --action A05`.)

Trang gồm 3 panel đồng bộ theo thanh trượt frame: ảnh RGB | pose 3D ground-truth
| pose 3D dự đoán, kèm bảng chẩn đoán (MPJPE, PA-MPJPE, PCK, g_PCK, tương quan
thời gian…). Thêm `--dry-run` (cùng `--subject/--action`) để chỉ in chỉ số rồi
thoát. Demo nạp được cả checkpoint của bản gốc HPE-Li-3D (tự dịch tên tham số,
tự denormalize z-score nếu checkpoint có `pose_normalization`) lẫn `best_model.pt`
do `train.py` ver2 lưu (mét thô, không cần denormalize), và đọc CSI bằng đúng hàm
`data/mmfi._load_csi` dùng khi train — một nguồn tiền xử lý duy nhất.

## Thứ tự đọc đề xuất

1. `train.py` → `main()` — thấy toàn bộ đường đi ở mức cao nhất.
2. `data/mmfi.py` — dữ liệu vào ra sao (CSI + pose, cách chia subject).
3. `model/dsknet3d.py` → `DSKNetTransMMFI3D.forward` → `DSKConv.forward`.
4. `model/channel_transformer.py` → `_Attention.forward` (dòng `Qᵀ·K` tạo ma
   trận `C×C` — điểm biến nó thành *channel* attention).
5. `metrics.py` — pose được chấm điểm thế nào.

> Khi đọc model, chỉ cần bám hai câu hỏi: **tensor đang có shape gì?** và **phép
> toán đang trộn thông tin theo chiều nào?**

## Khác biệt so với bản gốc `HPE-Li-3D`

Bản này ưu tiên dễ đọc, nên đã lược bỏ so với bản gốc:
- Code migrate checkpoint cũ, versioning, metadata huấn luyện chi tiết.
- Các nhánh split khác (`cross_scene`, `cross_subject`, `manual`) và chế độ
  `sequence` — chỉ giữ `random_split` + `frame` của P1-S1.
- Class `MMFi_Database` dựng index scene/subject/action (bản gốc không thực sự
  dùng tới khi nạp dữ liệu).
- Giá trị `elapsed_time` trả kèm ở `forward`, các tham số `vis`/`attn_weights`.

Và ver2 THAY ĐỔI so với cả bản gốc lẫn ver1 (đây là điểm thí nghiệm, không phải
rút gọn):
- Loss: MSE trên z-score → **SmoothL1(beta=0.1) + 0.2 × bone-length** trên mét thô.
- Bỏ hoàn toàn `compute_pose_stats`/normalize/denormalize (không còn z-score).
- Thêm lại chỉ số `g_PCK` (body-scale) và `rc_mpjpe_mm` (root-centered).

Khi cần con số chính xác để đối chiếu luận văn, vẫn tham chiếu code gốc.
