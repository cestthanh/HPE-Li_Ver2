"""
train.py
Điểm vào huấn luyện DSKNetTransMMFI3D — pose 3D từ WiFi-CSI trên dataset MMFi.

Quy trình (cấu hình P1-S1, ver2):
    1. Chia dữ liệu: train / (val + test).
    2. Huấn luyện TRỰC TIẾP trên tọa độ mét thô — KHÔNG chuẩn hóa z-score.
       Loss = SmoothL1(beta=0.1) + 0.2 × bone-length loss (xem losses.py).
    3. Mỗi epoch đánh giá trên val, giữ checkpoint có MPJPE thấp nhất.
    4. Cuối cùng đánh giá checkpoint tốt nhất trên test.

Khác ver1 (MSE trên z-score) ở hai điểm:
  • MSE kéo dự đoán về trung bình có điều kiện khi CSI không đủ thông tin
    (mean collapse) — SmoothL1 có phần tuyến tính nên giữ được biên độ cử động.
  • z-score chia mỗi trục cho std riêng, vô tình phạt cùng-1-cm-sai-số nặng nhẹ
    khác nhau giữa các trục — train trên mét thô thì loss thẳng hàng với MPJPE.

    python train.py --data-root <đường dẫn gốc dataset MMFi>
    (hoặc đặt biến môi trường MMFI_DATASET_ROOT)
"""
import argparse
import copy
import os

import numpy as np
import torch
from tqdm import tqdm

# Nhiều worker + dataset đọc hàng loạt .mat dễ chạm trần file descriptor
# ("Too many open files"). Chia sẻ tensor qua file_system thay vì file_descriptor
# để không phụ thuộc ulimit -n.
torch.multiprocessing.set_sharing_strategy("file_system")

from config import CONFIG
from data import make_datasets, make_loader, split_by_sequence
from losses import PoseLoss
from metrics import compute_metrics
from model import DSKNetTransMMFI3D


def parse_args():
    p = argparse.ArgumentParser(description="Train DSKNetTransMMFI3D (MMFi P1-S1, ver2).")
    p.add_argument("--data-root", default=os.getenv("MMFI_DATASET_ROOT"),
                   help="Thư mục gốc dataset MMFi.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Vòng huấn luyện & đánh giá ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        csi = batch["csi"].to(device)
        gt = batch["pose"].to(device)                      # (B, 17, 3), đơn vị mét
        loss = criterion(model(csi), gt)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        pred = model(batch["csi"].to(device))              # dự đoán đã ở đơn vị mét
        preds.append(pred.cpu().numpy())
        gts.append(batch["pose"].numpy())
    return compute_metrics(np.concatenate(preds), np.concatenate(gts))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not args.data_root:
        raise SystemExit("Cần --data-root hoặc biến môi trường MMFI_DATASET_ROOT.")
    set_seed(args.seed)
    device = torch.device(args.device)

    # Dữ liệu: chia subject (train / eval), rồi chia eval thành val/test theo sequence.
    train_ds, eval_ds = make_datasets(args.data_root, CONFIG["split_ratio"], CONFIG["split_seed"])
    val_ds, test_ds = split_by_sequence(eval_ds, test_ratio=0.5, seed=41)
    train_loader = make_loader(train_ds, CONFIG["train_batch_size"], True,  CONFIG["num_workers"])
    val_loader   = make_loader(val_ds,   CONFIG["eval_batch_size"],  False, CONFIG["num_workers"])
    test_loader  = make_loader(test_ds,  CONFIG["eval_batch_size"],  False, CONFIG["num_workers"])
    print(f"samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = DSKNetTransMMFI3D().to(device)
    criterion = PoseLoss(beta=CONFIG["smooth_l1_beta"],
                         bone_weight=CONFIG["bone_loss_weight"],
                         bone_beta=CONFIG["bone_loss_beta"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    best_mpjpe, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d} | train_loss={train_loss:.4f} | "
              f"val_mpjpe={val['mpjpe_mm']:.2f}mm rc={val['rc_mpjpe_mm']:.2f}mm "
              f"pa={val['pa_mpjpe_mm']:.2f} pck50={val['pck_50mm']:.1f}% "
              f"gpck20={val['g_pck_20']:.1f}%", flush=True)

        if val["mpjpe_mm"] < best_mpjpe:                    # chọn checkpoint theo val MPJPE
            best_mpjpe = val["mpjpe_mm"]
            best_state = copy.deepcopy(model.state_dict())
            print(f"  -> best (val_mpjpe={best_mpjpe:.2f}mm)", flush=True)

    # Đánh giá cuối cùng trên test bằng checkpoint tốt nhất.
    model.load_state_dict(best_state)
    torch.save({                                            # định dạng demo_web.py đọc được
        "model_state_dict": best_state,
        "model_config": model.get_model_config(),
        "pose_normalization": {"enabled": False},           # ver2: mét thô, không z-score
    }, "best_model.pt")
    test = evaluate(model, test_loader, device)
    print(f"\n[test] mpjpe={test['mpjpe_mm']:.2f}mm rc_mpjpe={test['rc_mpjpe_mm']:.2f}mm "
          f"pa_mpjpe={test['pa_mpjpe_mm']:.2f}mm", flush=True)
    print(f"       pck50={test['pck_50mm']:.1f}% pck100={test['pck_100mm']:.1f}% "
          f"g_pck20={test['g_pck_20']:.1f}% g_pck50={test['g_pck_50']:.1f}%", flush=True)


if __name__ == "__main__":
    main()
