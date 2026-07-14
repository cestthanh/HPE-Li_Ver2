"""
demo.py
Chạy thử DSKNetTransMMFI3D với input WiFi-CSI giả (random) để:
  • kiểm tra luồng forward chạy thông,
  • in shape tensor qua từng tầng — đối chiếu với bảng shape trong README.

    python demo.py
"""
import sys

import torch

from model import DSKNetTransMMFI3D

# Cho phép in tiếng Việt trên console Windows (mặc định cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    torch.manual_seed(0)
    batch = 2
    csi = torch.randn(batch, 3, 114, 10)   # (batch, antenna, sub-carrier, time)

    model = DSKNetTransMMFI3D().eval()

    print("Cấu hình model :", model.get_model_config())
    print(f"Số tham số      : {count_params(model):,}\n")

    print("Luồng shape qua từng tầng:")
    print(f"  input CSI          {tuple(csi.shape)}")
    with torch.no_grad():
        x = model.skunit1(csi);   print(f"  SKUnit1            {tuple(x.shape)}")
        x = model.bn(x)
        x = model.skunit2(x);     print(f"  SKUnit2            {tuple(x.shape)}")
        x = model.final_pool(x);  print(f"  final_pool         {tuple(x.shape)}")
        pose = model(csi)
    print(f"  regression+reshape {tuple(pose.shape)}   <- (B, 17 khớp, 3 tọa độ)\n")

    print("OK — forward chạy thành công.")


if __name__ == "__main__":
    main()
