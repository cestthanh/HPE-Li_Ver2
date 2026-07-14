"""
losses.py
Loss huấn luyện pose 3D: SmoothL1 vị trí + bone-length loss (ver2).

Train TRỰC TIẾP trên tọa độ MÉT THÔ (không z-score), theo cách của WiFlow
(arXiv 2602.08661 — pipeline WiFi-CSI 3D pose công khai trên MMFi):

  • SmoothL1(beta=0.1) cho vị trí khớp — beta có nghĩa vật lý = 10 cm:
    sai số < 10 cm phạt bậc hai (mượt quanh 0), sai số lớn phạt tuyến tính.
    Khác MSE: phần tuyến tính không bị các frame khó kéo dự đoán về trung bình
    có điều kiện (mean collapse — nguyên nhân tay/chân "lười" cử động).

  • bone-length loss (trọng số 0.2, beta=0.05 = 5 cm) — phạt chênh lệch ĐỘ DÀI
    XƯƠNG giữa pred và GT, giữ tỉ lệ chi thể, chống co ngắn tay/chân về thân.
"""
import torch
import torch.nn as nn

# Bộ xương 17 khớp MMFi (chuẩn Human3.6M — trùng danh sách BONES của demo_web).
BONES = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16],
]


class PoseLoss(nn.Module):
    """loss = SmoothL1(pred, gt) + bone_weight × SmoothL1(bone_len(pred), bone_len(gt))

    Input:  pred, gt — (B, 17, 3), đơn vị mét.
    Output: scalar loss.
    """

    def __init__(self, beta=0.1, bone_weight=0.2, bone_beta=0.05):
        super().__init__()
        self.position = nn.SmoothL1Loss(beta=beta)
        self.bone = nn.SmoothL1Loss(beta=bone_beta)
        self.bone_weight = bone_weight
        bones = torch.tensor(BONES, dtype=torch.long)      # (16, 2)
        self.register_buffer("bone_a", bones[:, 0], persistent=False)
        self.register_buffer("bone_b", bones[:, 1], persistent=False)

    def _bone_lengths(self, pose):
        """(B, 17, 3) -> (B, 16): độ dài từng xương (mét)."""
        return (pose[:, self.bone_a] - pose[:, self.bone_b]).norm(dim=-1)

    def forward(self, pred, gt):
        position_loss = self.position(pred, gt)
        bone_loss = self.bone(self._bone_lengths(pred), self._bone_lengths(gt))
        return position_loss + self.bone_weight * bone_loss
