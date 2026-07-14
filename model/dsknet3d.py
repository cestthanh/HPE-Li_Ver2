"""
dsknet3d.py
DSKNetTransMMFI3D — Ước lượng pose 3D người từ tín hiệu WiFi-CSI (dataset MMFi).

    Input : (B, 3, 114, 10)   3 antenna × 114 sub-carrier × 10 bước thời gian
    Output: (B, 17, 3)        tọa độ (x, y, z) của 17 khớp cơ thể

Ba ý tưởng chính:
    1. Multi-scale : 3 nhánh conv với dilation 1/2/3 nhìn CSI ở nhiều phạm vi.
    2. Dual select : chọn nhánh theo TỪNG channel VÀ theo TỪNG hàng tần số.
    3. Channel Transformer : học quan hệ giữa các channel sau khi hợp nhất.

Thứ tự nên đọc: DSKNetTransMMFI3D.forward -> DSKUnit -> DSKConv -> ChannelTransformer.

Ký hiệu: B = batch, C = channel, H×W = kích thước không gian, M = số nhánh dilation.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .channel_transformer import ChannelTransformer
from .regression import RegressionHead


# ─── Siêu tham số mặc định (cấu hình thí nghiệm P1-S1) ────────────────────────
DEFAULT_CONFIG = {
    "base_channels":       128,  # C: số channel đặc trưng cơ sở
    "reg_hidden":           32,  # chiều ẩn của regression head
    "sk_branches":           3,  # M: số nhánh dilation trong DSKConv
    "sk_groups":            32,  # số group của grouped-convolution
    "sk_reduction":          4,  # tỉ lệ nén channel trong SK-attention
    "sk_min_bottleneck":    32,  # chiều bottleneck tối thiểu
    "transformer_layers":    1,  # số block trong mỗi ChannelTransformer
    "transformer_heads":     6,  # số attention head (khớp cấu hình A3 để so sánh công bằng)
}


# ─── Khối DSKConv: chọn lọc kernel kép + Channel Transformer ──────────────────
class DSKConv(nn.Module):
    """Dual Selective Kernel Convolution, hợp nhất bằng Channel Transformer.

    Hai trục chọn nhánh:
      • Channel attention   — chọn nhánh dilation theo TỪNG channel.
      • Frequency attention — chọn nhánh dilation theo TỪNG hàng sub-carrier.
    Ghép hai kết quả theo chiều rộng W, đưa qua Transformer, rồi pool W về ban đầu.
    """

    def __init__(self, features, img_size, branches=3, groups=32,
                 reduction=4, min_bottleneck=32,
                 transformer_layers=1, transformer_heads=3):
        super().__init__()
        self.img_size = list(img_size)      # [H, 2W] — kích thước SAU khi ghép hai nhánh
        bottleneck = max(features // reduction, min_bottleneck)

        # M nhánh grouped-conv 3×3, dilation tăng dần 1, 2, 3, ... (vùng nhìn rộng dần).
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, padding=1 + i,
                          dilation=1 + i, groups=groups, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            )
            for i in range(branches)
        )

        # Nhánh channel-attention: GAP -> bottleneck -> một logit map cho mỗi nhánh.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(features, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
        )
        self.branch_fcs = nn.ModuleList(
            nn.Conv2d(bottleneck, features, kernel_size=1) for _ in range(branches)
        )
        self.softmax = nn.Softmax(dim=1)    # chuẩn hóa theo chiều nhánh M
        self.norm = nn.BatchNorm2d(features)

        self.transformer = ChannelTransformer(
            img_size=self.img_size, channels=features,
            num_layers=transformer_layers, num_heads=transformer_heads,
        )

    def forward(self, x):
        # Chạy M nhánh song song rồi stack: (B, M, C, H, W)
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)

        channel = self._channel_attention(feats)     # (B, C, H, W)
        freq = self._frequency_attention(feats)      # (B, C, H, W)

        # Ghép theo chiều rộng -> (B, C, H, 2W), rồi tinh chỉnh bằng Transformer.
        fused = torch.cat([channel, freq], dim=3)
        fused = self.transformer(self.norm(fused))
        # Giữ H, đưa W về lại một nửa -> (B, C, H, W).
        return F.avg_pool2d(fused, kernel_size=(1, 2))

    def _channel_attention(self, feats):
        """Chọn tổ hợp nhánh độc lập cho TỪNG channel."""
        context = self.fc(self.gap(feats.sum(dim=1)))                    # (B, bottleneck, 1, 1)
        logits = torch.stack([fc(context) for fc in self.branch_fcs], dim=1)  # (B, M, C, 1, 1)
        weights = self.softmax(logits)
        return (feats * weights).sum(dim=1)                              # (B, C, H, W)

    def _frequency_attention(self, feats):
        """Chọn tổ hợp nhánh độc lập cho TỪNG hàng sub-carrier."""
        descriptor = feats.sum(dim=2)                                    # (B, M, H, W): gộp channel
        pooled = F.adaptive_avg_pool2d(descriptor, (descriptor.size(2), 1))   # (B, M, H, 1): gộp thời gian
        weights = self.softmax(pooled)
        return (feats * weights.unsqueeze(2)).sum(dim=1)                 # (B, C, H, W)


# ─── DSKUnit: một stage trích xuất đặc trưng ─────────────────────────────────
class DSKUnit(nn.Module):
    """Một stage: conv1×1 -> AvgPool -> DSKConv -> BN -> conv1×1."""

    def __init__(self, in_features, mid_features, out_features, img_size, **dsk_kwargs):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d((2, 2))
        self.dsk = DSKConv(mid_features, img_size=img_size, **dsk_kwargs)
        self.norm = nn.BatchNorm2d(mid_features)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.conv1(x)     # đổi số channel
        x = self.pool(x)      # giảm H, W một nửa
        x = self.dsk(x)       # hợp nhất chọn lọc kép
        x = self.norm(x)
        return self.conv3(x)  # đổi số channel đầu ra


# ─── Model đầy đủ ─────────────────────────────────────────────────────────────
class DSKNetTransMMFI3D(nn.Module):
    """CSI -> 2 × SKUnit -> pool -> regression -> pose 3D (B, 17, 3)."""

    def __init__(self, config=None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._cfg = cfg
        C = cfg["base_channels"]
        dsk_kwargs = dict(
            branches=cfg["sk_branches"], groups=cfg["sk_groups"],
            reduction=cfg["sk_reduction"], min_bottleneck=cfg["sk_min_bottleneck"],
            transformer_layers=cfg["transformer_layers"],
            transformer_heads=cfg["transformer_heads"],
        )
        # img_size = [H, 2W]: kích thước SAU khi DSKConv ghép hai nhánh theo chiều rộng.
        self.skunit1 = DSKUnit(3, C, C, img_size=[57, 10], **dsk_kwargs)
        self.bn = nn.BatchNorm2d(C)
        self.skunit2 = DSKUnit(C, C * 2, C * 2, img_size=[28, 4], **dsk_kwargs)
        self.final_pool = nn.AvgPool2d((2, 2))
        self.regression = RegressionHead(input_dim=3584, output_dim=51,
                                         hidden_dim=cfg["reg_hidden"])

    def forward(self, x):
        x = self.skunit1(x)       # (B, 128, 57, 5)
        x = self.bn(x)
        x = self.skunit2(x)       # (B, 256, 28, 2)
        x = self.final_pool(x)    # (B, 256, 14, 1)
        pose = self.regression(x)                    # (B, 51)
        return pose.reshape(pose.size(0), 17, 3)     # (B, 17, 3)

    def get_model_config(self):
        return dict(self._cfg)
