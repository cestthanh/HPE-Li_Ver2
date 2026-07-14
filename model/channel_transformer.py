"""
channel_transformer.py
Channel-wise Transformer — Transformer hoạt động trên các CHANNEL đặc trưng.

Điểm mấu chốt: ma trận attention có shape (C, C) — mô hình hóa quan hệ GIỮA CÁC
CHANNEL, khác với ViT chuẩn (attention (N, N) giữa các vị trí không gian).
Module này được gọi bên trong DSKConv (xem dsknet3d.py) để hợp nhất đặc trưng
sau hai nhánh chọn lọc channel/frequency.

Ký hiệu: B = batch, C = số channel, H×W = kích thước không gian, N = H*W token,
         T = số attention head.
"""
import math

import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm, Softmax


class _ChannelEmbeddings(nn.Module):
    """(B, C, H, W) -> (B, N, C): mỗi vị trí (h, w) thành 1 token, cộng position embedding."""

    def __init__(self, img_size, channels):
        super().__init__()
        n_patches = img_size[0] * img_size[1]          # N = H * W
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        tokens = x.flatten(2).transpose(-1, -2)
        tokens = tokens + self.position_embeddings     # thêm thông tin vị trí
        return self.dropout(tokens)


class _Attention(nn.Module):
    """Channel-wise self-attention: attention map (C, C), lấy trung bình T head."""

    def __init__(self, channels, num_heads):
        super().__init__()
        # Mỗi head có Q/K/V riêng, đều chiếu C -> C (KHÔNG chia nhỏ C như MHA chuẩn).
        self.queries = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(num_heads))
        self.keys    = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(num_heads))
        self.values  = nn.ModuleList(nn.Linear(channels, channels, bias=False) for _ in range(num_heads))
        self.norm    = nn.InstanceNorm2d(num_heads)    # chuẩn hóa ma trận attention từng head
        self.softmax = Softmax(dim=3)
        self.out     = nn.Linear(channels, channels, bias=False)
        self.dropout = Dropout(0.1)

    def forward(self, emb):
        # emb: (B, N, C). Stack T head -> (B, T, N, C)
        Q = torch.stack([q(emb) for q in self.queries], dim=1)
        K = torch.stack([k(emb) for k in self.keys], dim=1)
        V = torch.stack([v(emb) for v in self.values], dim=1)

        # ★ ĐIỂM CỐT LÕI: transpose Q TRƯỚC khi nhân K.
        #   (B,T,C,N) @ (B,T,N,C) -> (B,T,C,C): quan hệ channel-với-channel.
        scores = torch.matmul(Q.transpose(-1, -2), K) / math.sqrt(emb.size(-1))
        scores = self.norm(scores)
        attn = self.dropout(self.softmax(scores))      # (B, T, C, C)

        # Áp attention lên V: (B,T,C,C) @ (B,T,C,N) -> (B,T,C,N)
        context = torch.matmul(attn, V.transpose(-1, -2))

        # Về lại layout token và LẤY TRUNG BÌNH các head (không concat).
        context = context.permute(0, 3, 2, 1).mean(dim=3)   # (B, N, C)
        return self.dropout(self.out(context))


class _MLP(nn.Module):
    """Feed-forward theo từng token: C -> 4C -> C."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), Dropout(0.1),
            nn.Linear(dim * 4, dim), Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class _TransformerBlock(nn.Module):
    """Pre-norm block với 2 residual: x + Attn(norm(x)), rồi x + MLP(norm(x))."""

    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm1 = LayerNorm(channels, eps=1e-6)
        self.attn  = _Attention(channels, num_heads)
        self.norm2 = LayerNorm(channels, eps=1e-6)
        self.ffn   = _MLP(channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _Reconstruct(nn.Module):
    """(B, N, C) -> (B, C, H, W), rồi Conv1x1 + BN + ReLU."""

    def __init__(self, channels, img_size):
        super().__init__()
        self.img_size = img_size
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        B, _, C = x.shape
        h, w = self.img_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, h, w)
        return self.act(self.norm(self.conv(x)))


class ChannelTransformer(nn.Module):
    """Nhúng token -> nhiều block channel-attention -> dựng lại feature map (có residual ngoài).

    Args:
        img_size (list[int]): [H, W] của feature map đầu vào.
        channels (int):       số channel C.
        num_layers (int):     số Transformer block.
        num_heads (int):      số attention head.
    """

    def __init__(self, img_size, channels, num_layers, num_heads):
        super().__init__()
        self.embed = _ChannelEmbeddings(img_size, channels)
        self.blocks = nn.ModuleList(
            _TransformerBlock(channels, num_heads) for _ in range(num_layers)
        )
        self.norm = LayerNorm(channels, eps=1e-6)
        self.reconstruct = _Reconstruct(channels, img_size)

    def forward(self, x):
        tokens = self.embed(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # Residual bao ngoài toàn bộ Transformer.
        return self.reconstruct(tokens) + x
