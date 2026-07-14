"""
metrics.py
Các chỉ số đánh giá pose 3D (đơn vị mm) cho benchmark MMFi 17 khớp.

  • MPJPE       — sai số vị trí trung bình mỗi khớp (chỉ số dùng để CHỌN checkpoint).
  • RC-MPJPE    — MPJPE sau khi trừ khớp gốc (0 — hông) khỏi cả pred lẫn GT: tách
                  "lỗi tư thế" khỏi "lỗi định vị toàn cục". Đây là quy ước của
                  WiFlow/HPE-Li trong bảng MMFi — chỉ so sánh với họ bằng số này.
  • PA-MPJPE    — MPJPE sau khi căn chỉnh Procrustes (bỏ khác biệt tư thế cứng: xoay,
                  tịnh tiến, co giãn) → phản ánh sai số hình dáng.
  • PCK@50/100  — tỉ lệ % khớp có sai số nằm trong ngưỡng cố định (50 / 100 mm).
  • g_PCK@k     — PCK theo body-scale: đúng nếu sai số ≤ k × khoảng cách GT giữa
                  R.Hip (1) và L.Shoulder (11) của từng frame (quy ước PCK WiFlow).
"""
import numpy as np

# Cặp khớp làm thước đo body-scale cho g_PCK: R.Hip (1) ↔ L.Shoulder (11).
G_PCK_SCALE_JOINTS = (1, 11)


def _check(pred, gt):
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    assert pred.shape == gt.shape and pred.shape[-1] == 3, "cần shape (B, số_khớp, 3)"
    return pred, gt


def mpjpe_mm(pred, gt):
    pred, gt = _check(pred, gt)
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000.0)


def root_centered_mpjpe_mm(pred, gt):
    pred, gt = _check(pred, gt)
    return mpjpe_mm(pred - pred[:, :1], gt - gt[:, :1])


def pck_mm(pred, gt, threshold_mm):
    pred, gt = _check(pred, gt)
    dist_mm = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    return float((dist_mm <= threshold_mm).mean() * 100.0)


def g_pck(pred, gt, threshold):
    pred, gt = _check(pred, gt)
    a, b = G_PCK_SCALE_JOINTS
    scale = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1)             # (B,)
    dist = np.linalg.norm(pred - gt, axis=-1)                        # (B, 17)
    valid = scale > 1e-8
    return float((dist[valid] <= threshold * scale[valid, None]).mean() * 100.0)


def _procrustes(pred, gt, eps=1e-8):
    """Căn chỉnh pred về gt bằng phép biến đổi tương tự tối ưu (xoay + co giãn + tịnh tiến)."""
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        return None
    mu_g, mu_p = gt.mean(0), pred.mean(0)
    g, p = gt - mu_g, pred - mu_p
    norm_g, norm_p = np.linalg.norm(g), np.linalg.norm(p)
    if norm_g < eps or norm_p < eps:
        return None
    try:
        u, s, vt = np.linalg.svd((g / norm_g).T @ (p / norm_p))
    except np.linalg.LinAlgError:
        return None
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:              # đảm bảo là phép xoay hợp lệ (không lật gương)
        vt[-1] *= -1
        s[-1] *= -1
        rotation = vt.T @ u.T
    scale = s.sum() * norm_g / norm_p
    return scale * (p @ rotation) + mu_g


def pa_mpjpe_mm(pred, gt):
    pred, gt = _check(pred, gt)
    errors = []
    for p, g in zip(pred, gt):
        aligned = _procrustes(p, g)
        if aligned is not None:
            errors.append(np.linalg.norm(aligned - g, axis=-1).mean() * 1000.0)
    return float(np.mean(errors)) if errors else float("nan")


def compute_metrics(pred, gt):
    """Gói toàn bộ chỉ số cho một tập dự đoán/nhãn (mỗi cái shape (B, 17, 3))."""
    return {
        "mpjpe_mm":    mpjpe_mm(pred, gt),
        "rc_mpjpe_mm": root_centered_mpjpe_mm(pred, gt),
        "pa_mpjpe_mm": pa_mpjpe_mm(pred, gt),
        "pck_50mm":    pck_mm(pred, gt, 50.0),
        "pck_100mm":   pck_mm(pred, gt, 100.0),
        "g_pck_20":    g_pck(pred, gt, 0.2),
        "g_pck_50":    g_pck(pred, gt, 0.5),
    }
