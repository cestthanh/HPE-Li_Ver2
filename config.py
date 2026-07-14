"""
config.py
Cấu hình thí nghiệm P1-S1 (MMFi Protocol 1, Setting 1 — random split).
Tách riêng để toàn bộ thiết lập nằm ở một chỗ, dễ đọc và dễ chỉnh.
"""
CONFIG = {
    # Dữ liệu
    "protocol":     "protocol1",   # 14 hành động sinh hoạt hằng ngày
    "split_ratio":  0.8,           # 80% subject cho train, 20% cho đánh giá
    "split_seed":   0,             # seed chia subject (tái lập được)

    # Huấn luyện
    "epochs":          50,
    "learning_rate":   1e-3,       # optimizer Adam
    "grad_clip_norm":  1.0,        # cắt norm gradient để ổn định

    # Loss (ver2): SmoothL1 trên MÉT THÔ + bone-length loss — xem losses.py
    "smooth_l1_beta":   0.1,       # ngưỡng Huber cho vị trí khớp = 0.1 m (10 cm)
    "bone_loss_weight": 0.2,       # trọng số bone-length loss trong tổng loss
    "bone_loss_beta":   0.05,      # ngưỡng Huber cho độ dài xương = 0.05 m (5 cm)

    # DataLoader
    "train_batch_size": 16,
    "eval_batch_size":  8,
    "num_workers":      4,
}
