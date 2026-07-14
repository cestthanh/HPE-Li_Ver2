"""
data/splits.py
Chia tập đánh giá thành val/test ở MỨC SEQUENCE.

Mỗi sequence = một file ground_truth (một cặp subject-action). Việc chia theo
sequence đảm bảo các frame của cùng một sequence không đồng thời rơi vào val và
test — tránh rò rỉ thông tin làm kết quả đánh giá bị lạc quan giả.
"""
import numpy as np
from torch.utils.data import Subset


def split_by_sequence(dataset, test_ratio=0.5, seed=41):
    """Trả về (val_subset, test_subset) từ một MMFiDataset."""
    # Gom chỉ số frame theo từng sequence (gt_path).
    groups = {}
    for idx, sample in enumerate(dataset.samples):
        groups.setdefault(sample["gt_path"], []).append(idx)

    seq_paths = sorted(groups)
    np.random.RandomState(seed).shuffle(seq_paths)

    n_test = int(round(len(seq_paths) * test_ratio))
    test_paths = set(seq_paths[:n_test])

    val_idx, test_idx = [], []
    for path, frame_indices in groups.items():
        (test_idx if path in test_paths else val_idx).extend(frame_indices)

    return Subset(dataset, sorted(val_idx)), Subset(dataset, sorted(test_idx))
