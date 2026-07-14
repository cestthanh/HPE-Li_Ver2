"""
data/mmfi.py
Nạp dữ liệu MMFi: đọc WiFi-CSI (.mat) và pose 3D ground-truth (.npy) từ đĩa.

Bản rút gọn — chỉ giữ đúng chế độ dùng cho thí nghiệm P1-S1:
  • đơn vị mẫu = frame (mỗi frame là một mẫu độc lập),
  • cách chia   = random_split (chia subject ngẫu nhiên, riêng cho từng action).
Bản gốc còn có class MMFi_Database dựng cây index scene/subject/action, nhưng
Dataset thực chất chỉ dùng đường dẫn gốc — nên đã được lược bỏ hoàn toàn.

Cây thư mục dataset kỳ vọng:
    <root>/<scene>/<subject>/<action>/wifi-csi/frameXXX.mat
    <root>/<scene>/<subject>/<action>/ground_truth.npy
"""
import os

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader, Dataset


# 40 subject của MMFi; mỗi 10 subject liên tiếp thuộc một scene E01..E04.
ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
SUBJECT_TO_SCENE = {s: f"E{(i // 10) + 1:02d}" for i, s in enumerate(ALL_SUBJECTS)}

# Protocol 1 = 14 hành động sinh hoạt hằng ngày.
PROTOCOL1_ACTIONS = ["A02", "A03", "A04", "A05", "A13", "A14", "A17",
                     "A18", "A19", "A20", "A21", "A22", "A23", "A27"]

FRAMES_PER_SEQUENCE = 297   # số frame mỗi (subject, action)


def build_random_split(ratio, seed, actions=PROTOCOL1_ACTIONS):
    """Chia subject thành train/val ngẫu nhiên, độc lập cho từng action.

    Trả về (train_form, val_form): mỗi cái là dict {subject: [danh sách action]}.
    Dùng seed tăng dần theo action để tái lập được đúng như bản gốc.
    """
    train_form, val_form = {}, {}
    n_train = int(np.floor(ratio * len(ALL_SUBJECTS)))
    for offset, action in enumerate(actions):
        perm = np.random.RandomState(seed + offset).permutation(len(ALL_SUBJECTS))
        train_subjects = {ALL_SUBJECTS[i] for i in perm[:n_train]}
        for subject in ALL_SUBJECTS:
            form = train_form if subject in train_subjects else val_form
            form.setdefault(subject, []).append(action)
    return train_form, val_form


def _load_csi(path):
    """Đọc 1 file CSI .mat, thay NaN/Inf, chuẩn hóa min-max về [0, 1]. -> (3, 114, 10)."""
    mat = scio.loadmat(path)["CSIamp"]
    mat[np.isinf(mat)] = np.nan
    for t in range(mat.shape[2]):                 # điền NaN theo trung bình từng lát thời gian
        sl = mat[:, :, t]
        nan = np.isnan(sl)
        if nan.any():
            sl[nan] = sl[~nan].mean()
    return (mat - mat.min()) / (mat.max() - mat.min())


class MMFiDataset(Dataset):
    """Mỗi mẫu = 1 frame: input CSI (3, 114, 10) + pose GT (17, 3)."""

    def __init__(self, data_root, data_form):
        self.data_root = data_root
        self.samples = self._index_frames(data_form)

    def _index_frames(self, data_form):
        samples = []
        for subject, actions in data_form.items():
            scene = SUBJECT_TO_SCENE[subject]
            for action in actions:
                base = os.path.join(self.data_root, scene, subject, action)
                gt_path = os.path.join(base, "ground_truth.npy")
                for frame in range(FRAMES_PER_SEQUENCE):
                    csi_path = os.path.join(base, "wifi-csi", f"frame{frame + 1:03d}.mat")
                    if os.path.exists(csi_path) and os.path.getsize(csi_path) > 0:
                        samples.append({"csi_path": csi_path, "gt_path": gt_path, "frame": frame})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        csi = _load_csi(s["csi_path"])                     # (3, 114, 10)
        gt = np.load(s["gt_path"])[s["frame"], :, :3]      # (17, 3)
        return {"csi": torch.from_numpy(csi).float(),
                "pose": torch.from_numpy(gt).float()}


def _collate(batch):
    return {"csi":  torch.stack([b["csi"] for b in batch]),
            "pose": torch.stack([b["pose"] for b in batch])}


def make_datasets(data_root, ratio=0.8, seed=0):
    """Trả về (train_dataset, eval_dataset) theo random_split của Protocol 1."""
    train_form, val_form = build_random_split(ratio, seed)
    return MMFiDataset(data_root, train_form), MMFiDataset(data_root, val_form)


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, collate_fn=_collate, num_workers=num_workers)
