"""
demo_web.py
Demo trực quan trên trình duyệt: 3 panel đồng bộ — RGB | pose 3D ground-truth |
pose 3D dự đoán từ WiFi-CSI — cho một sequence MMFi (một cặp subject-action).

Chạy hoàn toàn từ folder này: nạp checkpoint huấn luyện bởi HPE-Li-3D, đọc CSI
bằng ĐÚNG tiền xử lý lúc train (dùng lại data/mmfi._load_csi — một nguồn duy
nhất, nên không cần bước kiểm tra "loader consistency" như demo gốc).

Model nạp MỘT LẦN lúc khởi động; sau đó có thể chọn subject/action ngay trên
trình duyệt để chạy demo — không cần khởi động lại server.

    # Mở app, chọn subject/action trong web:
    python demo_web.py `
        --checkpoint "D:\\...\\best.pt" `
        --dataset-root "D:\\MMFi-dataset\\MMFi_Dataset" --device cpu

    # (Tùy chọn) nạp sẵn một sequence ngay khi mở:
    python demo_web.py ... --subject S21 --action A05

    # Chỉ chạy inference + in chẩn đoán rồi thoát (cần --subject/--action):
    python demo_web.py ... --subject S21 --action A05 --dry-run
"""
import argparse
import http.server
import json
import mimetypes
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.mmfi import SUBJECT_TO_SCENE, _load_csi
from metrics import compute_metrics
from model import DSKNetTransMMFI3D

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")


DEFAULT_RGB_ROOT = r"D:\RGB_image-MMFi\MMFi_Defaced_RGB"
HTML_PATH = Path(__file__).with_name("demo_web.html")

JOINT_NAMES = [
    "Bot Torso", "R.Hip", "R.Knee", "R.Foot", "L.Hip", "L.Knee", "L.Foot",
    "Center Torso", "Upper Torso", "Neck Base", "Center Head",
    "L.Shoulder", "L.Elbow", "L.Hand", "R.Shoulder", "R.Elbow", "R.Hand",
]
BONES = [
    [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
    [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16],
]
# Màu khớp/xương: GT tông lạnh (cyan/xanh), prediction tông nóng (cam/hồng).
GT_JOINT_COLORS = (["#00E5FF"] + ["#40C4FF"] * 3 + ["#69F0AE"] * 3
                   + ["#00E5FF"] * 4 + ["#69F0AE"] * 3 + ["#40C4FF"] * 3)
PRED_JOINT_COLORS = (["#FF6D00"] + ["#FFD740"] * 3 + ["#FF4081"] * 3
                     + ["#FF6D00"] * 4 + ["#FF4081"] * 3 + ["#FFD740"] * 3)
GT_EDGE_COLORS = (["#40C4FF"] * 3 + ["#69F0AE"] * 3 + ["#00E5FF"] * 4
                  + ["#69F0AE"] * 3 + ["#40C4FF"] * 3)
PRED_EDGE_COLORS = (["#FFD740"] * 3 + ["#FF4081"] * 3 + ["#FF6D00"] * 4
                    + ["#FF4081"] * 3 + ["#FFD740"] * 3)

# Cặp khớp làm thước đo body-scale cho g_PCK: R.Hip (1) -> L.Shoulder (11).
G_PCK_SCALE_JOINTS = (1, 11)


def parse_args():
    p = argparse.ArgumentParser(description="Demo 3 panel: RGB | GT 3D | Prediction 3D.")
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt do train.py lưu.")
    p.add_argument("--dataset-root", required=True, help="Thư mục gốc dataset MMFi.")
    p.add_argument("--subject", default=None, help="Tùy chọn — nạp sẵn subject này khi mở.")
    p.add_argument("--action", default=None, help="Tùy chọn — nạp sẵn action này khi mở.")
    p.add_argument("--rgb-root", default=DEFAULT_RGB_ROOT)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8086)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Chạy inference, in chẩn đoán rồi thoát (cần --subject/--action).")
    return p.parse_args()


# ─── Nạp checkpoint vào model clean ──────────────────────────────────────────
# Checkpoint được lưu bởi HPE-Li-3D (bản gốc) — tên config/tham số hơi khác bản
# clean, nên cần hai bảng dịch nhỏ dưới đây.

CONFIG_KEY_MAP = {
    "num_lay": "base_channels", "hidden_reg": "reg_hidden",
    "sk_m": "sk_branches", "sk_g": "sk_groups",
    "sk_r": "sk_reduction", "sk_l": "sk_min_bottleneck",
}
STATE_KEY_REPLACEMENTS = (
    (".fcs.", ".branch_fcs."),
    (".transformer.encoder.layers.", ".transformer.blocks."),
    (".transformer.encoder.norm.", ".transformer.norm."),
)


def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint phải là dict có model_state_dict (do train.py lưu).")

    config = {CONFIG_KEY_MAP.get(k, k): v
              for k, v in (ckpt.get("model_config") or {}).items()}
    model = DSKNetTransMMFI3D(config).to(device)

    state = {}
    for key, value in ckpt["model_state_dict"].items():
        key = key.removeprefix("module.")
        for old, new in STATE_KEY_REPLACEMENTS:
            key = key.replace(old, new)
        state[key] = value
    model.load_state_dict(state)   # strict: lệch key nào là báo lỗi ngay
    model.eval()

    pose_stats = ckpt.get("pose_normalization") or {"enabled": False}
    return model, pose_stats


# ─── Quét dataset: những subject/action nào có sẵn để chọn ───────────────────

def scan_catalog(dataset_root):
    """Duyệt cây thư mục dataset, chỉ liệt kê subject/action thực sự có ground_truth."""
    root = Path(dataset_root)
    subjects = []
    for scene_dir in sorted(root.glob("E*")):
        if not scene_dir.is_dir():
            continue
        for subj_dir in sorted(scene_dir.glob("S*")):
            if not subj_dir.is_dir():
                continue
            actions = [a.name for a in sorted(subj_dir.glob("A*"))
                       if (a / "ground_truth.npy").exists()]
            if actions:
                subjects.append({"id": subj_dir.name, "scene": scene_dir.name,
                                 "actions": actions})
    return subjects


# ─── Nạp sequence (CSI + GT) ─────────────────────────────────────────────────

def load_sequence(dataset_root, subject, action, scene=None, max_frames=None):
    """Đọc toàn bộ frame hợp lệ của một sequence, ghép CSI–GT theo CHỈ SỐ FRAME THẬT.

    Frame .mat rỗng/thiếu bị bỏ qua nhưng chỉ số thật vẫn được giữ trong
    frame_ids, nên GT (và ảnh RGB) luôn khớp đúng frame — kể cả khi có lỗ hổng.
    """
    scene = scene or SUBJECT_TO_SCENE[subject]

    base = Path(dataset_root) / scene / subject / action
    gt_all = np.load(base / "ground_truth.npy")[:, :, :3].astype(np.float32)

    csi_frames, gt_frames, frame_ids = [], [], []
    for idx in range(len(gt_all)):
        mat = base / "wifi-csi" / f"frame{idx + 1:03d}.mat"
        if not mat.exists() or mat.stat().st_size == 0:
            continue
        csi_frames.append(_load_csi(str(mat)))     # đúng tiền xử lý lúc train
        gt_frames.append(gt_all[idx])
        frame_ids.append(idx)
        if max_frames is not None and len(csi_frames) >= max_frames:
            break
    if not csi_frames:
        raise FileNotFoundError(f"Không có frame CSI hợp lệ trong {base}.")

    return (np.stack(csi_frames).astype(np.float32),
            np.stack(gt_frames).astype(np.float32), frame_ids, scene)


# ─── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, csi_sequence, device, batch_size, pose_stats):
    """Chạy model theo batch, đưa dự đoán từ z-score về tọa độ mét."""
    if pose_stats.get("enabled"):
        mean = torch.tensor(pose_stats["mean_xyz"], device=device).view(1, 1, 3)
        std = torch.tensor(pose_stats["std_xyz"], device=device).view(1, 1, 3)
    else:
        mean = std = None

    chunks = []
    for start in range(0, len(csi_sequence), batch_size):
        csi = torch.from_numpy(csi_sequence[start:start + batch_size]).to(device)
        pred = model(csi)                          # (B, 17, 3)
        if mean is not None:
            pred = pred * std + mean
        chunks.append(pred.cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


# ─── Chẩn đoán chất lượng dự đoán trên sequence ──────────────────────────────

def _g_pck(pred, gt, threshold):
    """PCK chuẩn hóa theo body-scale: đúng nếu sai số <= threshold × khoảng cách GT
    giữa R.Hip và L.Shoulder của từng frame."""
    a, b = G_PCK_SCALE_JOINTS
    scale = np.linalg.norm(gt[:, a] - gt[:, b], axis=-1)            # (N,)
    dist = np.linalg.norm(pred - gt, axis=-1)                       # (N, 17)
    valid = scale > 1e-8
    return float((dist[valid] <= threshold * scale[valid, None]).mean() * 100.0)


def _mean_step_ratio(pred, gt):
    """Tỉ lệ chuyển động trung bình giữa các frame: pred / GT (1.0 = khớp nhịp)."""
    if len(gt) < 2:
        return None
    gt_step = np.linalg.norm(np.diff(gt, axis=0), axis=-1).mean()
    pred_step = np.linalg.norm(np.diff(pred, axis=0), axis=-1).mean()
    return float(pred_step / gt_step) if gt_step > 1e-9 else None


def _temporal_correlation_mean(pred, gt):
    """Trung bình hệ số tương quan theo thời gian của từng (khớp, trục)."""
    corrs = []
    for j in range(gt.shape[1]):
        for c in range(gt.shape[2]):
            g, p = gt[:, j, c], pred[:, j, c]
            if g.std() <= 1e-8 or p.std() <= 1e-8:
                continue
            r = np.corrcoef(g, p)[0, 1]
            if np.isfinite(r):
                corrs.append(float(r))
    return float(np.mean(corrs)) if corrs else None


def compute_diagnostics(pred, gt):
    pred = np.asarray(pred, np.float64)
    gt = np.asarray(gt, np.float64)
    base = compute_metrics(pred, gt)               # MPJPE, PA-MPJPE, PCK@50/100

    # Baseline "đoán một pose trung bình cố định cho cả sequence" — model phải
    # thắng baseline này thì mới thực sự bám theo chuyển động.
    const_mean = np.broadcast_to(gt.mean(axis=0, keepdims=True), gt.shape)
    const_mpjpe = float(np.linalg.norm(const_mean - gt, axis=-1).mean() * 1000.0)

    root_pred, root_gt = pred - pred[:, :1], gt - gt[:, :1]   # khử vị trí gốc (hông)

    diag = {
        "num_frames": int(len(gt)),
        **base,
        "root_mpjpe_mm": float(np.linalg.norm(pred[:, 0] - gt[:, 0], axis=-1).mean() * 1000.0),
        "root_centered_mpjpe_mm": float(np.linalg.norm(root_pred - root_gt, axis=-1).mean() * 1000.0),
        "axis_mae_mm_by_name": {
            axis: float(np.abs(pred[..., i] - gt[..., i]).mean() * 1000.0)
            for i, axis in enumerate(("x", "y", "z"))
        },
        "constant_sequence_mean_gt_mpjpe_mm": const_mpjpe,
        "mpjpe_gain_over_constant_sequence_mean_mm": const_mpjpe - base["mpjpe_mm"],
        "temporal_correlation": {"mean": _temporal_correlation_mean(pred, gt)},
        "motion_ratios": {"mean_step_ratio": _mean_step_ratio(pred, gt)},
        "articulation_motion_ratios": {"mean_step_ratio": _mean_step_ratio(root_pred, root_gt)},
    }
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5):
        diag[f"g_PCK@{int(threshold * 100)}"] = _g_pck(pred, gt, threshold)
    return diag


# ─── Chuẩn bị toàn bộ dữ liệu cho trang web ──────────────────────────────────

def find_rgb_dir(rgb_root, scene, subject, action):
    for candidate in (Path(rgb_root) / scene / subject / action / "rgb",
                      Path(rgb_root) / scene / subject / action,
                      Path(rgb_root) / subject / action / "rgb"):
        if candidate.is_dir():
            return candidate
    return None


def build_payload(subject, action, scene=None):
    """Nạp sequence (subject/action), chạy inference bằng MODEL đã nạp sẵn, gói payload."""
    r = RUNTIME
    csi, gt, frame_ids, scene = load_sequence(
        r["dataset_root"], subject, action, scene, r["max_frames"]
    )
    print(f"  Đã nạp {len(csi)} frame CSI/GT ({scene}/{subject}/{action}).", flush=True)

    pred = run_inference(r["model"], csi, r["device"], r["batch_size"], r["pose_stats"])

    mpjpe_frames = (np.linalg.norm(pred - gt, axis=-1).mean(axis=1) * 1000.0).tolist()
    prefix = f"{scene}_{subject}_{action}"
    return {
        "gt_frames": gt.astype(float).tolist(),
        "pred_frames": pred.astype(float).tolist(),
        # Tên hiển thị dùng chỉ số frame THẬT (khớp file RGB, kể cả khi có frame bị bỏ).
        "frame_names": [f"{prefix} frame{i + 1:03d}" for i in frame_ids],
        "frame_ids": frame_ids,
        "mpjpe_frames_mm": mpjpe_frames,
        "diagnostics": compute_diagnostics(pred, gt),
        "paths": {
            "checkpoint": str(r["checkpoint"]),
            "dataset_root": str(r["dataset_root"]),
            "sequence": f"{scene}/{subject}/{action}",
            "device": str(r["device"]),
        },
        "rgb_dir": find_rgb_dir(r["rgb_root"], scene, subject, action),
        "pose_normalization": r["pose_stats"],
    }


# ─── Trạng thái server + nạp sequence theo yêu cầu ───────────────────────────

RUNTIME = {}   # model, pose_stats, device, dataset_root… — nạp một lần trong main()
CATALOG = []   # danh sách subject/action quét từ dataset
STATE = {"ready": False, "idle": True, "status": "Chọn subject + action rồi bấm Chạy.",
         "error": None, "payload": None}
STATE_LOCK = threading.Lock()


def snapshot():
    with STATE_LOCK:
        return dict(STATE)


def load_into_state(subject, action, scene=None):
    """Nạp một sequence vào STATE (chạy trong thread nền). Cập nhật cờ ready/idle/error."""
    with STATE_LOCK:
        STATE.update(ready=False, idle=False, status=f"Đang nạp {subject}/{action}…",
                     error=None)
    try:
        payload = build_payload(subject, action, scene)
        with STATE_LOCK:
            STATE.update(ready=True, idle=False, status="ready", error=None, payload=payload)
        print(f"  Sẵn sàng: {subject}/{action}.", flush=True)
    except Exception as exc:                       # noqa: BLE001 — báo lỗi lên trang web
        with STATE_LOCK:
            STATE.update(ready=False, idle=True, status=f"Lỗi: {exc}", error=str(exc),
                         payload=None)
        print(f"  Lỗi nạp {subject}/{action}: {exc}", flush=True)


# ─── HTTP server ─────────────────────────────────────────────────────────────

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, mime):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        state = snapshot()
        payload = state["payload"]

        # Các route luôn phục vụ được (không phụ thuộc đã nạp sequence hay chưa).
        if path in ("/", "/index.html"):
            return self.send_bytes(HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/status":
            return self.send_json({"ready": state["ready"], "idle": state["idle"],
                                   "status": state["status"], "error": state["error"]})
        if path == "/api/catalog":
            return self.send_json({"subjects": CATALOG})
        if path == "/api/load":
            return self.handle_load()

        # Các route dưới đây cần đã nạp xong sequence.
        if not state["ready"]:
            return self.send_json({"error": state["status"]}, 503)

        if path == "/api/config":
            return self.send_json({
                "joint_names": JOINT_NAMES, "edges": BONES,
                "gt_joint_colors": GT_JOINT_COLORS, "gt_edge_colors": GT_EDGE_COLORS,
                "pred_joint_colors": PRED_JOINT_COLORS, "pred_edge_colors": PRED_EDGE_COLORS,
                "paths": payload["paths"],
                "pose_normalization": payload["pose_normalization"],
            })
        if path == "/api/data":
            return self.send_json({"frames": payload["gt_frames"],
                                   "frame_names": payload["frame_names"],
                                   "num_frames": len(payload["gt_frames"])})
        if path == "/api/predict":
            return self.send_json({"frames": payload["pred_frames"],
                                   "num_frames": len(payload["pred_frames"]),
                                   "mpjpe_frames_mm": payload["mpjpe_frames_mm"]})
        if path == "/api/diagnostics":
            return self.send_json(payload["diagnostics"])
        if path == "/api/rgb":
            return self.serve_rgb(payload)
        self.send_error(404)

    def handle_load(self):
        """Nhận yêu cầu nạp sequence mới; đặt trạng thái 'đang nạp' rồi chạy nền."""
        query = parse_qs(urlparse(self.path).query)
        subject = (query.get("subject") or [""])[0]
        action = (query.get("action") or [""])[0]
        scene = SCENE_OF.get(subject)
        if not subject or not action:
            return self.send_json({"error": "Thiếu subject hoặc action."}, 400)
        if scene is None:
            return self.send_json({"error": f"Subject {subject} không có trong dataset."}, 404)

        # Đặt ready=False NGAY (đồng bộ) để lần poll /api/status kế tiếp thấy 'đang nạp',
        # tránh việc web còn đọc nhầm dữ liệu sequence cũ.
        with STATE_LOCK:
            STATE.update(ready=False, idle=False, status=f"Đang nạp {subject}/{action}…",
                         error=None)
        threading.Thread(target=load_into_state, args=(subject, action, scene),
                         daemon=True).start()
        self.send_json({"ok": True})

    def serve_rgb(self, payload):
        if payload["rgb_dir"] is None:
            return self.send_error(404)
        position = int(parse_qs(urlparse(self.path).query).get("frame", ["0"])[0])
        position = max(0, min(position, len(payload["frame_ids"]) - 1))
        frame_id = payload["frame_ids"][position]  # ánh xạ vị trí -> chỉ số frame thật
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            image = Path(payload["rgb_dir"]) / f"frame{frame_id + 1:03d}{ext}"
            if image.exists():
                mime = mimetypes.guess_type(str(image))[0] or "image/png"
                return self.send_bytes(image.read_bytes(), mime)
        self.send_error(404)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Nạp model MỘT LẦN — sau đó mỗi lần chọn sequence chỉ chạy inference lại.
    print("Đang nạp model…", flush=True)
    model, pose_stats = load_model_from_checkpoint(args.checkpoint, device)
    print(f"  Model: {model.get_model_config()}", flush=True)
    RUNTIME.update(model=model, pose_stats=pose_stats, device=device,
                   checkpoint=args.checkpoint, dataset_root=args.dataset_root,
                   rgb_root=args.rgb_root, batch_size=args.batch_size,
                   max_frames=args.max_frames)

    global CATALOG, SCENE_OF
    CATALOG = scan_catalog(args.dataset_root)
    SCENE_OF = {s["id"]: s["scene"] for s in CATALOG}
    print(f"  Dataset có {len(CATALOG)} subject để chọn.", flush=True)
    if not CATALOG:
        raise SystemExit(f"Không tìm thấy subject/action nào trong {args.dataset_root}.")

    if args.dry_run:
        if not (args.subject and args.action):
            raise SystemExit("--dry-run cần cả --subject và --action.")
        payload = build_payload(args.subject, args.action, SCENE_OF.get(args.subject))
        d = payload["diagnostics"]
        print(f"frames={d['num_frames']} first_frame_mpjpe={payload['mpjpe_frames_mm'][0]:.1f}mm")
        print(f"mpjpe={d['mpjpe_mm']:.1f}mm pa_mpjpe={d['pa_mpjpe_mm']:.1f}mm "
              f"pck50={d['pck_50mm']:.1f}% pck100={d['pck_100mm']:.1f}%")
        print(f"g_PCK@50={d['g_PCK@50']:.1f}% root_centered={d['root_centered_mpjpe_mm']:.1f}mm "
              f"model_gain={d['mpjpe_gain_over_constant_sequence_mean_mm']:.1f}mm")
        return

    # KHÔNG dùng SO_REUSEADDR: trên Windows nó cho phép 2 server bind trùng port,
    # trình duyệt sẽ nói chuyện với server CŨ và hiển thị sai sequence.
    http.server.ThreadingHTTPServer.allow_reuse_address = False
    try:
        server = http.server.ThreadingHTTPServer((args.host, args.port), DemoHandler)
    except OSError:
        raise SystemExit(
            f"Port {args.port} đang bị một server khác chiếm (có thể là demo cũ chưa tắt).\n"
            f"Tắt nó trước (Ctrl+C ở terminal cũ) hoặc chạy lại với --port khác, ví dụ "
            f"--port {args.port + 1}."
        )
    url = f"http://{args.host}:{args.port}/index.html"
    print(f"server={url}\nGiữ terminal này mở. Chọn subject/action trên web. Ctrl+C để dừng.")

    # Nếu chỉ định sẵn subject/action thì nạp luôn; nếu không, web chờ người chọn.
    if args.subject and args.action:
        threading.Thread(target=load_into_state,
                         args=(args.subject, args.action, SCENE_OF.get(args.subject)),
                         daemon=True).start()

    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDừng server.")
    finally:
        server.server_close()


SCENE_OF = {}   # subject -> scene, dựng từ CATALOG trong main()


if __name__ == "__main__":
    main()
