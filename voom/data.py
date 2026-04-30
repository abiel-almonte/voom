"""SemanticKITTI dataset for VOOMv2 training."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from PIL import Image

import config

_KITTI_W = config.train["data"]["width"]
_KITTI_H = config.train["data"]["height"]
_MEAN = torch.tensor(config.common["preprocess"]["mean"]).view(3, 1, 1)
_STD = torch.tensor(config.common["preprocess"]["std"]).view(3, 1, 1)

LEARNING_MAP = {
    0: 0,
    1: 0,
    10: 1,
    11: 2,
    13: 5,
    15: 3,
    16: 5,
    18: 4,
    20: 5,
    30: 6,
    31: 7,
    32: 8,
    40: 9,
    44: 10,
    48: 11,
    49: 12,
    50: 13,
    51: 14,
    52: 0,
    60: 15,
    70: 16,
    71: 17,
    72: 16,
    80: 18,
    81: 19,
    99: 0,
    252: 1,
    253: 6,
    254: 7,
    255: 0,
    259: 8,
}


def parse_calib(calib_path):
    with open(calib_path) as f:
        for line in f:
            if line.startswith("P2:"):
                vals = [float(x) for x in line.strip().split()[1:]]
                return np.array(vals).reshape(3, 4)
    raise RuntimeError(f"P2 not found in {calib_path}")


def label_to_occ(label_grid):
    return ((label_grid > 0) & (label_grid < 255)).astype(np.float32)


def label_to_sem(label_grid):
    out = np.zeros_like(label_grid, dtype=np.uint8)
    for raw_id, mapped_id in LEARNING_MAP.items():
        out[label_grid == raw_id] = mapped_id
    return out


def frustum_mask(gx, gy, gz, mpv, fx, cx, fy, cy, img_w, img_h, offset_m):
    offset_x_m, y_offset_m, _ = offset_m
    mask = np.zeros((gx, gy, gz), dtype=bool)

    for vk in range(gz):
        z = (vk + 0.5) * mpv
        if z < 1e-3:
            continue
        vi_left = (0 - cx) * z / fx + offset_x_m
        vi_right = (img_w - 1 - cx) * z / fx + offset_x_m
        vi_lo = max(0, int(np.floor(vi_left / mpv)))
        vi_hi = min(gx, int(np.ceil(vi_right / mpv)) + 1)

        vj_top = (0 - cy) * z / fy + y_offset_m
        vj_lo = max(0, int(np.floor(vj_top / mpv)))
        vj_hi = gy

        if vi_lo < vi_hi and vj_lo < vj_hi:
            mask[vi_lo:vi_hi, vj_lo:vj_hi, vk] = True
    return mask


def skitti_to_voom_grid(grid_256, gx, gz, offset_m, mpv, fmask=None):
    gy = 32
    j_lo = 128 - gx // 2
    j_hi = j_lo + gx
    i_hi = min(gz, 256)

    out = np.zeros((gx, gy, gz), dtype=np.float32)
    crop = grid_256[:i_hi, j_lo:j_hi, :]
    y_off = int(offset_m[1] / mpv)
    for k in range(32):
        vj = y_off - k
        if 0 <= vj < gy:
            out[:gx, vj, :i_hi] = crop[:i_hi, :gx, k].T

    if fmask is not None:
        out *= fmask
    return out


class SemanticKITTIDataset(Dataset):
    def __init__(self, root, seqs, resize, load_sem=False, load_depth=False):
        super().__init__()
        self.resize = tuple(resize)  # (H, W)
        self.root = Path(root)
        self.load_sem = load_sem
        self.load_depth = load_depth

        self.samples = []  # (img_path, label_path, calib_path, depth_path|None)
        for seq in seqs:
            seq_dir = self.root / "sequences" / f"{int(seq):02d}"
            img_dir = seq_dir / "image_2"
            vox_dir = seq_dir / "voxels"
            depth_dir = seq_dir / "depth"
            calib_path = seq_dir / "calib.txt"

            if not img_dir.exists() or not vox_dir.exists():
                print(f"Warning: {seq_dir} incomplete, skipping")
                continue

            for lf in sorted(f for f in os.listdir(vox_dir) if f.endswith(".label")):
                frame_id = lf.replace(".label", "")
                img_path = img_dir / f"{frame_id}.png"
                if not img_path.exists():
                    continue
                depth_path = depth_dir / f"{frame_id}.png"
                self.samples.append(
                    (
                        str(img_path),
                        str(vox_dir / lf),
                        str(calib_path),
                        str(depth_path) if depth_path.exists() else None,
                    )
                )

        gx, gy, gz = config.grid_dim
        if self.samples:
            P2 = parse_calib(self.samples[0][2])
            fx, fy = P2[0, 0], P2[1, 1]
            cx, cy = P2[0, 2], P2[1, 2]
            self.fmask = frustum_mask(
                gx,
                gy,
                gz,
                config.mpv,
                fx,
                cx,
                fy,
                cy,
                _KITTI_W,
                _KITTI_H,
                config.offset_m,
            )
        else:
            self.fmask = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, calib_path, depth_path = self.samples[idx]

        rgb = Image.open(img_path).convert("RGB")
        rgb = TF.resize(rgb, list(self.resize))
        rgb = TF.hflip(rgb)  # SKITTI training convention
        rgb = TF.to_tensor(rgb)
        rgb = (rgb - _MEAN) / _STD

        # K from P2, mirrored cx + scaled to resize
        P2 = parse_calib(calib_path)
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]
        cx = _KITTI_W - 1 - cx  # mirror cx to match hflip
        H, W = self.resize
        sx, sy = W / _KITTI_W, H / _KITTI_H
        K = torch.tensor(
            [
                [fx * sx, 0.0, cx * sx],
                [0.0, fy * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        gx, gy, gz = config.grid_dim
        label = np.fromfile(label_path, dtype=np.uint16).reshape(256, 256, 32)
        occ_voom = skitti_to_voom_grid(
            label_to_occ(label),
            gx,
            gz,
            config.offset_m,
            config.mpv,
            fmask=self.fmask,
        )
        occ = torch.from_numpy(occ_voom).unsqueeze(0)  # [1, gx, gy, gz]

        if self.load_sem:
            sem_voom = skitti_to_voom_grid(
                label_to_sem(label).astype(np.float32),
                gx,
                gz,
                config.offset_m,
                config.mpv,
                fmask=self.fmask,
            )
            sem = torch.from_numpy(sem_voom).long()  # [gx, gy, gz]
        else:
            sem = torch.zeros(gx, gy, gz, dtype=torch.long)

        if self.load_depth and depth_path is not None:
            depth_raw = cv2.imread(
                depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            )
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)
            depth_raw = depth_raw[:, ::-1].copy()  # mirror to match RGB
            depth = torch.tensor(depth_raw, dtype=torch.float32).unsqueeze(0) / 100.0
        else:
            depth = torch.zeros(1, H, W)

        return rgb, K, occ, sem, depth
