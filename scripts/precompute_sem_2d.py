import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from voom.data import (
    SemanticKITTIDataset,
    parse_calib,
    label_to_sem,
    label_to_occ,
    skitti_to_voom_grid,
)
from voom.ops import ray_marching
import config


N_CLASSES = 20  # 0=empty + 19 sem


def make_label_grid(label_grid, grid_dim):
    """Build a [20, gx, gy, gz] grid where ch0=alpha, ch1..19=one-hot of class c."""
    gx, _, gz = grid_dim

    sem = label_to_sem(label_grid).astype(np.float32)  # [256, 256, 32]
    occ = label_to_occ(label_grid)  # [256, 256, 32]

    sem_voom = skitti_to_voom_grid(sem, gx, gz)  # [gx, gy, gz]
    occ_voom = skitti_to_voom_grid(occ, gx, gz)  # [gx, gy, gz]

    one_hot = np.zeros((19,) + sem_voom.shape, dtype=np.float32)
    for c in range(1, N_CLASSES):
        one_hot[c - 1] = (sem_voom == c).astype(np.float32)

    alpha = (occ_voom * 0.99)[None]  # [1, gx, gy, gz]
    grid = np.concatenate([alpha, one_hot], axis=0)  # [20, gx, gy, gz]
    return torch.from_numpy(grid)


def build_K(calib_path, dataset_W, dataset_H, target_H, target_W):
    P2 = parse_calib(calib_path)
    fx, fy = P2[0, 0], P2[1, 1]
    cx, cy = P2[0, 2], P2[1, 2]
    cx = dataset_W - 1 - cx  # mirror to match TF.hflip
    sx, sy = target_W / dataset_W, target_H / dataset_H
    return torch.tensor(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def cast_one(label_grid, K, H, W, device):
    grid = make_label_grid(label_grid, config.grid_dim).unsqueeze(0).to(device)
    K_b = K.unsqueeze(0).to(device)
    rendered = ray_marching(
        grid=grid,
        K=K_b,
        size=(H, W),
        mpv=config.mpv,
        step_size=config.mpv,
        offset_m=config.offset_m,
    )  # [1, 19, H, W]

    # argmax over 19 class channels; +1 to remap to {1..19}
    cls = rendered[0].argmax(dim=0) + 1
    has_hit = rendered[0].sum(dim=0) > 0.5  # any class accumulated mass
    cls[~has_hit] = 0
    return cls.cpu().numpy().astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = config.train["height"], config.train["width"]
    dataset_W = config.train["data"]["width"]
    dataset_H = config.train["data"]["height"]

    ds = SemanticKITTIDataset(**config.dataset_config)
    print(f"{len(ds)} samples, output={H}x{W}, device={device}")

    skipped = 0
    written = 0
    for idx in range(len(ds)):
        _, label_path, calib_path, _ = ds.samples[idx]
        out_path = Path(label_path).with_suffix(".label_2d.npy")
        if out_path.exists():
            skipped += 1
            if (idx + 1) % 500 == 0:
                print(f"  [{idx + 1}/{len(ds)}] skipped={skipped} written={written}")
            continue

        label_grid = np.fromfile(label_path, dtype=np.uint16).reshape(256, 256, 32)
        K = build_K(calib_path, dataset_W, dataset_H, H, W)
        sem_2d = cast_one(label_grid, K, H, W, device)
        np.save(out_path, sem_2d)
        written += 1

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(ds)}] skipped={skipped} written={written}")

    print(f"\ndone. skipped={skipped}, written={written}")


if __name__ == "__main__":
    main()
