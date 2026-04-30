import argparse
import sys
from pathlib import Path

# Make repo root importable so `voom` and `config` resolve regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from voom import VOOMv2, SemanticKITTIDataset, load_state_dict
import config

CLASS_NAMES = config.train["semantic"]["enabled"]
N_SEM_CLASSES = len(CLASS_NAMES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.26)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SemanticKITTIDataset(**config.val_dataset_config)

    sd = load_state_dict(args.ckpt, device=device)

    model_cfg = dict(config.model_config)
    model = VOOMv2(**model_cfg).to(device).half()
    model.load_state_dict(sd, strict=False)
    model.eval()

    fmask = torch.from_numpy(ds.fmask.astype(np.float32)).to(device)
    fmask4 = fmask.unsqueeze(0).unsqueeze(0)  # [1, 1, gx, gy, gz]

    total_tp = total_fp = total_fn = 0
    total_sem_correct = total_sem_total = 0
    class_tp = np.zeros(N_SEM_CLASSES, dtype=np.float64)
    class_fp = np.zeros(N_SEM_CLASSES, dtype=np.float64)
    class_fn = np.zeros(N_SEM_CLASSES, dtype=np.float64)

    for i in range(len(ds)):
        rgb, K, gt_occ, gt_sem, _ = ds[i]
        rgb = rgb.unsqueeze(0).to(device).half()
        K = K.unsqueeze(0).to(device).half()

        gt_occ = gt_occ.unsqueeze(0).to(device)  # [1, 1, gx, gy, gz]
        gt_sem = gt_sem.to(device)  # [gx, gy, gz]

        with torch.no_grad():
            pred_grid = model(rgb, K)
            pred_act = pred_grid[:, :4].sigmoid()
            pred_occ = (pred_act[:, 0:1] > args.threshold).float() * fmask4
            gt_bin = (gt_occ > 0.5).float() * fmask4

            total_tp += (pred_occ * gt_bin).sum().item()
            total_fp += (pred_occ * (1 - gt_bin)).sum().item()
            total_fn += ((1 - pred_occ) * gt_bin).sum().item()

            pred_sem = pred_grid[:, 4:].argmax(dim=1)[0]  # [gx, gy, gz]

            occ_mask = gt_bin[0, 0] > 0.5
            if occ_mask.any():
                total_sem_correct += (
                    (pred_sem[occ_mask] == gt_sem[occ_mask]).sum().item()
                )
                total_sem_total += occ_mask.sum().item()

            pred_occ_mask = pred_occ[0, 0] > 0.5
            effective_pred = torch.where(
                pred_occ_mask,
                pred_sem,
                torch.zeros_like(pred_sem),
            )

            for c in range(1, N_SEM_CLASSES):
                pred_c = effective_pred == c
                gt_c = gt_sem == c
                class_tp[c] += (pred_c & gt_c).sum().item()
                class_fp[c] += (pred_c & ~gt_c).sum().item()
                class_fn[c] += (~pred_c & gt_c).sum().item()

        if (i + 1) % 100 == 0:
            running_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
            running_sem = total_sem_correct / (total_sem_total + 1e-6)
            print(
                f"  [{i+1}/{len(ds)}]  IoU={running_iou:.4f}  SemAcc={running_sem:.4f}"
            )

    iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"\n{'='*60}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Frames: {len(ds)}, Threshold: {args.threshold}")
    print(f"Occ:  IoU={iou:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

    sem_acc = total_sem_correct / (total_sem_total + 1e-6)
    print(f"Sem:  Accuracy={sem_acc:.4f} ({total_sem_correct}/{total_sem_total})")
    print(f"\nPer-class IoU:")
    valid_ious = []

    for c in range(1, N_SEM_CLASSES):
        c_iou = class_tp[c] / (class_tp[c] + class_fp[c] + class_fn[c] + 1e-6)
        if class_tp[c] + class_fn[c] > 0:
            valid_ious.append(c_iou)
            print(
                f"  {CLASS_NAMES[c]:15s}  IoU={c_iou:.4f}  (tp={int(class_tp[c])}, fp={int(class_fp[c])}, fn={int(class_fn[c])})"
            )
        else:
            print(f"  {CLASS_NAMES[c]:15s}  (no GT)")
    if valid_ious:
        print(f"\nmIoU (classes with GT): {np.mean(valid_ious):.4f}")


if __name__ == "__main__":
    main()
