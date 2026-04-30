import logging
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from voom import VOOMv2, SemanticKITTIDataset, ray_marching, render_gt
import config

log = logging.getLogger("voom.train")


def setup_logging(ckpt_dir):
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    file_h = logging.FileHandler(Path(ckpt_dir) / "train.log", mode="w")
    stream_h = logging.StreamHandler()

    for h in (file_h, stream_h):
        h.setFormatter(fmt)

    log.handlers = [file_h, stream_h]
    log.setLevel(logging.INFO)


def dump_config(ckpt_dir):
    with open(Path(ckpt_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(config.config, f, sort_keys=False)


def loss_occ(pred_grid, gt_occ):
    return F.binary_cross_entropy_with_logits(pred_grid[:, 0:1], gt_occ)


def loss_sem(pred_grid, gt_sem, gt_occ):
    sem_logits = pred_grid[:, 4:]  # [b, 20, gx, gy, gz]
    occ_mask = gt_occ[:, 0] > 0.5

    if occ_mask.any():
        return F.cross_entropy(
            sem_logits.permute(0, 2, 3, 4, 1)[occ_mask],
            gt_sem[occ_mask],
            ignore_index=0,
        )
    return torch.tensor(0.0, device="cuda")


def loss_photo(pred_grid, rgb, da_depth, K):
    rgb_pred = pred_grid[:, :4].sigmoid()
    H, W = rgb.shape[-2:]

    rendered_pred = ray_marching(
        grid=rgb_pred,
        K=K,
        size=(H, W),
        mpv=config.mpv,
        step_size=config.mpv,
        offset_m=config.offset_m,
    )

    rendered_gt = render_gt(
        rgb_norm=rgb,
        depth=da_depth,
        K=K,
        grid_dim=tuple(config.grid_dim),
        mpv=config.mpv,
        depth_bins=config.model["depth_bins"],
        offset_m=config.offset_m,
    )

    return F.l1_loss(rendered_pred, rendered_gt)


def loss_depth(depth_logits, da_depth):
    n_bins = config.model["depth_bins"]
    gt_depth_m = da_depth[:, 0]
    valid = (gt_depth_m >= config.mpv) & (gt_depth_m <= config.max_depth_m)

    gt_bins = (
        ((gt_depth_m - config.mpv) / (config.max_depth_m - config.mpv) * (n_bins - 1))
        .long()
        .clamp(0, n_bins - 1)
    )
    gt_bins[~valid] = -1

    gt_resized = (
        F.interpolate(
            gt_bins.unsqueeze(1).float(),
            size=depth_logits.shape[-2:],
            mode="nearest",
        )
        .long()
        .squeeze(1)
    )

    return F.cross_entropy(depth_logits, gt_resized, ignore_index=-1)


def build_optimizer(model, lrs):
    vit_params = [p for n, p in model.named_parameters() if "backbone" in n]
    dpt_params = [p for n, p in model.named_parameters() if "dpt_head" in n]
    lss_params = [
        p
        for n, p in model.named_parameters()
        if "backbone" not in n and "dpt_head" not in n
    ]

    return torch.optim.AdamW(
        [
            {"params": vit_params, "lr": lrs["vit"]},
            {"params": dpt_params, "lr": lrs["dpt"]},
            {"params": lss_params, "lr": lrs["lss"]},
        ],
        weight_decay=1e-5,
    )


def save_ckpt(model, optimizer, step, ckpt_dir):
    weights_dir = Path(ckpt_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        weights_dir / f"step_{step}.pt",
    )


def load_ckpt(model, optimizer, init_path):
    if not init_path or not Path(init_path).exists():
        return 0
    ckpt = torch.load(init_path, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


def log_losses(metrics, step, n_batches):
    metrics_str = "  ".join(f"{k}={v / step:.4f}" for k, v in metrics.items())
    log.info(f"  [{step}/{n_batches}]  {metrics_str}")


def main():
    device = "cuda"
    ckpt_dir = config.train["ckpt_dir"]
    setup_logging(ckpt_dir)
    dump_config(ckpt_dir)

    dataset = SemanticKITTIDataset(**config.dataset_config)
    model = VOOMv2(**config.model_config).to(device)
    log.info(f"dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=config.train["batch_size"],
        num_workers=config.train["num_workers"],
        shuffle=True,
    )

    optimizer = build_optimizer(model, config.train["learning_rates"])
    step = load_ckpt(model, optimizer, config.train.get("init_path"))
    if step:
        log.info(f"resumed from step {step}")

    weights = config.train["loss_weights"]
    for epoch in range(config.train["epochs"]):
        epoch_metrics = defaultdict(float)
        epoch_steps = 0

        for batch in loader:
            rgb, K, gt_occ, gt_sem, da_depth = [x.to(device) for x in batch]

            pred_grid, depth_logits = model(rgb, K)
            losses = {
                "occ": loss_occ(pred_grid, gt_occ),
                "sem": loss_sem(pred_grid, gt_sem, gt_occ),
                "photo": loss_photo(pred_grid, rgb, da_depth, K),
                "depth": loss_depth(depth_logits, da_depth),
            }
            total = sum(weights.get(k, 0.0) * v for k, v in losses.items())

            total.backward()
            optimizer.step()
            optimizer.zero_grad()

            for k, v in losses.items():
                epoch_metrics[k] += v.item()

            step += 1
            epoch_steps += 1

            if step % 10 == 0:
                log_losses(epoch_metrics, epoch_steps, len(loader))

            if step % config.train["save_every"] == 0:
                save_ckpt(model, optimizer, step, config.train["ckpt_dir"])

        log.info(f"epoch {epoch}: {dict(epoch_metrics)}")


if __name__ == "__main__":
    main()
