from collections import defaultdict

import torch
import torch.nn.functional as F

import config

_MEAN = torch.tensor(config.common["preprocess"]["mean"]).view(3, 1, 1)
_STD = torch.tensor(config.common["preprocess"]["std"]).view(3, 1, 1)


def compute_rays(K, size):
    h, w = size
    dev = K.device

    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    K_inv = torch.zeros_like(K)
    K_inv[:, 0, 0] = 1.0 / fx
    K_inv[:, 1, 1] = 1.0 / fy
    K_inv[:, 0, 2] = -cx / fx
    K_inv[:, 1, 2] = -cy / fy
    K_inv[:, 2, 2] = 1.0
    u, v = torch.meshgrid(
        torch.arange(0, w, device=dev), torch.arange(0, h, device=dev), indexing="xy"
    )

    pts_h = torch.stack([u, v, torch.ones_like(u)], dim=-1).to(K.dtype)
    rays = torch.einsum("hwi,...ji->...hwj", pts_h, K_inv)  # [b, h, w, 3]
    return rays


def pts_to_voxels(pts_3d, grid_dim, mpv, offset_m):
    gx, gy, gz = grid_dim
    offset = torch.tensor(offset_m, device=pts_3d.device)

    pts_vox = ((pts_3d + offset) / mpv).long()

    xi, yi, zi = pts_vox[..., 0], pts_vox[..., 1], pts_vox[..., 2]
    valid = (xi >= 0) & (xi < gx) & (yi >= 0) & (yi < gy) & (zi >= 0) & (zi < gz)
    flat_idx = (xi * gy * gz + yi * gz + zi).clamp(0, gx * gy * gz - 1)
    return flat_idx, valid


def ray_marching(grid, K, size, mpv, step_size=0.5, extrinsic=None, offset_m=None):
    h, w = size
    dev = grid.device

    rays = compute_rays(K, size)

    if extrinsic is not None:
        R_grid, t_grid, R_cam, t_cam = extrinsic
        R_rel = torch.einsum("...ij,...kj->...ik", R_grid, R_cam)
        t_rel = t_grid - torch.einsum("...ij,...j->...i", R_rel, t_cam)

    b, c, gx, gy, gz = grid.shape
    offset = torch.tensor(offset_m, device=dev)

    n_steps = int((gz * mpv) // step_size)
    rendered = torch.zeros(b, c - 1, h, w, device=dev)
    alpha_sum = torch.zeros(b, 1, h, w, device=dev)

    for step in range(1, n_steps):
        distance = step * step_size
        pts = rays * distance

        if extrinsic is not None:
            pts = (
                torch.einsum("...ij,...hwj->...hwi", R_rel, pts)
                + t_rel[:, None, None, :]
            )

        voxel_coords = (pts + offset) / mpv

        norm = torch.stack(
            [
                voxel_coords[..., 2] / (gz - 1) * 2 - 1,
                voxel_coords[..., 1] / (gy - 1) * 2 - 1,
                voxel_coords[..., 0] / (gx - 1) * 2 - 1,
            ],
            dim=-1,
        )

        sample_grid = norm.unsqueeze(1)
        sampled = F.grid_sample(
            grid, sample_grid, padding_mode="zeros", align_corners=True, mode="nearest"
        ).squeeze(2)

        alpha = sampled[:, 0:1]
        feats = sampled[:, 1:]

        weight = (1 - alpha_sum) * alpha
        rendered = rendered + feats * weight
        alpha_sum = alpha_sum + weight

    return rendered


def render_gt(rgb_norm, depth, K, grid_dim, mpv, depth_bins, offset_m):
    H, W = rgb_norm.shape[-2:]
    dev = rgb_norm.device

    rgb_gt = (rgb_norm * _STD.to(dev) + _MEAN.to(dev)).clamp(0, 1)
    gt_ones = torch.ones(rgb_gt.shape[0], 1, H, W, device=dev)
    gt_feats = torch.cat([gt_ones, rgb_gt], dim=1)  # [b, 4, H, W]

    gz = grid_dim[2]
    max_depth = gz * mpv
    gt_depth_m = depth[:, 0]
    bin_idx = (
        ((gt_depth_m - mpv) / (max_depth - mpv) * (depth_bins - 1))
        .long()
        .clamp(0, depth_bins - 1)
    )
    depth_dist = torch.zeros(depth.shape[0], depth_bins, H, W, device=dev)
    depth_dist.scatter_(1, bin_idx.unsqueeze(1), 1.0)

    gt_raw = lift_splat(
        context=gt_feats,
        depth_dist=depth_dist,
        K=K,
        size=(H, W),
        grid_dim=grid_dim,
        mpv=mpv,
        offset_m=offset_m,
    )

    gt_count = gt_raw[:, 0:1].clamp(min=1.0)
    gt_rgb = gt_raw[:, 1:4] / gt_count
    gt_occ = (gt_raw[:, 0:1] > 0.5).float()
    gt_grid = torch.cat([gt_occ * 0.9, gt_rgb * gt_occ], dim=1)

    rendered_gt = ray_marching(
        grid=gt_grid,
        K=K,
        size=(H, W),
        mpv=mpv,
        step_size=mpv,
        offset_m=offset_m,
    )
    return rendered_gt


def lift_splat(context, depth_dist, K, size, grid_dim, mpv, offset_m):
    b, c, h, w = context.shape
    d = depth_dist.shape[1]
    gx, gy, gz = grid_dim
    dev = context.device

    orig_h, orig_w = size
    K_scaled = K.clone()
    K_scaled[:, 0] *= w / orig_w
    K_scaled[:, 1] *= h / orig_h

    rays = compute_rays(K_scaled, (h, w))
    depth_vals = torch.linspace(mpv, gz * mpv, d, device=dev)

    grid = torch.zeros(b, c, gx * gy * gz, device=dev, dtype=depth_dist.dtype)

    for di in range(d):
        pts_d = rays * depth_vals[di]
        flat_idx, valid = pts_to_voxels(pts_d, grid_dim, mpv, offset_m=offset_m)

        weight = depth_dist[:, di]
        weighted = context * weight.unsqueeze(1)
        weighted = weighted * valid.unsqueeze(1).to(context.dtype)

        idx = flat_idx.reshape(b, 1, -1).expand(b, c, -1)
        src = weighted.reshape(b, c, -1)
        grid.scatter_add_(2, idx, src)

    return grid.reshape(b, c, gx, gy, gz)


def precompute_vox2pix(K, h, w, orig_h, orig_w, grid_dim, mpv, depth_bins, offset_m):
    gx, gy, gz = grid_dim
    dev = K.device

    K_scaled = K.clone()
    K_scaled[:, 0] *= w / orig_w
    K_scaled[:, 1] *= h / orig_h

    rays = compute_rays(K_scaled, (h, w))
    depth_vals = torch.linspace(mpv, gz * mpv, depth_bins, device=dev)

    pts = rays.unsqueeze(1) * depth_vals.view(1, depth_bins, 1, 1, 1)
    offset = torch.tensor(offset_m, device=dev)
    pts_vox = ((pts + offset) / mpv).long()

    vox_x = pts_vox[0, ..., 0]
    vox_y = pts_vox[0, ..., 1]
    vox_z = pts_vox[0, ..., 2]

    valid = (
        (vox_x >= 0)
        & (vox_x < gx)
        & (vox_y >= 0)
        & (vox_y < gy)
        & (vox_z >= 0)
        & (vox_z < gz)
    )  # [D, H, W]

    vox_flat = vox_x * gy * gz + vox_y * gz + vox_z  # [D, H, W]

    di_idx = (
        torch.arange(depth_bins, device=dev)
        .view(depth_bins, 1, 1)
        .expand(depth_bins, h, w)
    )
    py_idx = torch.arange(h, device=dev).view(1, h, 1).expand(depth_bins, h, w)
    px_idx = torch.arange(w, device=dev).view(1, 1, w).expand(depth_bins, h, w)
    packed = (di_idx.long() << 32) | (py_idx.long() << 16) | px_idx.long()

    vox_flat = vox_flat[valid]
    packed = packed[valid]

    sort_idx = torch.argsort(vox_flat)
    vox_sorted = vox_flat[sort_idx]
    pixs_tensor = packed[sort_idx].contiguous()

    n_voxels = gx * gy * gz
    counts = torch.bincount(vox_sorted, minlength=n_voxels)
    offsets_tensor = torch.zeros(n_voxels + 1, device=dev, dtype=torch.long)
    offsets_tensor[1:] = torch.cumsum(counts, dim=0)
    return offsets_tensor, pixs_tensor


_C = None


def _get_cuda_ext():
    global _C
    if _C is None:
        from torch.utils.cpp_extension import load

        _C = load(name="ops_C", sources=["voom/ops.cu"])
    return _C


def lift_splat_gather(context, depth_dist, offsets, pixs, grid_dim):
    gx, gy, gz = grid_dim
    return _get_cuda_ext().lift_splat_gather(
        context, depth_dist, offsets, pixs, gx, gy, gz
    )


def lift_splat_gather_fp16_nhwc_ch64(
    context, depth_dist, offsets, pixs, grid_dim
) -> torch.Tensor:
    gx, gy, gz = grid_dim
    return _get_cuda_ext().lift_splat_gather_fp16_nhwc_ch64(
        context, depth_dist, offsets, pixs, gx, gy, gz
    )
