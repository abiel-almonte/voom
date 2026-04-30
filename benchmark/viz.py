import sys
import time

import argparse
import atexit
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

import torch

from voom import VOOMv2, SemanticKITTIDataset, load_state_dict
from voom.ops import ray_marching
import config


def voxel_grid_to_mesh(alpha_3d, mpv, offset, colors, sem_3d, threshold=0.1):
    occupied = (alpha_3d > threshold).nonzero(as_tuple=False).cpu().numpy()
    if len(occupied) == 0:
        return o3d.geometry.TriangleMesh()

    offset_np = offset.cpu().numpy() if hasattr(offset, "cpu") else np.array(offset)
    colors_f = np.array(colors, dtype=np.float64) / 255.0
    gy = alpha_3d.shape[1]
    brightness = np.linspace(3.0, 0.4, gy)

    meshes = o3d.geometry.TriangleMesh()
    for i, j, k in occupied:
        cube = o3d.geometry.TriangleMesh.create_box(width=mpv, height=mpv, depth=mpv)
        pos = np.array([i, j, k], dtype=np.float64) * mpv - offset_np
        cube.translate(pos)

        cls = int(sem_3d[i, j, k])
        c = (colors_f[cls] * brightness[j]).clip(0, 1).tolist()
        cube.paint_uniform_color(c)
        meshes += cube

    meshes.compute_vertex_normals()
    return meshes


def render_mesh_topdown(mesh, center, width=576, height=400):
    vis = o3d.visualization.rendering.OffscreenRenderer(width, height)
    vis.scene.set_background([0, 0, 0, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    vis.scene.add_geometry("mesh", mesh, mat)

    eye = center + np.array([0, -10, -12])
    up = np.array([0, 0, 1])
    vis.scene.camera.look_at(center, eye, up)

    img = vis.render_to_image()
    return np.asarray(img)


def compose_layout(rgb_np, render_np, sem_3d_np, H, W):
    rgb_img = Image.fromarray(rgb_np).resize((W, H), Image.LANCZOS)
    render_img = Image.fromarray(render_np).resize((W, H), Image.LANCZOS)
    sem_img = Image.fromarray(sem_3d_np)

    gap = 6
    left_h = H + gap + H
    sem_w = sem_img.size[0]

    pad = 50
    total_w = pad + W + pad + sem_w
    total_h = pad + left_h + pad

    total_w = (total_w // 2) * 2
    total_h = (total_h // 2) * 2

    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    canvas.paste(rgb_img, (pad, pad))
    canvas.paste(render_img, (pad, pad + H + gap))
    canvas.paste(sem_img, (pad + W + pad, pad))

    return canvas


def render_frame(rgb_norm, pred_grid, K, threshold=0.26):
    device = pred_grid.device
    H, W = rgb_norm.shape[-2:]
    mpv = config.mpv
    offset = torch.tensor(config.offset_m, device=device)
    grid_center = np.array(config.grid_center_m)

    pred_occ_rgb = pred_grid[:, :4].sigmoid()
    pred_alpha = pred_occ_rgb[0, 0].cpu()

    sem_logits = pred_grid[:, 4:]
    pred_sem = sem_logits.argmax(dim=1)[0].cpu().numpy()
    occ_mask = pred_alpha.numpy() > threshold
    pred_sem_masked = pred_sem * occ_mask

    # Mesh render
    sem_bev_w, sem_bev_h = int(W * 1.5), int(H * 1.5)
    sem_mesh = voxel_grid_to_mesh(
        pred_alpha, mpv, offset, config.colors, pred_sem_masked, threshold=threshold
    )
    sem_3d_np = render_mesh_topdown(
        sem_mesh, grid_center, width=sem_bev_w, height=sem_bev_h
    )

    # RGB camera-view render
    rendered_pred = ray_marching(
        grid=pred_occ_rgb.float(),
        K=K.float(),
        size=(H, W),
        mpv=mpv,
        step_size=mpv,
        offset_m=config.offset_m,
    )
    render_np = (
        rendered_pred[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)

    mean = torch.tensor(config.common["preprocess"]["mean"]).cuda().view(3, 1, 1)
    std = torch.tensor(config.common["preprocess"]["std"]).cuda().view(3, 1, 1)
    rgb_denorm = (rgb_norm[0] * std + mean).clamp(0, 1)
    rgb_np = (rgb_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return compose_layout(rgb_np, render_np, sem_3d_np, H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.26)
    p.add_argument("--max_frames", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SemanticKITTIDataset(**config.val_dataset_config)

    sd = load_state_dict(args.ckpt, device=device)

    model_cfg = dict(config.model_config)
    model = VOOMv2(**model_cfg).to(device).half().eval()
    model.load_state_dict(sd, strict=False)

    n_frames = len(ds) if args.max_frames is None else min(args.max_frames, len(ds))

    if args.out is None:
        ckpt_dir = Path(args.ckpt).parent
        date = time.strftime("%m-%d-%Y_%H-%M-%S")
        args.out = str(ckpt_dir / f"viz_{date}.mp4")

    rgb, K, _, _, _ = ds[0]
    rgb_batch = rgb.unsqueeze(0).to(device).half()
    K_batch = K.unsqueeze(0).to(device).half()
    with torch.no_grad():
        pred_grid = model(rgb_batch, K_batch)

    test_img = render_frame(rgb_batch, pred_grid, K_batch, threshold=args.threshold)
    w, h = test_img.size

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            "10",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-loglevel",
            "error",
            args.out,
        ],
        stdin=subprocess.PIPE,
    )

    def close_ffmpeg():
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        try:
            ffmpeg_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()

    atexit.register(close_ffmpeg)

    frame_bgr = cv2.cvtColor(np.array(test_img), cv2.COLOR_RGB2BGR)
    ffmpeg_proc.stdin.write(frame_bgr.tobytes())

    for i in range(1, n_frames):
        rgb, K, _, _, _ = ds[i]
        rgb_batch = rgb.unsqueeze(0).to(device).half()
        K_batch = K.unsqueeze(0).to(device).half()

        with torch.no_grad():
            pred_grid = model(rgb_batch, K_batch)

        img = render_frame(rgb_batch, pred_grid, K_batch, threshold=args.threshold)
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ffmpeg_proc.stdin.write(frame_bgr.tobytes())

        if (i + 1) % 50 == 0 or i == n_frames - 1:
            print(f"  [{i + 1}/{n_frames}]")

    close_ffmpeg()
    print(f"Video saved to {args.out}")


if __name__ == "__main__":
    main()
