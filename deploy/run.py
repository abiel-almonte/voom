import os

os.environ["MODERNGL_WINDOW"] = "glfw"

import atexit

import torch
import moderngl_window as mglw

import config
from .camera import Camera
from .preprocess import preprocess
from .trt_model import VOOMv2TRT
from .viewer import VoomViewer


def _gen(camera, voom):
    while True:
        result = camera.read()
        if result is None:
            continue

        pil, K_cam = result
        rgb, K = preprocess(pil, K_cam)

        with torch.inference_mode():
            grid = voom(rgb.half(), K.half())

        occ = grid[0, 0].sigmoid() > 0.26
        sem = grid[0, 4:].argmax(dim=0)
        yield occ, sem, pil


def _warmup(voom):
    H, W = config.train["height"], config.train["width"]
    dummy_rgb = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float16)
    dummy_K = torch.tensor(
        [[[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]],
        device="cuda",
        dtype=torch.float16,
    )
    with torch.no_grad():
        for _ in range(3):
            voom(dummy_rgb, dummy_K)

    voom.offsets = None
    voom.pixs = None


def main():
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XAUTHORITY", "/run/user/1000/gdm/Xauthority")

    print("[voom] opening camera")
    camera = Camera(**config.camera_config)
    atexit.register(camera.release)

    print("[voom] loading model")
    voom = VOOMv2TRT(**config.trt_config)

    print("[voom] warming up")
    _warmup(voom)
    print("[voom] ready")

    VoomViewer.gen = _gen(camera, voom)
    mglw.run_window_config(VoomViewer)


if __name__ == "__main__":
    main()
