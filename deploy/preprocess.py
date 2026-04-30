import numpy as np
import torch
from torchvision.transforms import functional as TF

import config

_KITTI_W = config.train["data"]["width"]
_KITTI_H = config.train["data"]["height"]
_KITTI_F = 718.856
_KITTI_HFOV = 2 * np.degrees(np.arctan(_KITTI_W / (2 * _KITTI_F)))  # 81.6°
_KITTI_VFOV = 2 * np.degrees(np.arctan(_KITTI_H / (2 * _KITTI_F)))  # 29.2°

_MEAN = torch.tensor(config.common["preprocess"]["mean"]).view(3, 1, 1)
_STD = torch.tensor(config.common["preprocess"]["std"]).view(3, 1, 1)

_INPUT_H = config.train["height"]
_INPUT_W = config.train["width"]


def preprocess(rgb_pil, K_cam):
    orig_w, orig_h = rgb_pil.size
    K = np.asarray(K_cam, dtype=np.float64).copy()
    src_f = float(K[0, 0])

    crop_w = min(int(2 * src_f * np.tan(np.radians(_KITTI_HFOV / 2))), orig_w)
    crop_h = min(int(2 * src_f * np.tan(np.radians(_KITTI_VFOV / 2))), orig_h)

    left = (orig_w - crop_w) // 2
    top = (orig_h - crop_h) // 2
    rgb_pil = rgb_pil.crop((left, top, left + crop_w, top + crop_h))
    K[0, 2] -= left
    K[1, 2] -= top

    rgb_pil = TF.resize(rgb_pil, [_INPUT_H, _INPUT_W])
    sx = _INPUT_W / crop_w
    sy = _INPUT_H / crop_h
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    rgb_tensor = TF.to_tensor(rgb_pil)
    rgb_norm = ((rgb_tensor - _MEAN) / _STD).unsqueeze(0).to("cuda")

    K_t = torch.tensor(K, dtype=torch.float32).unsqueeze(0).to("cuda")
    return rgb_norm, K_t
