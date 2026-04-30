from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file


def load_state_dict(path, device="cpu"):
    if Path(path).suffix == ".safetensors":
        return load_file(str(path), device=device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt["model"] if "model" in ckpt else ckpt


def _make_norm(kind, ch):
    if kind == "group":
        groups = min(8, ch)
        while ch % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, ch)
    elif kind == "instance":
        return nn.InstanceNorm2d(ch)
    return nn.Identity()


def _make_act(kind):
    if kind == "relu":
        return nn.ReLU(inplace=True)
    elif kind == "tanh":
        return nn.Tanh()
    elif kind == "sigmoid":
        return nn.Sigmoid()
    return nn.Identity()


def size_to_model(size: str) -> str:

    mapping = {
        "s": "dinov2_vits14",
        "small": "dinov2_vits14",
        "b": "dinov2_vitb14",
        "base": "dinov2_vitb14",
        "l": "dinov2_vitl14",
        "large": "dinov2_vitl14",
        "g": "dinov2_vitg14",
        "giant": "dinov2_vitg14",
    }

    size = size.lower()
    if size in mapping:
        return mapping[size]

    raise ValueError(f"Unknown ViT backbone size: '{size}'")


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, final_act="relu", norm="group"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.final_act = _make_act(final_act)

        self.norm1 = _make_norm(norm, out_ch)
        self.norm2 = _make_norm(norm, out_ch)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                _make_norm(norm, out_ch),
            )

    def forward(self, x):
        res = self.downsample(x) if self.downsample is not None else x
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        return self.final_act(y + res)
