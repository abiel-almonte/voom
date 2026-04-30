import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DA_ROOT = _REPO_ROOT / "third_party" / "Depth-Anything-V2"
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
}


def load_depth_model(encoder, max_depth=80.0, device="cuda"):
    sys.path.insert(0, str(_DA_ROOT / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    weights = (
        _DA_ROOT
        / "metric_depth"
        / "checkpoints"
        / f"depth_anything_v2_metric_vkitti_{encoder}.pth"
    )
    if not weights.exists():
        raise FileNotFoundError(
            f"weights not found: {weights}\nsee setup instructions in script docstring"
        )

    model = DepthAnythingV2(**_CONFIGS[encoder], max_depth=max_depth)
    model.load_state_dict(
        torch.load(str(weights), map_location="cpu", weights_only=True)
    )
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def infer_depth(model, img_rgb, device):
    img_t = TF.to_tensor(img_rgb).unsqueeze(0).to(device)
    x = (img_t - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)

    h, w = x.shape[-2:]
    h14, w14 = math.ceil(h / 14) * 14, math.ceil(w / 14) * 14
    x = F.interpolate(x, size=(h14, w14), mode="bilinear", align_corners=True)

    with torch.inference_mode():
        depth = model(x)

    depth = F.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True
    )
    return depth[0, 0].clamp(min=0.01).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skitti_root", type=str, required=True)
    p.add_argument("--encoder", type=str, default="vits", choices=list(_CONFIGS))
    p.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_depth_model(args.encoder, device=device)
    print(f"Loaded DA v2 metric depth ({args.encoder}, vkitti weights)")

    root = Path(args.skitti_root)
    for seq in args.sequences:
        img_dir = root / "sequences" / seq / "image_2"
        if not img_dir.exists():
            print(f"skip {seq}: {img_dir} not found")
            continue

        out_dir = root / "sequences" / seq / "depth"
        out_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(img_dir.glob("*.png"))
        print(f"seq {seq}: {len(images)} images")

        for i, img_path in enumerate(images):
            out_path = out_dir / f"{img_path.stem}.png"
            if out_path.exists():
                continue

            img = Image.open(img_path).convert("RGB")
            depth_m = infer_depth(model, img, device)
            depth_cm = (depth_m * 100.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(out_path), depth_cm)

            if (i + 1) % 100 == 0:
                print(f"  {seq}: {i + 1}/{len(images)}")

        print(f"  {seq}: done")


if __name__ == "__main__":
    main()
