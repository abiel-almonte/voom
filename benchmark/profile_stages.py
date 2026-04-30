import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from deploy.trt_model import VOOMv2TRT
import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    H, W = config.train["height"], config.train["width"]
    model = VOOMv2TRT(**config.trt_config).cuda().eval()

    rgb = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float16)
    K = torch.tensor(
        [[[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]],
        device="cuda",
        dtype=torch.float16,
    )

    ph, pw = math.ceil(H / 14) * 14, math.ceil(W / 14) * 14
    inp = F.interpolate(
        rgb, size=(ph, pw), align_corners=True, mode="bilinear"
    ).contiguous()

    with torch.no_grad():
        for _ in range(args.warmup):
            model(rgb, K)
    torch.cuda.synchronize()

    stages = ["backbone", "prelift", "lift_splat", "postlift", "total"]
    times = {s: 0.0 for s in stages}
    events = {
        s: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for s in stages
    }

    with torch.no_grad():
        for _ in range(args.iters):
            with torch.cuda.stream(model.stream):
                events["total"][0].record(model.stream)

                events["backbone"][0].record(model.stream)
                layers = model.backbone.get_intermediate_layers(
                    inp,
                    n=model.sampled_layers,
                    reshape=True,
                    return_class_token=True,
                    norm=False,
                )
                events["backbone"][1].record(model.stream)

                pre_args = tuple(t.contiguous() for layer in layers for t in layer) + (
                    inp,
                )
                events["prelift"][0].record(model.stream)
                context, depth = model.prelift(*pre_args)
                events["prelift"][1].record(model.stream)

                events["lift_splat"][0].record(model.stream)
                grid = model._lift(rgb, context, depth, K)
                events["lift_splat"][1].record(model.stream)

                events["postlift"][0].record(model.stream)
                _ = model.postlift(grid)
                events["postlift"][1].record(model.stream)

                events["total"][1].record(model.stream)

            torch.cuda.synchronize()
            for s in stages:
                times[s] += events[s][0].elapsed_time(events[s][1])

    total_ms = times["total"] / args.iters
    print(f"\n{H}x{W} fp16  iters={args.iters}")
    print(f"{'stage':<12}  {'ms':>7}  {'%':>6}")
    for s in ["backbone", "prelift", "lift_splat", "postlift"]:
        ms = times[s] / args.iters
        print(f"{s:<12}  {ms:>6.2f}  {100 * ms / total_ms:>5.1f}%")
    print(f"{'total':<12}  {total_ms:>6.2f}  100.0%")
    print(f"\nFPS: {1000 / total_ms:.1f}")


if __name__ == "__main__":
    main()
