import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from voom import VOOMv2, load_state_dict
import config


def build_pt(ckpt):
    model = VOOMv2(**config.model_config).cuda().half().eval()
    model.load_state_dict(load_state_dict(ckpt, device="cuda"), strict=False)
    return model


def build_trt():
    from deploy.trt_model import VOOMv2TRT

    return VOOMv2TRT(**config.trt_config).cuda().eval()


def run(model, inp, K, warmup, secs):
    lats = []
    with torch.no_grad():
        for _ in range(warmup):
            model(inp, K)
        torch.cuda.synchronize()

        t_end = time.perf_counter() + secs
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            model(inp, K)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)

    lats.sort()
    n = len(lats)
    return {
        "n": n,
        "mean": sum(lats) / n,
        "p50": lats[n // 2],
        "p99": lats[min(n - 1, int(n * 0.99))],
        "max": lats[-1],
        "fps": 1000 / (sum(lats) / n),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="release/voom.safetensors")
    p.add_argument("--mode", choices=["pt", "trt", "all"], default="all")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--secs", type=float, default=60.0)
    args = p.parse_args()

    H, W = config.train["height"], config.train["width"]
    device_name = torch.cuda.get_device_name(0)
    print(f"{device_name}  {H}x{W}  secs={args.secs}\n")

    rgb = torch.randn(1, 3, H, W, device="cuda").half()
    K = torch.tensor(
        [[[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]], device="cuda"
    ).half()

    modes = ["pt", "trt"] if args.mode == "all" else [args.mode]

    results = {}
    for mode in modes:
        try:
            model = build_trt() if mode == "trt" else build_pt(args.ckpt)
        except Exception as e:
            print(f"{mode}: skipped ({e})")
            continue

        r = run(model, rgb, K, args.warmup, args.secs)
        results[mode] = r

        del model
        torch.cuda.empty_cache()

    label_w = max(len(m) for m in results) if results else 4
    print(f"\n{device_name}")
    print(f"{'mode':<{label_w}}   {'mean':>7}   {'p50':>7}   {'p99':>7}   {'fps':>6}")
    for mode, r in results.items():
        print(
            f"{mode:<{label_w}}   "
            f"{r['mean']:>6.2f}ms  {r['p50']:>6.2f}ms  {r['p99']:>6.2f}ms  "
            f"{r['fps']:>5.1f}"
        )


if __name__ == "__main__":
    main()
