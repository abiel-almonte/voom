import argparse
import torch
from safetensors.torch import save_file
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    
    args = parser.parse_args()
    ckpt_path = Path(args.ckpt)

    ckpt = torch.load(ckpt_path, weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    save_file(state, "voom.safetensors")

if __name__ == "__main__":
    main()
