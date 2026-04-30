import torch
from safetensors.torch import save_file

ckpt = torch.load("voom.pt", weights_only=True)
state = ckpt["model"] if "model" in ckpt else ckpt
save_file(state, "voom.safetensors")
