import subprocess

import torch

import config
from voom import VOOMv2
from .trt_model import PreLift, PostLift


def export():
    config.engine_dir.mkdir(parents=True, exist_ok=True)

    voom = VOOMv2(**config.model_config)
    voom.load_state_dict(torch.load(config.weights_path, weights_only=True))
    voom = voom.cuda().eval().half()

    H, W = config.train["height"], config.train["width"]
    dummy_rgb = torch.randn(1, 3, H, W, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        layers = voom.backbone.get_intermediate_layers(
            dummy_rgb,
            n=voom.sampled_layers,
            reshape=True,
            return_class_token=True,
            norm=False,
        )
    feats = [l[0].contiguous() for l in layers]
    clses = [l[1].contiguous() for l in layers]

    gx, gy, gz = voom.grid_dim
    ch = voom.rproj.weight.shape[1]

    pre = PreLift(voom.dpt_head, voom.cproj, voom.dproj).eval()
    post = PostLift(voom.rproj, voom.rblok, voom.rout).eval()

    pre_args = (
        feats[0],
        clses[0],
        feats[1],
        clses[1],
        feats[2],
        clses[2],
        feats[3],
        clses[3],
        dummy_rgb,
    )
    post_arg = torch.randn(1, ch, gx, gy, gz, dtype=torch.float16, device="cuda")

    torch.onnx.export(
        pre,
        pre_args,
        str(config.prelift_onnx),
        input_names=["f0", "c0", "f1", "c1", "f2", "c2", "f3", "c3", "rgb"],
        output_names=["context", "depth"],
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        post,
        post_arg,
        str(config.postlift_onnx),
        input_names=["grid_in"],
        output_names=["grid_out"],
        opset_version=17,
        dynamo=False,
    )

    for onnx_path, trt_path in [
        (config.prelift_onnx, config.prelift_trt),
        (config.postlift_onnx, config.postlift_trt),
    ]:
        subprocess.run(
            [
                config.trtexec,
                f"--onnx={onnx_path}",
                f"--saveEngine={trt_path}",
                "--fp16",
            ],
            check=True,
        )


if __name__ == "__main__":
    export()
