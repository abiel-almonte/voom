import gc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from voom.ops import precompute_vox2pix, lift_splat_gather_fp16_nhwc_ch64
from voom.utils import size_to_model
from .trt_module import TRTModule


class PreLift(nn.Module):
    def __init__(self, dpt_head, cproj, dproj):
        super().__init__()
        self.dpt_head = dpt_head
        self.cproj = cproj
        self.dproj = dproj

    def _dpt_feat(self, layers):
        dh = self.dpt_head
        x = dh.reassemble_blocks(list(layers))
        x = [dh.convs[i](f) for i, f in enumerate(x)]
        out = dh.fusion_blocks[0](x[-1])
        for i in range(1, len(dh.fusion_blocks)):
            out = dh.fusion_blocks[i](out, x[-(i + 1)])
        return out

    def forward(self, f0, c0, f1, c1, f2, c2, f3, c3, rgb):
        feat = self._dpt_feat([(f0, c0), (f1, c1), (f2, c2), (f3, c3)])
        depth = F.softmax(self.dproj(feat), dim=1)
        rgb_resized = F.interpolate(
            rgb,
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        context = self.cproj(torch.cat([rgb_resized, feat], dim=1))
        return context, depth


class PostLift(nn.Module):
    def __init__(self, rproj, rblok, rout):
        super().__init__()
        self.rproj = rproj
        self.rblok = rblok
        self.rout = rout

    def forward(self, grid):
        grid = self.rproj(grid)
        grid = F.relu(grid + self.rblok(grid))
        return self.rout(grid)


class VOOMv2TRT(nn.Module):
    def __init__(
        self,
        backbone_size="s",
        sampled_layers=(2, 5, 8, 11),
        depth_bins=128,
        grid_dim=(128, 32, 128),
        mpv=0.2,
        offset_m=(0, 0, 0),
        weights_path=None,
        prelift_engine=None,
        postlift_engine=None,
    ):
        super().__init__()

        weights_path = weights_path or config.weights_path
        prelift_engine = prelift_engine or config.prelift_trt
        postlift_engine = postlift_engine or config.postlift_trt

        dinov2 = torch.hub.load(
            "facebookresearch/dinov2",
            f"{size_to_model(backbone_size)}_dd",
            weights="KITTI",
        )
        backbone = dinov2.backbone
        del dinov2
        gc.collect()

        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        bb_sd = {
            k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")
        }
        backbone.load_state_dict(bb_sd)
        del sd, bb_sd
        gc.collect()

        self.backbone = backbone.cuda().eval().half()
        self.sampled_layers = list(sampled_layers)
        torch.cuda.empty_cache()

        self.prelift = TRTModule(str(prelift_engine))
        self.postlift = TRTModule(str(postlift_engine))
        self.stream = torch.cuda.Stream()

        self.depth_bins = depth_bins
        self.grid_dim = grid_dim
        self.mpv = mpv
        self.offset_m = offset_m

        self.offsets = None
        self.pixs = None

    def _lift(self, rgb, context, depth, K):
        if self.offsets is None:
            h, w = context.shape[-2:]
            orig_h, orig_w = rgb.shape[-2:]
            self.offsets, self.pixs = precompute_vox2pix(
                K,
                h,
                w,
                orig_h,
                orig_w,
                self.grid_dim,
                self.mpv,
                self.depth_bins,
                self.offset_m,
            )

        context_nhwc = context.permute(0, 2, 3, 1).contiguous()
        return lift_splat_gather_fp16_nhwc_ch64(
            context_nhwc,
            depth,
            self.offsets,
            self.pixs,
            self.grid_dim,
        )

    def forward(self, rgb, K):
        with torch.cuda.stream(self.stream):
            b, c, h, w = rgb.shape
            ph = math.ceil(h / 14) * 14
            pw = math.ceil(w / 14) * 14
            inp = F.interpolate(rgb, size=(ph, pw), align_corners=True, mode="bilinear")

            with torch.no_grad():
                layers = self.backbone.get_intermediate_layers(
                    inp,
                    n=self.sampled_layers,
                    reshape=True,
                    return_class_token=True,
                    norm=False,
                )

            prelift_args = (
                layers[0][0].contiguous(),
                layers[0][1].contiguous(),
                layers[1][0].contiguous(),
                layers[1][1].contiguous(),
                layers[2][0].contiguous(),
                layers[2][1].contiguous(),
                layers[3][0].contiguous(),
                layers[3][1].contiguous(),
                inp.contiguous(),
            )
            context, depth = self.prelift(*prelift_args)
            grid = self._lift(rgb, context, depth, K)
            return self.postlift(grid)
