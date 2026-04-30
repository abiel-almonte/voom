import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import lift_splat, precompute_vox2pix, lift_splat_gather_fp16_nhwc_ch64
from .utils import size_to_model, BasicBlock


class VOOMv2(nn.Module):
    def __init__(
        self,
        backbone_size="s",
        sampled_layers=[2, 5, 8, 11],
        embed_dim=64,
        refine_dim=48,
        depth_bins=128,
        out_ch=4,
        grid_dim=(128, 32, 128),
        mpv=0.2,
        offset_m=(0, 0, 0),
    ) -> None:

        super().__init__()

        dinov2 = torch.hub.load(
            "facebookresearch/dinov2",
            f"{size_to_model(backbone_size)}_dd",
            weights="KITTI",
        )

        self.backbone = dinov2.backbone
        self.dpt_head = dinov2.decode_head
        self.sampled_layers = sampled_layers

        self.depth_feat = None

        def _hook(mod, inp, out):
            self.depth_feat = out

        self.dpt_head.fusion_blocks[-1].register_forward_hook(_hook)

        self.depth_bins = depth_bins
        self.mpv = mpv
        self.grid_dim = grid_dim
        self.offset_m = offset_m

        self.dproj = nn.Sequential(
            nn.Conv2d(256, embed_dim, 1),
            nn.ReLU(),
            BasicBlock(embed_dim, embed_dim),
            BasicBlock(embed_dim, embed_dim),
            nn.Conv2d(embed_dim, depth_bins, 1),
        )

        self.cproj = nn.Sequential(
            nn.Conv2d(256 + 3, embed_dim, 1),
            nn.ReLU(),
            BasicBlock(embed_dim, embed_dim),
            BasicBlock(embed_dim, embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

        self.rproj = nn.Conv3d(embed_dim, refine_dim, 1)
        self.rblok = nn.Sequential(
            nn.Conv3d(refine_dim, refine_dim, 3, padding=1),
            nn.GroupNorm(min(8, refine_dim), refine_dim),
        )
        self.rout = nn.Conv3d(refine_dim, out_ch, 1)

        self.offsets = None
        self.pixs = None

    def _vit_backbone(self, rgb):
        b, c, h, w = rgb.shape

        ph = math.ceil(h / 14) * 14
        pw = math.ceil(w / 14) * 14

        inp = F.interpolate(rgb, size=(ph, pw), align_corners=True, mode="bilinear")
        layers = self.backbone.get_intermediate_layers(
            inp,
            n=self.sampled_layers,
            reshape=True,
            return_class_token=True,
            norm=False,
        )

        return layers

    def _lift_splat_refine(self, rgb, K, feat):
        depth = self.dproj(feat)

        rgb_resized = F.interpolate(
            rgb, size=depth.shape[-2:], mode="bilinear", align_corners=True
        )

        context = self.cproj(torch.cat([rgb_resized, feat], dim=1))

        if self.training or not context.is_cuda:
            grid = lift_splat(
                context,
                F.softmax(depth, dim=1),
                K,
                rgb.shape[-2:],
                self.grid_dim,
                self.mpv,
                self.offset_m,
            )
        else:
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
            grid = lift_splat_gather_fp16_nhwc_ch64(
                context_nhwc,
                F.softmax(depth, dim=1),
                self.offsets,
                self.pixs,
                self.grid_dim,
            )

        grid = self.rproj(grid)
        grid = F.relu(grid + self.rblok(grid))
        grid = self.rout(grid)

        if self.training:
            return grid, depth

        return grid

    def forward(self, rgb, K):
        layers = self._vit_backbone(rgb)
        _ = self.dpt_head(list(layers), img_metas=None)

        return self._lift_splat_refine(rgb, K, self.depth_feat)
