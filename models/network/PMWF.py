# =====================================================================
# PMWF: Progressive Multi-Scale Weighted Fusion Module
# =====================================================================
# Core Innovation: Progressive fusion of multi-scale backbone features with
# adaptive weight allocation. Low-level discriminative details are preserved
# through sequential fusion, avoiding being overwhelmed by high-level semantics.
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from models.backbone.utils import LayerNorm
from models.network.conv_cross_att import LSKA


class PMWF(nn.Module):
    def __init__(self, in_channels, output_size=(14, 14), **kwargs):
        super().__init__()
        drop_path = kwargs.get('drop_path', 0.1)
        layer_scale = kwargs.get('layer_scale', 1e-6)
        compress_c = kwargs.get('compress_channels', 8)
        dim = in_channels[3] // 8

        # Feature projectors
        self.projectors = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, c // 8, 1, bias=False),
                         LayerNorm(c // 8, eps=1e-6, data_format="channels_first"),
                         nn.GELU()) for c in in_channels
        ])

        # Spatial aligners
        self.aligners = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels[0] // 8, dim, 4, 4, 0)),
            nn.Sequential(nn.Conv2d(in_channels[1] // 8, dim, 2, 2, 0)),
            nn.Sequential(nn.Conv2d(in_channels[2] // 8, dim, 1)),
            nn.Upsample(scale_factor=2, mode='bilinear')
        ])

        # Weight generators for adaptive fusion
        self.weight_gens = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, compress_c, 1),
                         LayerNorm(compress_c, eps=1e-6, data_format="channels_first"),
                         nn.GELU()) for _ in range(4)
        ])
        self.fusion_weights = nn.ModuleList([nn.Conv2d(compress_c * 2, 2, 1) for _ in range(3)])
        self.weight_forwards = nn.ModuleList([nn.Conv2d(compress_c * 2, compress_c, 1) for _ in range(2)])

        # Fusion refiners
        self.refiners = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1),
                         LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                         nn.GELU()) for _ in range(3)
        ])

        # Enhancement
        self.lkfe = LSKA(dim, k_size=7)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.ffn1 = nn.Linear(dim, 4 * dim)
        self.ffn2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.output_proj = nn.Sequential(nn.Conv2d(dim, in_channels[3], 1),
                                         LayerNorm(in_channels[3], eps=1e-6, data_format="channels_first"),
                                         nn.GELU())

    def forward(self, features):
        x0, x1, x2, x3 = features

        # Project and align
        x = [p(f) for p, f in zip(self.projectors, [x0, x1, x2, x3])]
        x = [a(x[i]) for i, a in enumerate(self.aligners)]
        skip = x[3]

        # Progressive fusion
        w = [g(x[i]) for i, g in enumerate(self.weight_gens)]

        w_cat = torch.cat([w[0], w[1]], dim=1)
        alpha = F.softmax(self.fusion_weights[0](w_cat), dim=1)
        f1 = self.refiners[0](x[0] * alpha[:, 0:1] + x[1] * alpha[:, 1:2])

        w_cat = torch.cat([self.weight_forwards[0](w_cat), w[2]], dim=1)
        alpha = F.softmax(self.fusion_weights[1](w_cat), dim=1)
        f2 = self.refiners[1](f1 * alpha[:, 0:1] + x[2] * alpha[:, 1:2])

        w_cat = torch.cat([self.weight_forwards[1](w_cat), w[3]], dim=1)
        alpha = F.softmax(self.fusion_weights[2](w_cat), dim=1)
        f3 = self.refiners[2](f2 * alpha[:, 0:1] + x[3] * alpha[:, 1:2])

        # Enhancement
        out = f3 + skip
        out = self.lkfe(out) + skip
        out = self.norm(out).permute(0, 2, 3, 1)
        out = self.ffn2(F.gelu(self.ffn1(out)))
        if self.gamma is not None:
            out = self.gamma * out
        return self.output_proj(out.permute(0, 3, 1, 2))
