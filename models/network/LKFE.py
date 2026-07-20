# =====================================================================
# LKFE: Large-Kernel Feature Enhancement Module
# =====================================================================
# Core Innovation: Efficient large-kernel attention via directional
# decomposition. Uses separable horizontal/vertical convolutions with
# bidirectional propagation to capture long-range spatial dependencies
# without the computational cost of standard large-kernel convolutions.
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.utils import LayerNorm


class LKFE(nn.Module):
    """Large-kernel feature enhancement via directional decomposition."""

    def __init__(self, dim, k_size=7):
        super().__init__()

        config = self._get_conv_config(k_size, dim)
        self.conv_h = config['h']
        self.conv_v = config['v']
        self.conv_h2 = config['h2']
        self.conv_v2 = config['v2']
        self.conv_h_rev = config['h_rev']
        self.conv_v_rev = config['v_rev']
        self.conv_h2_rev = config['h2_rev']
        self.conv_v2_rev = config['v2_rev']

        self.proj = nn.Conv2d(dim, dim, 1)
        self.weights = nn.Parameter(torch.randn(2, 1, 1, 1) * 1e-3)
        self.norms = nn.ModuleList([
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),
            LayerNorm(dim, eps=1e-6, data_format="channels_first")
        ])

    def _get_conv_config(self, k_size, dim):
        configs = {
            7: {'kh': 3, 'kv': 3, 'dh': 2, 'dv': 2, 'sh': 3, 'sv': 3},
            11: {'kh': 3, 'kv': 5, 'dh': 2, 'dv': 2, 'sh': 3, 'sv': 5},
            23: {'kh': 5, 'kv': 7, 'dh': 3, 'dv': 3, 'sh': 5, 'sv': 7},
            35: {'kh': 5, 'kv': 11, 'dh': 3, 'dv': 3, 'sh': 5, 'sv': 11},
            41: {'kh': 5, 'kv': 13, 'dh': 3, 'dv': 3, 'sh': 5, 'sv': 13},
            53: {'kh': 5, 'kv': 17, 'dh': 3, 'dv': 3, 'sh': 5, 'sv': 17},
        }
        c = configs.get(k_size, configs[7])
        return {
            'h': nn.Conv2d(dim, dim, (1, c['kh']), 1, (0, c['kh']//2), groups=dim),
            'v': nn.Conv2d(dim, dim, (c['kv'], 1), 1, (c['kv']//2, 0), groups=dim),
            'h2': nn.Conv2d(dim, dim, (1, c['sh']), 1, c['dh'], groups=dim, dilation=c['dh']),
            'v2': nn.Conv2d(dim, dim, (c['sv'], 1), 1, c['dv'], groups=dim, dilation=c['dv']),
            'h_rev': nn.Conv2d(dim, dim, (1, c['kh']), 1, (0, c['kh']//2), groups=dim),
            'v_rev': nn.Conv2d(dim, dim, (c['kv'], 1), 1, (c['kv']//2, 0), groups=dim),
            'h2_rev': nn.Conv2d(dim, dim, (1, c['sh']), 1, c['dh'], groups=dim, dilation=c['dh']),
            'v2_rev': nn.Conv2d(dim, dim, (c['sv'], 1), 1, c['dv'], groups=dim, dilation=c['dv']),
        }

    def forward(self, x):
        residual = x.clone()

        # Forward path
        out = self.conv_v2(self.conv_h2(self.conv_v(self.conv_h(x))))

        # Reverse path
        rev = x.flip([-1])
        rev = self.conv_h_rev(rev).flip([-1])
        rev = rev.flip([-2])
        rev = self.conv_v_rev(rev).flip([-2])
        rev = rev.flip([-1])
        rev = self.conv_h2_rev(rev).flip([-1])
        rev = rev.flip([-2])
        rev = self.conv_v2_rev(rev).flip([-2])

        # Aggregation
        paths = [out, rev]
        if self.training:
            paths = [n(p) for n, p in zip(self.norms, paths)]
            attn = sum(p * w for p, w in zip(paths, F.softmax(self.weights, dim=0)))
        else:
            attn = sum(paths)

        return residual * self.proj(attn)


LSKA1 = LKFE
