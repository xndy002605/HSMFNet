# =====================================================================
# SGCA: Semantic-Guided Cross-Attention Calibration Module
# =====================================================================
# Core Innovation: Calibrate fused features using high-purity semantic
# features from the final backbone layer as an anchor. Employs dual
# attention mechanisms (channel + spatial) for comprehensive refinement.
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from models.backbone.utils import LayerNorm
from models.network.conv_cross_att import LSKA


class SemanticGuide(nn.Module):
    """Semantic guidance for feature fusion via learned weights."""

    def __init__(self, dim):
        super().__init__()
        self.decoder = nn.Sequential(
            _BasicConv(dim, dim // 4, 1),
            LSKA(dim // 4, k_size=7),
            _BasicConv(dim // 4, dim // 4, 3, pad=1)
        )
        self.weight_gen = nn.Sequential(nn.Conv2d(dim // 4, dim, 1), nn.Sigmoid())
        self.strength = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, base_feat, guide_feat):
        return base_feat * (1 + self.strength * self.weight_gen(self.decoder(guide_feat)))


class CrossAttentionCalibrator(nn.Module):
    """Cross-attention calibration using semantic anchor."""

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, kv, mask=None):
        H, W = query.shape[2], query.shape[3]
        q = rearrange(self.W_q(query), 'b c h w -> b (h w) c')
        k = rearrange(self.W_k(kv), 'b c h w -> b (h w) c')
        v = rearrange(self.W_v(kv), 'b c h w -> b (h w) c')
        q = rearrange(q, 'b n (h hc) -> b h n hc', h=self.num_heads)
        k = rearrange(k, 'b n (h hc) -> b h n hc', h=self.num_heads)
        v = rearrange(v, 'b n (h hc) -> b h n hc', h=self.num_heads)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        out = rearrange((F.softmax(attn, dim=-1) @ v), 'b h n hc -> b n (h hc)')
        return rearrange(self.W_o(out), 'b (h w) c -> b c h w', h=H, w=W)


class SemanticTransition(nn.Module):
    """Semantic transition between hierarchy levels with background suppression."""

    def __init__(self, dim, target_size):
        super().__init__()
        self.norm_high = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm_low = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.target_size = target_size

    def forward(self, high_feat, low_feat):
        B, C, H, W = low_feat.shape
        high = F.normalize(rearrange(self.norm_high(high_feat), 'b c h w -> b c (h w)'), p=2, dim=1)
        low = F.normalize(rearrange(self.norm_low(low_feat), 'b c h w -> b c (h w)'), p=2, dim=1)

        self_map = torch.topk(torch.bmm(low.transpose(-1, -2), low), k=int(H*W*0.1), dim=-1)[0].mean(dim=-1)
        self_map = rearrange(self_map, 'b (h w) -> b h w', h=H, w=W)

        cross_map = rearrange(torch.sum(high * low, dim=1), 'b (h w) -> b h w', h=H, w=W)
        patch_size = 2
        cross_patched = rearrange(cross_map, 'b (h ph) (w pw) -> b h w ph pw', ph=patch_size, pw=patch_size)
        patch_sum = cross_patched.sum(dim=(-1, -2), keepdim=True)
        contrib_map = rearrange(cross_patched / (patch_sum + 1e-8) * patch_sum,
                                'b h w ph pw -> b (h ph) (w pw)', h=H//patch_size, w=W//patch_size)

        mask = torch.sigmoid(self_map) * (1 - torch.sigmoid(contrib_map))
        channel_w = torch.sigmoid(F.normalize(mask.reshape(B, -1), p=2, dim=1).unsqueeze(1)
                                  .expand(-1, C, -1).sum(dim=-1))
        return low_feat * mask.unsqueeze(1) * channel_w.unsqueeze(-1).unsqueeze(-1) + low_feat


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(),
                                  nn.Linear(channels // reduction, channels))

    def forward(self, x):
        return x * F.sigmoid(self.mlp(F.adaptive_avg_pool2d(x, 1).view(x.shape[0], x.shape[1]))
                            ).view(x.shape[0], x.shape[1], 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        return x * F.sigmoid(self.conv(torch.cat([x.mean(dim=1, keepdim=True),
                                                   x.max(dim=1, keepdim=True)[0]], dim=1)))


class CBAM(nn.Module):
    """Combined channel and spatial attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, feat, ref=None):
        return feat * self.ca(ref or feat) * self.sa(ref or feat)


class _BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, pad=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, (k-1)//2 if pad is None else pad, bias=False)
        self.norm = LayerNorm(out_ch, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class MutipCrossAttention(nn.Module):
    """Multi-part cross-attention for hierarchical alignment."""

    def __init__(self, dim, query_dim, heads_dim=4, drop=0.2):
        super().__init__()
        self.num_heads = dim // heads_dim
        self.scale = heads_dim ** -0.5
        self.q = nn.Linear(query_dim, query_dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.o = nn.Linear(query_dim, query_dim)
        self.pos_bias = nn.Parameter(torch.zeros(1, self.num_heads, query_dim // self.num_heads, dim // self.num_heads))
        self.dropout = nn.Dropout(drop)

    def forward(self, query, kv):
        B, P, _ = query.shape
        q = rearrange(self.q(query), 'b p c -> b c p')
        kv = rearrange(self.kv(kv), 'b p c -> b c p')
        q = rearrange(q, 'b (h hc) n -> b h hc n', h=self.num_heads)
        kv = rearrange(kv, 'b (kv h hc) p -> kv b h hc p', kv=2, h=self.num_heads)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.pos_bias
        out = rearrange(self.dropout(F.softmax(attn, dim=-1)) @ v, 'b h hc n -> b (h hc) n')
        return self.o(rearrange(out, 'b c p -> b p c'))


class Attentions(nn.Module):
    """Ensemble channel attention across scales."""

    def __init__(self, channels):
        super().__init__()
        self.attentions = nn.ModuleList([ChannelAttention(channels) for _ in range(4)])

    def forward(self, features):
        out = self.attentions[0](features[0])
        for i, feat in enumerate(features[1:], 1):
            out = (self.attentions[i](feat) + out) / 2
        return features[0] * out
