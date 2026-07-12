from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# PMFE / PMWF: PROGRESSIVE MULTI-SCALE WEIGHTED FUSION MODULE
# Purpose: Fuse multi-stage backbone features (x0, x1, x2, x3) via:
#   - Adaptive dual-branch weight allocation (channel + spatial)
#   - Progressive fusion from low-level to high-level features
#   - Large-Kernel Feature Enhancement (LKFE) for long-range associations
#   Core innovation: Low-level discriminative details are preserved by
#   progressive fusion, avoiding being overwhelmed by high-level semantics.
# =====================================================================
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from models.backbone.utils import LayerNorm
from models.network.conv_cross_att import LSKA, CC_LSKA


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(in_channels=filter_in, out_channels=filter_out,
                           kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", LayerNorm(filter_out, eps=1e-6, data_format="channels_first")),
        ("gelu", nn.GELU()),
    ]))

class BasicBlock(nn.Module):
    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=filter_out, out_channels=filter_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=filter_out, momentum=0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()
        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x):
        x = self.downsample(x)
        return x

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_str = f"{total_params:,} ({total_params/1e6:.2f}M)"
    trainable_str = f"{trainable_params:,} ({trainable_params/1e6:.2f}M)"
    return total_str, trainable_str


# =====================================================================
# CrossLayerFusion: Core PMFE forward pass
# Purpose: Progressive multi-scale weighted fusion with LKFE enhancement.
# =====================================================================
class CrossLayerFusion(nn.Module):


    def __init__(self, in_c=[80,160,320,640], out_size=(14, 14),
                 drop_path=0.1, layer_scale_init_value=1e-6, compress_c=8):
        super().__init__()

        self.conv0 = BasicConv(in_c[0], in_c[0] // 8, 1)
        self.conv1 = BasicConv(in_c[1], in_c[1] // 8, 1)
        self.conv2 = BasicConv(in_c[2], in_c[2] // 8, 1)
        self.conv3 = BasicConv(in_c[3], in_c[3] // 8, 1)

        self.layer0 = nn.Sequential(BasicConv(in_c[0] // 8, in_c[0] // 8, 1))
        self.layer1 = nn.Sequential(BasicConv(in_c[1] // 8, in_c[1] // 8, 1))
        self.layer2 = nn.Sequential(BasicConv(in_c[2] // 8, in_c[2] // 8, 1))
        self.layer3 = nn.Sequential(BasicConv(in_c[3] // 8, in_c[3] // 8, 1))

        self.d0 = Downsample_x4(in_channels=in_c[0] // 8, out_channels=in_c[3] // 8)
        self.d1 = Downsample_x2(in_channels=in_c[1] // 8, out_channels=in_c[3] // 8)
        self.t2 = BasicConv(in_c[2] // 8, in_c[3] // 8,1)
        self.u3=nn.Upsample(scale_factor=2, mode='bilinear')
        self.out_size = out_size

        self.weight0 = BasicConv(in_c[3] // 8, compress_c, 1, 1)
        self.weight1 = BasicConv(in_c[3] // 8, compress_c, 1, 1)
        self.weight2 = BasicConv(in_c[3] // 8, compress_c, 1, 1)
        self.weight3 = BasicConv(in_c[3] // 8, compress_c, 1, 1)

        self.weight_f0 = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.fusion_c0 = BasicConv(in_c[3] // 8, in_c[3] // 8, 3, 1)

        self.weight_f1 = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.fusion_c1 = BasicConv(in_c[3] // 8, in_c[3] // 8, 3, 1)

        self.weight_f2 = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.fusion_c2 = BasicConv(in_c[3] // 8, in_c[3] // 8, 3, 1)

        self.lska = LSKA(in_c[3] // 8, k_size=7)
        # self.lska2 = LSKA(in_c[3], k_size=7)
        # self.norm = LayerNorm(in_c[3] // 8, eps=1e-6)
        # self.pwconv1 = nn.Conv2d(in_c[3] // 8, 4 * (in_c[3] // 8), kernel_size=1, stride=1, padding=0)
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Conv2d(4 * (in_c[3] // 8), in_c[3] // 8,kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_c[3] // 8)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.conv00 = BasicConv(in_c[3] // 8,in_c[3], 1)
        self.weight1_forward=nn.Conv2d(compress_c * 2, compress_c, kernel_size=1, stride=1, padding=0)
        self.weight0_forward=nn.Conv2d(compress_c * 2, compress_c, kernel_size=1, stride=1, padding=0)


        # self.norm_feat0 = LayerNorm(in_c[3] // 8, eps=1e-6, data_format="channels_first")
        # self.norm_feat1 = LayerNorm(in_c[3] // 8, eps=1e-6, data_format="channels_first")
        self.norm_feat = LayerNorm(in_c[3] // 8, eps=1e-6, data_format="channels_first")
        # self.self.norm_f = LayerNorm(in_c[3] // 8, eps=1e-6, data_format="channels_first")

        self.norm_clong = LayerNorm(in_c[3], eps=1e-6, data_format="channels_first")

        self.pwconv1 = nn.Linear(in_c[3] // 8, 4 * (in_c[3] // 8))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * (in_c[3] // 8), in_c[3] // 8)


    def forward(self, x: torch.Tensor):
        x0,x1,x2,x3=x
        # x3_clone = x3
        # x3_clone = self.u3(x3)
        # x3_clone = self.norm_clong(x3_clone)

        x0=self.conv0(x0)
        x1=self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)


        x0=self.layer0(x0)
        x1=self.layer1(x1)
        x2=self.layer2(x2)
        x3=self.layer3(x3)

        x0=self.d0(x0)
        x1=self.d1(x1)
        x2=self.t2(x2)
        x3=self.u3(x3)
        x3_clone = x3

        x0_weight=self.weight0(x0)
        x1_weight = self.weight1(x1)
        x2_weight = self.weight2(x2)
        x3_weight = self.weight3(x3)

        weight_cat0 = torch.cat([x0_weight, x1_weight], dim=1)
        weight0_forward=self.weight0_forward(weight_cat0)
        weight0 = self.weight_f0(weight_cat0)
        weight0 = F.softmax(weight0, dim=1)  # 权重归一化，动态分配贡献
        fused_feat0 = x0 * weight0[:, 0:1, :, :] + x1 * weight0[:, 1:2, :, :]
        fused_feat0 = self.fusion_c0(fused_feat0)  # 融合后卷积增强
        # fused_feat0 = self.norm_feat0(fused_feat0)

        weight_cat1 = torch.cat([weight0_forward, x2_weight], dim=1)
        weight1_forward = self.weight1_forward(weight_cat1)
        weight1 = self.weight_f1(weight_cat1)
        weight1 = F.softmax(weight1, dim=1)  # 权重归一化，动态分配贡献
        fused_feat1 = fused_feat0 * weight1[:, 0:1, :, :] + x2 * weight1[:, 1:2, :, :]
        fused_feat1 = self.fusion_c1(fused_feat1)  # 融合后卷积增强
        # fused_feat1 = self.norm_feat1(fused_feat1)

        weight_cat2 = torch.cat([weight1_forward, x3_weight], dim=1)
        weight2 = self.weight_f2(weight_cat2)
        weight2 = F.softmax(weight2, dim=1)  # 权重归一化，动态分配贡献
        fused_feat2 = fused_feat1 * weight2[:, 0:1, :, :] + x3 * weight2[:, 1:2, :, :]
        fused_feat2 = self.fusion_c2(fused_feat2)  # 融合后卷积增强


        final_feat2 = fused_feat2 + x3_clone
        final_feat2 = self.lska(final_feat2)
        final_feat2 = final_feat2 + x3_clone
        final_feat2 = self.norm_feat(final_feat2)
        final_feat2 = final_feat2.permute(0, 2, 3, 1)

        final_feat2 = self.pwconv1(final_feat2)
        final_feat2 = self.act(final_feat2)
        final_feat2 = self.pwconv2(final_feat2)
        if self.gamma is not None:
            final_feat2 = self.gamma * final_feat2
        final_feat2 = final_feat2.permute(0, 3, 1, 2)  # (b,h,w,d)->(b,d,h,w)
        final_feat = final_feat2



        # final_feat = self.drop_path(fused_feat2) + x2


        # feat_enhance = self.lska(fused_feat)
        # feat_enhance = feat_enhance + fused_feat  # 残差
        # feat_enhance = fused_feat.permute(0, 2, 3, 1)  # (B,14,14,640)
        # feat_enhance = self.norm(feat_enhance)
        # feat_enhance = feat_enhance.permute(0, 3, 1, 2)
        # ffn_out = self.pwconv1(feat_enhance)
        # ffn_out = self.act(ffn_out)
        # ffn_out = self.pwconv2(ffn_out)
        # ffn_out = ffn_out.permute(0, 2, 3, 1)
        # if self.gamma is not None:
        #     ffn_out = self.gamma * ffn_out
        # ffn_out = ffn_out.permute(0, 3, 1, 2)  # (B,640,14,14)
        # ffn_out = self.norm_f(ffn_out)
        # final_feat = fused_feat + self.drop_path(ffn_out)
        #
        final_feat=self.conv00(final_feat)

        # fused_feat2 = self.norm_feat(fused_feat2)
        # final_feat = self.drop_path(fused_feat2) + x3_clone
        # final_feat = self.lska(final_feat)
        # final_feat = final_feat+x3_clone

        return final_feat
