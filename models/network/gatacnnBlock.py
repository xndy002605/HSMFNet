from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath


class GatedCNNBlock(nn.Module):

    def __init__(self, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):

        super().__init__()
        self.norm = norm_layer(dim)

        hidden = int(expansion_ratio * dim)
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)

        self.fc1 = nn.Linear(dim, hidden * 2)

        self.act = act_layer()

        self.conv = nn.Conv2d(
            conv_channels, conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_channels,
            bias=False
        )

        self.fc2 = nn.Linear(hidden, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        shortcut = x

        x = self.norm(x)  # [B, H, W, C] → [B, H, W, C]

        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)

        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1)

        fused = self.act(g) * torch.cat((i, c), dim=-1)

        x = self.fc2(fused)
        x = self.drop_path(x)
        return x + shortcut

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # bhwc
    x = torch.randn(1, 32, 32, 64).to(device)
    model = GatedCNNBlock(64)

    model.to(device)
    y = model(x)

    print("微信公众号：十小大的底层视觉工坊")
    print("知乎、CSDN：十小大")

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
