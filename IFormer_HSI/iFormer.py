from timm.models.layers import PatchEmbed, Mlp, DropPath
from timm.models.vision_transformer import Attention, Block

from timm.models.layers import make_divisible

import torch
import torch.nn as nn

class InceptionMixer(nn.Module):
    def __init__(self, input_channels, tran_ratio, pool_stride, img_size):
        super().__init__()
        # have to be even, not odd
        # conv_chan = int(input_channels*conv_ratio/2) * 2
        tran_chans = make_divisible(input_channels * tran_ratio, 32)
        conv_chans = input_channels - tran_chans
        self.high = conv_chans
        self.low = tran_chans
        self.maxpool_fc = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(self.high // 2, self.high // 2, 1),
            nn.BatchNorm2d(self.high // 2),
            nn.ReLU6(inplace=True),
        )

        self.fc_dw = nn.Sequential(
            nn.Conv2d(self.high // 2, self.high // 2, 1),
            nn.BatchNorm2d(self.high // 2),
            nn.ReLU6(inplace=True),
            nn.Conv2d(self.high // 2, self.high // 2, 3, padding=1, groups=self.high // 2),
            nn.BatchNorm2d(self.high // 2),
            # nn.ReLU6(inplace=True),
        )

        self.pool_stride = pool_stride

        self.attn = Attention(self.low, num_heads=self.low // 32)  # 8)
        H = W = img_size
        patch_size = int(H // pool_stride * W // pool_stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_size, self.low))

        self.fuse_dw = nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels)
        self.fuse_linear = nn.Conv2d(input_channels, input_channels, 1)
        # print(conv_chan, input_channels-conv_chan)

    def forward(self, x):
        B, C, H, W = x.shape

        X_h1 = x[:, :self.high // 2, ...]
        X_h2 = x[:, self.high // 2:self.high, ...]
        X_l = x[:, -self.low:, ...]

        Y_h1 = self.maxpool_fc(X_h1)
        Y_h2 = self.fc_dw(X_h2)

        Y_l = nn.AdaptiveAvgPool2d((H // self.pool_stride, W // self.pool_stride))(X_l)
        Y_l = Y_l.flatten(2).transpose(1, 2)
        Y_l = Y_l + self.pos_embed
        # print(Y_l.shape, self.pos_embed.shape)
        # print(attn.shape)
        Y_l = self.attn(Y_l)
        Y_l = Y_l.reshape(B, -1, H // self.pool_stride, W // self.pool_stride)
        Y_l = nn.UpsamplingBilinear2d((H, W))(Y_l)

        Y_c = torch.cat([Y_l, Y_h1, Y_h2], dim=1)
        out = self.fuse_linear(Y_c + self.fuse_dw(Y_c))
        # print("conv", conv_out1.shape, conv_out2.shape, "attn", attn_out.shape)
        # print(out.shape)
        return out


class iFormerBlock(nn.Module):
    def __init__(
            self, input_channels, tran_ratio, pool_stride, img_size,
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        dim = input_channels
        self.norm1 = norm_layer(dim)
        self.inceptionmixer = InceptionMixer(input_channels, tran_ratio, pool_stride, img_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x = x + self.drop_path(
            self.inceptionmixer(
                self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            )
        )
        x = x + self.drop_path(
            self.mlp(
                self.norm2(x.permute(0, 2, 3, 1))
            ).reshape(B, C, H, W)
        )
        return x


class InceptionTransformer(nn.Module):
    def __init__(self, img_size, in_channels):
        super().__init__()

        self.iBlock = nn.Sequential(
            iFormerBlock(in_channels, 0.5, 1, img_size)
        )

    def forward(self, x):
        x = self.iBlock(x)
        return x

if __name__ == "__main__":
    input = torch.rand(6, 64, 7, 7)
    print(input.shape)
    iFormer = InceptionTransformer(img_size=7, in_channels=64)
    output = iFormer(input)
    print(output.shape)
