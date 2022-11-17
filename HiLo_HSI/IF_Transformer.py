from functools import partial
import torch
from torch import nn
import math
from iFormer import InceptionTransformer


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # cheap operation, Noteth use of grouped convolution for channel separation
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)  # The main convolution operation
        x2 = self.cheap_operation(x1)  # cheap transform operation
        out = torch.cat([x1, x2], dim=1)  # The main convolution operation
        return out[:, :self.oup, :]



class IF_Transformer(nn.Module):
    def __init__(self, image_size=224, in_channels=200, num_classes=16,
                 embed_dim=256, depth=3, norm_layer=None):
        super(IF_Transformer, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.CNN_denoise = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 128, kernel_size=(1, 1)),
            nn.LeakyReLU()
        )

        self.ghostmodule = GhostModule(128, embed_dim)
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, image_size * image_size + 1, embed_dim))
        torch.nn.init.normal_(self.spa_pos_embed, std=.02)
        self.depth = depth
        self.iftransformers = nn.ModuleList([])
        for _ in range(self.depth):
            self.iftransformers.append(InceptionTransformer(img_size=image_size, in_channels=embed_dim))
        self.norm = norm_layer(embed_dim)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.CNN_denoise(x)
        x_spa = self.ghostmodule(x)
        for IFtransformers in self.iftransformers:
            x_spa = IFtransformers(x_spa)
        x_spa = self.pooling(x_spa).flatten(1)
        x_spa = self.classifier(x_spa)

        return x_spa
