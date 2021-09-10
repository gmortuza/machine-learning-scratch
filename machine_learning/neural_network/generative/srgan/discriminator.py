import torch
from config import *
from torch import Tensor
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(.2, inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super(Discriminator, self).__init__()

        blocks = []
        for idx, out_channel in enumerate([64, 64, 128, 128, 256, 256, 512, 512]):
            blocks.append(
                ConvBlock(
                    in_channel,
                    out_channel,
                    3,
                    1 + idx % 2,
                    padding=1,
                )
            )
            in_channel = out_channel
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return self.classifier(x)


def test():
    input_ = torch.randn((16, 3, HEIGHT, WIDTH))
    dis = Discriminator()
    out = dis(input_)
    print(out.shape)


if __name__ == '__main__':
    test()
