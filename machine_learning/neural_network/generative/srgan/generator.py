import torch
import torch.nn as nn
from config import *


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.rs_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor):
        return x + self.rs_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4, bias=False),
            nn.PReLU()
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64)
        )

        self.up_scaler = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 9, 1, 4),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block_1(x)
        for _ in range(16):
            out = ResBlock(64)(out)
        out = out + self.conv_block_2(out)
        out = self.up_scaler(out)
        out = self.final_conv(out)
        return out


def test():
    gen = Generator()
    x = torch.randn((32, 3, HEIGHT, WIDTH))
    out = gen(x)
    print(out.shape)


if __name__ == '__main__':
    test()
