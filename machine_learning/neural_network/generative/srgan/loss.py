import torch
import torch.nn as nn
import torchvision.models as models
from config import *

bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        for params in self.vgg19.parameters():
            params.requires_grad = False

    def forward(self, fake: torch.Tensor, high_res: torch.Tensor) -> torch.Tensor:
        vgg_fake = self.vgg19(fake)
        vgg_high_res = self.vgg19(high_res)
        return mse_loss(vgg_fake, vgg_high_res)


def get_vgg_loss(fake, high_res):
    vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()

    extracted_feature = nn.Sequential(*list(vgg19.features.children())[:36])
    # Freeze model param
    for parameters in extracted_feature.parameters():
        parameters.requires_grad = False
    vgg_fake_feature = vgg19(fake)
    vgg_target_feature = vgg19(high_res)
    return mse_loss(vgg_fake_feature, vgg_target_feature)


def fetch_disc_loss():
    def disc_loss(fake, real):
        fake_loss = bce_loss(fake, torch.zeros_like(fake))
        real_loss = bce_loss(real, torch.ones_like(real))
        return fake_loss + real_loss

    return disc_loss


def fetch_gen_loss():
    def gen_loss(disc_fake, gen_fake, high_res):
        adversarial_loss = 1e-3 * bce_loss(disc_fake, torch.ones_like(disc_fake))
        vgg_loss = .006 * VGGLoss().to(gen_fake.device)(gen_fake, high_res)
        return adversarial_loss + vgg_loss




    return gen_loss