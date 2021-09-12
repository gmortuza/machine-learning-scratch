import torch
import torch.nn as nn
from tqdm import tqdm

from discriminator import Discriminator
from generator import Generator
from loss import fetch_disc_loss, fetch_gen_loss
from config import *
from torch import optim
from dataset import fetch_data_loader


def train(gen, disc: nn.Module, disc_optim: torch.optim, gen_optim: torch.optim, gen_loss_fn, disc_loss_fn, train_data_loader: torch.utils.data.DataLoader):
    gen_loss_history = []
    disc_loss_history = []
    accuracy_history = []

    with tqdm(total=len(train_data_loader)) as progress_bar:
        for idx, (low_res, high_res) in enumerate(train_data_loader):
            low_res = low_res.to(DEVICE)
            high_res = high_res.to(DEVICE)

            fake_img = gen(low_res)

            # Train Discriminator
            disc_real = disc(high_res)
            disc_fake = disc(fake_img.detach())
            disc_loss = disc_loss_fn(disc_fake, disc_real)

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            # train Generator
            disc_fake = disc(fake_img)
            gen_loss = gen_loss_fn(disc_fake, fake_img, high_res)
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            gen_loss_history.append(gen_loss.item())
            disc_loss_history.append(disc_loss.item())

            bar_text = f'gen : {round(sum(gen_loss_history) / len(gen_loss_history), 3)}, disc : {round(sum(disc_loss_history) / len(disc_loss_history), 3)}'
            progress_bar.set_postfix(loss=bar_text)
            progress_bar.update()


def train_evaluation():
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    gen_optim = optim.Adam(gen.parameters(), lr=LEARNING_RATE)
    disc_optim = optim.Adam(disc.parameters(), lr=LEARNING_RATE)
    gen_loss_fn = fetch_gen_loss()
    disc_loss_fn = fetch_disc_loss()
    # Data loader
    train_data_loader, val_data_loader = fetch_data_loader('dataset/')
    train(gen, disc, disc_optim, gen_optim, gen_loss_fn, disc_loss_fn, train_data_loader)


if __name__ == '__main__':
    train_evaluation()
