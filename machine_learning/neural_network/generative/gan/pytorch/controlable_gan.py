import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# ============= CREATE DISCRIMINATOR ============= #


class Discriminator(nn.Module):
    """
    Discriminator will take an image and differentiate if that image is real or fake
    """
    def __init__(self, image_channel=1, hidden_dims=16):
        """

        :param image_dims: Input image shape
        """
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_single_block(image_channel, hidden_dims),
            self.get_single_block(hidden_dims, hidden_dims * 2),
            nn.Conv2d(hidden_dims * 2, 1, kernel_size=4, stride=2)
        )

    def get_single_block(self, input_channels, output_channels, kernel_size=4, stride=2):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(.2)
        )

    def forward(self, x):
        predict = self.disc(x)
        return predict.view(len(x), -1)


# ============= CREATE GENERATOR ============= #
class Generator(nn.Module):
    """
    Generator will take a noise vector and generate fake images from that noise vector
    """
    def __init__(self, noise_dim, image_channel=1, hidden_dim=64):
        """

        :param noise_dim: Dimension of the noise vector (batch_size, noise_vector_dim)
        :param image_dim: Dimension of the output image (batch_size, image_dim)
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.gen = nn.Sequential(
            self.get_single_block(noise_dim, hidden_dim * 4),
            self.get_single_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.get_single_block(hidden_dim * 2, hidden_dim),
            # output layer
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dim, image_channel, kernel_size=4, stride=2),
                nn.Tanh()
            )

        )

    def get_single_block(self, input_channel, output_channel, kernel_size=3, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(len(x), self.noise_dim, 1, 1)
        return self.gen(x)

def combine_vector(x, y):
    """

    :param x:
    :param y:
    :return:
    """
    return torch.cat((x.float(), y.float()), dim=1)

def get_noise(n_samples, noise_dims):
    """
    This will generate noise vector based on given dimension
    :param n_samples: batch size
    :param noise_dims: Dimension of each noise vector
    :return:
    """
    return torch.randn(n_samples, noise_dims).to(device)


def get_disc_loss(generator, discriminator, real, image_one_hot_labels, one_hot_labels, batch_size):
    """
    Calculate discriminator loss
    :param generator:
    :param discriminator:
    :param real:
    :param batch_size:
    :return:
    """
    # Generating the noise vector
    noise = get_noise(batch_size, noise_dim)
    # Creating the fake image from the generator
    noise_and_label = combine_vector(noise, one_hot_labels)
    # We won't train the generator so we are detaching it
    fake = generator(noise_and_label).detach()
    fake_image_and_labels = combine_vector(fake, image_one_hot_labels)
    # See our discriminator prediction about these fake images
    disc_prediction = discriminator(fake_image_and_labels)
    # Getting the discriminator loss for fake images
    disc_fake_loss = criterion(disc_prediction, torch.zeros_like(disc_prediction))

    # train discriminator for real images
    real_image_and_labels = combine_vector(real, image_one_hot_labels)
    disc_real = discriminator(real_image_and_labels)
    # Discriminator loss for real image
    disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))
    # Total loss will be the average of real loss and fake loss
    return (disc_fake_loss + disc_real_loss) / 2


def get_gen_loss(generator, discriminator, image_one_hot_labels, one_hot_labels, num_images):
    """
    Calculate generator loss
    :param generator:
    :param discriminator:
    :param num_images:
    :return:
    """
    # Get random noise
    noise = get_noise(num_images, noise_dim)
    noise_and_label = combine_vector(noise, one_hot_labels)
    # Generate fake image
    fake_image = generator(noise_and_label)
    # Predict the fake image using discriminator
    fake_image_and_labels = combine_vector(fake_image, image_one_hot_labels)
    disc_prediction = discriminator(fake_image_and_labels)
    # These are fake image but we want to fool the discriminator so we put ones in there
    return criterion(disc_prediction, torch.ones_like(disc_prediction))


if __name__ == '__main__':
    # ============= HYPER PARAMETER ============= #
    # If there is GPU available then we will choose GPU else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 1e-3
    # Random noise shape from which generator will create fake images
    noise_dim = 64
    batch_size = 32
    epochs = 10
    input_shape = (1, 28, 28)
    input_size = np.prod(input_shape)
    n_classes = 10
    # This noise will be used generate image for visualization. These images will be written in the tensorboard grid
    fixed_noise = get_noise(batch_size, noise_dim+n_classes)
    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/real")
    writer = SummaryWriter(f"logs/")

    # Generator
    gen = Generator(noise_dim+n_classes).to(device)
    # Discriminator
    disc = Discriminator(image_channel=input_shape[0]+n_classes).to(device)

    # Optimizer for generator and discriminator
    gen_optim = optim.Adam(params=gen.parameters(), lr=learning_rate)
    disc_optim = optim.Adam(params=disc.parameters(), lr=learning_rate)

    # we will use binary cross entropy loss
    criterion = nn.BCEWithLogitsLoss()

    # ============= DATASETS ============= #
    dataset = datasets.MNIST(root="dataset", train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ============= TRAINING LOOP ============= #
    for epoch in range(epochs):
        pbar_batch = tqdm(data_loader, total=len(data_loader))
        for batch_idx, (real, labels), in enumerate(pbar_batch):
            current_batch_size = len(real)
            real = real.to(device)
            #
            one_hot_labels = torch.nn.functional.one_hot(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, input_shape[1], input_shape[2])
            # Update discriminator
            disc_optim.zero_grad()
            disc_loss = get_disc_loss(gen, disc, real, image_one_hot_labels, one_hot_labels, current_batch_size)
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            # Update generator
            gen_optim.zero_grad()
            gen_loss = get_gen_loss(gen, disc, image_one_hot_labels, one_hot_labels, current_batch_size)
            gen_loss.backward(retain_graph=True)
            gen_optim.step()

            text = f"Epoch {epoch} | Generator loss: {gen_loss.item():.4f} \t Discriminator loss: " \
                   f"{disc_loss.item():.4f}\t"
            pbar_batch.set_description(text)
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            data = real.reshape(-1, 1, 28, 28)
            img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "Mnist Fake Images", img_grid_fake, global_step=epoch
            )
            writer_real.add_image(
                "Mnist Real Images", img_grid_real, global_step=epoch
            )

            writer.add_scalars("loss", {'generator loss': gen_loss.item(), 'Discriminator loss': disc_loss.item()}, global_step=epoch)
writer.close()
writer_fake.close()
writer_real.close()

# Generate control label
import matplotlib.pyplot as plt
generate_this_label = 5
noise = torch.randn(1, noise_dim)
one_hot = torch.zeros(1, n_classes)
one_hot[0][generate_this_label] = 1.0
noise_and_label = combine_vector(noise, one_hot).to(device)
fake = gen(noise_and_label).detach().cpu()
plt.imshow(fake[0][0], cmap="gray")
plt.show()


