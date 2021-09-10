import os
import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import glob
from PIL import Image
from config import *
import torch
from torchvision.transforms import ToTensor


class Dataset(Dataset):
    def __init__(self, data_dir):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.file_names = glob.glob(self.data_dir + '/*.jpg')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.file_names[idx]))
        low_res_image = degrade_res_transform(image=image)['image']
        return low_res_image, normalize_transform(image=image)['image']


def fetch_data_loader(directory):
    directory = os.path.join(os.getcwd(), directory)
    dataset = Dataset(directory)
    # return both train and validation
    train_indices, val_indices = train_test_split(range(len(dataset)), test_size=.1)

    train_split = Subset(dataset, train_indices)
    val_split = Subset(dataset, val_indices)

    # create batch
    train_batch = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
    val_batch = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=True)

    return train_batch, val_batch


def test():
    train, val = fetch_data_loader('dataset/')
    for (low_res, high_res) in train:
        print(high_res.shape)
        print(low_res.shape)


if __name__ == '__main__':
    test()
