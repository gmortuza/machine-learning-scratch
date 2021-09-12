import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

# All config
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SCALE = 4.
LOAD_MODEL = True
SAVE_MODEL = True
EPOCHS = 100
HIGH_RES = 96
LOW_RES = int(HIGH_RES // SCALE)

normalize_transform = A.Compose(
    [
        A.Normalize(mean=[0., 0., 0.], std=[.5, .5, .5]),
        A.CenterCrop(HIGH_RES, HIGH_RES, always_apply=True, p=1),
    ]
)

low_res_transform = A.Compose(
    [
        A.Resize(LOW_RES, LOW_RES, interpolation=Image.BICUBIC),
        ToTensorV2()
    ]
)
