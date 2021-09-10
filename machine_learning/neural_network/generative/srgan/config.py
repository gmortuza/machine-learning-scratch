import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# All config
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SCALE = 4.
LOAD_MODEL = True
SAVE_MODEL = True
EPOCHS = 100
HEIGHT = 321
WIDTH = 500

normalize_transform = A.Compose(
    [
        A.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        A.Resize(HEIGHT, WIDTH, always_apply=True, p=1),
        ToTensorV2()
    ]
)
# degrade resolution
degrade_res_transform = A.Compose(
    [
        A.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        # lower the resolution
        A.Resize(int(HEIGHT // SCALE), int(WIDTH // SCALE), always_apply=True, p=1),
        # higher the resolution
        A.Resize(HEIGHT, WIDTH, always_apply=True, p=1),
        ToTensorV2()
    ]
)