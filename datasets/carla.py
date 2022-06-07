import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return torch.swapaxes(t, 0, 1)

    if f > frames:
        return torch.swapaxes(t[:, :frames], 0, 1)

    return torch.swapaxes(F.pad(t, (0, 0, 0, 0, 0, frames - f)), 0, 1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


# class CARLADataset(Dataset):
#     def __init__(self, folder, image_size, channels=3, frames_per_sample=16, horizontal_flip=False,
#                  force_num_frames=True, exts=['gif']):
#         super().__init__()
#         self.folder = folder
#         assert os.path.exists(self.folder)
#         self.image_size = image_size
#         self.channels = channels
#         self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
#
#         self.cast_num_frames_fn = partial(cast_num_frames, frames=frames_per_sample) if force_num_frames else identity
#
#         self.transform = T.Compose([
#             T.Resize(image_size),
#             T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
#             T.CenterCrop(image_size),
#             T.ToTensor(),
#             T.Lambda(normalize_img)
#         ])
#
#     def __len__(self):
#         return len(self.paths)
#
#     def __getitem__(self, index):
#         path = self.paths[index]
#         tensor = gif_to_tensor(path, self.channels, transform=self.transform)
#         return self.cast_num_frames_fn(tensor), torch.tensor(1)


class CARLADataset(Dataset):
    def __init__(
            self,
            folder,
            image_size,
            channels=3,
            frames_per_sample=16,
            horizontal_flip=False,
            force_num_frames=True,
            exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = os.listdir(self.folder)
        # [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames=frames_per_sample) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        forward_path = os.path.join(self.folder, path, 'forward.gif')
        action_path = os.path.join(self.folder, path, 'action.txt')

        action_tensor = torch.Tensor(np.loadtxt(action_path))
        gif_tensor = self.cast_num_frames_fn(gif_to_tensor(forward_path, self.channels, transform=self.transform))
        return gif_tensor, action_tensor
