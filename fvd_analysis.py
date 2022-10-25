import itertools

import numpy as np

from models.fvd.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
import torch
from PIL import Image
from torchvision import transforms as T
from functools import partial
from torch.utils import data
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

modes = ['action_conformity', 'obstacle_conformity', 'vehicle-conformity']

reduce_framerate = False


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def identity(t, *args, **kwargs):
    return t


class Dataset(data.Dataset):
    def __init__(
            self,
            folder,
            image_size,
            channels=3,
            num_frames=16,
            force_num_frames=True,
            exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, transform=self.transform)
        return self.cast_num_frames_fn(tensor)


def seek_all_images(img):
    mode = 'RGB'
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def gif_to_tensor(path, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img)))
    return torch.stack(tensors, dim=1)


i3d = load_i3d_pretrained('cuda')

first_folder_path = 'C:\\Users\\chint\\Downloads\\gifs_old\\generated\\throttle'
second_folder_path = 'C:\\Users\\chint\\Downloads\\gifs\\train\\throttle_left'

first_ds = Dataset(first_folder_path, 224, channels=3, num_frames=16)
first_dl = iter(data.DataLoader(first_ds, batch_size=4, shuffle=True))

second_ds = Dataset(second_folder_path, 224, channels=3, num_frames=16)
second_dl = iter(data.DataLoader(second_ds, batch_size=4, shuffle=True))

fake_embeddings = []
real_embeddings = []
for nt in tqdm(range(14)):
    first_set_of_videos = next(first_dl).cuda()
    second_set_of_videos = next(second_dl).cuda()

    if reduce_framerate:
        second_set_of_videos = torch.index_select(second_set_of_videos, 2,
                                                  torch.tensor([0, 1, 2, 3, 4, 6, 8, 10, 12, 14]).cuda())

    fake_embeddings.append(get_fvd_feats(first_set_of_videos, i3d=i3d, device='cuda'))
    real_embeddings.append(get_fvd_feats(second_set_of_videos, i3d=i3d, device='cuda'))
    del first_set_of_videos
    del second_set_of_videos

dist = frechet_distance(np.concatenate(fake_embeddings), np.concatenate(real_embeddings))
print(dist)
