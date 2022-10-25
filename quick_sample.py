import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

try:
    from torchvision.transforms.functional import resize, InterpolationMode

    interp = InterpolationMode.NEAREST
except:
    from torchvision.transforms.functional import resize

    interp = 0

from datasets import data_transform, inverse_data_transform
from main import dict2namespace
from models.ema import EMAHelper
from runners.ncsn_runner import get_model, conditioning_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')

from models import ddpm_sampler


# Make and load model
def load_model(ckpt_path, device):
    # Parse config file
    with open(os.path.join(os.path.dirname(ckpt_path), 'config.yml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Load config file
    config = dict2namespace(config)
    config.device = device
    # Load model
    scorenet = get_model(config)
    if config.device != torch.device('cpu'):
        scorenet = torch.nn.DataParallel(scorenet)
        states = torch.load(ckpt_path, map_location=config.device)
    else:
        states = torch.load(ckpt_path, map_location='cpu')
        states[0] = OrderedDict([(k.replace('module.', ''), v) for k, v in states[0].items()])
    scorenet.load_state_dict(states[0], strict=False)
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(scorenet)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(scorenet)
    scorenet.eval()
    return scorenet, config


# data_path = '/path/to/data/CIFAR10'
ckpt_path = r'C:\Users\chint\Documents\Malta documents\research\latent diffusion\mcvd\carla_action\final\checkpoint.pt'
data_path = r'C:\Users\chint\Downloads\gifs_old\test\throttle_right'
save_path = r'C:/Users/chint/Documents/Malta documents/research/latent diffusion/mcvd/carla_action/final/v2/'

scorenet, config = load_model(ckpt_path, device)
net = scorenet.module if hasattr(scorenet, 'module') else scorenet


def seek_all_images(img, channels=3):
    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert('RGB')
        except EOFError:
            break
        i += 1


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return torch.swapaxes(t, 0, 1)

    if f > frames:
        time_idx = np.random.choice(f - frames)
        return torch.swapaxes(t[:, time_idx:time_idx + frames], 0, 1)

    return torch.swapaxes(F.pad(t, (0, 0, 0, 0, 0, frames - f)), 0, 1)


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


# drivegan
# turn_right = [1.611762561798095703e-01, 1.254118710692751435e+01, 9.157459855079650879e-01,
#               1.597717523574829102e+00, -4.675447642803192139e-01, -1.249152183532714844e+01]
# turn_left = [1.611762561798095703e-01, 1.254118710692751435e+01, -2.157459855079650879e-01,
#              1.597717523574829102e+00, -4.675447642803192139e-01, -1.249152183532714844e+01]
# go_straight = [1.611762561798095703e-01, 1.254118710692751435e+01, 0.000000000000000000e+00,
#                1.597717523574829102e+00, -4.675447642803192139e-01, -1.249152183532714844e+01]

# customdata
turn_left = [0., 1., 0.]
go_straight = [0., 0., 1.]
turn_right = [1., 0., 0.]


class CarRacingDataset(Dataset):
    def __init__(self, folder, image_size, channels=3, frames_per_sample=16, exts=['gif']):
        super().__init__()
        self.folder = folder
        assert os.path.exists(self.folder)
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.cast_num_frames_fn = partial(cast_num_frames, frames=frames_per_sample)

        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor), torch.tensor(turn_right, dtype=torch.float)


# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration=200, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop,
                   optimize=optimize)
    return images


test_dataset = CarRacingDataset(folder=data_path,
                                frames_per_sample=config.data.num_frames_cond + config.data.num_frames,
                                image_size=config.data.image_size)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
conditional = config.data.num_frames_cond > 0

for batch, (X, y) in enumerate(test_loader):
    X = X.to(config.device)
    X = data_transform(config, X)
    cond = None
    if conditional:
        X, cond, _ = conditioning_fn(config, X)

    chain_total = 1
    mega_samples = torch.zeros(len(X), config.data.num_frames_cond + config.data.num_frames * chain_total,
                               config.data.channels, config.data.image_size, config.data.image_size,
                               device=config.device)

    for chain in tqdm(range(chain_total)):
        init_samples = torch.randn(len(X), config.data.channels * config.data.num_frames,
                                   config.data.image_size, config.data.image_size,
                                   device=config.device)
        all_samples = ddpm_sampler(init_samples, scorenet, cond=cond[:len(init_samples)],
                                   actions=y, n_steps_each=config.sampling.n_steps_each,
                                   step_lr=config.sampling.step_lr, just_beta=False,
                                   final_only=True, denoise=config.sampling.denoise,
                                   subsample_steps=50, verbose=True)

        sample = all_samples[-1].reshape(all_samples[-1].shape[0], config.data.num_frames, config.data.channels,
                                         config.data.image_size, config.data.image_size)
        sample = inverse_data_transform(config, sample)
        cond = cond.reshape(cond.shape[0], config.data.num_frames_cond, config.data.channels,
                            config.data.image_size, config.data.image_size)
        cond = inverse_data_transform(config, cond)

        # combined_sample =
        if chain == 0:
            mega_samples[:, :config.data.num_frames_cond + config.data.num_frames, :, :, :] = torch.cat((cond, sample),
                                                                                                        dim=1)
        else:
            mega_samples[:,
            config.data.num_frames_cond + config.data.num_frames * chain:config.data.num_frames_cond + config.data.num_frames * chain + config.data.num_frames,
            :, :, :] = sample

        cond = torch.cat((cond[:, 1:3, :, :, :], sample[:, -1, :, :, :].unsqueeze(dim=1)), dim=1)
        cond = cond.reshape(len(cond), -1, config.data.image_size, config.data.image_size)

    # concatenate X and combined_sample tensors
    for step in range(mega_samples.shape[0]):
        video_tensor_to_gif(torch.swapaxes(mega_samples[step], 0, 1),
                            path=os.path.join(save_path, f'sample{batch}{step}.gif'))
    print('saved')
    if batch > 12:
        break
