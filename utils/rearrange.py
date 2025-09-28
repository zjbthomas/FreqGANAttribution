import random

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

import torch

from PIL import Image
import numpy as np

class Rearrange(ImageOnlyTransform):
    def __init__(self, crop_size, always_apply=False, p=1.0):
        super(Rearrange, self).__init__(always_apply, p)
        self.crop_size = crop_size

    def apply(self, img, **params):
        C, H, W = img.shape
        assert H % self.crop_size == 0 and W % self.crop_size == 0 and H == W

        n_block = H // self.crop_size

        reshaped = torch.reshape(img, (C, n_block, self.crop_size, n_block, self.crop_size))
        permuted = torch.permute(reshaped, (0, 1, 3, 2, 4))
        blocks = torch.reshape(permuted, (C, n_block * n_block, self.crop_size, self.crop_size))

        return blocks

class RandomPick(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(RandomPick, self).__init__(always_apply, p)

    def apply(self, blocks, **params):
        C, N, S, S = blocks.shape

        pick = random.randint(0, N - 1)
        b = blocks[:, pick, :, :] # no need to squeeze

        return b

class Flatten(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Flatten, self).__init__(always_apply, p)

    def apply(self, blocks, **params):
        C, N, S, S = blocks.shape

        permuted = torch.permute(blocks, (1, 0, 2, 3))
        reshaped = torch.reshape(permuted, (C * N, S, S))

        return reshaped
    
class Avg(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(Avg, self).__init__(always_apply, p)

    def apply(self, blocks, **params):
        C, N, S, S = blocks.shape

        return torch.mean(blocks, dim = 1)