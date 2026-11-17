"""Torchvision-transform subset used by the repository's PPU training path.

The host torchvision wheel is not ABI-compatible with the PPU PyTorch build;
the training path only needs these PIL/tensor transforms.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np
import torch
from PIL import Image


class InterpolationMode(Enum):
    NEAREST = Image.Resampling.NEAREST
    BILINEAR = Image.Resampling.BILINEAR
    BICUBIC = Image.Resampling.BICUBIC
    BOX = Image.Resampling.BOX
    HAMMING = Image.Resampling.HAMMING
    LANCZOS = Image.Resampling.LANCZOS


def _pil_interpolation(interpolation):
    return interpolation.value if isinstance(interpolation, InterpolationMode) else interpolation


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, image):
        if torch.rand((), device="cpu").item() < self.p:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class Resize:
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=True):
        self.size = size
        self.interpolation = _pil_interpolation(interpolation)
        self.max_size = max_size
        self.antialias = antialias

    def __call__(self, image):
        if isinstance(self.size, int):
            width, height = image.size
            short, long = (width, height) if width <= height else (height, width)
            if short == self.size:
                return image
            new_short = self.size
            new_long = int(self.size * long / short)
            size = (new_short, new_long) if width <= height else (new_long, new_short)
            if self.max_size is not None and max(size) > self.max_size:
                scale = self.max_size / max(size)
                size = (int(round(size[0] * scale)), int(round(size[1] * scale)))
        else:
            height, width = self.size
            size = (int(width), int(height))
        return image.resize(size, self.interpolation)


class CenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))

    def __call__(self, image):
        crop_h, crop_w = self.size
        width, height = image.size
        if width < crop_w or height < crop_h:
            image = image.resize((max(width, crop_w), max(height, crop_h)), Image.Resampling.BILINEAR)
            width, height = image.size
        left = int(round((width - crop_w) / 2.0))
        top = int(round((height - crop_h) / 2.0))
        return image.crop((left, top, left + crop_w, top + crop_h))


class ToTensor:
    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image
            if tensor.ndim == 3 and tensor.shape[0] not in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            return tensor.float().div(255.0) if not tensor.is_floating_point() else tensor
        array = np.asarray(image)
        if array.ndim == 2:
            array = array[:, :, None]
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor.float().div(255.0)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False):
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)
        self.inplace = inplace

    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return tensor.sub(mean).div(std)


def crop(image, top: int, left: int, height: int, width: int):
    return image.crop((left, top, left + width, top + height))


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class _Functional:
    crop = staticmethod(crop)


functional = _Functional()
