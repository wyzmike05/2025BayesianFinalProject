# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pathlib
import pickle
import zipfile
from typing import Union

import numpy as np
import requests
import torch
import torchvision
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from utils_model import quantize


def bin_mnist_transform(x):
    return torch.bernoulli(x.permute(1, 2, 0).contiguous()).int()


def rgb_image_transform(x, num_bins=256):
    return quantize((x * 2) - 1, num_bins).permute(1, 2, 0).contiguous()


class MyLambda(torchvision.transforms.Lambda):
    def __init__(self, lambd, arg1):
        super().__init__(lambd)
        self.arg1 = arg1

    def __call__(self, x):
        return self.lambd(x, self.arg1)


class MNIST(torchvision.datasets.MNIST):
    def __getitem__(self, idx):
        return super().__getitem__(idx)


def make_datasets(cfg: DictConfig) -> tuple[Dataset, Dataset, Dataset]:
    """
    Mandatory keys: dataset (must be cifar10, mnist, bin_mnist, bin_mnist_cts or text8), data_dir
    Optional for vision: num_bins (default 256), val_frac (default 0.01), horizontal_flip (default: False)
    Mandatory for text: seq_len
    """
    if cfg.dataset == "bin_mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(bin_mnist_transform)]
        )
        train_set = MNIST(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
        val_set = MNIST(
            root=cfg.data_dir, train=True, download=True, transform=transform
        )
        test_set = MNIST(
            root=cfg.data_dir, train=False, download=True, transform=transform
        )

    if cfg.dataset != "text8":
        # For vision datasets we split the train set into train and val
        val_frac = cfg.get("val_frac", 0.01)
        train_val_split = [1.0 - val_frac, val_frac]
        seed = 2147483647
        train_set = random_split(
            train_set, train_val_split, generator=torch.Generator().manual_seed(seed)
        )[0]
        val_set = random_split(
            val_set, train_val_split, generator=torch.Generator().manual_seed(seed)
        )[1]

    return train_set, val_set, test_set


def batch_to_images(image_batch: torch.Tensor, ncols: int = None) -> plt.Figure:
    if ncols is None:
        ncols = math.ceil(math.sqrt(len(image_batch)))
    if image_batch.size(-1) == 3:  # for color images (CIFAR-10)
        image_batch = (image_batch + 1) / 2
    grid = make_grid(image_batch.permute(0, 3, 1, 2), ncols, pad_value=1).permute(
        1, 2, 0
    )
    fig = plt.figure(figsize=(grid.size(1) / 30, grid.size(0) / 30))
    plt.imshow(grid.cpu().clip(min=0, max=1), interpolation="nearest")
    plt.grid(False)
    plt.axis("off")
    return fig
