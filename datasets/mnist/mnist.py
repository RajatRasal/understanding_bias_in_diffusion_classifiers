import os
import numpy as np
from PIL import Image, ImageColor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F

import torchvision
from torch import tensor
from torchvision import transforms
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F_vision
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset

from datasets.mnist.io import load_idx


class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        resize: int = 32,
    ) -> None:
        transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        super().__init__(root, transform=transform)
        self.train = train
        self.data, self.labels = self._load_data()

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = load_idx(os.path.join(self.root, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = load_idx(os.path.join(self.root, label_file))

        return data, targets
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target = int(self.labels[index]) 

        img = Image.fromarray(img, mode="L")
        img = self.transform(img)

        return img, {"y": target}

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


if __name__ == "__main__":
    textures_folder = MNIST(root="/vol/biomedic3/rrr2417/understanding_bias_in_diffusion_classifiers/datasets/mnist/raw/")
    dl = DataLoader(textures_folder, batch_size=20)
    for img, label in dl:
        print(img.shape, img.min(), img.max())
        break
    grid = torchvision.utils.make_grid(img, nrow=5, normalize=True, value_range=(-1, 1))
    grid_img = F_vision.to_pil_image(grid)
    grid_img.save("grid.png")
