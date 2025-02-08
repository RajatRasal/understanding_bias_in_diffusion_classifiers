from itertools import cycle
from typing import Literal

from torch.utils.data import DataLoader

from datasets.mnist.mnist import MNIST


def load_data(
    dataset: Literal["mnist"],
    batch_size: int = 32,
    image_size: int = 32,
    train: bool = True,
):
    if dataset == "mnist":
        root = "/vol/biomedic3/rrr2417/understanding_bias_in_diffusion_classifiers/datasets/mnist/raw/"
        train = MNIST(root=root, train=train, resize=image_size)
        n_classes = 10
        n_channels = 1
    else:
        raise NotImplementedError

    return cycle(DataLoader(train, batch_size=batch_size, shuffle=train)), n_classes, n_channels
