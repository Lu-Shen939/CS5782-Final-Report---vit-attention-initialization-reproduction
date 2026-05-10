import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASET_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}


def _subset(dataset, fraction: float, seed: int):
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("subset fraction must be in (0, 1].")
    count = max(1, int(len(dataset) * fraction))
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:count])


def build_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    subset_fraction: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")
    mean, std = DATASET_STATS[dataset_name]

    train_tfms = [
        transforms.RandomCrop(32, padding=4),
    ]
    if dataset_name != "svhn":
        train_tfms.append(transforms.RandomHorizontalFlip())
    train_tfms.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_tfms = transforms.Compose(train_tfms)

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tfms)
        test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tfms)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_set = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tfms)
        test_set = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tfms)
        num_classes = 100
    else:
        train_set = datasets.SVHN(data_root, split="train", download=True, transform=train_tfms)
        test_set = datasets.SVHN(data_root, split="test", download=True, transform=test_tfms)
        num_classes = 10

    train_set = _subset(train_set, subset_fraction, seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader, num_classes
