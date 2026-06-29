from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class TwoViewTransform:
    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def build_transforms(
    dataset: str,
    img_size: int,
    train: bool,
    random_erasing: float = 0.25,
) -> transforms.Compose:
    dataset = dataset.lower()
    if dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    elif dataset == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if train:
        ops = [
            transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0), ratio=(0.75, 1.3333)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if random_erasing > 0:
            ops.append(transforms.RandomErasing(p=random_erasing, value="random"))
        return transforms.Compose(ops)

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_dataloaders(
    dataset: str = "cifar100",
    data_dir: str = "data",
    img_size: int = 224,
    batch_size: int = 128,
    workers: int = 4,
    download: bool = True,
    random_erasing: float = 0.25,
    two_views: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    dataset = dataset.lower()
    if dataset == "cifar100":
        dataset_cls = datasets.CIFAR100
        num_classes = 100
    elif dataset == "cifar10":
        dataset_cls = datasets.CIFAR10
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_transform = build_transforms(dataset, img_size, train=True, random_erasing=random_erasing)
    if two_views:
        train_transform = TwoViewTransform(train_transform)
    val_transform = build_transforms(dataset, img_size, train=False)

    train_set = dataset_cls(root=data_dir, train=True, transform=train_transform, download=download)
    val_set = dataset_cls(root=data_dir, train=False, transform=val_transform, download=download)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader, num_classes


def build_eval_loader(
    dataset: str = "cifar100",
    data_dir: str = "data",
    img_size: int = 224,
    batch_size: int = 128,
    workers: int = 4,
    download: bool = True,
) -> Tuple[DataLoader, int]:
    dataset = dataset.lower()
    if dataset == "cifar100":
        dataset_cls = datasets.CIFAR100
        num_classes = 100
    elif dataset == "cifar10":
        dataset_cls = datasets.CIFAR10
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    val_set = dataset_cls(
        root=data_dir,
        train=False,
        transform=build_transforms(dataset, img_size, train=False),
        download=download,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return val_loader, num_classes
