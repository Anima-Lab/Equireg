from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from ..groups import TransformationGroup, get_group


__all__ = [
    "build_transform",
    "ImageFileDataset",
    "GroupAugmentedDataset",
    "build_dataset",
    "split_dataset",
    "limit_dataset",
    "wrap_with_group",
    "RotationDataset",
    "wrap_with_rotation",
    "build_contrastive_transform",
    "ContrastiveImageDataset",
    "ContrastiveMNISTDataset",
    "build_contrastive_dataset",
    "register_dataset",
    "get_dataset",
    "get_dataloader",
]


_DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _default_data_root() -> Path:
    return Path(os.getcwd()) / "data"


def _list_image_paths(root: Path, extensions: Iterable[str]) -> List[Path]:
    exts = tuple(ext.lower() for ext in extensions)
    paths = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    if not paths:
        raise FileNotFoundError(f"No images with extensions {extensions} found in {root}.")
    return paths


def build_transform(
    image_size: int,
    channels: int,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    norm_mean = mean if mean is not None else [0.5] * channels
    norm_std = std if std is not None else [0.5] * channels
    pipeline: List = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    pipeline.append(transforms.Normalize(norm_mean, norm_std))
    return transforms.Compose(pipeline)


class ImageFileDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose,
        channels: int = 3,
        extensions: Iterable[str] = _DEFAULT_EXTENSIONS,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.channels = channels
        self.mode = "RGB" if channels == 3 else "L"
        self.paths = _list_image_paths(self.root, extensions)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            img = img.convert(self.mode)
            tensor = self.transform(img)
        return tensor


class GroupAugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, group: TransformationGroup) -> None:
        self.dataset = dataset
        self.group = group

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.dataset[idx]
        if isinstance(sample, (tuple, list)):
            image = sample[0]
        else:
            image = sample
        op_idx = torch.randint(low=0, high=self.group.order, size=(1,)).item()
        augmented = self.group.apply_to_input(image, op_idx)
        return augmented, op_idx


def build_dataset(
    name: str,
    data_path: str | None,
    image_size: int,
    channels: int,
    split: str = "train",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Dataset:
    dataset_name = name.lower()
    target_dir = Path(data_path) if data_path else _default_data_root() / dataset_name
    transform = build_transform(image_size=image_size, channels=channels, mean=mean, std=std)

    if dataset_name == "mnist":
        if target_dir.exists():
            try:
                return ImageFileDataset(target_dir, transform=transform, channels=channels)
            except FileNotFoundError:
                pass
        is_train = split != "test"
        norm_mean = mean if mean is not None else [0.5] * channels
        norm_std = std if std is not None else [0.5] * channels
        mnist_transforms: List = [transforms.Resize(image_size), transforms.ToTensor()]
        if channels == 3:
            mnist_transforms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        mnist_transforms.append(transforms.Normalize(norm_mean, norm_std))
        dataset = datasets.MNIST(
            root=str(target_dir),
            train=is_train,
            download=True,
            transform=transforms.Compose(mnist_transforms),
        )
        return dataset

    if not target_dir.exists():
        raise FileNotFoundError(f"Dataset path {target_dir} does not exist.")
    return ImageFileDataset(target_dir, transform=transform, channels=channels)


def split_dataset(dataset: Dataset, val_fraction: float, seed: int = 0) -> Tuple[Dataset, Dataset | None]:
    if not 0 < val_fraction < 1:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    return train_ds, val_ds


def limit_dataset(
    dataset: Dataset,
    max_samples: int | None,
    seed: int = 0,
    strategy: str = "random",
) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    if strategy == "first":
        indices = list(range(max_samples))
    elif strategy == "random":
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    else:
        raise ValueError(f"Unknown limit strategy '{strategy}'. Expected 'random' or 'first'.")
    return Subset(dataset, indices)


def wrap_with_group(dataset: Dataset, group_name: str) -> GroupAugmentedDataset:
    group = get_group(group_name)
    return GroupAugmentedDataset(dataset, group)


class RotationDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.dataset[idx]
        if isinstance(sample, (tuple, list)):
            image = sample[0]
        else:
            image = sample
        rot_idx = torch.randint(0, 4, (1,)).item()
        rotated = torch.rot90(image, rot_idx, dims=(-2, -1))
        return rotated, rot_idx


def wrap_with_rotation(dataset: Dataset) -> RotationDataset:
    return RotationDataset(dataset)


def build_contrastive_transform(image_size: int, channels: int, mean: Sequence[float], std: Sequence[float]):
    rotation_ops = [transforms.RandomRotation((deg, deg)) for deg in (0, 90, 180, 270)]
    pipeline: List = [
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomChoice(rotation_ops),
        transforms.ToTensor(),
    ]
    pipeline.append(transforms.Normalize(mean, std))
    return transforms.Compose(pipeline)


class ContrastiveImageDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: transforms.Compose,
        channels: int = 3,
        extensions: Iterable[str] = _DEFAULT_EXTENSIONS,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.channels = channels
        self.mode = "RGB" if channels == 3 else "L"
        self.paths = _list_image_paths(self.root, extensions)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            img = img.convert(self.mode)
            view1 = self.transform(img)
            view2 = self.transform(img)
        return view1, view2


class ContrastiveMNISTDataset(Dataset):
    def __init__(self, base_dataset: datasets.MNIST, transform: transforms.Compose, channels: int):
        self.dataset = base_dataset
        self.transform = transform
        self.channels = channels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        if self.channels == 3 and view1.shape[0] == 1:
            view1 = view1.repeat(3, 1, 1)
            view2 = view2.repeat(3, 1, 1)
        return view1, view2


def build_contrastive_dataset(
    name: str,
    data_path: str | None,
    image_size: int,
    channels: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> Dataset:
    dataset_name = name.lower()
    transform = build_contrastive_transform(image_size, channels, mean, std)

    if dataset_name == "mnist":
        target_dir = Path(data_path) if data_path else _default_data_root() / dataset_name
        base_dataset = datasets.MNIST(root=str(target_dir), train=True, download=True, transform=None)
        return ContrastiveMNISTDataset(base_dataset, transform, channels)

    target_dir = Path(data_path) if data_path else _default_data_root() / dataset_name
    if not target_dir.exists():
        raise FileNotFoundError(f"Dataset path {target_dir} does not exist.")
    return ContrastiveImageDataset(target_dir, transform=transform, channels=channels)


# --- Registry helpers ported from src/equireg_data/dataloader.py ---

_REGISTERED_DATASETS: dict[str, type[Dataset]] = {}


def register_dataset(name: str):
    """Decorator used by legacy code to register dataset classes by name."""

    def wrapper(cls):
        key = name.lower()
        if key in _REGISTERED_DATASETS:
            raise NameError(f"Dataset '{name}' already registered.")
        _REGISTERED_DATASETS[key] = cls
        return cls

    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    key = (name or "").lower()
    cls = _REGISTERED_DATASETS.get(key)
    if cls is None:
        return ImageFileDataset(root=root, **kwargs)
    return cls(root=root, **kwargs)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    train: bool,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=torch.cuda.is_available(),
    )
