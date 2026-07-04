"""
dataset_utils.py

Handles loading and preparing the EuroSAT dataset (Sentinel-2 land cover
patches) for fine-tuning a ResNet-50 classifier.

EuroSAT: 27,000 labeled 64x64 RGB image patches across 10 land-use /
land-cover classes, derived from Sentinel-2 satellite imagery.
Reference: Helber et al., 2019, "EuroSAT: A Novel Dataset and Deep
Learning Benchmark for Land Use and Land Cover Classification."

This mirrors the workflow used in NARSS-adjacent research (e.g. CNN-based
classification of EgyptSat-1 imagery into land cover classes such as
urban, vegetation, desert, water, soil, and roads).
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    resize_dim = 224
    if train:
        return transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_eurosat_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    download: bool = True,
    seed: int = 42,
):
    full_dataset = datasets.EuroSAT(
        root=data_dir, download=download, transform=get_transforms(train=True)
    )

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    val_set.dataset.transform = get_transforms(train=False)
    test_set.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, EUROSAT_CLASSES


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_eurosat_dataloaders(batch_size=8, download=True)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:8].tolist()}")
    print(f"Classes: {classes}")
