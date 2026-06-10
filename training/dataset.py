import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.config import IMAGE_SIZE, MEAN, STD, CLASSES


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])


def get_dataloaders(
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4
) -> dict:
    data_path = Path(data_dir)
    loaders = {}

    for split in ["train", "val", "test"]:
        split_path = data_path / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Expected '{split_path}'."
                f"Download the Kaggle chest X-ray dataset and place it under '{data_dir}/'."
            )

        dataset = datasets.ImageFolder(
            root=str(split_path),
            transform=get_transforms(split)
        )

        shuffle = split == "train"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        loaders[split] = (loader, len(dataset))
        print(f"  [{split:5s}] {len(dataset):>5,} images | classes: {dataset.classes}")

    return loaders


if __name__ == '__main__':
    print("Loading datasets...")
    loaders = get_dataloaders()
    train_loader, n_train = loaders["train"]
    images, labels = next(iter(train_loader))
    print(f"\nSample batch - images: {images.shape}, labels: {labels.shape}")