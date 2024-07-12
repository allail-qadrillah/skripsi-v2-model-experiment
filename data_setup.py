from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

NUM_WORKERS = os.cpu_count()
# NUM_WORKERS = 0

def create_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    transforms: transforms,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    
    # use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(train_dir, transform=transforms)
    val_data = datasets.ImageFolder(val_dir, transform=transforms)
    test_data = datasets.ImageFolder(test_dir, transform=transforms)

    # get class names
    class_names = train_data.classes

    # turn images into data loaders
    train_data_loaders = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    val_data_loaders = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_data_loaders = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_data_loaders, val_data_loaders, test_data_loaders, class_names

