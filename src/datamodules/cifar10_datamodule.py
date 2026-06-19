"""
CIFAR-10 DataModule for PyTorch Lightning

This module handles the CIFAR-10 dataset. Provides a 50,000/5,000 train/validation split
 and a test set of 10,000 images. It also includes data augmentation and normalization.
The split and the train dataloader are seeded for reproducibility by using a generator with 
a fixed seed.
"""

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        # Download CIFAR-10 dataset if not already downloaded
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if stage == 'fit' or stage is None:
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                CIFAR10(root=self.data_dir, train=True, transform=transform),
                [45000, 5000],
                generator=generator,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(root=self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# Test code to verify that the DataModule works as expected
if __name__ == "__main__":
    # Example usage
    data_module = CIFAR10DataModule()
    data_module.prepare_data()
    data_module.setup('fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")