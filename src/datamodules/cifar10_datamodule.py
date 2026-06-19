"""
CIFAR-10 DataModule for PyTorch Lightning

This module handles the CIFAR-10 dataset. Provides a 45,000/5,000 train/validation split
and a test set of 10,000 images. It also includes normalization transforms for the dataset.
The split and the train dataloader are seeded for reproducibility by using a generator with
a fixed seed.

Uses a mirror URL to download the dataset to avoid issues with the original source.

TODO:
- Add support for augmentations. Pass the configuration or the transforms. To be implemented
when the augmentations are defined in the config.
"""

import hashlib
import os
import tarfile
import urllib.request

import lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(L.LightningDataModule):
    download_url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
    checksum = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
    expected_batch_md5 = {
        "data_batch_1": "c99cafc152244af753f735de768cd75f",
        "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
        "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
        "data_batch_4": "634d18415352ddfa80567beed471001a",
        "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
        "test_batch": "40351d587109b95175f43aff81a1287e",
    }

    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        """
        Download and verify the CIFAR-10 dataset using a mirror URL.

        1. If extracted files already exist, verify expected batch-file MD5 hashes.
        2. Otherwise download the archive from the mirror.
        3. Remove partial archive file if download fails.
        4. Verify downloaded archive SHA-256 before extraction.
        5. Extract the archive, remove it, then verify extracted batch-file MD5 hashes.
        """
        os.makedirs(self.data_dir, exist_ok=True)

        # Check if existing files match the expected MD5 hashes
        extracted_dir = os.path.join(self.data_dir, "cifar-10-batches-py")
        if os.path.isdir(extracted_dir):
            self._verify_extracted_files(extracted_dir)
            return

        # Download the archive from the mirror URL
        tar_path = os.path.join(self.data_dir, "cifar-10-python.tar.gz")

        print("Downloading CIFAR-10 from mirror...")
        try:
            urllib.request.urlretrieve(self.download_url, tar_path)
        except Exception:
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise

        # Verify archive integrity before extracting it.
        file_hash = hashlib.sha256()
        with open(tar_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                file_hash.update(chunk)

        downloaded_checksum = file_hash.hexdigest()
        if downloaded_checksum != self.checksum:
            os.remove(tar_path)
            raise ValueError(
                f"Checksum mismatch for CIFAR-10 archive: expected {self.checksum}, got {downloaded_checksum}"
            )

        # Extract the archive and verify the extracted files
        print("Extracting files...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir)

        os.remove(tar_path)
        self._verify_extracted_files(extracted_dir)
        print("CIFAR-10 dataset ready")

    @staticmethod
    def _md5sum(file_path: str) -> str:
        file_hash = hashlib.md5()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def _verify_extracted_files(self, extracted_dir: str) -> None:
        for filename, expected_md5 in self.expected_batch_md5.items():
            file_path = os.path.join(extracted_dir, filename)
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Missing CIFAR-10 file: {file_path}")

            actual_md5 = self._md5sum(file_path)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"MD5 mismatch for {filename}: expected {expected_md5}, got {actual_md5}"
                )

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        if stage == 'fit' or stage == 'validate' or stage is None:
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
