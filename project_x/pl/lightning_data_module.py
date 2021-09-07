from typing import Optional
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()
import pytorch_lightning as pl
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

class LitDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = (3, 32, 32)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.num_classes = 10

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset_full = CIFAR10(self.data_dir, train=True,
                                   transform=self.transform)
            self.dataset_train, self.dataset_val = random_split(
                dataset_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.dataset_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)