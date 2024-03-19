import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.dataset.components.hyperspectral_dataset import HyperspectralDataset
from src.dataset.components.utils import download_dataset, load_dataset


class BaseHyperspectralDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, dataset_name, batch_size=32,
                 patch_size=5, transform=None, hyperparams=None):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.transform = transform
        self.hyperparams = hyperparams or {}

    # rest of the class remains the same

    def setup_datasets(self, img, gt, hyperparams):
        self.dataset = HyperspectralDataset(
            img, gt, transform=self.transform, **hyperparams)
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [int(0.8 * len(self.dataset)),
                           len(self.dataset) - int(0.8 * len(self.dataset))]
        )

    def setup(self, stage=None, hyperparams={}):
        if stage == 'fit' or stage is None:
            dataset_dir = os.path.join(self.data_dir, self.dataset_name)
            if not os.path.exists(dataset_dir):
                download_dataset(self.dataset_name, dataset_dir)

            img, gt, _, ignored_labels, _, _ = load_dataset(
                self.dataset_name, dataset_dir)
            self.setup_datasets(img, gt, hyperparams)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class SalinasDataModule(BaseHyperspectralDataModule):
    def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
        super().__init__(data_dir, "Salinas", batch_size,
                         patch_size, transform, hyperparams=kwargs)


class PaviaUDataModule(BaseHyperspectralDataModule):
    def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
        super().__init__(data_dir, "PaviaU", batch_size,
                         patch_size, transform, hyperparams=kwargs)
