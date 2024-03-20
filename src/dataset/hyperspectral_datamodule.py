import glob
import os
import random
import shutil
import zipfile

import numpy as np
import pytorch_lightning as pl
import spectral.io.envi as envi
from torch.utils.data import DataLoader, random_split

from src.dataset.components.hyperspectral_dataset import HyperspectralDataset
from src.dataset.components.utils import (download_dataset,
                                          download_from_zenodo, load_dataset)


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

    def setup_datasets(self, img, gt, hyperparams):
        self.dataset = HyperspectralDataset(
            img, gt, transform=self.transform, **hyperparams)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [int(
            0.8 * len(self.dataset)), len(self.dataset) - int(0.8 * len(self.dataset))])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def prepare_data(self):
        # Indicate that subclasses should implement this method
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def setup(self, stage=None):
        # Indicate that subclasses should implement this method
        raise NotImplementedError(
            "This method should be implemented by subclasses.")
