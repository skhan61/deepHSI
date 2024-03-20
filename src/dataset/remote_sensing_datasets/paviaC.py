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
from src.dataset.hyperspectral_datamodule import BaseHyperspectralDataModule


class PaviaCDataModule(BaseHyperspectralDataModule):
    def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
        super().__init__(data_dir, "PaviaU", batch_size,
                         patch_size, transform, hyperparams=kwargs)

    def prepare_data(self):
        # Check if the dataset directory exists
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"Dataset '{self.dataset_name}' not found. Downloading...")
            os.makedirs(dataset_dir, exist_ok=True)
            download_dataset(self.dataset_name, dataset_dir)
        else:
            print(f"Dataset '{
                  self.dataset_name}' already exists. Skipping download.")

    def setup(self, stage=None):
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        img, gt, _, ignored_labels, _, _ = load_dataset(
            self.dataset_name, dataset_dir)
        self.setup_datasets(img, gt, self.hyperparams)
