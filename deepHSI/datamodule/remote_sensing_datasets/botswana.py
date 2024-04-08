import glob
import os
import random
import shutil
import zipfile

import numpy as np
import pytorch_lightning as pl
import spectral.io.envi as envi
from torch.utils.data import DataLoader, random_split

from deepHSI.datamodule import (BaseHyperspectralDataModule,
                                HyperspectralDataset)
from deepHSI.datamodule.components.utils import (download_dataset,
                                                 download_from_zenodo,
                                                 load_dataset)


class BotswanaDataModule(BaseHyperspectralDataModule):
    """
    A PyTorch Lightning DataModule for the Botswana hyperspectral image classification dataset.

    The Botswana dataset consists of hyperspectral remote sensing images acquired by the NASA EO-1 satellite
    over the Okavango Delta, Botswana. The dataset features a diverse set of land cover types and has been
    preprocessed to mitigate various sensor-related anomalies, resulting in 145 spectral bands suitable for analysis.

    Attributes:
        data_dir (str): The directory where the dataset is stored or will be downloaded.
        batch_size (int): The size of batches to be used during training.
        patch_size (int): The size of the patches to be extracted from hyperspectral images.
        transform (callable, optional): Optional transform to be applied on a sample.
        hyperparams (dict, optional): Additional hyperparameters for dataset handling.

    Methods:
        prepare_data: Downloads the dataset to the specified directory if it does not already exist.
        setup: Prepares the dataset for use in the model by loading the images, applying any specified transforms,
               and splitting into training and validation sets.

    Dataset Source:
        The dataset was acquired by the NASA EO-1 satellite's Hyperion sensor over the Okavango Delta, Botswana.
        It has been preprocessed by the UT Center for Space Research to address sensor-related anomalies.

    Land Cover Types:
        The dataset includes observations from 14 land cover types in seasonal swamps, occasional swamps, and drier woodlands.

    Usage Example:
        ```python
        botswana_data_module = BotswanaDataModule(data_dir='/path/to/data', batch_size=32, patch_size=5)
        botswana_data_module.prepare_data()
        botswana_data_module.setup(stage='fit')
        train_loader = botswana_data_module.train_dataloader()
        ```
    """

    def __init__(self, data_dir, transform=None, hyperparams=None):
        """
        Initializes the BotswanaDataModule with the provided parameters.

        Args:
            data_dir (str): The directory path for the dataset.
            transform (callable, optional): An optional function/transform to apply to the data.
            hyperparams (dict, optional): Dictionary containing additional hyperparameters for the dataset or model,
                                          including 'batch_size', 'patch_size', and 'num_workers'.
        """
        super().__init__(data_dir, "Botswana",
                         transform=transform, hyperparams=hyperparams)

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
