import glob
import os
import random
import shutil
import zipfile

import lightning as L
import numpy as np
import spectral.io.envi as envi
from torch.utils.data import DataLoader, random_split

from src.dataset.components.hyperspectral_dataset import HyperspectralDataset
from src.dataset.components.utils import (
    download_dataset,
    download_from_zenodo,
    load_dataset,
)


class BaseHyperspectralDataModule(L.LightningDataModule):
    """A foundational class for managing hyperspectral datasets within a PyTorch Lightning
    framework. This base class streamlines the data handling process, offering a structured
    approach to organizing, preparing, and utilizing hyperspectral data for deep learning
    applications.

    Attributes:
        data_dir (str): Directory where the dataset is stored or will be downloaded.
        dataset_name (str): Identifiable name of the dataset for organizational purposes.
        batch_size (int): Number of samples per batch.
        patch_size (int): Dimension of square patches extracted from hyperspectral images.
        transform (callable, optional): Optional transformations to apply to each data sample.
        hyperparams (dict, optional): Dictionary of hyperparameters specific to the dataset or modeling approach.

    Main Methods:
        train_dataloader(): Returns a DataLoader for the training set.
        val_dataloader(): Returns a DataLoader for the validation set.
        setup_datasets(img, gt, hyperparams): Prepares and splits the dataset into training and validation sets.

    Abstract Methods to be Implemented by Subclasses:
        prepare_data(): Contains logic for downloading, processing, and saving the dataset.
        setup(stage=None): Prepares the data for training/validation/testing stages.

    Example Usage:
        To utilize this base class, one must subclass it and provide implementations for the abstract methods.
        Below is a simplified example demonstrating how to create a custom data module for a
        new hyperspectral dataset:

        ```python
        class CustomHyperspectralDataModule(BaseHyperspectralDataModule):
            def prepare_data(self):
                # Custom logic to download/process the dataset
                pass

            def setup(self, stage=None):
                # Custom logic to prepare datasets for training/validation
                # Typically involves loading data, applying transforms, and splitting
                self.setup_datasets(self.image_data, self.annotation, self.hyperparams)
        ```

        Once the subclass is defined, it can be instantiated and used within a PyTorch Lightning training routine:

        ```python
        data_module = CustomHyperspectralDataModule(data_dir='/path/to/data', batch_size=32, patch_size=5)
        data_module.prepare_data()
        data_module.setup(stage='fit')
        train_loader = data_module.train_dataloader()
        ```
    """

    def __init__(self, data_dir, dataset_name, transform=None, hyperparams=None):
        """Initializes the BaseHyperspectralDataModule with the provided parameters.

        Args:
            data_dir (str): The directory path for the dataset.
            dataset_name (str): The name of the dataset.
            batch_size (int): The number of samples per batch.
            patch_size (int): The size of the patches to be extracted from hyperspectral images.
            transform (callable, optional): An optional function/transform to apply to the data.
            hyperparams (dict, optional): Additional hyperparameters for the dataset or model.
        """
        super().__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.hyperparams = hyperparams or {}

        # Set default values for hyperparameters if not specified
        self.hyperparams.setdefault("batch_size", 32)
        self.hyperparams.setdefault("patch_size", 5)
        self.hyperparams.setdefault("num_workers", 4)

        self.batch_size = self.hyperparams["batch_size"]

    def setup_datasets(self, img, gt, hyperparams):
        self.dataset = HyperspectralDataset(img, gt, transform=self.transform, **hyperparams)

        # Adjust the splitting to include test data
        train_size = int(0.7 * len(self.dataset))
        test_val_size = len(self.dataset) - train_size
        # Splitting the remaining into half for validation and test
        val_size = int(0.5 * test_val_size)
        test_size = test_val_size - val_size  # The rest goes into the test

        self.train_dataset, test_val_dataset = random_split(
            self.dataset, [train_size, test_val_size]
        )
        self.val_dataset, self.test_dataset = random_split(test_val_dataset, [val_size, test_size])

    def train_dataloader(self):
        """Creates a DataLoader for the training dataset.

        Returns:
            DataLoader: The DataLoader object for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            num_workers=self.hyperparams["num_workers"],
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        """Creates a DataLoader for the validation dataset.

        Returns:
            DataLoader: The DataLoader object for the validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            num_workers=self.hyperparams["num_workers"],
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        """Creates a DataLoader for the test dataset.

        Returns:
            DataLoader: The DataLoader object for the test dataset.
        """

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.hyperparams["num_workers"],
        )

        # pass

    def prepare_data(self):
        """Placeholder method for data preparation logic.

        This method should be implemented by subclasses to handle dataset-specific preparation
        steps such as downloading or extracting data.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def setup(self, stage=None):
        """Placeholder method for setup logic.

        This method should be implemented by subclasses to handle dataset-specific loading and
        preprocessing steps, such as creating datasets for training, validation, and testing.

        Args:
            stage (str, optional): The stage for which setup is being called. This could be 'fit', 'validate',
                                   'test', or 'predict'. The stage can be used to differentiate which datasets
                                   need to be setup in the current context.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
