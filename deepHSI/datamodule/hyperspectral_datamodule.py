import glob
import os
import random
import shutil
import zipfile

import lightning as L
import numpy as np
import spectral.io.envi as envi
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# from deepHSI.datamodule import HyperspectralDataset
from deepHSI.datamodule.components.utils import (
    download_dataset,
    download_from_zenodo,
    load_dataset,
)

#    """Generic class for a hyperspectral scene."""


class HyperspectralDataset(Dataset):
    """Provides a PyTorch dataset for hyperspectral images, facilitating the extraction and use of
    spatial patches and their corresponding labels for model training and evaluation.

    Hyperspectral images are 3D data cubes with two spatial dimensions and one spectral dimension.
    This class enables the extraction of small 3D patches from these data cubes for deep learning purposes,
    particularly useful in remote sensing applications.

    Args:
        data (np.ndarray): The hyperspectral image data as a 3D array of shape (Height, Width, Bands),
                           where Bands is the number of spectral bands.
        gt (np.ndarray): The ground truth labels as a 2D array of shape (Height, Width), where each
                         element represents the class label of the corresponding pixel in 'data'.
        transform (callable, optional): An optional function/transform that takes in a 3D patch and
                                        returns a transformed version. This is useful for data augmentation.
        **hyperparams (dict): A dictionary containing additional hyperparameters:
            - patch_size (int): The size of the square patches to be extracted from the hyperspectral image.
                                Defaults to 5.
            - ignored_labels (list): A list of label values to ignore during training and evaluation. Defaults to [].
            - center_pixel (bool): If True, the label for a patch is determined solely by the label of its center pixel.
                                   Defaults to True.
            - supervision (str): Specifies the supervision mode - 'full' for fully supervised learning where only
                                 pixels with labels not in ignored_labels are used, and 'semi' for semi-supervised
                                 learning where all pixels are used. Defaults to 'full'.

    Attributes:
        data (np.ndarray): Stored hyperspectral image data.
        label (np.ndarray): Stored ground truth labels.
        patch_size (int): The size of the patches to be extracted.
        ignored_labels (set): Set of labels to be ignored.
        center_pixel (bool): Flag indicating whether to use only the center pixel label for patches.
        indices (list): List of valid pixel indices for patch extraction.
        labels (list): List of labels corresponding to the patches extracted at 'indices'.

    Example:
        >>> dataset = HyperspectralDataset(data, gt, patch_size=7, ignored_labels=[0], center_pixel=True)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for patches, labels in dataloader:
        >>>     # Forward pass through your model
        >>>     outputs = model(patches)
        >>>     # Compute loss and backpropagate
        >>>     loss = criterion(outputs, labels)
        >>>     loss.backward()

    Note:
        This dataset class is intended for use with PyTorch models and data loaders. It handles the
        preprocessing of hyperspectral data into suitable forms for Convolutional Neural Networks (CNNs)
        and other deep learning models.
    """

    def __init__(self, data, gt, transform=None, **hyperparams):
        super().__init__()

        self.data = data
        self.label = gt

        # self.name = hyperparams.get("dataset", "Unknown")

        self.patch_size = hyperparams.get("patch_size", 5)

        self.ignored_labels = set(hyperparams.get("ignored_labels", []))

        self.center_pixel = hyperparams.get("center_pixel", True)

        supervision = hyperparams.get("supervision", "full")

        # self.transform = transform

        # Handling transform
        if isinstance(transform, str) and transform.lower() != "none":
            module_name, func_name = transform.rsplit(".", 1)
            module = importlib.import_module(module_name)
            # print()
            # print(module)
            # print()
            self.transform = getattr(module, func_name)
        else:
            self.transform = None  # Set to None if not specified or explicitly set to null in YAML

        # Fully supervised: use all pixels with labels not in ignored_labels
        if supervision == "full":
            mask = np.ones_like(gt, dtype=bool)
            for l in self.ignored_labels:
                mask[gt == l] = 0

        # Semi-supervised: use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt, dtype=bool)

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = [
            (x, y)
            for x, y in zip(x_pos, y_pos)
            if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
        ]
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Ensure consistent shape for all patch sizes
        if self.patch_size == 1:
            # Squeeze out the single spatial dimension but keep the channel dimension
            data = data.squeeze(axis=(0, 1))  # Shape (C,)
            data = np.expand_dims(data, axis=(1, 2))  # Shape (C, 1, 1)
            label = label.item()  # Extract scalar value
        else:
            # Rearrange dimensions to CxHxW
            data = np.copy(data).transpose((2, 0, 1))

        # Convert the data to PyTorch tensors
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            data = self.transform(data)

        # If center_pixel is True, return the label of the center pixel
        if self.center_pixel and self.patch_size > 1:
            # Extract the label of the center pixel
            center = self.patch_size // 2
            label = label[center, center].item()  # Convert to scalar
        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label


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
