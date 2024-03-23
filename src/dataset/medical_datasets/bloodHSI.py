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


class BloodDetectionHSIDataModule(BaseHyperspectralDataModule):
    """
    A data module for handling the "Blood Detection in Hyperspectral Images" dataset within a
    PyTorch Lightning workflow. This dataset is aimed at facilitating the development of
    machine learning algorithms for hyperspectral blood detection.

    The dataset consists of 14 hyperspectral images capturing mock-up scenes with blood and
    visually similar substances under varying conditions. It serves as a challenging testbed for
    hyperspectral image analysis due to the high dimensionality and complexity of the data.

    Creators:
        - Michał Romaszewski
        - Przemysław Głomb
        - Arkadiusz Sochan
        - Michał Cholewa

    Dataset Details:
        - Images are provided in ENVI format.
        - The dataset includes annotations for pixels where blood and similar substances are visible.
        - Variations in background composition and lighting intensity are present across different images.

    Reference:
        This dataset and its use are documented in the following publications:
        - Preprint: arXiv:2008.10254 (https://arxiv.org/abs/2008.10254)
        - Journal article: Sensors 2020, 20(22), 6666; https://doi.org/10.3390/s20226666
        - Journal article: Forensic Science International, Volume 319, April 2021, 110701;
          https://doi.org/10.1016/j.forsciint.2021.110701

    Dataset DOI: 10.5281/zenodo.3984905

    Args:
        data_dir (str): The directory path where the dataset is stored or will be downloaded to.
        doi (str): The DOI of the dataset for downloading using Zenodo.
        batch_size (int): The number of samples per batch to load.
        patch_size (int): The size of the patches to be extracted from the hyperspectral images.
        transform (callable, optional): Optional transform to be applied on a sample.
        selected_image (str, optional): The specific image to be loaded from the dataset. If not specified,
                                        a random image from the predefined list will be chosen.
        **kwargs: Additional hyperparameters relevant to the dataset or model.

    Methods:
        prepare_data: Implements the data preparation logic including dataset downloading and extraction.
        setup: Sets up the data module by loading the selected hyperspectral image and its annotations
               for subsequent processing.
        load_image_and_annotation: Loads a specific hyperspectral image and its corresponding annotations
                                   from the dataset.

    Usage:
        To use this data module, instantiate it with the desired configuration and call the `prepare_data`
        method followed by the `setup` method. After setup, the `train_dataloader` and `val_dataloader`
        methods can be used to obtain DataLoaders for training and validation.

    Example:
        >>> dm = BloodDetectionHSIDataModule(data_dir='/path/to/data', doi='10.5281/zenodo.3984905',
        ...                                  batch_size=32, patch_size=5, selected_image='A_1')
        >>> dm.prepare_data()
        >>> dm.setup(stage='fit')
        >>> train_loader = dm.train_dataloader()
        >>> val_loader = dm.val_dataloader()
    """

    IMAGES = ['A_1', 'B_1', 'C_1', 'D_1', 'E_1', 'E_7', 'E_21',
              'F_1', 'F_1a', 'F_1s', 'F_2', 'F_2k', 'F_7', 'F_21']

    def __init__(self, data_dir, doi, transform=None, selected_image=None, **kwargs):
        self.dataset_name = "BloodDetectionHSI"
        super().__init__(data_dir, self.dataset_name,
                         transform=transform, hyperparams=kwargs)
        self.doi = doi
        self.selected_image = selected_image

    def prepare_data(self):
        dataset_dir = os.path.join(self.data_dir, self.dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"Creating directory for dataset '{
                  self.dataset_name}' at {dataset_dir}")
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"Downloading dataset '{self.dataset_name}'...")
            download_from_zenodo(self.doi, dataset_dir)
            self.extract_and_cleanup(dataset_dir)
        else:
            print(f"Dataset '{self.dataset_name}' already exists at {
                  dataset_dir}. Checking for ZIP files...")
            # Ensure handling of ZIP files
            self.extract_and_cleanup(dataset_dir)

    def extract_and_cleanup(self, dataset_dir):
        # Handle ZIP files
        zip_path = os.path.join(dataset_dir, "HyperBlood.zip")
        if os.path.exists(zip_path):
            print(f"Found ZIP file: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    print(f"Extracting {zip_path} to {dataset_dir}")
                    zip_ref.extractall(dataset_dir)
                os.remove(zip_path)
                print(f"Deleted ZIP file {zip_path}")
            except zipfile.BadZipFile:
                print(
                    f"Error: {zip_path} is a bad ZIP file and cannot be extracted.")
            except Exception as e:
                print(f"An error occurred while extracting {zip_path}: {e}")

        # Cleanup non-directory items (like the md5sums.txt)
        for item in os.listdir(dataset_dir):
            item_path = os.path.join(dataset_dir, item)
            if not os.path.isdir(item_path):
                print(f"Deleting non-directory item: {item_path}")
                os.remove(item_path)

        self.list_directory_contents(dataset_dir)

    def list_directory_contents(self, dir_path):
        print(f"Contents of {dir_path}:")
        for item in os.listdir(dir_path):
            print(f" - {item}")

    def setup(self, stage=None):
        # Added "HyperBlood" here
        dataset_dir = os.path.join(
            self.data_dir, self.dataset_name, "HyperBlood")
        data_dir = os.path.join(dataset_dir, "data")
        anno_dir = os.path.join(dataset_dir, "anno")

        # If a specific image is selected, use it; otherwise,
        # pick one randomly from the list
        image_to_load = self.selected_image if \
            self.selected_image in self.IMAGES else random.choice(
                self.IMAGES)

        # # This should now print the correct path including "HyperBlood"
        # print(data_dir)
        # # This should now print the correct path including "HyperBlood"
        # print(anno_dir)

        self.image_data, self.annotation = self.load_image_and_annotation(
            data_dir, anno_dir, image_to_load)

        self.setup_datasets(self.image_data, self.annotation, self.hyperparams)

    def load_image_and_annotation(self, data_dir, anno_dir, image_name):
        # Construct file paths for the hyperspectral image and its annotation
        hs_file = os.path.join(data_dir, f"{image_name}.float")
        hdr_file_path = os.path.join(data_dir, f"{image_name}.hdr")
        anno_file = os.path.join(anno_dir, f"{image_name}.npz")

        # # Print out the expected file paths for verification
        # print(f"Expected hyperspectral image file: {hs_file}")
        # print(f"Expected header file: {hdr_file_path}")
        # print(f"Expected annotation file: {anno_file}")

        # Check existence of files and print results
        hs_file_exists = os.path.exists(hs_file)
        hdr_file_exists = os.path.exists(hdr_file_path)
        anno_file_exists = os.path.exists(anno_file)
        # print(f"Hyperspectral image file exists: {hs_file_exists}")
        # print(f"Header file exists: {hdr_file_exists}")
        # print(f"Annotation file exists: {anno_file_exists}")

        # Load the hyperspectral image
        if hs_file_exists and hdr_file_exists:
            hs_image = envi.open(hdr_file_path, hs_file)
            image_data = hs_image.load()
            print(f"Loaded hyperspectral image data.")
        else:
            raise FileNotFoundError(
                f"Hyperspectral image or header file not found for {image_name}")

        # Load the annotation
        if anno_file_exists:
            with np.load(anno_file) as data:
                annotation = next(iter(data.values()))
            print(f"Loaded annotation data.")
        else:
            raise FileNotFoundError(
                f"Annotation file not found for {image_name}")

        return image_data, annotation
