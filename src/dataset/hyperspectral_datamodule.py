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


# class BloodDetectionHSIDataModule(BaseHyperspectralDataModule):
#     IMAGES = ['A_1', 'B_1', 'C_1', 'D_1', 'E_1', 'E_7', 'E_21',
#               'F_1', 'F_1a', 'F_1s', 'F_2', 'F_2k', 'F_7', 'F_21']

#     def __init__(self, data_dir, doi, batch_size=32,
#                  patch_size=5, transform=None, selected_image=None, **kwargs):
#         self.dataset_name = "BloodDetectionHSI"
#         super().__init__(data_dir, self.dataset_name, batch_size,
#                          patch_size, transform, hyperparams=kwargs)
#         self.doi = doi
#         self.selected_image = selected_image

#     def prepare_data(self):
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         if not os.path.exists(dataset_dir):
#             print(f"Creating directory for dataset '{
#                   self.dataset_name}' at {dataset_dir}")
#             os.makedirs(dataset_dir, exist_ok=True)
#             print(f"Downloading dataset '{self.dataset_name}'...")
#             download_from_zenodo(self.doi, dataset_dir)
#             self.extract_and_cleanup(dataset_dir)
#         else:
#             print(f"Dataset '{self.dataset_name}' already exists at {
#                   dataset_dir}. Checking for ZIP files...")
#             # Ensure handling of ZIP files
#             self.extract_and_cleanup(dataset_dir)

#     def extract_and_cleanup(self, dataset_dir):
#         # Handle ZIP files
#         zip_path = os.path.join(dataset_dir, "HyperBlood.zip")
#         if os.path.exists(zip_path):
#             print(f"Found ZIP file: {zip_path}")
#             try:
#                 with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                     print(f"Extracting {zip_path} to {dataset_dir}")
#                     zip_ref.extractall(dataset_dir)
#                 os.remove(zip_path)
#                 print(f"Deleted ZIP file {zip_path}")
#             except zipfile.BadZipFile:
#                 print(
#                     f"Error: {zip_path} is a bad ZIP file and cannot be extracted.")
#             except Exception as e:
#                 print(f"An error occurred while extracting {zip_path}: {e}")

#         # Cleanup non-directory items (like the md5sums.txt)
#         for item in os.listdir(dataset_dir):
#             item_path = os.path.join(dataset_dir, item)
#             if not os.path.isdir(item_path):
#                 print(f"Deleting non-directory item: {item_path}")
#                 os.remove(item_path)

#         self.list_directory_contents(dataset_dir)

#     def list_directory_contents(self, dir_path):
#         print(f"Contents of {dir_path}:")
#         for item in os.listdir(dir_path):
#             print(f" - {item}")

#     def setup(self, stage=None):
#         # Added "HyperBlood" here
#         dataset_dir = os.path.join(
#             self.data_dir, self.dataset_name, "HyperBlood")
#         data_dir = os.path.join(dataset_dir, "data")
#         anno_dir = os.path.join(dataset_dir, "anno")

#         # If a specific image is selected, use it; otherwise,
#         # pick one randomly from the list
#         image_to_load = self.selected_image if \
#             self.selected_image in self.IMAGES else random.choice(
#                 self.IMAGES)

#         # # This should now print the correct path including "HyperBlood"
#         # print(data_dir)
#         # # This should now print the correct path including "HyperBlood"
#         # print(anno_dir)

#         self.image_data, self.annotation = self.load_image_and_annotation(
#             data_dir, anno_dir, image_to_load)

#         self.setup_datasets(self.image_data, self.annotation, self.hyperparams)

#     def load_image_and_annotation(self, data_dir, anno_dir, image_name):
#         # Construct file paths for the hyperspectral image and its annotation
#         hs_file = os.path.join(data_dir, f"{image_name}.float")
#         hdr_file_path = os.path.join(data_dir, f"{image_name}.hdr")
#         anno_file = os.path.join(anno_dir, f"{image_name}.npz")

#         # # Print out the expected file paths for verification
#         # print(f"Expected hyperspectral image file: {hs_file}")
#         # print(f"Expected header file: {hdr_file_path}")
#         # print(f"Expected annotation file: {anno_file}")

#         # Check existence of files and print results
#         hs_file_exists = os.path.exists(hs_file)
#         hdr_file_exists = os.path.exists(hdr_file_path)
#         anno_file_exists = os.path.exists(anno_file)
#         # print(f"Hyperspectral image file exists: {hs_file_exists}")
#         # print(f"Header file exists: {hdr_file_exists}")
#         # print(f"Annotation file exists: {anno_file_exists}")

#         # Load the hyperspectral image
#         if hs_file_exists and hdr_file_exists:
#             hs_image = envi.open(hdr_file_path, hs_file)
#             image_data = hs_image.load()
#             print(f"Loaded hyperspectral image data.")
#         else:
#             raise FileNotFoundError(
#                 f"Hyperspectral image or header file not found for {image_name}")

#         # Load the annotation
#         if anno_file_exists:
#             with np.load(anno_file) as data:
#                 annotation = next(iter(data.values()))
#             print(f"Loaded annotation data.")
#         else:
#             raise FileNotFoundError(
#                 f"Annotation file not found for {image_name}")

#         return image_data, annotation


# class SalinasDataModule(BaseHyperspectralDataModule):
#     def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
#         super().__init__(data_dir, "Salinas", batch_size,
#                          patch_size, transform, hyperparams=kwargs)

#     def prepare_data(self):
#         # Check if the dataset directory exists
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         if not os.path.exists(dataset_dir):
#             print(f"Dataset '{self.dataset_name}' not found. Downloading...")
#             os.makedirs(dataset_dir, exist_ok=True)
#             download_dataset(self.dataset_name, dataset_dir)
#         else:
#             print(f"Dataset '{
#                   self.dataset_name}' already exists. Skipping download.")

#     def setup(self, stage=None):
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         img, gt, _, ignored_labels, _, _ = load_dataset(
#             self.dataset_name, dataset_dir)
#         self.setup_datasets(img, gt, self.hyperparams)


# class PaviaUDataModule(BaseHyperspectralDataModule):
#     def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
#         super().__init__(data_dir, "PaviaU", batch_size,
#                          patch_size, transform, hyperparams=kwargs)

#     def prepare_data(self):
#         # Check if the dataset directory exists
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         if not os.path.exists(dataset_dir):
#             print(f"Dataset '{self.dataset_name}' not found. Downloading...")
#             os.makedirs(dataset_dir, exist_ok=True)
#             download_dataset(self.dataset_name, dataset_dir)
#         else:
#             print(f"Dataset '{
#                   self.dataset_name}' already exists. Skipping download.")

#     def setup(self, stage=None):
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         img, gt, _, ignored_labels, _, _ = load_dataset(
#             self.dataset_name, dataset_dir)
#         self.setup_datasets(img, gt, self.hyperparams)


# class PaviaCDataModule(BaseHyperspectralDataModule):
#     def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
#         super().__init__(data_dir, "PaviaC", batch_size,
#                          patch_size, transform, hyperparams=kwargs)

#     def prepare_data(self):
#         # Check if the dataset directory exists
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         if not os.path.exists(dataset_dir):
#             print(f"Dataset '{self.dataset_name}' not found. Downloading...")
#             os.makedirs(dataset_dir, exist_ok=True)
#             download_dataset(self.dataset_name, dataset_dir)
#         else:
#             print(f"Dataset '{
#                   self.dataset_name}' already exists. Skipping download.")

#     def setup(self, stage=None):
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         img, gt, _, ignored_labels, _, _ = load_dataset(
#             self.dataset_name, dataset_dir)
#         self.setup_datasets(img, gt, self.hyperparams)


# class KSCDataModule(BaseHyperspectralDataModule):
#     def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
#         super().__init__(data_dir, "KSC", batch_size,
#                          patch_size, transform, hyperparams=kwargs)

#     def prepare_data(self):
#         # Check if the dataset directory exists
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         if not os.path.exists(dataset_dir):
#             print(f"Dataset '{self.dataset_name}' not found. Downloading...")
#             os.makedirs(dataset_dir, exist_ok=True)
#             download_dataset(self.dataset_name, dataset_dir)
#         else:
#             print(f"Dataset '{
#                   self.dataset_name}' already exists. Skipping download.")

#     def setup(self, stage=None):
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         img, gt, _, ignored_labels, _, _ = load_dataset(
#             self.dataset_name, dataset_dir)
#         self.setup_datasets(img, gt, self.hyperparams)


# class IndianPinesDataModule(BaseHyperspectralDataModule):
    # def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
    #     super().__init__(data_dir, "IndianPines", batch_size,
    #                      patch_size, transform, hyperparams=kwargs)

    # def prepare_data(self):
    #     # Check if the dataset directory exists
    #     dataset_dir = os.path.join(self.data_dir, self.dataset_name)
    #     if not os.path.exists(dataset_dir):
    #         print(f"Dataset '{self.dataset_name}' not found. Downloading...")
    #         os.makedirs(dataset_dir, exist_ok=True)
    #         download_dataset(self.dataset_name, dataset_dir)
    #     else:
    #         print(f"Dataset '{
    #               self.dataset_name}' already exists. Skipping download.")

    # def setup(self, stage=None):
    #     dataset_dir = os.path.join(self.data_dir, self.dataset_name)
    #     img, gt, _, ignored_labels, _, _ = load_dataset(
    #         self.dataset_name, dataset_dir)
    #     self.setup_datasets(img, gt, self.hyperparams)


# class BotswanaDataModule(BaseHyperspectralDataModule):
#     def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None, **kwargs):
#         super().__init__(data_dir, "Botswana", batch_size,
#                          patch_size, transform, hyperparams=kwargs)

#     def prepare_data(self):
#         # Check if the dataset directory exists
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         if not os.path.exists(dataset_dir):
#             print(f"Dataset '{self.dataset_name}' not found. Downloading...")
#             os.makedirs(dataset_dir, exist_ok=True)
#             download_dataset(self.dataset_name, dataset_dir)
#         else:
#             print(f"Dataset '{
#                   self.dataset_name}' already exists. Skipping download.")

#     def setup(self, stage=None):
#         dataset_dir = os.path.join(self.data_dir, self.dataset_name)
#         img, gt, _, ignored_labels, _, _ = load_dataset(
#             self.dataset_name, dataset_dir)
#         self.setup_datasets(img, gt, self.hyperparams)
