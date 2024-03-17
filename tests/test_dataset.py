import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset.components.hyperspectral_dataset import HyperspectralDataset
from src.dataset.components.utils import *


@pytest.fixture(scope="session")
def dataset():
    dataset_name = "Salinas"
    root_temp_folder = Path("/home/sayem/Desktop/deepHSI/temp_data")
    target_folder = root_temp_folder / dataset_name
    target_folder.mkdir(parents=True, exist_ok=True)

    # Ensure the dataset is downloaded only once for the session
    download_dataset(dataset_name, str(target_folder))

    # Load the dataset components
    img, gt, label_values, ignored_labels, rgb_bands, _ = load_dataset(
        dataset_name, str(target_folder))

    # Yield the loaded components for use in tests
    yield img, gt, label_values, ignored_labels, rgb_bands, target_folder

    # Cleanup after the session is done
    shutil.rmtree(root_temp_folder)


def test_hyperspectral_dataset_basic(dataset):
    img, gt, label_values, ignored_labels, rgb_bands, target_folder = dataset

    # Pack the hyperparameters into a dictionary
    hyperparams = {
        "patch_size": 5,
        "ignored_labels": ignored_labels,
        "center_pixel": True,
        "supervision": "full"  # Assuming you want to use full supervision for this test
        # You can add more hyperparameters here as needed
    }

    # Unpack the hyperparams dictionary when creating the dataset instance
    dataset_instance = HyperspectralDataset(img, gt, **hyperparams)

    # Initial dataset checks
    assert len(dataset_instance) > 0, "The dataset should not be empty"
    for i in range(len(dataset_instance)):
        data, label = dataset_instance[i]
        assert isinstance(data, torch.Tensor), "Data should be a torch.Tensor"
        assert isinstance(
            label, torch.Tensor), "Label should be a torch.Tensor"
        if dataset_instance.center_pixel:
            assert label.ndim == 0, "Label should be a scalar when center_pixel is True"
        else:
            assert label.ndim == 2, "Label should be a 2D tensor when center_pixel is False"


@pytest.mark.parametrize("patch_size", [1, 3, 5])
def test_patch_size_variation(dataset, patch_size):
    img, gt, label_values, ignored_labels, rgb_bands, target_folder = dataset

    # Pass hyperparameters as a dictionary
    hyperparams = {
        "patch_size": patch_size,
        "ignored_labels": ignored_labels,
        "center_pixel": True,
        "supervision": "full"  # Assuming full supervision for the test
    }

    dataset_instance = HyperspectralDataset(img, gt, **hyperparams)
    data, _ = dataset_instance[0]

    # For patch sizes > 1, an extra dimension for the batch size (set to 1) is added
    # For a patch size of 1, only the spectral dimension is expected
    expected_shape = (
        1, img.shape[2], patch_size, patch_size) if patch_size > 1 else (img.shape[2],)
    assert data.shape == expected_shape, (
        f"Expected data shape {expected_shape}, but got {data.shape}. "
        f"Patch size: {patch_size}, Center pixel: {
            hyperparams['center_pixel']}"
    )


@pytest.mark.parametrize("center_pixel", [True, False])
def test_center_pixel_functionality(dataset, center_pixel):
    img, gt, label_values, ignored_labels, rgb_bands, target_folder = dataset

    hyperparams = {
        "patch_size": 5,
        "ignored_labels": ignored_labels,
        "center_pixel": center_pixel
    }

    dataset_instance = HyperspectralDataset(img, gt, **hyperparams)
    _, label = dataset_instance[0]

    if center_pixel:
        assert label.ndim == 0, "Label should be a scalar when center_pixel is True"
    else:
        assert label.ndim == 2, "Label should be a 2D tensor when center_pixel is False"
