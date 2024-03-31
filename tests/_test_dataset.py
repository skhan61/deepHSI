import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from deepHSI.dataset.components.hyperspectral_dataset import \
    HyperspectralDataset
from deepHSI.dataset.components.utils import *


# @pytest.fixture(params=DATASETS_CONFIG.keys())
# def setup_dataset(request):
#     dataset_name = request.param
@pytest.fixture()
def setup_dataset():
    dataset_name = "Salinas"  # "KSC" # request.param
    root_temp_folder = Path("/home/sayem/Desktop/deepHSI/temp_data")
    target_folder = root_temp_folder / dataset_name
    target_folder.mkdir(parents=True, exist_ok=True)
    yield dataset_name, str(target_folder)
    shutil.rmtree(root_temp_folder)


def test_hyperspectral_dataset_integration(setup_dataset):
    # Set up the dataset using the fixture
    dataset_name, target_folder = setup_dataset

    # Ensure the dataset is downloaded
    download_dataset(dataset_name, target_folder)

    # Load the dataset components
    img, gt, label_values, ignored_labels, rgb_bands, _ = load_dataset(
        dataset_name, target_folder)

    # # Use the loaded data to build the dataset for training/testing
    # input_imgs, levels = build_dataset(img, gt)

    # # Basic checks on the dataset components
    # assert img is not None, "Image data should not be None"
    # assert gt is not None, "Ground truth data should not be None"
    # assert len(label_values) > 0, "Label values should not be empty"
    # assert len(
    #     ignored_labels) >= 0, "Ignored labels should be defined, even if empty"
    # assert len(rgb_bands) == 3, "RGB bands should be a tuple of 3 elements"
    # assert not np.isnan(img).any(), "Image data should not contain NaN"
    # assert not np.isnan(gt).any(), "Ground truth data should not contain NaN"

    # # # Additional checks for input_imgs and levels
    # # assert input_imgs is not None, "Input images should not be None"
    # # assert levels is not None, "Levels should not be None"
    # # assert len(input_imgs) > 0, "Input images should not be empty"
    # # assert len(levels) > 0, "Levels should not be empty"
    # # assert not np.isnan(input_imgs).any(
    # # ), "Input images should not contain NaN"
    # # assert not np.isnan(levels).any(), "Levels should not contain NaN"

    hyperparams = {
        "dataset": dataset_name,
        "patch_size": 5,
        "ignored_labels": ignored_labels,
        "flip_augmentation": True,  # Enable augmentation for testing
        "radiation_augmentation": True,
        "mixture_augmentation": True,
        "center_pixel": True,
        "supervision": "full",
    }

    dataset = HyperspectralDataset(img, gt, **hyperparams)

    # Initial assertions as provided before
    assert len(
        dataset) > 0, "The HyperspectralDataset instance should not be empty"
    assert all(
        label not in dataset.ignored_labels for label in dataset.labels
    ), "Ignored labels should not be present in the dataset labels"

    p = hyperparams["patch_size"] // 2
    valid_indices = [
        (x, y)
        for x, y in dataset.indices
        if x > p and x < img.shape[0] - p and y > p and y < img.shape[1] - p
    ]
    assert len(valid_indices) == len(
        dataset.indices
    ), "All indices in the dataset should be valid considering the patch size"

    # Test augmentations
    original_data, _ = dataset[0]
    # Assuming augmentation randomness can lead to different results
    augmented_data, _ = dataset[0]
    assert not torch.equal(
        original_data, augmented_data), "Augmentation should modify the data"

    # Test label consistency
    for i in range(len(dataset)):
        patch, label = dataset[i]
        if dataset.center_pixel:
            assert (
                label == dataset.labels[i]
            ), "Label should match the center pixel for the corresponding patch"
        else:
            assert label in np.unique(
                patch), "Label should be present in the patch"

    # Test DataLoader integration
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for batch in dataloader:
        data, labels = batch
        assert data.shape == (
            10,
            1,
            dataset.patch_size,
            dataset.patch_size,
        ), "Data batch shape should be (batch_size, 1, patch_size, patch_size)"
        assert labels.shape[0] == 10, "Labels batch size should match the data batch size"

    # Test ignoring labels
    for label in dataset.ignored_labels:
        assert label not in dataset.labels, "Ignored labels should not be present in the dataset"

    # # Specify hyperparameters for the HyperspectralDataset class
    # hyperparams = {
    #     "dataset": dataset_name,
    #     "patch_size": 5,
    #     "ignored_labels": ignored_labels,
    #     "flip_augmentation": False,
    #     "radiation_augmentation": False,
    #     "mixture_augmentation": False,
    #     "center_pixel": True,
    #     "supervision": "full"
    # }

    # # Instantiate the HyperspectralDataset class
    # dataset = HyperspectralDataset(img, gt, **hyperparams)

    # # Conduct tests to validate the HyperspectralDataset instance
    # assert len(
    #     dataset) > 0, "The HyperspectralDataset instance should not be empty"
    # assert all(label not in dataset.ignored_labels for label in dataset.labels), "Ignored labels should not be present in the dataset labels"

    # # Test to ensure the proper handling of patch size and indices
    # p = hyperparams['patch_size'] // 2
    # valid_indices = [
    #     (x, y) for x, y in dataset.indices
    #     if x > p and x < img.shape[0] - p and y > p and y < img.shape[1] - p
    # ]
    # assert len(valid_indices) == len(
    #     dataset.indices), "All indices in the dataset should be valid considering the patch size"
