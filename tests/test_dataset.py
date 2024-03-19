import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.dataset.components.hyperspectral_dataset import HyperspectralDataset
from src.dataset.components.utils import *


# @pytest.fixture(params=DATASETS_CONFIG.keys())
# def setup_dataset(request):
#     dataset_name = request.param
@pytest.fixture(scope="session")
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

    hyperparams = {
        "dataset": dataset_name,
        "patch_size": 5,
        "ignored_labels": ignored_labels,
        "center_pixel": True,
        "supervision": "full",
    }

    dataset = HyperspectralDataset(img, gt, transform=None, **hyperparams)


@pytest.mark.parametrize("patch_size", [1, 3, 5, 7])
def test_patch_sizes(setup_dataset, patch_size):
    dataset_name, target_folder = setup_dataset
    img, gt, _, ignored_labels, _, _ = load_dataset(
        dataset_name, target_folder)

    # Ensure center_pixel is set to False for this test to make sense
    dataset = HyperspectralDataset(
        img, gt, patch_size=patch_size, ignored_labels=ignored_labels, center_pixel=False)
    sample_data, sample_label = dataset[0]

    print(sample_data.shape)

    # Assert the patch size in the data sample matches the expected size
    assert sample_data.shape[-2] == patch_size  # H dimension
    assert sample_data.shape[-1] == patch_size  # W dimension


def test_ignored_labels(setup_dataset):
    dataset_name, target_folder = setup_dataset
    img, gt, _, _, _, _ = load_dataset(dataset_name, target_folder)

    ignored_labels = [0]  # Typically, 0 is used for background/ignored class
    dataset = HyperspectralDataset(img, gt, ignored_labels=ignored_labels)

    for _, label in dataset:
        assert label not in ignored_labels


def test_center_pixel_extraction(setup_dataset):
    dataset_name, target_folder = setup_dataset
    img, gt, _, ignored_labels, _, _ = load_dataset(
        dataset_name, target_folder)

    dataset = HyperspectralDataset(
        img, gt, patch_size=3, center_pixel=True, ignored_labels=ignored_labels)
    _, sample_label = dataset[0]

    # Assert the label is a single value, not an array
    assert isinstance(sample_label, int) or sample_label.ndim == 0


@pytest.mark.parametrize("supervision", ["full", "semi"])
def test_supervision_modes(setup_dataset, supervision):
    dataset_name, target_folder = setup_dataset
    img, gt, _, ignored_labels, _, _ = load_dataset(
        dataset_name, target_folder)

    dataset = HyperspectralDataset(
        img, gt, supervision=supervision, ignored_labels=ignored_labels)
    assert len(dataset) > 0  # Check if the dataset is not empty

    if supervision == "full":
        # In full supervision, none of the labels should be in ignored_labels
        for _, label in dataset:
            assert label not in ignored_labels
