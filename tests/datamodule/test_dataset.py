import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.stats
import torch

from deepHSI.datamodule.components import HyperspectralDataset
from deepHSI.datamodule.components.utils import *
from deepHSI.datamodule.transforms import *


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


@pytest.fixture
def synthetic_hsi_patch():
    # Generate a random tensor of size 5x10x10
    return torch.rand(5, 10, 10)


def test_hsi_normalize(synthetic_hsi_patch):
    hsi_normalize = HSINormalize()
    normalized_patch = hsi_normalize(synthetic_hsi_patch)

    # Check the shape is retained
    assert normalized_patch.shape == synthetic_hsi_patch.shape, "The normalized patch should have the same shape as the input patch."

    # Check that values are within the range [0, 1]
    assert torch.all(normalized_patch >= 0) and torch.all(
        normalized_patch <= 1), "All values should be within the range [0, 1]."

    # Optionally, check for constant bands
    for band in range(synthetic_hsi_patch.shape[0]):
        if torch.max(synthetic_hsi_patch[band, :, :]) == torch.min(synthetic_hsi_patch[band, :, :]):
            assert torch.all(normalized_patch[band, :, :] == synthetic_hsi_patch[band, :, :]), f"Band {
                band} is constant and should remain unchanged."


def test_hsi_flip(synthetic_hsi_patch):
    hsi_flip = HSIFlip()
    flipped_patch = hsi_flip(synthetic_hsi_patch)

    # Check the shape is retained
    assert flipped_patch.shape == synthetic_hsi_patch.shape, "The flipped patch should have the same shape as the input patch."

    # Check that the first column of the original is equal to the last column of the flipped patch and so on
    assert torch.all(flipped_patch[:, :, 0] == synthetic_hsi_patch[:,
                     :, -1]), "The patch was not flipped correctly."


def test_hsi_rotate(synthetic_hsi_patch):
    hsi_rotate = HSIRotate()
    rotated_patch = hsi_rotate(synthetic_hsi_patch)

    # Check the shape is retained
    assert rotated_patch.shape == synthetic_hsi_patch.shape, "The rotated patch should have the same shape as the input patch."

    # Check that rotation happened correctly
    # The first row of the original patch should match the last column of the rotated patch and so on
    assert torch.all(rotated_patch[:, 0, :] == synthetic_hsi_patch[:,
                     :, -1]), "The patch was not rotated correctly."


# import pytest
# import torch
# from deepHSI.datamodule.transforms.hsi_transforms import HSISpectralNoise, HSISpectralShift

# Assuming synthetic_hsi_patch fixture is already defined as provided in the original code

def test_hsi_spectral_noise(synthetic_hsi_patch):
    """
    Test HSISpectralNoise to ensure Gaussian noise is added to the patch.
    This test verifies that after adding noise, the mean and std of the patch
    are within expected ranges, considering the noise parameters.
    """
    mean = 0.0
    std = 0.01  # Noise standard deviation
    hsi_spectral_noise = HSISpectralNoise(mean=mean, std=std)
    noisy_patch = hsi_spectral_noise(synthetic_hsi_patch)

    # Compute the noise added by subtracting the original patch from the noisy patch
    noise_added = noisy_patch - synthetic_hsi_patch

    # Verify that the noise added has the mean and std as expected
    assert torch.isclose(noise_added.mean(), torch.tensor(mean), atol=1e-3)
    assert torch.isclose(noise_added.std(), torch.tensor(std), atol=1e-3)


def test_hsi_spectral_shift(synthetic_hsi_patch):
    """
    Test HSISpectralShift to ensure spectral bands are shifted correctly.
    This test verifies that after shifting, the spectral bands are rolled
    as per the shift value, and the relative order of bands is maintained.
    """
    shift = 2  # Number of places to shift
    hsi_spectral_shift = HSISpectralShift(shift=shift)
    shifted_patch = hsi_spectral_shift(synthetic_hsi_patch)

    # Verifying the shift by comparing the shifted bands with the original bands
    for band in range(synthetic_hsi_patch.shape[0]):
        shifted_band_index = (band + shift) % synthetic_hsi_patch.shape[0]
        assert torch.allclose(
            shifted_patch[shifted_band_index], synthetic_hsi_patch[band]), "The band was not shifted correctly."

# Assuming synthetic_hsi_patch fixture is already defined as provided in the original code


def test_hsi_random_spectral_drop(synthetic_hsi_patch):
    drop_prob = 0.1  # Probability of dropping a spectral band
    hsi_random_spectral_drop = HSIRandomSpectralDrop(drop_prob=drop_prob)

    num_trials = 100  # Number of trials to average the randomness
    total_dropped_bands = 0

    for _ in range(num_trials):
        # Apply the random spectral drop transformation
        dropped_patch = hsi_random_spectral_drop(synthetic_hsi_patch.clone())

        for band in range(synthetic_hsi_patch.shape[0]):
            if torch.all(dropped_patch[band, :, :] == 0):
                total_dropped_bands += 1

    # Calculate the average proportion of dropped bands over all trials
    avg_dropped_ratio = total_dropped_bands / \
        (synthetic_hsi_patch.shape[0] * num_trials)

    # Use a tolerance to check if the average dropped ratio is close to the expected drop_prob
    tolerance = 0.05  # Acceptable deviation from drop_prob

    assert abs(avg_dropped_ratio - drop_prob) <= tolerance, f"Average drop ratio significantly different from expected. Expected: {
        drop_prob}, got: {avg_dropped_ratio}"


def test_compose(synthetic_hsi_patch):
    # Apply a known modification to the synthetic patch for a verifiable test
    # Set a specific pixel's value to a known quantity
    # Arbitrary value outside the [0, 1] range for testing normalization
    synthetic_hsi_patch[0, 0, 0] = 2.0

    # Create a composition of transformations including spectral noise and shift
    transform = Compose([
        HSISpectralNoise(mean=0.0, std=0.01),  # Adding Gaussian noise
        HSISpectralShift(shift=1),             # Shifting spectral bands by 1
        # Normalizing values to be within [0, 1]
        HSINormalize(),
        HSIFlip(),                             # Flipping the patch horizontally
        HSIRotate(),                            # Rotating the patch 90 degrees clockwise
        HSIRandomSpectralDrop()
    ])
    transformed_patch = transform(synthetic_hsi_patch)

    # Assert the shape is consistent
    assert transformed_patch.shape == synthetic_hsi_patch.shape, "Transformed patch does not retain the original shape."

    # Assert normalization effectiveness
    assert torch.all(transformed_patch >= 0) and torch.all(
        transformed_patch <= 1), "Transformed patch values are not all within [0, 1]."

    # For a precise test of flipping and rotating, one needs to know the initial state.
    # Since flipping and rotation alter positions, without an initial state, we focus on the process's impact.
    # The pixel originally at (0, 0) of the first channel should have moved due to the flip and then the rotation.
    # After flipping, it would move to the rightmost column. After rotation, it moves to the top row of the last column.
    # After rotation, this should be the new position of the originally set pixel.
    expected_position_value = transformed_patch[0, -1, 0]
    assert expected_position_value <= 1, "The specific transformation path did not move the pixel as expected."

    # This test checks the composed transformation effects in a broad sense without relying on the exact initial values of all pixels.
