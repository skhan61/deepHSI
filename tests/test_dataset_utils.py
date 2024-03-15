import os

import pytest

from src.data.components.utils import get_dataset


@pytest.fixture
def setup_dataset():
    # Fixture to setup dataset parameters
    dataset_name = "PaviaC"  # Example dataset name
    target_folder = "/tmp/test_data"  # Temporary folder for testing
    yield dataset_name, target_folder
    # Cleanup after tests run
    if os.path.exists(target_folder):
        for filename in os.listdir(target_folder):
            file_path = os.path.join(target_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def test_get_dataset(setup_dataset):
    # Test the get_dataset function with the setup_dataset fixture
    dataset_name, target_folder = setup_dataset
    img, gt, label_values, ignored_labels, rgb_bands, _ = get_dataset(dataset_name, target_folder)

    # Assertions to verify the dataset is loaded correctly
    assert img is not None, "Image data should not be None"
    assert gt is not None, "Ground truth data should not be None"
    assert len(label_values) > 0, "Label values should not be empty"
    assert len(ignored_labels) >= 0, "Ignored labels should be defined, even if empty"
    assert len(rgb_bands) == 3, "RGB bands should be a tuple of 3 elements"

    # Optional: Add more specific tests such as checking for correct shapes, specific data values, etc.
