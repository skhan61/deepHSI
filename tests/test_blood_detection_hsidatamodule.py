import shutil
from pathlib import Path

import pytest
import torch

from src.dataset.medical_datasets.bloodHSI import BloodDetectionHSIDataModule

# You might need to adjust the import according to your project structure


def test_blood_detection_hsidatamodule():
    # Specify a temporary directory for testing
    temp_data_dir = Path("/home/sayem/Desktop/deepHSI/tmp/deepHSI")
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    # Define DOI for the dataset if needed (you can use a dummy DOI for testing)
    doi = '10.5281/zenodo.3984905'

    # Instantiate the data module
    dm = BloodDetectionHSIDataModule(data_dir=str(temp_data_dir), doi=doi, batch_size=32,
                                     patch_size=5, transform=None, selected_image="A_1")

    # Call the prepare_data method
    dm.prepare_data()

    # Call the setup method for the 'fit' stage
    dm.setup(stage='fit')

    # Check if datasets are initialized
    assert dm.train_dataset is not None, "Train dataset not initialized."
    assert dm.val_dataset is not None, "Validation dataset not initialized."

    # Get a batch from the train dataloader and check the dimensions
    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    x, y = batch

    assert len(x) == dm.batch_size, "Batch size for data does not match expected."
    assert len(
        y) == dm.batch_size, "Batch size for labels does not match expected."
    assert x.dtype == torch.float32, "Data dtype is not float32."
    assert y.dtype == torch.long, "Label dtype is not long."

    # Clean up the temporary directory after the test
    shutil.rmtree(temp_data_dir)


if __name__ == "__main__":
    pytest.main([__file__])
