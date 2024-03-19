import shutil  # Add this import at the beginning of your file
from pathlib import Path

import pytest
import torch

from src.dataset.hyperspectral_datamodule import *  # (PaviaUDataModule,

#   SalinasDataModule)


@pytest.mark.parametrize("DataModule, dataset_name, hyperparams", [
    (SalinasDataModule, "Salinas", {
        "patch_size": 5, "center_pixel": True, "supervision": "full"}),
    (PaviaUDataModule, "PaviaU", {
        "patch_size": 7, "center_pixel": False, "supervision": "semi"}),
    (PaviaCDataModule, "PaviaC", {
        "patch_size": 5, "center_pixel": True, "supervision": "full"}),
    (KSCDataModule, "KSC", {
        "patch_size": 7, "center_pixel": False, "supervision": "semi"}),
    (IndianPinesDataModule, "IndianPines", {
        "patch_size": 5, "center_pixel": True, "supervision": "full"}),
    (BotswanaDataModule, "Botswana", {
        "patch_size": 7, "center_pixel": False, "supervision": "semi"}),
])
def test_hyperspectral_datamodule(DataModule, dataset_name, hyperparams):
    data_dir = Path("/home/sayem/Desktop/deepHSI/temp/") / dataset_name

    # Use the batch_size variable directly instead of accessing it from hyperparams
    batch_size = 32

    dm = DataModule(data_dir=str(data_dir),
                    batch_size=batch_size, **hyperparams)
    dm.prepare_data()
    dm.setup(stage='fit')

    assert dm.train_dataset and dm.val_dataset, "Datasets not initialized properly."

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    x, y = batch

    # Use the batch_size variable in the assertion
    assert len(x) == batch_size, "Batch size for data does not match expected."
    assert len(y) == batch_size, "Batch size for labels does not match expected."
    assert x.dtype == torch.float32, "Data dtype is not float32."
    assert y.dtype == torch.long, "Label dtype is not long."

    # Clean up after test using shutil.rmtree()
    shutil.rmtree(data_dir)
