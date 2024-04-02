import shutil  # Add this import at the beginning of your file
from pathlib import Path

import pytest
import torch

from deepHSI.datamodule.remote_sensing_datasets import *

# from deepHSI.datamodule.remote_sensing_datasets.indianpine import \
#     IndianPinesDataModule
# from deepHSI.datamodule.remote_sensing_datasets.ksc import KSCDataModule
# from deepHSI.datamodule.remote_sensing_datasets.paviaC import PaviaCDataModule
# from deepHSI.datamodule.remote_sensing_datasets.paviaU import PaviaUDataModule
# from deepHSI.datamodule.remote_sensing_datasets.salinas import \
#     SalinasDataModule

# from deepHSI.datamodule import *  # (PaviaUDataModule,

# #   SalinasDataModule)


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
    data_dir = Path("/home/sayem/Desktop/deepHSI/temp/")

    dataset_name = dataset_name

    dm = DataModule(data_dir=str(data_dir), hyperparams=hyperparams)
    dm.prepare_data()
    dm.setup(stage='fit')

    assert dm.train_dataset and dm.val_dataset, "Datasets not initialized properly."

    train_dataloader = dm.train_dataloader()
    batch = next(iter(train_dataloader))
    x, y = batch

    # Use 'batch_size' from 'hyperparams' in the assertion
    batch_size = hyperparams['batch_size']
    assert len(x) == batch_size, f"Batch size for data does not match expected: {
        batch_size}."
    assert len(y) == batch_size, f"Batch size for labels does not match expected: {
        batch_size}."
    assert x.dtype == torch.float32, "Data dtype is not float32."
    assert y.dtype == torch.long, "Label dtype is not long."

    # Test dataloader (if applicable)
    if hasattr(dm, 'test_dataloader') and callable(getattr(dm, 'test_dataloader')):
        dm.setup(stage='test')
        test_dataloader = dm.test_dataloader()
        test_batch = next(iter(test_dataloader))
        test_x, test_y = test_batch

        assert len(test_x) == hyperparams['batch_size'], f"Test batch size does not match expected: {
            hyperparams['batch_size']}."
        assert len(test_y) == hyperparams['batch_size'], f"Test label batch size does not match expected: {
            hyperparams['batch_size']}."
        assert test_x.dtype == torch.float32, "Test data dtype is not float32."
        assert test_y.dtype == torch.long, "Test label dtype is not long."

    # Clean up after test
    shutil.rmtree(data_dir)
