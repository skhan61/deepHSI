from functools import partial

import pytest
import torch

from deepHSI.models.task_algos import BaseModule


@pytest.fixture
def optimizer():
    # Creating a partial function for the SGD optimizer with
    # predefined learning rate and weight decay
    return partial(torch.optim.SGD, lr=0.001, weight_decay=0.001)


@pytest.fixture
def scheduler():
    # Here, the optimizer must be instantiated first in the test function
    return partial(torch.optim.lr_scheduler.StepLR, step_size=10)


@pytest.fixture
def lit_module(optimizer, scheduler):
    # Create a BaseModule instance with the optimizer and scheduler partial functions
    return BaseModule(optimizer=optimizer,
                      scheduler=scheduler)


def test_initialization(lit_module):
    # Verify optimizer and scheduler are stored as partial functions
    assert isinstance(lit_module.optimizer,
                      partial), "Optimizer should be a partial function"
    assert isinstance(lit_module.scheduler,
                      partial), "Scheduler should be a partial function"

    # # Assuming BaseModule extracts the learning rate from the optimizer partial
    # # and stores it upon initialization
    # expected_lr = 0.001  # As defined in the optimizer_partial fixture
    # assert lit_module.learning_rate == expected_lr, \
    #     "Learning rate should match the expected value"

    # Verify that hyperparameters are saved excluding 'net'
    # This assumes 'save_hyperparameters' works correctly as per PyTorch Lightning's design
    # and that you have added custom logic to exclude 'net'
    assert 'optimizer' in lit_module.hparams, \
        "Optimizer should be in saved hyperparameters"
    assert 'scheduler' in lit_module.hparams, \
        "Scheduler should be in saved hyperparameters"
    assert 'net' not in lit_module.hparams, \
        "'net' should be excluded from saved hyperparameters"
