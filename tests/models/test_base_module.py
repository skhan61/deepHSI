from functools import partial
from unittest.mock import Mock, create_autospec

import pytest
import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler

from deepHSI.models.task_algos import BaseModule


@pytest.fixture
def mock_model():
    # Create a mock model with a parameters method
    model = Mock()
    # Empty iterator for parameters
    model.parameters = Mock(return_value=iter([]))
    return model


@pytest.fixture
def mock_optimizer(mock_model):
    # Create a mock optimizer class spec'ed from the Optimizer class
    mock_optimizer_class = create_autospec(Adam, instance=False)
    # Create a mock optimizer instance
    mock_optimizer_instance = Mock(spec=Adam)
    # Configure the optimizer instance with necessary attributes
    mock_optimizer_instance.param_groups = [{}]
    # Configure the optimizer class to return the mock optimizer instance when called
    mock_optimizer_class.return_value = mock_optimizer_instance
    return mock_optimizer_class(mock_model.parameters())


@pytest.fixture
def mock_scheduler(mock_optimizer):
    # Create a mock learning rate scheduler spec'ed from the _LRScheduler class
    scheduler = create_autospec(
        StepLR, instance=True, optimizer=mock_optimizer, step_size=10)
    return scheduler


@pytest.fixture
def base_module(mock_optimizer, mock_scheduler):
    # Instantiate the BaseModule with the mock optimizer and mock scheduler
    return BaseModule(optimizer=mock_optimizer, scheduler=mock_scheduler)


def test_base_module_configure_optimizers(base_module):
    # Test the configure_optimizers method
    config = base_module.configure_optimizers()

    # Check that the config dictionary contains the expected keys
    assert 'optimizer' in config
    assert isinstance(config['optimizer'], Optimizer)

    # Check that the lr_scheduler configuration is correctly set up
    assert 'lr_scheduler' in config
    assert 'scheduler' in config['lr_scheduler']
    assert isinstance(config['lr_scheduler']['scheduler'], _LRScheduler)
    assert config['lr_scheduler']['monitor'] == 'val/loss'
    assert config['lr_scheduler']['interval'] == 'epoch'
    assert config['lr_scheduler']['frequency'] == 1


# @pytest.fixture
# def model():
#     return torch.nn.Linear(10, 2)  # A simple model for testing


# @pytest.fixture
# def optimizer():
#     return partial(torch.optim.SGD, lr=0.001, weight_decay=0.001)


# @pytest.fixture
# def scheduler(optimizer):
#     # Since the scheduler requires an instantiated optimizer, we instantiate
#     # the optimizer here with dummy model parameters
#     opt = optimizer(params=model().parameters())
#     return partial(torch.optim.lr_scheduler.StepLR, optimizer=opt, step_size=10)


# @pytest.fixture
# def lit_module(optimizer, scheduler):
#     return BaseModule(optimizer=optimizer, scheduler=scheduler)


# def test_configure_optimizers(lit_module):
#     # Execute configure_optimizers to retrieve the configuration
#     config = lit_module.configure_optimizers()

#     # Check that the optimizer was instantiated correctly
#     assert isinstance(config["optimizer"], torch.optim.Optimizer), \
#         "Optimizer should be an instance of torch.optim.Optimizer"

#     # Check that the scheduler was instantiated correctly, if present
#     if "lr_scheduler" in config:
#         assert isinstance(config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler._LRScheduler), \
#             "Scheduler should be an instance of torch.optim.lr_scheduler._LRScheduler"

#     # Verify that hyperparameters are saved
#     assert 'optimizer' in lit_module.hparams, \
#         "Optimizer should be in saved hyperparameters"
#     assert 'scheduler' in lit_module.hparams, \
#         "Scheduler should be in saved hyperparameters"


# from functools import partial

# import pytest
# import torch

# from deepHSI.models.task_algos import BaseModule


# @pytest.fixture
# def optimizer():
#     # Creating a partial function for the SGD optimizer with
#     # predefined learning rate and weight decay
#     return partial(torch.optim.SGD, lr=0.001, weight_decay=0.001)


# @pytest.fixture
# def scheduler():
#     # Here, the optimizer must be instantiated first in the test function
#     return partial(torch.optim.lr_scheduler.StepLR, step_size=10)


# @pytest.fixture
# def lit_module(optimizer, scheduler):
#     # Create a BaseModule instance with the optimizer and scheduler partial functions
#     return BaseModule(optimizer=optimizer,
#                       scheduler=scheduler)


# def test_initialization(lit_module):
#     # Verify optimizer and scheduler are stored as partial functions
#     assert isinstance(lit_module.optimizer,
#                       partial), "Optimizer should be a partial function"
#     assert isinstance(lit_module.scheduler,
#                       partial), "Scheduler should be a partial function"

#     # # Assuming BaseModule extracts the learning rate from the optimizer partial
#     # # and stores it upon initialization
#     # expected_lr = 0.001  # As defined in the optimizer_partial fixture
#     # assert lit_module.learning_rate == expected_lr, \
#     #     "Learning rate should match the expected value"

#     # Verify that hyperparameters are saved excluding 'net'
#     # This assumes 'save_hyperparameters' works correctly as per PyTorch Lightning's design
#     # and that you have added custom logic to exclude 'net'
#     assert 'optimizer' in lit_module.hparams, \
#         "Optimizer should be in saved hyperparameters"
#     assert 'scheduler' in lit_module.hparams, \
#         "Scheduler should be in saved hyperparameters"
#     assert 'net' not in lit_module.hparams, \
#         "'net' should be excluded from saved hyperparameters"
