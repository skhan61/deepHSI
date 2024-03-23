import pytest
import torch
from torch.optim import Adam
from torchmetrics import Accuracy

from src.models.hsi_classification_module import \
    HSIClassificationLitModule  # Replace with your actual import


@pytest.fixture
def simple_model():
    return torch.nn.Linear(10, 2)  # A simple linear model for testing


@pytest.fixture
def simple_optimizer():
    return Adam


@pytest.fixture
def simple_loss_fn():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def simple_scheduler_cls():
    return torch.optim.lr_scheduler.StepLR


@pytest.fixture
def simple_scheduler_params():
    return {"step_size": 1, "gamma": 0.1}


@pytest.fixture
def simple_custom_metrics():
    return {"test_accuracy": Accuracy(num_classes=2, task="binary")}


@pytest.fixture
def lit_module(simple_model, simple_optimizer,
               simple_loss_fn, simple_scheduler_cls,
               simple_scheduler_params, simple_custom_metrics):
    return HSIClassificationLitModule(
        model=simple_model,
        optimizer_cls=simple_optimizer,
        optimizer_params={"lr": 0.001},
        loss_fn=simple_loss_fn,
        scheduler_cls=simple_scheduler_cls,
        scheduler_params=simple_scheduler_params,
        num_classes=2,
        custom_metrics=simple_custom_metrics
    )


def test_initialization(lit_module):
    assert lit_module.model is not None
    assert isinstance(lit_module.loss_fn, torch.nn.Module)
    assert lit_module.optimizer_cls == Adam
    assert lit_module.scheduler_cls == torch.optim.lr_scheduler.StepLR
    assert "lr" in lit_module.optimizer_params
    assert lit_module.metrics["test_accuracy"]


def test_forward_pass(lit_module):
    x = torch.randn(5, 10)  # 5 samples, 10 features
    logits = lit_module(x)
    assert logits.shape == (5, 2)  # 5 samples, 2 classes


def test_training_step(lit_module):
    # 5 samples, 10 features, labels in {0, 1}
    batch = (torch.randn(5, 10), torch.randint(0, 2, (5,)))
    loss, _, _ = lit_module.model_step(batch)
    assert loss is not None
    assert loss.item() > 0

# Add more tests as needed for other methods and functionalities
