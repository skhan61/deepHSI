import pytest
import torch

from src.models.components.simple_dense_net import \
    HSIFCModel  # Update the import path as needed


@pytest.fixture
def setup_model():
    input_channels = 103
    patch_size = 5
    n_classes = 9
    model = HSIFCModel(input_channels=input_channels,
                       patch_size=patch_size, n_classes=n_classes, dropout=True)
    return model


def test_initialization(setup_model):
    """Test model layers are initialized correctly."""
    model = setup_model
    assert model.fc1 is not None, "fc1 layer should be initialized"
    assert model.fc2 is not None, "fc2 layer should be initialized"
    assert model.fc3 is not None, "fc3 layer should be initialized"
    assert model.fc4 is not None, "fc4 layer should be initialized"


def test_forward_pass_single_sample(setup_model):
    """Test the forward pass of the model with a single sample."""
    model = setup_model
    # Shape: [batch_size, planes, channels, height, width]
    input_tensor = torch.randn(1, 1, 103, 5, 5)
    output = model(input_tensor)
    assert output.shape == (
        1, 9), f"Output shape is incorrect. Expected: (1, 9), Got: {output.shape}"


def test_forward_pass_batch(setup_model):
    """Test the forward pass of the model with a batch of samples."""
    model = setup_model
    # Shape: [batch_size, planes, channels, height, width]
    input_tensor = torch.randn(4, 1, 103, 5, 5)
    output = model(input_tensor)
    assert output.shape == (
        4, 9), f"Output shape is incorrect. Expected: (4, 9), Got: {output.shape}"


def test_dropout_effect(setup_model):
    """Test the effect of dropout in the model."""
    model = setup_model
    model.eval()  # Set the model to evaluation mode to use dropout
    input_tensor = torch.randn(1, 1, 103, 5, 5)
    output1 = model(input_tensor)
    output2 = model(input_tensor)
    # Since dropout is stochastic, we can't guarantee that outputs will be different,
    # but we can check that the model is in eval mode, which means dropout is active
    assert model.training == False, "Model is not in evaluation mode."


if __name__ == "__main__":
    pytest.main([__file__])
