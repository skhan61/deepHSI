import pytest
import torch

from src.models.components.simple_dense_net import \
    HSIFCResNetModel  # Update the import path as needed


@pytest.fixture
def setup_resnet_model():
    input_channels = 103  # Hyperspectral image channels
    patch_size = 7  # Adjusted for compatibility with ResNet
    n_classes = 9
    model = HSIFCResNetModel(input_channels=input_channels,
                             patch_size=patch_size, n_classes=n_classes,
                             dropout=True, base_model='resnet50')
    return model, patch_size  # Return both model and patch_size


def test_resnet_initialization(setup_resnet_model):
    """Test ResNet model layers are initialized correctly."""
    model, _ = setup_resnet_model  # Unpack the model and ignore patch_size
    assert model.fc1 is not None, "ResNet fc1 layer should be initialized"
    assert model.fc2 is not None, "ResNet fc2 layer should be initialized"
    assert model.resnet_model is not None, "ResNet backbone should be initialized"


def test_resnet_forward_pass_single_sample(setup_resnet_model):
    """Test the forward pass of the ResNet model with a single sample."""
    model, patch_size = setup_resnet_model  # Unpack both model and patch_size

    # Ensure model is in evaluation mode to disable dropout and batch normalization effects
    model.eval()

    # Shape: [batch_size, channels, height, width]
    input_tensor = torch.randn(1, 103, patch_size, patch_size)

    print(f"Input tensor shape: {input_tensor.shape}")  # Debugging print

    try:
        output = model(input_tensor)
        print(f"Output tensor shape: {output.shape}")  # Debugging print
        assert output.shape == (
            1, 9), f"Output shape is incorrect. Expected: (1, 9), Got: {output.shape}"
    except Exception as e:
        print(f"Error during forward pass: {e}")
        raise  # Reraise the exception to fail the test and show the traceback


def test_resnet_forward_pass_batch(setup_resnet_model):
    """Test the forward pass of the ResNet model with a batch of samples."""
    model, patch_size = setup_resnet_model  # Unpack both model and patch_size
    input_tensor = torch.randn(4, 103, patch_size, patch_size)
    output = model(input_tensor)
    assert output.shape == (
        4, 9), f"Output shape is incorrect. Expected: (4, 9), Got: {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__])
