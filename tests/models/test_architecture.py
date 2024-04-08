import pytest
import torch

from deepHSI.models.architectures import (DeepGRUHSIClassifier, HSIFCModel,
                                          HSIModelBase,
                                          HyperspectralCNNDetector,
                                          SpectralSpatialCNN)


@pytest.fixture(params=[
    # Modified Dummy subclass to produce an output shape of (batch_size, n_classes)
    (type("DummyHSIModel", (HSIModelBase,), {
        "forward": lambda self, x: torch.randn(x.size(0), self.n_classes)}),
     {"input_channels": 103, "patch_size": 5, "n_classes": 9, "dropout": True}),
    (HSIFCModel, {"input_channels": 103,
     "patch_size": 5, "n_classes": 9, "dropout": True}),
    (SpectralSpatialCNN, {"input_channels": 103, "patch_size": 5,
     "n_classes": 9, "n_planes": 2, "dropout": True}),
    # Adding DeepGRUHSIClassifier with its specific parameters
    (DeepGRUHSIClassifier, {"input_channels": 103, "patch_size": 5,
     "n_classes": 9, "dropout": True})
], ids=["DummyModel", "FCModel", "SpectralSpatialCNN", "DeepGRUHSIClassifier"])
def setup_model(request):
    model_cls, model_kwargs = request.param
    model = model_cls(**model_kwargs)
    return model, model_kwargs["patch_size"]


def test_model_initialization(setup_model):
    model, patch_size = setup_model
    assert model is not None, "Model initialization failed."


# def test_model_forward_pass(setup_model):
#     model, patch_size = setup_model
#     # Create a dummy input tensor appropriate for the model
#     dummy_input = torch.rand(
#         (1, 1, model.input_channels, patch_size, patch_size))
#     # Forward pass through the model
#     output = model(dummy_input)
#     # Check if the output is of the expected shape
#     expected_shape = (1, model.n_classes)
#     assert output.shape == expected_shape, f"Output shape mismatch. Expected: {
#         expected_shape}, Got: {output.shape}"

def test_model_forward_pass(setup_model):
    model, patch_size = setup_model
    # Create a dummy input tensor appropriate for the model
    dummy_input = torch.rand(
        (1, 1, model.input_channels, patch_size, patch_size))
    print(f"Input tensor shape before model: {dummy_input.shape}")

    # Forward pass through the model
    output = model(dummy_input)
    print(f"Output tensor shape: {output.shape}")


def test_base_model_initialization(setup_model):
    model, patch_size = setup_model
    assert hasattr(
        model, "input_channels"), "Model should have an attribute 'input_channels'"
    assert hasattr(
        model, "patch_size"), "Model should have an attribute 'patch_size'"
    assert hasattr(
        model, "n_classes"), "Model should have an attribute 'n_classes'"
    assert hasattr(
        model, "use_dropout"), "Model should have an attribute 'use_dropout'"


def test_model_forward_pass_single_sample(setup_model):
    model, patch_size = setup_model
    if model.__class__.__name__ == "DummyHSIModel":
        pytest.skip("Skipping forward pass test for DummyHSIModel")
    model.eval()  # Set model to evaluation mode if it contains dropout or batch normalization layers
    # Adjust input tensor for SpectralSpatialCNN
    if isinstance(model, SpectralSpatialCNN):
        input_tensor = torch.randn(
            1, 1, model.input_channels, patch_size, patch_size)  # 5D tensor for 3D CNN
    else:
        input_tensor = torch.randn(
            1, model.input_channels, patch_size, patch_size)  # 4D tensor for 2D CNN
    output = model(input_tensor)
    assert output.shape == (
        1, 9), f"Output shape is incorrect. Expected: (1, 9), Got: {output.shape}"


def test_model_forward_pass_batch(setup_model):
    model, patch_size = setup_model
    if model.__class__.__name__ == "DummyHSIModel":
        pytest.skip("Skipping forward pass test for DummyHSIModel")
    # Adjust input tensor for SpectralSpatialCNN
    if isinstance(model, SpectralSpatialCNN):
        input_tensor = torch.randn(
            4, 1, model.input_channels, patch_size, patch_size)  # 5D tensor for 3D CNN
    else:
        input_tensor = torch.randn(
            4, model.input_channels, patch_size, patch_size)  # 4D tensor for 2D CNN
    output = model(input_tensor)
    assert output.shape == (
        4, 9), f"Output shape is incorrect. Expected: (4, 9), Got: {output.shape}"
