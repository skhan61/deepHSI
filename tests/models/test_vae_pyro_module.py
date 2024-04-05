import pytest
import torch
import torch.nn as nn

from deepHSI.models.task_algos import VAEPyroModule


class DummyEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, latent_dim * 2)

    def forward(self, x):
        return self.fc(x)


class DummyDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return torch.sigmoid(self.fc(z))


@pytest.fixture
def dummy_vae_module():
    input_dim, latent_dim, output_dim = 784, 2, 784  # Example dimensions
    encoder = DummyEncoder(input_dim, latent_dim)
    decoder = DummyDecoder(latent_dim, output_dim)
    return VAEPyroModule(encoder, decoder, latent_dim)


def test_reconstruction_shape(dummy_vae_module):
    batch_size, input_dim = 16, 784  # Example batch size and input dimension
    x = torch.randn(batch_size, input_dim)
    reconstructed_x = dummy_vae_module.reconstruct_img(x)
    assert reconstructed_x.shape == (batch_size, input_dim), \
        f"Expected reconstructed shape to be {(batch_size, input_dim)}, got {
        reconstructed_x.shape}"


def test_latent_sampling_shape(dummy_vae_module):
    batch_size, input_dim = 16, 784  # Example batch size and input dimension
    x = torch.randn(batch_size, input_dim)
    hidden = dummy_vae_module.encoder(x)
    z_loc, z_scale = hidden[:, :dummy_vae_module.latent_dim], hidden[:,
                                                                     dummy_vae_module.latent_dim:].exp()

    assert z_loc.shape == (batch_size, dummy_vae_module.latent_dim) \
        and z_scale.shape == (batch_size, dummy_vae_module.latent_dim), \
        "Latent variable means and scales do not match expected shapes."
