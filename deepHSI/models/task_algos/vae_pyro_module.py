import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule


class VAEPyroModule(PyroModule):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.size(0)):
            # Prior for latent variables
            z_prior = dist.Normal(torch.zeros(x.size(0), self.latent_dim),
                                  torch.ones(x.size(0), self.latent_dim)).to_event(1)
            z = pyro.sample("latent", z_prior)
            # Decode the latent variables
            x_recon = self.decoder(z)
            pyro.sample("obs", dist.Bernoulli(x_recon).to_event(1), obs=x)

    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.size(0)):
            # Use the encoder to get parameters for q(z|x)
            hidden = self.encoder(x)
            z_loc, z_scale = hidden[:,
                                    :self.latent_dim], hidden[:, self.latent_dim:].exp()
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        # Use the encoder to get parameters for q(z|x)
        hidden = self.encoder(x)
        z_loc, z_scale = hidden[:,
                                :self.latent_dim], hidden[:, self.latent_dim:].exp()
        # Sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # Decode the latent variables
        loc_img = self.decoder(z)
        return loc_img
