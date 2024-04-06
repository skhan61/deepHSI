from functools import partial

import pyro
import pyro.distributions as dist
import pyro.optim as pyro_opt
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule
from pyro.optim import PyroLRScheduler, PyroOptim

from .base_class_module import BaseModule


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


class VAEModule(BaseModule):
    def __init__(self, encoder,
                 decoder, latent_dim, **kwargs):

        # Initialize the BaseModule with the remaining kwargs
        super().__init__(**kwargs)

        self.loss_fn = kwargs.pop('loss_fn', Trace_ELBO())
        self.vae = VAEPyroModule(encoder=encoder,
                                 decoder=decoder,
                                 latent_dim=latent_dim)

        # Initialize the optimizer using the provided constructor and parameters
        self.optimizer = self.optimizer_constructor(self.vae.parameters(),
                                                    **self.optimizer_params)

        # Initialize the scheduler if a constructor is provided, else set to None
        self.scheduler = self.scheduler_constructor(
            self.optimizer, **self.scheduler_params) if \
            self.scheduler_constructor else None

        # Use PyroOptim with the initialized optimizer from BaseModule
        self.pyro_optim = PyroOptim(lambda: self.optimizer, {})

        # print(self.pyro_optim)

        # Check if a scheduler is provided and initialize it
        if self.scheduler_constructor is not None:
            # Wrap the PyTorch scheduler with Pyro's scheduler wrapper
            # The 'optim_args' dictionary is required for PyroLRScheduler 
            # and contains the optimizer arguments
            self.scheduler = self.scheduler_constructor(
                self.optimizer, **self.scheduler_params)
            self.pyro_scheduler = PyroLRScheduler(
                self.scheduler,
                {'optimizer': self.pyro_optim, 'optim_args': self.optimizer_params}
            )
            # Use PyroLRScheduler with SVI
            self.svi = SVI(self.vae.model, self.vae.guide,
                           self.pyro_scheduler, loss=Trace_ELBO())
        else:
            # If no scheduler is provided, use PyroOptim with SVI directly
            self.svi = SVI(self.vae.model, self.vae.guide,
                           self.pyro_optim, loss=Trace_ELBO())

    def training_step(self, batch, batch_idx):
        x, _ = batch
        elbo_loss = self.svi.step(x)
        self.log('train/loss', elbo_loss)
        return {'loss': elbo_loss}
