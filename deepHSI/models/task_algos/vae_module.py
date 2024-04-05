from functools import partial

import pyro
import pyro.distributions as dist
import pyro.optim as pyro_opt
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule

from .base_class_module import BaseModule
from .vae_pyro_module import VAEPyroModule


class VAEModule(BaseModule):
    def __init__(self, vae, optimizer, scheduler):
        # Utilize the partial optimizer and potentially scheduler provided
        optimizer = optimizer  # partial function
        scheduler = scheduler  # partial function

        # Initialize the BaseModule with these
        super().__init__(optimizer=optimizer, scheduler=scheduler)

        self.vae = vae
        # Initialize Pyro's SVI here with the model and guide from VAEPyroModule and the optimizer
        self.svi = SVI(self.vae.model,
                       self.vae.guide,
                       self.optimizer,
                       loss=Trace_ELBO())

    def training_step(self, batch, batch_idx):
        x, _ = batch
        elbo_loss = self.svi.step(x)
        self.log('train/loss', elbo_loss)
        return {'loss': elbo_loss}
