from functools import partial
from typing import Any, Callable, Dict, Optional, Type

import pyro
import pyro.distributions as dist
import pyro.optim as pyro_opt
import torch
import torch.nn as nn
import torch.optim as optim
from pyro.distributions import Independent, Normal
from pyro.infer import (ELBO, SVI,  # Use ELBO as the base class for typing
                        Trace_ELBO)
from pyro.infer.elbo import ELBO
from pyro.nn import PyroModule
from pyro.optim import PyroLRScheduler, PyroOptim
from torch import nn, optim
from torch.optim import Adam  # If you're using PyTorch's Adam optimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.optim.optimizer import Optimizer

from .base_class_module import BaseModule


class VAEPyroModule(PyroModule):
    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()  # Move the entire module's parameters and buffers to the GPU

    def model(self, x):
        # If CUDA is available, ensure the input tensor is on GPU
        if self.use_cuda:
            x = x.cuda()

        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.size(0)):
            z_prior = dist.Normal(torch.zeros(x.size(0), self.latent_dim).to(x.device),
                                  torch.ones(x.size(0),
                                             self.latent_dim).to(x.device)).to_event(1)
            z = pyro.sample("latent", z_prior)
            x_loc = self.decoder(z)
            sigma = torch.ones_like(x_loc)
            obs_dist = Independent(Normal(x_loc, sigma), 4)
            pyro.sample("obs", obs_dist, obs=x)

    def guide(self, x):
        # If CUDA is available, ensure the input tensor is on GPU
        if self.use_cuda:
            x = x.cuda()

        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.size(0)):
            hidden = self.encoder(x)
            z_loc, z_logvar = hidden[:,
                                     :self.latent_dim], hidden[:, self.latent_dim:]
            z_scale = torch.exp(0.5 * z_logvar)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        if self.use_cuda:
            x = x.cuda()

        hidden = self.encoder(x)
        z_loc, z_logvar = hidden[:,
                                 :self.latent_dim], hidden[:, self.latent_dim:]
        z_scale = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(z_scale)
        z = z_loc + epsilon * z_scale
        loc_img = self.decoder(z)
        return loc_img


class VAEModule(BaseModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        optimizer_constructor: Type[Optimizer],
        optimizer_params: Dict[str, Any],
        pyro_loss_function: Type[ELBO] = Trace_ELBO,
        scheduler_constructor: Optional[Type[_LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            optimizer_constructor=optimizer_constructor,
            optimizer_params=optimizer_params,
            scheduler_constructor=scheduler_constructor,
            scheduler_params=scheduler_params,
            **kwargs
        )

        pyro.enable_validation(True)
        self.vae = VAEPyroModule(encoder, decoder, latent_dim)

        # Assuming optimizer_constructor is passed as an argument to your function or class
        optimizer_name = optimizer_constructor.__name__
        print(f"The optimizer name is: {type(optimizer_name)}")

        optimizer_cls = getattr(torch.optim, optimizer_name)

        # assert optimizer_cls == torch.optim.SGD, 'Dont Match'

        scheduler_name = scheduler_constructor.__name__
        print(f"Then scheduler name is: {scheduler_name}")

        scheduler_cls = getattr(pyro.optim, scheduler_name)

        self.scheduler = scheduler_cls(
            {'optimizer': optimizer_cls,
             'optim_args': optimizer_params,
             **scheduler_params})

        # Initialize the SVI object for inference using the PyroLRScheduler
        self.svi = SVI(self.vae.model, self.vae.guide,
                       self.scheduler, loss=Trace_ELBO())

        # Disable automatic optimization
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, _ = batch
        elbo_loss = self.svi.step(x)
        elbo_loss_tensor = torch.tensor(
            elbo_loss, dtype=torch.float32, device=self.device)
        self.log('train_loss', elbo_loss_tensor)
        return {'loss': elbo_loss_tensor}

    def on_train_epoch_end(self):
        if self.scheduler:
            self.scheduler.step()

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        elbo_loss = self.svi.evaluate_loss(x)
        self.log('val_loss', elbo_loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return {'val_loss': elbo_loss}
