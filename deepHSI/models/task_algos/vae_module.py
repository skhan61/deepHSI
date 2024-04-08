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
    """
    A PyTorch Module for a Variational Autoencoder (VAE) utilizing Pyro probabilistic models.

    Attributes:
        encoder (nn.Module): The encoder network that maps input data to the latent space.
        decoder (nn.Module): The decoder network that maps latent space vectors back to the input data space.
        latent_dim (int): Dimensionality of the latent space.
        device (torch.device): Device on which the module's tensors will be allocated.

    Methods:
        model(x): Defines the generative model and samples from the prior and likelihood distributions.
        guide(x): Defines the variational approximation (guide) to the posterior.
        reconstruct_img(x): Reconstructs input data from the latent space representation.
        reconstruction_loss(x): Computes the reconstruction loss (MSE) between input and reconstructed data.
        generate_samples(num_samples): Generates new data samples from the latent space.
        infer(x): Performs inference, reconstructing given input and generating new samples.

    Example:
        >>> encoder = MyEncoder()
        >>> decoder = MyDecoder()
        >>> latent_dim = 10
        >>> vae_module = VAEPyroModule(encoder, decoder, latent_dim)
        >>> x = torch.randn(64, 1, 28, 28)  # Example batch of data
        >>> reconstructed_x, generated_x = vae_module.infer(x)
    """

    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def model(self, x):
        """
        Defines the generative model, specifying the prior distribution over the latent space and 
        the likelihood of the data given latent variables.

        Args:
            x (torch.Tensor): Input data tensor.
        """
        # Ensure the input tensor is on the correct device
        x = x.to(self.device)
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
        """
        Defines the variational guide (approximate posterior) for inference, specifying how to sample
        latent variables given the observed data.

        Args:
            x (torch.Tensor): Input data tensor.
        """
        x = x.to(self.device)
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.size(0)):
            hidden = self.encoder(x)
            z_loc, z_logvar = hidden[:,
                                     :self.latent_dim], hidden[:, self.latent_dim:]
            z_scale = torch.exp(0.5 * z_logvar)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        """
        Reconstructs the input data by encoding it into the latent space and 
        then decoding it back to the data space.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Reconstructed data tensor.
        """

        hidden = self.encoder(x)
        z_loc, z_logvar = hidden[:,
                                 :self.latent_dim], hidden[:, self.latent_dim:]
        z_scale = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(z_scale)
        z = z_loc + epsilon * z_scale
        loc_img = self.decoder(z)
        return loc_img

    def reconstruction_loss(self, x):
        """
        Calculates the reconstruction loss (MSE) between the input data and its reconstruction.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: The MSE reconstruction loss.
        """
        # Reconstruct the images
        reconstructed_x = self.reconstruct_img(x)
        # Calculate MSE as the reconstruction loss
        loss = torch.nn.functional.mse_loss(
            reconstructed_x, x, reduction='sum')
        return loss

    def generate_samples(self, num_samples=1):
        """
        Generates new data samples from the latent space by sampling from the prior distribution
        and decoding the sampled latent variables.

        Args:
            num_samples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            torch.Tensor: Generated data samples.
        """
        # Generate new data samples from the learned latent space
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            generated_x = self.decoder(z)
            return generated_x

    def infer(self, x):
        """
        Performs inference by reconstructing the input data and generating new data samples. 
        This function combines reconstruction and sample generation in one step.

        Args:
            x (torch.Tensor): Input data tensor to be reconstructed.

        Returns:
            tuple: A tuple containing reconstructed data and generated data samples.
        """
        # Perform inference by reconstructing given input and generating new samples
        reconstructed_x = self.reconstruct_img(x)
        generated_x = self.generate_samples(num_samples=x.size(0))
        return reconstructed_x, generated_x


# class VAEModule(BaseModule):
class VAEModule(BaseModule):
    """
    A PyTorch Lightning module for training and evaluating a Variational Autoencoder (VAE) using Pyro.

    Attributes:
        encoder (nn.Module): The encoder network.
        decoder (nn.Module): The decoder network.
        latent_dim (int): Dimensionality of the latent space.
        optimizer_constructor (Type[Optimizer]): The optimizer class to be used for training.
        optimizer_params (Dict[str, Any]): Parameters to initialize the optimizer.
        pyro_loss_function (Type[ELBO]): The Pyro loss function class (e.g., Trace_ELBO).
        scheduler_constructor (Optional[Type[_LRScheduler]]): The learning rate scheduler class.
        scheduler_params (Optional[Dict[str, Any]]): Parameters to initialize the scheduler.

    Methods:
        training_step(batch, batch_idx): Performs a single training step.
        validation_step(batch, batch_idx): Performs a single validation step.
        on_train_epoch_end(): Called at the end of the training epoch to update the learning rate scheduler.

    Example:
        >>> encoder = MyEncoder()
        >>> decoder = MyDecoder()
        >>> latent_dim = 10
        >>> optimizer_params = {'lr': 1e-3}
        >>> scheduler_params = {'step_size': 10, 'gamma': 0.5}
        >>> vae = VAEModule(encoder, decoder, latent_dim, optim.Adam, optimizer_params, \
            Trace_ELBO, StepLR, scheduler_params)
        >>> trainer = pl.Trainer()
        >>> trainer.fit(vae, train_dataloader, val_dataloader)
    """

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = VAEPyroModule(encoder, decoder, latent_dim).to(device)

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
        self.automatic_optimization = False  # svi will take care of this

    def training_step(self, batch, batch_idx):
        """
        Executes the training step for a single batch.

        Args:
            batch (tuple): The batch of data. First element is the data tensor, 
            second element can be the target tensor.
            batch_idx (int): The index of the batch in the dataset.

        Returns:
            dict: A dictionary with the ELBO loss ('loss') and reconstruction loss ('rec_loss').
        """
        x, _ = batch
        x = x.to(self.vae.device)
        elbo_loss = self.svi.step(x)

        # Additionally, calculate the reconstruction loss
        rec_loss = self.vae.reconstruction_loss(x)
        self.log('train_reconstruction_loss', rec_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        # Convert ELBO loss to tensor and log it
        elbo_loss_tensor = torch.tensor(
            elbo_loss, dtype=torch.float32, device=self.device)
        self.log('train_loss', elbo_loss_tensor)
        return {'loss': elbo_loss_tensor, 'rec_loss': rec_loss}

    def validation_step(self, batch, batch_idx):
        """
        Executes the validation step for a single batch.

        Args:
            batch (tuple): The batch of data. First element is the data tensor, 
            second element can be the target tensor.
            batch_idx (int): The index of the batch in the dataset.

        Returns:
            dict: A dictionary with the validation ELBO loss ('val_loss') and 
            validation reconstruction loss ('rec_loss').
        """
        x, _ = batch
        # Evaluate the ELBO loss
        elbo_loss = self.svi.evaluate_loss(x)

        # Additionally, calculate the reconstruction loss
        rec_loss = self.vae.reconstruction_loss(x)
        self.log('val_reconstruction_loss', rec_loss,
                 on_step=True, on_epoch=True, prog_bar=True)

        self.log('val_loss', elbo_loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return {'val_loss': elbo_loss, 'rec_loss': rec_loss}

    def on_train_epoch_end(self):
        """
        Hook called at the end of a training epoch. Here, we step the learning rate scheduler.
        """
        if self.scheduler:
            self.scheduler.step()
