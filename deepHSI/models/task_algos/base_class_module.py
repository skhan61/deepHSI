import functools
import importlib
import inspect  # Add this import statement at the beginning of your script
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
# from torch.optim.lr_scheduler import LinearLR, SequentialLR
# from pytorch_lightning import LightningModule
from torchmetrics import (F1Score, MaxMetric, MeanMetric, Metric, Precision,
                          Recall)
from torchmetrics.classification.accuracy import Accuracy


class BaseModule(L.LightningModule):
    """
    A base module for PyTorch Lightning that provides a structured way to define
    optimizers and learning rate schedulers for various tasks such as classification,
    autoencoders, etc. It's designed to be a flexible foundation that can be extended
    to train different models by specifying custom optimizers, schedulers, and other
    configurations.

    This module centralizes the common setup for optimizers and schedulers, making
    it easier to manage training configurations and hyperparameters.

    Attributes:
        optimizer_constructor (Type[Optimizer]): A constructor for the optimizer to be used.
        optimizer_params (Dict[str, Any]): A dictionary containing parameters for the optimizer.
        scheduler_constructor (Optional[Type[_LRScheduler]]): An optional constructor for the learning rate scheduler.
        scheduler_params (Optional[Dict[str, Any]]): An optional dictionary containing parameters for the scheduler.

    Args:
        optimizer_constructor (Type[Optimizer]): The class of the optimizer to use for training.
        optimizer_params (Dict[str, Any]): Parameters to initialize the optimizer with.
        scheduler_constructor (Optional[Type[_LRScheduler]]): Optional. The class of the learning rate scheduler to use.
        scheduler_params (Optional[Dict[str, Any]]): Optional. Parameters to initialize the scheduler with.
        **kwargs: Additional keyword arguments that can be passed to the LightningModule.

    Example:
        ```python
        # Define the optimizer and scheduler constructors along with their parameters
        optimizer_constructor = torch.optim.Adam
        optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        scheduler_constructor = torch.optim.lr_scheduler.StepLR
        scheduler_params = {"step_size": 10, "gamma": 0.1}

        # Initialize the base module
        base_module = BaseModule(
            optimizer_constructor=optimizer_constructor,
            optimizer_params=optimizer_params,
            scheduler_constructor=scheduler_constructor,
            scheduler_params=scheduler_params
        )

        # Now, `base_module` can be extended to implement specific tasks like classification
        ```
    """

    def __init__(
        self,
        optimizer_constructor: Type[Optimizer],
        optimizer_params: Dict[str, Any],
        scheduler_constructor: Optional[Type[_LRScheduler]] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        # Save the constructors and parameters directly
        self.optimizer_constructor = optimizer_constructor
        self.optimizer_params = optimizer_params
        self.scheduler_constructor = scheduler_constructor
        self.scheduler_params = scheduler_params \
            if scheduler_params is not None else {}

        # Save hyperparameters, excluding certain parameters
        # like 'encoder', 'decoder', etc., for clarity
        self.save_hyperparameters(ignore=[
                                  'encoder',
                                  'decoder', 'net'])

        # print(self.hparams)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizers and (optionally) learning rate schedulers for training.

        This method is automatically called by PyTorch Lightning during the training setup.

        Returns:
            A dictionary containing the optimizer and (optionally) the scheduler along with
            their configurations. This dictionary is used by PyTorch Lightning to manage the
            optimization process.
        """
        # Instantiate the optimizer
        optimizer = self.optimizer_constructor(
            self.trainer.model.parameters(), **self.optimizer_params)
        config = {"optimizer": optimizer}

        # Check if a scheduler constructor was provided
        # and instantiate it
        if self.scheduler_constructor is not None:
            scheduler = self.scheduler_constructor(
                optimizer, **self.scheduler_params)

            # Add the scheduler configuration
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }

        return config
