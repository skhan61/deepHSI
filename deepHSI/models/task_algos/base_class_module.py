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
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}

        # Save hyperparameters, excluding certain parameters like 'encoder', 'decoder', etc., for clarity
        self.save_hyperparameters(ignore=[
                                  'encoder',
                                  'decoder', 'net'])

        # print(self.hparams)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Instantiate the optimizer
        optimizer = self.optimizer_constructor(
            self.trainer.model.parameters(), **self.optimizer_params)
        config = {"optimizer": optimizer}

        # Check if a scheduler constructor was provided and instantiate it
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

    # def configure_optimizers(self) -> Dict[str, Any]:
    #     """Choose what optimizers and learning-rate schedulers to use in your optimization.
    #     Normally you'd need one. But in the case of GANs or similar you might have multiple.

    #     Examples:
    #         https://lightning.ai/docs/pytorch/latest/common/lightningmodule.html#configure-optimizers

    #     :return: A dict containing the configured optimizers and learning-rate
    #             schedulers to be used for training.
    #     """
    #     # print(self.hparams)

    #     # Check if the optimizer is a partial function and instantiate it if necessary
    #     if callable(self.hparams.optimizer):
    #         optimizer = self.hparams.optimizer(
    #             params=self.trainer.model.parameters())
    #     else:
    #         optimizer = self.hparams.optimizer

    #     config = {"optimizer": optimizer}

    #     # Check if a scheduler is provided and is a partial function, then instantiate it
    #     if self.hparams.scheduler is not None:
    #         if callable(self.hparams.scheduler):
    #             scheduler = self.hparams.scheduler(optimizer=optimizer)
    #         else:
    #             scheduler = self.hparams.scheduler

    #         # Add the scheduler configuration
    #         config["lr_scheduler"] = {
    #             "scheduler": scheduler,
    #             "monitor": "val/loss",
    #             "interval": "epoch",
    #             "frequency": 1,
    #         }

    #     return config
