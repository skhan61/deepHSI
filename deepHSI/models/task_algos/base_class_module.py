import functools
import importlib
import inspect  # Add this import statement at the beginning of your script
from typing import Any, Dict, Optional, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
# from torch.optim.lr_scheduler import LinearLR, SequentialLR
# from pytorch_lightning import LightningModule
from torchmetrics import (F1Score, MaxMetric, MeanMetric, Metric, Precision,
                          Recall)
from torchmetrics.classification.accuracy import Accuracy


# TODO: rename: HyperNetModule
class BaseModule(L.LightningModule):
    def __init__(
        self,
        # net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # loss_fn: torch.nn.Module = F.cross_entropy,
        # custom_metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,  # Additional parameters for flexibility
    ):

        super().__init__()
        self.optimizer = optimizer
        # self.optimizer_params = optimizer_params
        self.scheduler = scheduler

        # # Ensure learning rate is saved in hparams for easy access
        # Default to 1e-3 if not specified
        # self.learning_rate = optimizer.keywords.get("lr", 1e-3)
        # print(self.learning_rate)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightningmodule.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate 
                schedulers to be used for training.
        """
        # print(self.hparams)

        # Check if the optimizer is a partial function and instantiate it if necessary
        if callable(self.hparams.optimizer):
            optimizer = self.hparams.optimizer(
                params=self.trainer.model.parameters())
        else:
            optimizer = self.hparams.optimizer

        config = {"optimizer": optimizer}

        # Check if a scheduler is provided and is a partial function, then instantiate it
        if self.hparams.scheduler is not None:
            if callable(self.hparams.scheduler):
                scheduler = self.hparams.scheduler(optimizer=optimizer)
            else:
                scheduler = self.hparams.scheduler

            # Add the scheduler configuration
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            }

        return config
