import functools
import importlib
import inspect  # Add this import statement at the beginning of your script
from typing import Any, Callable, Dict, Optional, Tuple, Union

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
# from torch.optim.lr_scheduler import LinearLR, SequentialLR
# from pytorch_lightning import LightningModule
from torchmetrics import (F1Score, MaxMetric, MeanMetric, Metric, Precision,
                          Recall)
from torchmetrics.classification.accuracy import Accuracy

from .base_class_module import BaseModule


class HSIClassificationModule(BaseModule):
    """
    A PyTorch Lightning module for Hyperspectral Image (HSI) Classification tasks. This module encapsulates 
    the neural network model, loss function, and evaluation metrics for a classification task.

    Attributes:
        net (torch.nn.Module): The neural network model used for classification.
        loss_fn (torch.nn.Module): The loss function. Default is `torch.nn.functional.cross_entropy`.
        num_classes (int, optional): The number of classes in the classification task. Default is None.
        custom_metrics (dict, optional): A dictionary of custom metrics (name: metric) to be tracked during training/validation.

    Example:
        >>> from torch import nn
        >>> from torch.nn import functional as F
        >>> from pytorch_lightning.metrics import Accuracy
        >>> from deephsi.models import HSIClassificationModule
        >>>
        >>> # Define a simple neural network for classification
        >>> class SimpleNet(nn.Module):
        ...     def __init__(self, num_classes):
        ...         super(SimpleNet, self).__init__()
        ...         self.fc1 = nn.Linear(1024, 512)
        ...         self.fc2 = nn.Linear(512, num_classes)
        ...
        ...     def forward(self, x):
        ...         x = F.relu(self.fc1(x))
        ...         x = self.fc2(x)
        ...         return x
        >>>
        >>> net = SimpleNet(num_classes=10)
        >>> custom_metrics = {'accuracy': Accuracy()}
        >>>
        >>> # Initialize the HSIClassificationModule with the network, custom loss function, and metrics
        >>> module = HSIClassificationModule(net=net, loss_fn=F.cross_entropy, num_classes=10, custom_metrics=custom_metrics)
        >>>
        >>> # Example usage with a PyTorch Lightning Trainer
        >>> # trainer = pl.Trainer(max_epochs=10)
        >>> # trainer.fit(module, train_dataloader, val_dataloader)

    Args:
        net (torch.nn.Module): The neural network model for classification.
        loss_fn (torch.nn.Module): The loss function for the classification task. Defaults to `torch.nn.functional.cross_entropy`.
        num_classes (Optional[int]): The number of unique classes in the dataset. Required for certain metrics. Defaults to None.
        custom_metrics (Optional[Dict[str, Metric]]): Custom metrics to be tracked during training/validation. Defaults to None.
        **kwargs: Additional keyword arguments for the BaseModule, such as optimizer and scheduler configurations.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss_fn: torch.nn.Module = F.cross_entropy,
        num_classes: Optional[int] = None,
        custom_metrics: Optional[Dict[str, Metric]] = None,
        **kwargs,  # kwargs will include optimizer and scheduler among other possible arguments
    ):
        # Implementation omitted for brevity.

        # Pass kwargs to BaseModule, which includes optimizer and scheduler
        super().__init__(**kwargs)

        # Initialize specific attributes for HSIClassificationModule
        self.net = net
        self.loss_fn = loss_fn
        self.num_classes = num_classes

        # Initialize custom metrics if provided
        self.metrics = {}
        if custom_metrics:
            for name, metric in custom_metrics.items():
                setattr(self, name, metric)
                self.metrics[name] = getattr(self, name)

        # Optionally, save hyperparameters, excluding 'net' if it's not serializable
        self.save_hyperparameters(logger=False, ignore=['net'])
        # self.save_hyperparameters()
        # self.save_hyperparameters(logger=False, ignore=['model'])

        # Initialize a place to store predictions and targets
        self.all_test_preds = []
        self.all_test_targets = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit 
        (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or 
        adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if getattr(self.hparams, "compile", False) and stage == "fit":
            self.net = torch.compile(self.net)

        # if hasattr(self.hparams, 'compile') and self.hparams.compile
        # and stage == "fit":

    def _model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Private method to process a single batch through the model during training or validation.

        This method is responsible for the forward pass, computing the loss, and predicting the class labels.
        It is intended to be used internally within the class during the training and validation steps.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A tuple containing a tensor of input data `x` and a tensor
                of corresponding true labels `y`. The input tensor `x` is expected to have the shape
                [batch_size, channels, height, width], and the label tensor `y` is expected to have the shape [batch_size].

        Returns:
            loss (torch.Tensor): The computed loss value as a result of comparing the model's predictions to the true labels.
            preds (torch.Tensor): The predicted class labels for the input data. Shape: [batch_size].
            y (torch.Tensor): The tensor of true labels provided in the input. Shape: [batch_size].

        Note:
            This method is an internal utility function and should not be called directly from outside the class.
        """
        x, y = batch

        print(type(x))
        # # Check input and label shapes
        # print(f"x shape: {x.shape}, y shape: {y.shape}")

        logits = self.forward(x)  # logits: [batch_size, num_classes]
        # print(f"logits shape: {logits.shape}")  # Check model output shape

        # # Check for any unexpected values in y
        # print(f"y unique values: {torch.unique(y)}")

        loss = self.loss_fn(logits, y)

        # print(f"Loss: {loss.item()}")  # Print the loss value

        preds = torch.argmax(logits, dim=1)  # preds: [batch_size]
        # Check prediction values
        # print(f"Predictions unique values: {torch.unique(preds)}")

        return loss, preds, y

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset all custom metrics at the start of training to ensure they
        # don't store results from validation sanity checks or previous runs.
        for metric_name, metric_obj in self.metrics.items():
            if hasattr(metric_obj, 'reset'):
                metric_obj.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels. batch[0] (inputs) shape: [batch_size, channels, height, width] batch[1]
            (labels) shape: [batch_size]
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        # Assuming model_step is defined elsewhere
        loss, preds, targets = self._model_step(batch)

        # preds shape: [batch_size, num_classes] - Assuming a classification task
        # targets shape: [batch_size]

        # Ensure predictions and targets are on the same device as the model
        device = next(self.net.parameters()).device
        # preds after this line have the same shape: [batch_size, num_classes]
        preds = preds.to(device)
        # targets after this line have the same shape: [batch_size]
        targets = targets.to(device)

        # Log the training loss
        self.log("train/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Update and log custom metrics for each step
        # and aggregate them over the epoch
        for metric_name, metric_obj in self.metrics.items():
            metric_obj.update(preds, targets)
            metric_value = metric_obj.compute()  # Ensure you compute the metric
            self.log(
                f"train/{metric_name}", metric_value, on_step=False,
                on_epoch=True, prog_bar=False
            )
            metric_obj.reset()  # Reset the metric for the next batch/epoch if needed

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # update sampler weight
        pass

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int):
        """Perform a single validation step. This method will be called 
        for each batch of the
        validation set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current batch of
                                        data in the validation set.

            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self._model_step(batch)

        # Log the validation loss for each batch
        self.log("val/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Update and log custom metrics for each step
        # and aggregate them over the epoch
        for metric_name, metric_obj in self.metrics.items():
            metric_obj.update(preds, targets)
            metric_value = metric_obj.compute()  # Ensure you compute the metric
            self.log(
                f"val/{metric_name}", metric_value,
                on_step=False, on_epoch=True, prog_bar=False
            )
            metric_obj.reset()  # Reset the metric for the next batch/epoch if needed

        return loss

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Perform a single test step. This method will be called for each batch of the
        test set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current 
            batch of data in the test set.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self._model_step(batch)

        # Log the test loss for each batch
        self.log("test/loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)

        # Update custom metrics with predictions and targets
        # Temporarily store predictions and targets for potential further analysis
        self.all_test_preds.append(preds.detach())
        self.all_test_targets.append(targets.detach())

        # Update and log custom metrics for each step and aggregate them over the epoch
        for metric_name, metric_obj in self.metrics.items():
            metric_obj.update(preds, targets)
            metric_value = metric_obj.compute()  # Ensure you compute the metric
            self.log(
                f"test/{metric_name}", metric_value,
                on_step=False, on_epoch=True, prog_bar=False
            )
            metric_obj.reset()  # Reset the metric for the next batch/epoch if needed

        return loss
