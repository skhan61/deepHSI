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
from torch.optim.lr_scheduler import LinearLR, SequentialLR
# from pytorch_lightning import LightningModule
from torchmetrics import (F1Score, MaxMetric, MeanMetric, Metric, Precision,
                          Recall)
from torchmetrics.classification.accuracy import Accuracy


class HSIClassificationLitModule(L.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: str,
        optimizer_params: Dict[str, Any],
        loss_fn: torch.nn.Module = F.cross_entropy,
        scheduler: Optional[type] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        num_classes: int = None,

        # New parameter for custom metrics
        custom_metrics: Optional[Dict[str, Metric]] = None,
    ):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params or {}
        self.num_classes = num_classes

        # Setup metrics
        self.metrics = {}
        # Initialize custom metrics as attributes
        if custom_metrics:
            for metric_name, metric_obj in custom_metrics.items():
                # Set each metric as an attribute
                setattr(self, metric_name, metric_obj)
                # Reference the attribute in the metrics dict
                self.metrics[metric_name] = getattr(self, metric_name)

        # Ensure learning rate is saved in hparams for easy access
        self.learning_rate = optimizer_params.get(
            'lr', 1e-3)  # Default to 1e-3 if not specified

        self.save_hyperparameters(ignore=['net'])

        # self.save_hyperparameters()
        # self.save_hyperparameters(logger=False, ignore=['model'])

        self.validation_step_outputs = []

    def configure_optimizers(self):
        OptimizerClass = getattr(torch.optim, self.optimizer)
        optimizer_params = self.optimizer_params.copy()

        if 'learning_rate' in self.hparams:
            optimizer_params['lr'] = self.hparams.learning_rate

        optimizer = OptimizerClass(self.net.parameters(), **optimizer_params)

        # Scheduler setup
        if self.scheduler:
            SchedulerClass = getattr(torch.optim.lr_scheduler, self.scheduler)

            # Exclude 'epoch' parameter and other non-scheduler init args to prevent warnings
            valid_scheduler_params = {k: v for k, v in self.scheduler_params.items()
                                      if k in inspect.signature(SchedulerClass).parameters
                                      and k != 'epoch'}

            scheduler = SchedulerClass(optimizer, **valid_scheduler_params)

            scheduler_config = {
                "scheduler": scheduler,
                # Default to 'epoch' interval
                "interval": self.scheduler_params.get("interval", "epoch"),
                # Default frequency
                "frequency": self.scheduler_params.get("frequency", 1)
            }

            # For schedulers like ReduceLROnPlateau that need a 'monitor' metric
            if 'monitor' in self.scheduler_params:
                scheduler_config["monitor"] = self.scheduler_params["monitor"]

            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        return {"optimizer": optimizer}

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

        This is a good hook when you need to build models dynamically 
        or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if getattr(self.hparams, 'compile', False) and stage == "fit":
            self.net = torch.compile(self.net)

        # if hasattr(self.hparams, 'compile') and self.hparams.compile
        # and stage == "fit":

    def _model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [batch_size, channels, height, width], y: [batch_size]
        x, y = batch
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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images 
                    and target labels.
                    batch[0] (inputs) shape: [batch_size, channels, height, width]
                    batch[1] (labels) shape: [batch_size]

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
        self.log("train/loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)

        # Update and log custom metrics for each step
        # and aggregate them over the epoch
        for metric_name, metric_obj in self.metrics.items():
            metric_obj.update(preds, targets)
            metric_value = metric_obj.compute()  # Ensure you compute the metric
            self.log(f"train/{metric_name}", metric_value,
                     on_step=False, on_epoch=True, prog_bar=False)
            metric_obj.reset()  # Reset the metric for the next batch/epoch if needed

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a single validation step. This method will be called 
        for each batch of the validation set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current batch of 
                                                       data in the validation set.

            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self._model_step(batch)

        # Log the validation loss for each batch
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update custom metrics with predictions and targets

        # Update and log custom metrics for each step
        # and aggregate them over the epoch
        for metric_name, metric_obj in self.metrics.items():
            metric_obj.update(preds, targets)
            metric_value = metric_obj.compute()  # Ensure you compute the metric
            self.log(f"val/{metric_name}", metric_value,
                     on_step=False, on_epoch=True, prog_bar=False)
            metric_obj.reset()  # Reset the metric for the next batch/epoch if needed

        return loss

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # Perform the forward pass and calculate the loss (if needed)
        loss, preds, targets = self._model_step(batch)

        # Convert predictions and targets to CPU and detach them from the computation graph
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Return predictions and targets for this batch
        return {'preds': preds, 'targets': targets}

    def test_epoch_end(self, outputs):
        # Concatenate all predictions and targets
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_targets = torch.cat([x['targets'] for x in outputs], dim=0)

        # Compute the confusion matrix
        cm = confusion_matrix(all_targets.numpy(), all_preds.numpy())

        # Plot the confusion matrix (optional)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

        # Log the confusion matrix as an image in TensorBoard (optional)
        self.logger.experiment.add_figure(
            "Confusion Matrix", plt.gcf(), self.current_epoch)

        # Return the confusion matrix if you want to see it in the test results
        return {'confusion_matrix': cm}
