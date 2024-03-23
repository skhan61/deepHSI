from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import (F1Score, MaxMetric, MeanMetric, Metric, Precision,
                          Recall)
from torchmetrics.classification.accuracy import Accuracy


class HSIClassificationLitModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cls: type,
        optimizer_params: Dict[str, Any],
        loss_fn: torch.nn.Module = F.cross_entropy,
        scheduler_cls: Optional[type] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        num_classes: int = None,
        # New parameter for custom metrics
        custom_metrics: Optional[Dict[str, Metric]] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params
        self.scheduler_cls = scheduler_cls
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

        # self.save_hyperparameters(ignore=['model'])
        self.save_hyperparameters()
        # self.save_hyperparameters(logger=False, ignore=['model'])

    def setup(self, stage: str) -> None:
        if hasattr(self.hparams, 'compile') and self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        :return: A dict containing the configured optimizers and 
                 learning-rate schedulers to be used for training.
        """
        # Ensure the model has parameters
        model_parameters = list(self.model.parameters())
        if len(model_parameters) == 0:
            raise ValueError(
                "The model does not have any parameters. Please check the model architecture.")

        # Initialize the optimizer with the model's parameters
        optimizer = self.optimizer_cls(
            params=model_parameters, **self.optimizer_params)

        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(
                optimizer=optimizer, **self.scheduler_params)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) \
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

        :param batch: A batch of data (a tuple) containing the input 
                    tensor of images and target labels.

        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss, preds, targets = self.model_step(batch)

        # Ensure predictions and targets are on the same device as the model
        device = next(self.model.parameters()).device
        preds = preds.to(device)
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

    def on_train_epoch_end(self, unused=None) -> None:
        print(f"\nEpoch {self.current_epoch} - Summary of Training Metrics:")

        # Check and print the average training loss for the epoch
        avg_epoch_loss_key = "train/loss_epoch"
        if avg_epoch_loss_key in self.trainer.logged_metrics:
            avg_epoch_loss = self.trainer.logged_metrics[avg_epoch_loss_key].item(
            )
            print(f"  Average Train Loss: {avg_epoch_loss:.4f}")
        else:
            print("  Average Train Loss: Not available for this epoch.")

        # Iterate through all custom metrics
        for metric_name in self.metrics.keys():
            # Ensure the metric name used here matches the one used in training_step
            metric_epoch_key = f"train/{metric_name}"

            if metric_epoch_key in self.trainer.logged_metrics:
                avg_metric_value = self.trainer.logged_metrics[metric_epoch_key].item(
                )
                print(f"  {metric_name.capitalize()}: {avg_metric_value:.4f}")
            else:
                print(f"  {metric_name.capitalize()
                           }: Not available for this epoch.")

        super().on_train_epoch_end()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Perform a single validation step. This method will be called for each batch of the validation set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The current batch of data in the validation set.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # Log the validation loss for each batch
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Update custom metrics with predictions and targets
        for metric_name, metric_obj in self.metrics.items():
            metric_obj.update(preds, targets)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to log the computed values of custom metrics
        and print them for immediate visibility in the console or logs.
        """
        print(f"\nEpoch {self.current_epoch} - Summary of Validation Metrics:")

        # Log and print the computed values of custom metrics at the end of the validation epoch
        for metric_name, metric_obj in self.metrics.items():
            # Ensure the metric has a 'compute' method
            if hasattr(metric_obj, 'compute'):
                metric_value = metric_obj.compute()
                self.log(f"val/{metric_name}", metric_value,
                         prog_bar=True, logger=True)
                metric_obj.reset()  # Reset the metric for the next epoch

                # Print the metric value if it has been logged successfully
                metric_epoch_key = f"val/{metric_name}"
                if metric_epoch_key in self.trainer.logged_metrics:
                    avg_metric_value = self.trainer.logged_metrics[metric_epoch_key].item(
                    )
                    print(f"  {metric_name.capitalize()}: {
                          avg_metric_value:.4f}")
                else:
                    print(f"  {metric_name.capitalize()
                               }: Not available for this epoch.")

        # Call the parent class's method if needed
        super().on_validation_epoch_end()
