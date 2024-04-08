import datetime
import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
import wandb
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from deepHSI.datamodule.remote_sensing_datasets import IndianPinesDataModule
from deepHSI.datamodule.transforms import (Compose, HSIFlip, HSINormalize,
                                           HSIRandomSpectralDrop, HSIRotate,
                                           HSISpectralNoise, HSISpectralShift)
from deepHSI.utils import (RankedLogger, extras, get_metric_value,
                           instantiate_callbacks, instantiate_loggers,
                           log_hyperparameters, task_wrapper)

torch.set_float32_matmul_precision('medium')


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = RankedLogger(__name__, rank_zero_only=True)

# # Assuming the project root is set correctly in the environment by `rootutils.setup_root`
# project_root_dir = os.getenv("PROJECT_ROOT")
# print(f"Project Root Directory: {project_root_dir}")


@task_wrapper
def train(cfg: DictConfig):
    # Instantiate model from the Hydra config
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    print(model)

    # # Accessing and printing the 'paths' part of the configuration
    # paths_config = cfg['paths']
    print(cfg)

    # # # Printing individual paths from the 'paths' configuration
    # # print(f"Root Directory: {paths_config.root_dir}")
    # # print(f"Data Directory: {paths_config.data_dir}")
    # # print(f"Log Directory: {paths_config.log_dir}")
    # # print(f"Output Directory: {paths_config.output_dir}")
    # # print(f"Work Directory: {paths_config.work_dir}")

    # # Data transformations
    # transform = Compose([
    #     HSINormalize(),
    #     HSIFlip(),
    #     HSIRotate(),
    #     HSISpectralNoise(mean=0.0, std=0.01),
    #     HSISpectralShift(shift=2),
    #     HSIRandomSpectralDrop(drop_prob=0.1)
    # ])

    # # Include 'batch_size', 'num_workers', and
    # # 'num_classes' within the hyperparams dictionary
    # hyperparams = {
    #     "batch_size": 64,
    #     "num_workers": 24,
    #     "patch_size": 10,
    #     "center_pixel": True,
    #     "supervision": "full",
    #     "num_classes": 10,  # Define the number of classes in your dataset
    # }

    # # Data module setup
    # datamodule = IndianPinesDataModule(
    #     data_dir=cfg.paths.data_dir,
    #     transform=transform,
    #     hyperparams=hyperparams,
    # )

    # # Initialize WandbLogger with dynamic configurations
    # logger = WandbLogger(
    #     name="Run",
    #     project="IndianPines",
    #     save_dir="/home/sayem/Desktop/deepHSI/notebooks/wandb",
    #     offline=False,
    #     tags="exp",
    #     # notes=experiment_notes,  # Add the experiment notes
    # )

    # # Callbacks: Pytorch Callbacks: See L doc
    # # Define the EarlyStopping callback
    # callbacks = L.pytorch.callbacks.EarlyStopping(
    #     monitor="val/f1",  # Specify the metric to monitor
    #     patience=20,  # Number of epochs with no improvement after which training will be stopped
    #     verbose=True,  # Whether to print logs to stdout
    #     mode="max",  # In 'min' mode, training will stop when the quantity monitored has stopped decreasing
    #     check_on_train_epoch_end=False,
    # )

    # # Initialize the PyTorch Lightning Trainer with fast_dev_run enabled
    # trainer = L.Trainer(
    #     fast_dev_run=True,  # Enable fast_dev_run
    #     precision="16-mixed",  # Use 16-bit precision
    #     accelerator="auto",  # Specify the accelerator as GPU
    #     max_epochs=100,
    #     log_every_n_steps=3,
    #     callbacks=callbacks,
    #     logger=logger,
    #     deterministic=False,
    # )

    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # print("-------------")
    # print(cfg.get("ckpt_path"))

    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(model=model, datamodule=datamodule,)
    #                 # ckpt_path=None)

    # train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning(
    #             "Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # # merge train and test metrics
    # metric_dict = {**train_metrics, **test_metrics}

    metric_dict, object_dict = 0, 0

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
