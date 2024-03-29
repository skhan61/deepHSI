import io

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_tensor


class ConfusionMatrixLogger(Callback):
    def on_test_epoch_end(self, trainer, pl_module):
        # Concatenate all predictions and targets across all test batches
        test_preds = torch.cat(pl_module.all_test_preds, dim=0)
        test_targets = torch.cat(pl_module.all_test_targets, dim=0)

        # Convert predictions and targets to CPU and numpy arrays
        test_preds_np = test_preds.cpu().numpy() if test_preds.is_cuda else test_preds.numpy()
        test_targets_np = test_targets.cpu().numpy(
        ) if test_targets.is_cuda else test_targets.numpy()

        # Print the shapes and first few elements for debugging
        print(f"test_preds_np shape: {
              test_preds_np.shape}, first few elements: {test_preds_np[:5]}")
        print(f"test_targets_np shape: {
              test_targets_np.shape}, first few elements: {test_targets_np[:5]}")

        # Check if test_preds_np is 2-dimensional before using np.argmax
        if test_preds_np.ndim == 2:
            # If 2-dimensional, it likely contains class probabilities, and we use np.argmax
            preds_labels = np.argmax(test_preds_np, axis=1)
        elif test_preds_np.ndim == 1:
            # If 1-dimensional, it likely contains direct class labels
            preds_labels = test_preds_np
        else:
            raise ValueError("Unexpected dimensionality of test_preds_np")

        # Compute the confusion matrix
        cm = confusion_matrix(test_targets_np, preds_labels)

        # Plot the confusion matrix
        fig = self.plot_confusion_matrix(cm)

        # Log the confusion matrix
        self.log_confusion_matrix(trainer, fig)

        # Clear the lists to prepare for the next test epoch
        pl_module.all_test_preds.clear()
        pl_module.all_test_targets.clear()

    def plot_confusion_matrix(self, cm):
        class_names = np.arange(cm.shape[0])
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return plt.gcf()

    def log_confusion_matrix(self, trainer, fig):
        # Convert the matplotlib figure to an RGB image
        fig.canvas.draw()
        cm_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        cm_image = cm_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Close the figure to free memory
        plt.close(fig)

        # Log the confusion matrix image using the trainer's logger
        if trainer.logger:
            # Check for Weights & Biases logger
            if isinstance(trainer.logger, WandbLogger):
                # Directly use wandb.Image to log the image
                trainer.logger.experiment.log(
                    {"confusion_matrix": [wandb.Image(cm_image)]})
            else:
                # For other loggers: Convert numpy image to PIL Image and then to BytesIO
                cm_image_pil = Image.fromarray(cm_image)
                buffer = io.BytesIO()
                cm_image_pil.save(buffer, format='PNG')
                buffer.seek(0)

                # Attempt to use a generic image logging method if available
                if hasattr(trainer.logger, 'log_image'):
                    trainer.logger.log_image(
                        'confusion_matrix', buffer, step=trainer.current_epoch)
                else:
                    # Attempt to log directly or print a message if direct logging is not supported
                    try:
                        trainer.logger.log({'confusion_matrix': buffer})
                    except AttributeError:
                        print("Logger does not support direct image logging.")
