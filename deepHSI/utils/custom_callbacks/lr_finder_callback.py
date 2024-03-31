# Customize LearningRateFinder callback to run at different epochs.
# This feature is useful while fine-tuning models.
from lightning.pytorch.callbacks import LearningRateFinder


class InitialLRFinder(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        # Run the LR finder only at the start of the first training epoch
        if trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)
            # # To prevent multiple runs, disable the callback after the first use
            # trainer.remove_callback(self)


class DynamicLRFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, trainer, pl_module):
        # Optionally, run the LR finder at the start of training
        if 0 in self.milestones:
            self.lr_find(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        # Run the LR finder at the start of specified epochs
        if trainer.current_epoch in self.milestones:
            self.lr_find(trainer, pl_module)
