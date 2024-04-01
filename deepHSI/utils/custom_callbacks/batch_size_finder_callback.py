from lightning.pytorch.callbacks import BatchSizeFinder


class InitialBatchSizeFinder(BatchSizeFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, trainer, pl_module, *args, **kwargs):
        pass  # We don't need to do anything special at the fit start

    def on_train_epoch_start(self, trainer, pl_module, *args, **kwargs):
        # Run the batch size finder only at the start
        # of the first training epoch
        if trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)
