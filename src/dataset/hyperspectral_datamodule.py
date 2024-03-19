import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.dataset.components.hyperspectral_dataset import HyperspectralDataset
from src.dataset.components.utils import \
    load_dataset  # Assuming load_dataset is defined in utils.py


class BaseHyperspectralDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.transform = transform

    def setup_datasets(self, img, gt, ignored_labels):
        hyperparams = {
            "patch_size": self.patch_size,
            "ignored_labels": ignored_labels,
            "center_pixel": True,
            "supervision": "full",
        }
        dataset = HyperspectralDataset(
            img, gt, transform=self.transform, **hyperparams)
        self.train_dataset, self.val_dataset = random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class SalinasDataModule(BaseHyperspectralDataModule):
    def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None):
        super().__init__(data_dir, batch_size, patch_size, transform)
        self.dataset_name = "Salinas"

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            img, gt, _, ignored_labels, _, _ = load_dataset(
                self.dataset_name, self.data_dir)
            self.setup_datasets(img, gt, ignored_labels)


class PaviaUDataModule(BaseHyperspectralDataModule):
    def __init__(self, data_dir, batch_size=32, patch_size=5, transform=None):
        super().__init__(data_dir, batch_size, patch_size, transform)
        self.dataset_name = "PaviaU"

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            img, gt, _, ignored_labels, _, _ = load_dataset(
                self.dataset_name, self.data_dir)
            self.setup_datasets(img, gt, ignored_labels)
