import numpy as np
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    """Generic class for a hyperspectral scene."""

    def __init__(self, data, gt, **hyperparams):
        """Initializes the dataset.

        Args:
            data (np.ndarray): 3D hyperspectral image.
            gt (np.ndarray): 2D array of labels.
            hyperparams (dict): Dictionary of hyperparameters including:
                                - dataset: Name of the dataset.
                                - patch_size: Size of the spatial neighbourhood.
                                - ignored_labels: Labels to ignore.
                                - flip_augmentation: Whether to perform flip augmentation.
                                - radiation_augmentation: Whether to perform radiation augmentation.
                                - mixture_augmentation: Whether to perform mixture augmentation.
                                - center_pixel: Whether to consider only the label of the center pixel.
                                - supervision: Type of supervision ('full' or 'semi').
        """
        super().__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams.get("dataset", "Unknown")
        self.patch_size = hyperparams.get("patch_size", 5)
        self.ignored_labels = set(hyperparams.get("ignored_labels", []))
        self.flip_augmentation = hyperparams.get("flip_augmentation", False)
        self.radiation_augmentation = hyperparams.get("radiation_augmentation", False)
        self.mixture_augmentation = hyperparams.get("mixture_augmentation", False)
        self.center_pixel = hyperparams.get("center_pixel", True)

        supervision = hyperparams.get("supervision", "full")

        # Fully supervised: use all pixels with labels not in ignored_labels
        if supervision == "full":
            mask = np.ones_like(gt, dtype=bool)
            for l in self.ignored_labels:
                mask[gt == l] = 0

        # Semi-supervised: use all pixels, except padding
        elif supervision == "semi":
            mask = np.ones_like(gt, dtype=bool)

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = [
            (x, y)
            for x, y in zip(x_pos, y_pos)
            if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p
        ]
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1.0, size=2)
        noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert self.labels[l_indice] == value
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
        label = np.asarray(np.copy(label), dtype="int64")

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)
        return data, label
