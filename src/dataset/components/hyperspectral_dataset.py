import numpy as np
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):
    """Generic class for a hyperspectral scene."""

    def __init__(self, data, gt, transform=None, **hyperparams):
        """Initializes the dataset.

        Args:
            data (np.ndarray): 3D hyperspectral image.
            gt (np.ndarray): 2D array of labels.
            hyperparams (dict): Dictionary of hyperparameters including:
                                - dataset: Name of the dataset.
                                - patch_size: Size of the spatial neighbourhood.
                                - ignored_labels: Labels to ignore.
                                - center_pixel: Whether to consider only the label of the center pixel.
                                - supervision: Type of supervision ('full' or 'semi').
        """
        super().__init__()

        self.data = data
        self.label = gt

        # self.name = hyperparams.get("dataset", "Unknown")

        self.patch_size = hyperparams.get("patch_size", 5)

        self.ignored_labels = set(hyperparams.get("ignored_labels", []))

        self.center_pixel = hyperparams.get("center_pixel", True)

        supervision = hyperparams.get("supervision", "full")

        self.transform = transform

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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        # Ensure consistent shape for all patch sizes
        if self.patch_size == 1:
            # Squeeze out the single spatial dimension but keep the channel dimension
            data = data.squeeze(axis=(0, 1))  # Shape (C,)
            data = np.expand_dims(data, axis=(1, 2))  # Shape (C, 1, 1)
            label = label.item()  # Extract scalar value
        else:
            # Rearrange dimensions to CxHxW
            data = np.copy(data).transpose((2, 0, 1))

        # Convert the data to PyTorch tensors
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply transformations if provided
        if self.transform:
            data = self.transform(data)

        # If center_pixel is True, return the label of the center pixel
        if self.center_pixel and self.patch_size > 1:
            # Extract the label of the center pixel
            center = self.patch_size // 2
            label = label[center, center].item()  # Convert to scalar

        # # Remove unused dimensions when we work with invidual spectrums
        # elif self.patch_size == 1:
        #     data = data[:, 0, 0]
        #     label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label

    # def __getitem__(self, i):
    #     x, y = self.indices[i]
    #     x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
    #     x2, y2 = x1 + self.patch_size, y1 + self.patch_size

    #     data = self.data[x1:x2, y1:y2]
    #     label = self.label[x1:x2, y1:y2]

    #     # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
    #     data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype="float32")
    #     label = np.asarray(np.copy(label), dtype="int64")
    #     # Load the data into PyTorch tensors
    #     data = torch.from_numpy(data)
    #     label = torch.from_numpy(label)

    #     # Apply transformations if provided
    #     if self.transform:
    #         data = self.transform(data_patch)

    #     # Extract the center label if needed
    #     if self.center_pixel and self.patch_size > 1:
    #         label = label[self.patch_size // 2, self.patch_size // 2]

    #     # Remove unused dimensions when we work with invidual spectrums
    #     elif self.patch_size == 1:
    #         data = data[:, 0, 0]
    #         label = label[0, 0]

    #     # Add a fourth dimension for 3D CNN
    #     if self.patch_size > 1:
    #         # Make 4D data ((Batch x) Planes x Channels x Width x Height)
    #         data = data.unsqueeze(0)
    #     return data, label
