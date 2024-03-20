import numpy as np
import torch
from torch.utils.data import Dataset

#    """Generic class for a hyperspectral scene."""


class HyperspectralDataset(Dataset):
    """
    Provides a PyTorch dataset for hyperspectral images, facilitating the extraction and use of spatial
    patches and their corresponding labels for model training and evaluation.

    Hyperspectral images are 3D data cubes with two spatial dimensions and one spectral dimension.
    This class enables the extraction of small 3D patches from these data cubes for deep learning purposes,
    particularly useful in remote sensing applications.

    Args:
        data (np.ndarray): The hyperspectral image data as a 3D array of shape (Height, Width, Bands),
                           where Bands is the number of spectral bands.
        gt (np.ndarray): The ground truth labels as a 2D array of shape (Height, Width), where each
                         element represents the class label of the corresponding pixel in 'data'.
        transform (callable, optional): An optional function/transform that takes in a 3D patch and
                                        returns a transformed version. This is useful for data augmentation.
        **hyperparams (dict): A dictionary containing additional hyperparameters:
            - patch_size (int): The size of the square patches to be extracted from the hyperspectral image.
                                Defaults to 5.
            - ignored_labels (list): A list of label values to ignore during training and evaluation. Defaults to [].
            - center_pixel (bool): If True, the label for a patch is determined solely by the label of its center pixel.
                                   Defaults to True.
            - supervision (str): Specifies the supervision mode - 'full' for fully supervised learning where only
                                 pixels with labels not in ignored_labels are used, and 'semi' for semi-supervised
                                 learning where all pixels are used. Defaults to 'full'.

    Attributes:
        data (np.ndarray): Stored hyperspectral image data.
        label (np.ndarray): Stored ground truth labels.
        patch_size (int): The size of the patches to be extracted.
        ignored_labels (set): Set of labels to be ignored.
        center_pixel (bool): Flag indicating whether to use only the center pixel label for patches.
        indices (list): List of valid pixel indices for patch extraction.
        labels (list): List of labels corresponding to the patches extracted at 'indices'.

    Example:
        >>> dataset = HyperspectralDataset(data, gt, patch_size=7, ignored_labels=[0], center_pixel=True)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for patches, labels in dataloader:
        >>>     # Forward pass through your model
        >>>     outputs = model(patches)
        >>>     # Compute loss and backpropagate
        >>>     loss = criterion(outputs, labels)
        >>>     loss.backward()

    Note:
        This dataset class is intended for use with PyTorch models and data loaders. It handles the
        preprocessing of hyperspectral data into suitable forms for Convolutional Neural Networks (CNNs)
        and other deep learning models.
    """

    def __init__(self, data, gt, transform=None, **hyperparams):
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
        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label
