import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .base_model import HSIModelBase


class HyperspectralCNNDetector(HSIModelBase):
    """A CNN for hyperspectral image detection and classification.

    This model extracts deep features from hyperspectral images to perform classification tasks. Inspired by Chen et al.,
    it uses 3D convolutional layers to process both spatial and spectral information. The design is adapted to suit
    different application needs and computational constraints, inheriting from HSIModelBase for a standardized model
    interface.

    Attributes:
        conv1: The first 3D convolutional layer (nn.Conv3d).
        conv2: The second 3D convolutional layer (nn.Conv3d).
        conv3: The third 3D convolutional layer (nn.Conv3d).
        pool1: The first max-pooling layer (nn.MaxPool3d).
        pool2: The second max-pooling layer (nn.MaxPool3d).
        fc: A fully connected layer for classification (nn.Linear).
        dropout: An optional dropout layer for regularization (nn.Dropout).

    Args:
        input_channels (int): The number of spectral bands in the input image.
        n_classes (int): The number of target classes for the output.
        patch_size (int, optional): The spatial size of the processed patches. Defaults to 27.
        n_planes (int, optional): The number of output channels (feature maps) in the convolutional layers. Defaults to 32.
        dropout (bool, optional): Whether to include dropout for regularization. Defaults to False.

    Example:
        >>> model = HyperspectralCNNDetector(input_channels=102, n_classes=9, \
                                                patch_size=27, n_planes=32, dropout=False)
        >>> inputs = torch.rand(64, 1, 102, 10, 10)  # [Batch size, 1, Spectral bands, Height, Width]
        >>> output = model(inputs)
        >>> print(f'Input shape: {inputs.shape}')
        >>> print(f'Output shape: {output.shape}')
        # Input shape: torch.Size([64, 1, 102, 10, 10])
        # Output shape: torch.Size([64, 9])

    Note:
        Input tensor should be 4D with the shape [Batch Size, 1, Spectral Bands, Height, Width]. 
        The output tensor shape is [Batch Size, Number of Classes], representing class probabilities.

    Reference:
        Chen, Y., Jiang, H., Li, C., Jia, X., & Ghamisi, P. (2017). Deep Feature Extraction and Classification of 
        Hyperspectral Images Based on Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing,
        55(10), 6239-6255. DOI: 10.1109/TGRS.2017.2719629
    """

    @staticmethod
    def weight_init(m):
        """
        Initializes weights with a standard deviation of 0.001 for Conv3d and Linear layers.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=27, n_planes=32, dropout=False):
        """
        Initializes the HyperspectralCNNDetector model 
        with the specified parameters.
        """
        super(HyperspectralCNNDetector, self).__init__(
            input_channels, patch_size, n_classes, dropout)
        self.n_planes = n_planes

        # Convolutional layers
        self.conv1 = nn.Conv3d(
            1, n_planes, (input_channels, 4, 4), padding='same')
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.conv2 = nn.Conv3d(n_planes, n_planes, (1, 4, 4), padding='same')
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.conv3 = nn.Conv3d(n_planes, n_planes, (1, 4, 4), padding='same')

        # Calculate the size of the flattened feature vector
        self.features_size = self._get_final_flattened_size()

        # Fully connected layer for classification
        self.fc = nn.Linear(self.features_size, n_classes)

        # Optional dropout for regularization
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

        # Weight initialization
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        """
        Computes the size of the tensor just before the fully connected layer.
        """
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels,
                            self.patch_size, self.patch_size)
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x):
        """
        Defines the forward pass of the HyperspectralCNNDetector model.
        """
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.conv3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x
