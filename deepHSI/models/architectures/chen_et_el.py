import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .base_model import HSIModelBase


class HyperspectralCNNDetector(HSIModelBase):
    """
    A Convolutional Neural Network (CNN) architecture for hyperspectral image detection and classification, 
    inspired by the work of Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia, and Pedram Ghamisi in their paper:

    "Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks."

    Published in IEEE Transactions on Geoscience and Remote Sensing (TGRS), Volume 55, Issue 10, October 2017.

    This model is designed to extract deep features from hyperspectral images and perform classification tasks,
    leveraging the unique spatial-spectral characteristics of hyperspectral data. The original architecture
    proposed by Chen et al. introduces a novel approach to processing hyperspectral images using 3D convolutional
    layers, aiming to effectively capture both spatial and spectral information inherent in the data.

    The implementation herein is adapted from the described architecture, incorporating modifications to suit
    specific application needs and computational constraints. This class inherits from HSIModelBase, providing
    a standardized interface for models tailored to hyperspectral image analysis.

    Reference:
    Chen, Y., Jiang, H., Li, C., Jia, X., & Ghamisi, P. (2017). Deep Feature Extraction and Classification of Hyperspectral 
    Images Based on Convolutional Neural Networks. IEEE Transactions on Geoscience and Remote Sensing, 55(10), 6239-6255.
    DOI: 10.1109/TGRS.2017.2719629
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
        Initializes the HyperspectralCNNDetector model with the specified parameters.
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
