import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .base_model import HSIModelBase


class SpectralSpatialCNN(HSIModelBase):
    """
    Implementation of a 3D Convolutional Neural Network (3D CNN) for the spectral-spatial classification
    of hyperspectral imagery. This model is based on the architecture proposed by Ying Li, Haokui Zhang, 
    and Qiang Shen in "Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional 
    Neural Network," published in MDPI Remote Sensing, Volume 9, Issue 1, January 2017, Article 67.

    The 3D CNN approach captures both spectral and spatial information in hyperspectral images by 
    treating the spectral channels as sequential input layers to 3D convolutions. This enables the 
    extraction of features that are inherently spectral-spatial in nature, significantly advancing 
    classification performance.

    The network architecture leverages the depth of hyperspectral data, utilizing the spectral channels 
    effectively for feature extraction. This implementation adapts the described architecture for 
    spectral-spatial classification tasks within the hyperspectral imaging domain.

    Reference:
    Li, Y., Zhang, H., & Shen, Q. (2017). Spectral–Spatial Classification of Hyperspectral Imagery with 
    3D Convolutional Neural Network. MDPI Remote Sensing, 9(1), 67. 
    Available at: http://www.mdpi.com/2072-4292/9/1/67

    Parameters:
    - input_channels (int): Number of spectral channels in the input hyperspectral image.
    - n_classes (int): Number of classification labels or classes.
    - n_planes (int): Number of output channels for the first 3D convolution layer.
    - patch_size (int): Spatial size (height and width) of the input patch. Assumes square patches.
    - dropout (bool): Whether to include dropout in the network. If True, a dropout layer is added.

    Input Shape:
    - Input tensor shape: (batch_size, 1, input_channels, patch_size, patch_size)
      where `batch_size` is the number of hyperspectral image patches, `1` is the number of input channels
      (single-band, grayscale patches, typically pre-processed hyperspectral data),
      `input_channels` corresponds to the number of spectral bands, and `patch_size` is the spatial dimension
      of each patch.

    Output Shape:
    - Output tensor shape: (batch_size, n_classes)
      where `batch_size` matches the input, and `n_classes` is the number of target classification labels.

    Example Usage:
    >>> model = SpectralSpatialCNN(input_channels=103, n_classes=9, n_planes=1, patch_size=5, dropout=True)
    >>> input_tensor = torch.rand(64, 1, 103, 5, 5)  # Example input (batch, depth/plane, C, H, W)
    >>> output = model(input_tensor)
    >>> print(output.shape)  # Expected output shape: torch.Size([64, 9])
    """

# class SpectralSpatialCNN(HSIModelBase):
#     # The class documentation remains the same as provided before.

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5, dropout=False):
        super(SpectralSpatialCNN, self).__init__(
            input_channels, patch_size, n_classes, dropout)

        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(
            3, 1, 1))  # Adjusted padding for proper shape
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               # Adjusted kernel size & padding for demonstration
                               (5, 3, 3), padding=(2, 1, 1))
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            # Initializing a dummy input to calculate the size after convolutions
            x = torch.zeros(1, 1, self.input_channels,
                            self.patch_size, self.patch_size)
            # print(f"Initial dummy shape: {x.shape}")
            x = F.relu(self.conv1(x))
            # print(f"After conv1 dummy shape: {x.shape}")
            x = F.relu(self.conv2(x))
            # print(f"After conv2 dummy shape: {x.shape}")
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"After conv1 shape: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After conv2 shape: {x.shape}")

        x = x.view(-1, self.features_size)
        # print(f"After flattening shape: {x.shape}")

        if self.use_dropout:
            x = self.dropout(x)
            # print(f"After dropout shape: {x.shape}")

        x = self.fc(x)
        # print(f"After fc shape: {x.shape}")

        return x
