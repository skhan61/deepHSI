import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .base_model import HSIModelBase


class SpectralSpatialCNN(HSIModelBase):
    """
    Implementation of a 3D Convolutional Neural Network (3D CNN) for the spectral-spatial classification
    of hyperspectral imagery, based on the architecture proposed by Ying Li, Haokui Zhang, and Qiang Shen in:

    "Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network."

    This paper was published in MDPI Remote Sensing, Volume 9, Issue 1, January 2017, Article 67.

    The key contribution of the paper is the introduction of a 3D CNN approach that simultaneously captures
    the spectral and spatial information in hyperspectral images, offering a significant advancement in
    classification performance. The proposed network architecture leverages the depth of hyperspectral data
    by treating the spectral channels as sequential input layers to the 3D convolutions, thereby enabling
    the extraction of features that are inherently spectral-spatial in nature.

    This implementation adapts the described architecture to facilitate spectral-spatial classification
    tasks within the hyperspectral imaging domain, ensuring that the unique characteristics of hyperspectral
    data are effectively utilized.

    Reference:
    Li, Y., Zhang, H., & Shen, Q. (2017). Spectral–Spatial Classification of Hyperspectral Imagery with 3D 
    Convolutional Neural Network. MDPI Remote Sensing, 9(1), 67. Available at: http://www.mdpi.com/2072-4292/9/1/67
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight)
            init.constant_(m.bias, 0)

    def __init__(self, input_channels, n_classes, n_planes=2, patch_size=5, dropout=False):
        super(SpectralSpatialCNN, self).__init__(
            input_channels, patch_size, n_classes, dropout)

        self.conv1 = nn.Conv3d(1, n_planes, (7, 3, 3), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(n_planes, 2 * n_planes,
                               (3, 3, 3), padding=(1, 0, 0))
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_channels,
                            self.patch_size, self.patch_size)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.features_size)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc(x)
        return x
