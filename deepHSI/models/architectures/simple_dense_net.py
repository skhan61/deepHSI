import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init

from .base_model import HSIModelBase

# from torchvision.models.resnet import ResNet50_Weights


# # Assuming HSIModelBase is in the same directory,
# otherwise adjust the import path.
# from base_model import HSIModelBase


class HSIFCModel(HSIModelBase):
    """
    A Fully Connected (FC) Neural Network model designed specifically for the classification of 
    hyperspectral image (HSI) data. The model takes 3D input tensors representing hyperspectral 
    image patches and processes them through a series of fully connected layers to perform classification. 
    The model architecture includes four fully connected layers with ReLU activations and optional dropout for 
    regularization.

    The model is built upon the HSIModelBase to leverage common initialization routines and utilities tailored 
    for HSI analysis. This design facilitates easy integration into HSI processing pipelines and ensures 
    compatibility with standard HSI preprocessing and data handling practices.

    Parameters:
        input_channels (int): The number of spectral bands in each input HSI patch.
        patch_size (int): The spatial dimensions (height and width) of the input HSI patch, assumed to be square.
        n_classes (int): The number of classes for the classification task.
        dropout (bool, optional): If True, dropout layers are added after each fully connected layer except 
        the last one, for regularization. Default is False.
        **kwargs: Additional keyword arguments passed to the base class constructor (HSIModelBase).

    Attributes:
        fc1, fc2, fc3, fc4 (nn.Linear): Fully connected layers comprising the network architecture.
        dropout (nn.Dropout, optional): Dropout layer applied after each fully connected layer except the last one, 
                                        included if `dropout=True`.

    Example:
        # Initialize the FC model for a HSI patch with 102 spectral bands, 
        # 10x10 spatial dimensions, and 9 target classes.
        >>> model = HSIFCModel(input_channels=102, patch_size=10, n_classes=9, dropout=True)

        # Creating a dummy input tensor with dimensions: [Batch Size, Channels, Height, Width]
        # Batch Size: 64, Channels (Spectral Bands): 102, Height: 10, Width: 10
        >>> inputs = torch.rand(64, 1, 102, 10, 10)  # Input shape: torch.Size([64, 1, 102, 10, 10])

        # Forward pass: Pass the HSI patches through the model to obtain class logits
        >>> output = model(inputs)  # Output shape (logits): torch.Size([64, 9])

        # The output is a tensor of shape [Batch Size, Number of Classes], where each element 
        # represents the logit scores for each class, for each HSI patch in the batch.

    Note:
        The input tensor to the model should be 4-dimensional, following the format 
        [Batch Size, Channels, Height, Width], where 
        'Channels' corresponds to the spectral bands of the HSI patch. 
        The model internally flattens the spatial and spectral 
        dimensions before passing them through the fully connected layers.    """

    def __init__(self, input_channels, patch_size, n_classes, dropout=False, **kwargs):
        """
        Initializes the HSIFCModel.
        Inherits from HSIModelBase to make use of common initialization and utilities.
        """
        super(HSIFCModel, self).__init__(input_channels,
                                         patch_size, n_classes,
                                         dropout=dropout, **kwargs)

        # Calculate the total number of features after flattening
        input_features = self.input_channels * self.patch_size * self.patch_size

        # Fully connected layers
        self.fc1 = nn.Linear(input_features, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

        # Assuming weight_init is appropriately defined in HSIModelBase,
        # or you can define a custom method if needed.
        self.apply(self.weight_init)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = x.reshape(x.size(0), -1)  # Flatten the input tensor

        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc4(x)
        return x

    def weight_init(self, m):
        """Weight initialization callback."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
