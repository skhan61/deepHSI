import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class HSIFCModel(nn.Module):
    """
    Fully Connected Neural Network for Hyperspectral Image Classification.
    Designed to work with 3D input tensors representing hyperspectral image patches.
    """

    @staticmethod
    def weight_init(m):
        """
        Initializes the weights of the linear layers using Kaiming Normal initialization
        and sets biases to zero.
        """
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels,
                 patch_size, n_classes,
                 dropout=False):
        """
        Initializes the HSIFCModel.

        Args:
            input_channels (int): Number of input spectral channels.
            patch_size (int): The size of the spatial dimensions (assumed square patches).
            n_classes (int): Number of output classes.
            dropout (bool, optional): If True, applies dropout with p=0.5. Defaults to False.
        """
        super(HSIFCModel, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size
        self.use_dropout = dropout

        # Calculate the total number of features after flattening
        input_features = input_channels * patch_size * patch_size

        # print('==========')
        # print(input_features)
        # print('===========')

        # Fully connected layers
        self.fc1 = nn.Linear(input_features, 2048)
        self.fc2 = nn.Linear(2048, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, n_classes)

        if dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.apply(self.weight_init)  # Apply weight initialization

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, planes, channels, height, width].
                            The spatial dimensions (height, width) and planes are assumed to be flattened.

        Returns:
            torch.Tensor: Output logits.
        """
        # Flatten the input tensor except for the batch dimension
        # print('from model...')
        # print(x.shape)
        x = x.reshape(x.size(0), -1)  # Use .reshape() instead of .view()

        # print(x.shape)

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
