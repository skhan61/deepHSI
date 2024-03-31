import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torchvision.models.resnet import ResNet50_Weights


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


class HSIFCResNetModel(HSIFCModel):
    """
    Extends HSIFCModel with a ResNet backbone for feature extraction 
    from hyperspectral image patches.
    """

    def __init__(self, input_channels, patch_size, n_classes,
                 dropout=False, base_model='resnet50'):
        """
        Initializes the HSIFCResNetModel.

        Args:
            input_channels (int): Number of input spectral channels.
            patch_size (int): The size of the spatial dimensions 
            (assumed square patches).
            n_classes (int): Number of output classes.
            dropout (bool, optional): If True, applies dropout with p=0.5. 
            Defaults to False.
            base_model (str, optional): Specifies the ResNet variant to use. 
            Defaults to 'resnet50'.
        """
        # Initialize the HSIFCModel
        super(HSIFCResNetModel, self).__init__(
            input_channels, patch_size, n_classes, dropout)

        # Load the specified base ResNet model
        # Load the specified base ResNet model with new `weights` argument
        if base_model == 'resnet50':
            self.resnet_model = models.resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1)
        # Add other ResNet variants as needed
        # elif base_model == 'resnet18':
        #     self.resnet_model = models.resnet18(pretrained=True)
        # ...

        # Adjust the first convolutional layer of the ResNet model to
        # accept hyperspectral input
        self.resnet_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the classifier of the ResNet model with a new one
        num_features = self.resnet_model.fc.in_features
        self.resnet_model.fc = nn.Identity()

        # Modify the first fully connected layer of HSIFCModel to
        # accept features extracted by ResNet
        self.fc1 = nn.Linear(num_features, 2048)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor with shape 
            [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output logits.
        """
        # Check if input is 5D and remove the extra dimension
        if x.dim() == 5:
            # Assuming the extra dimension is the second one, which is usually 1
            x = x.squeeze(1)  # This removes dimensions of size 1 at index 1

        # Now x should be [batch_size, channels, height, width]
        # Proceed with the ResNet model
        x = self.resnet_model(x)

        # Pass the features through the fully connected layers for classification
        return super().forward(x)
