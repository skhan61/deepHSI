import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base_model import HSIModelBase


class DeepGRUHSIClassifier(HSIModelBase):
    """
    A model for hyperspectral image classification leveraging Gated Recurrent Units (GRUs) 
    to capture spectral-spatial 
    features. Inspired by Mou, Lichao, Pedram Ghamisi, and Xiao Xang Zhu's work, this model 
    is particularly effective 
    in handling HSI data where understanding the sequential nature of spectral bands is crucial.

    The model processes the spectral dimension as a sequence using GRU layers, followed by 
    batch normalization and 
    a fully connected layer for classification. This setup is designed to 
    extract meaningful features from both the 
    spectral and spatial dimensions of HSI data.

    Parameters:
        input_channels (int): The number of spectral bands in the input HSI data.
        n_classes (int): The number of unique classes or labels in the classification task.
        patch_size (int, optional): The size of the spatial dimension (height and width) of the 
                                    input HSI patches, assumed to be square. Defaults to 1, 
                                    which means the model primarily focuses on 
                                    spectral features.
        **kwargs: Additional keyword arguments passed to the HSIModelBase constructor.

    Attributes:
        gru (nn.GRU): GRU layer for processing the spectral sequence.
        gru_bn (nn.BatchNorm1d): Batch normalization layer applied to the GRU outputs.
        tanh (nn.Tanh): Activation function providing non-linearity.
        fc (nn.Linear): Fully connected layer mapping GRU outputs to class scores.
        dropout (nn.Dropout, optional): Dropout layer for regularization, included if 
        `use_dropout` is set to True.

    Example:
        # Initialize the classifier for HSI data with 102 spectral bands and 5 target classes.
        >>> model = DeepGRUHSIClassifier(input_channels=102, n_classes=5)

        # Assuming `hsi_data` is a tensor representing a batch of HSI data 
        # with dimensions [Batch, Plane, Channel, Height, Width]
        # where Plane is always 1.
        >>> inputs = torch.rand(64, 1, 102, 10, 10)  # Input shape: torch.Size([64, 1, 102, 10, 10])

        # Forward pass: Process the HSI data through the model to obtain class logits
        >>> output = model(inputs)  # Output shape: torch.Size([64, 5])
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            init.uniform_(m.weight.data, -0.1, 0.1)
            if m.bias is not None:
                init.uniform_(m.bias.data, -0.1, 0.1)
        elif isinstance(m, nn.GRU):
            # Initialize input-hidden weights
            init.uniform_(m.weight_ih_l0.data, -0.1, 0.1)
            # Initialize hidden-hidden weights
            init.uniform_(m.weight_hh_l0.data, -0.1, 0.1)
            # Initialize biases, if they are used
            if m.bias:
                init.uniform_(m.bias_ih_l0.data, -0.1, 0.1)
                init.uniform_(m.bias_hh_l0.data, -0.1, 0.1)

    def __init__(self, input_channels, n_classes, patch_size=1, **kwargs):
        super(DeepGRUHSIClassifier, self).__init__(
            input_channels, patch_size, n_classes, **kwargs)
        self.height = patch_size
        self.width = patch_size
        self.gru = nn.GRU(self.height * self.width, 64, 1, bidirectional=False)
        self.gru_bn = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(64, n_classes)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.apply(self.weight_init)

    def forward(self, x):
        # print(f"Initial shape: {x.shape}")

        # Squeeze the plane dimension out as it's not needed
        # Now shape is [batch_size, input_channels, height, width]
        x = x.squeeze(1)
        # print(f"After squeezing plane dimension: {x.shape}")

        # Reshape for GRU processing:
        # Flatten the spatial dimensions and treat the spectral dimension as the sequence
        # This makes the input shape compatible with what GRU expects: [seq_len, batch, features]
        batch_size, channels, height, width = x.size()
        # Rearrange so channels are last: [batch_size, height, width, channels]
        x = x.permute(0, 2, 3, 1)
        # Flatten spatial dimensions and keep channels as features
        x = x.reshape(batch_size, height*width, channels)
        # Permute to get [seq_len (channels), batch_size, spatial_features]
        x = x.permute(2, 0, 1)
        # print(f"After reshape for GRU: {x.shape}")

        # GRU processing
        x, _ = self.gru(x)
        # print(f"After GRU: {x.shape}")

        # Selecting the last output for classification
        x = x[-1, :, :]
        # print(f"After selecting the last output: {x.shape}")

        # Conditional BatchNorm1d
        if x.size(0) > 1:
            x = self.gru_bn(x)
            # print(f"After BatchNorm1d: {x.shape}")
        # else:
        #     print("Skipping BatchNorm1d due to batch size 1")

        # Activation and optional dropout
        x = self.tanh(x)
        if self.use_dropout:
            x = self.dropout(x)
        # print(f"After Tanh and optional dropout: {x.shape}")

        # Fully connected layer for final output
        x = self.fc(x)
        # print(f"Final output shape: {x.shape}")

        return x
