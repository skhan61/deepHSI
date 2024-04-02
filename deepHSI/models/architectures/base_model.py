import torch.nn as nn
import torch.nn.functional as F


class HSIModelBase(nn.Module):
    """
    Base class for Hyperspectral Image Classification models.
    """

    def __init__(self, input_channels,
                 patch_size,
                 n_classes,
                 dropout=False, **kwargs):
        """
        Base initialization for HSI models.

        Args:
            input_channels (int): Number of input spectral channels.
            patch_size (int): The size of the spatial dimensions (assumed square patches).
            n_classes (int): Number of output classes.
            dropout (bool, optional): If True, applies dropout with p=0.5. Defaults to False.
            **kwargs: Additional keyword arguments for future parameters.
        """
        super(HSIModelBase, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.use_dropout = dropout
        self.extra_params = kwargs  # Store any additional parameters

    # def weight_init(self, m):
    #     """
    #     Weight initialization callback.
    #     """
    #     if isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model. To be implemented by subclasses.
        """
        raise NotImplementedError("Forward pass is not implemented.")
