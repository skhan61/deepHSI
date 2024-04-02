import torch


class HSITransform:
    def __call__(self, patch):
        raise NotImplementedError(
            "This method needs to be implemented by subclasses.")


class Compose(HSITransform):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, patch):
        for t in self.transforms:
            patch = t(patch)
        return patch


class HSINormalize(HSITransform):
    """
    Normalize each spectral band of the HSI data to the range [0, 1].
    Assumes the input is always a PyTorch Tensor.
    """

    def __call__(self, patch):
        normalized_patch = torch.zeros_like(patch)
        for band in range(patch.shape[0]):
            min_val = torch.min(patch[band, :, :])
            max_val = torch.max(patch[band, :, :])

            # Avoid division by zero
            if max_val - min_val > 0:
                normalized_patch[band, :, :] = (
                    patch[band, :, :] - min_val) / (max_val - min_val)
            else:
                # If max equals min, the output is a tensor of zeros
                # This condition might need handling depending on the use case
                normalized_patch[band, :, :] = patch[band, :, :]

        return normalized_patch


class HSIFlip(HSITransform):
    """
    Horizontally flip the hyperspectral image.
    """

    def __call__(self, patch):
        # Assuming dims are [channels, height, width]
        return torch.flip(patch, dims=[2])


class HSIRotate(HSITransform):
    """
    Rotate the hyperspectral image by 90 degrees.
    """

    def __call__(self, patch):
        # Rotate 90 degrees clockwise
        return torch.rot90(patch, k=1, dims=[1, 2])


class HSISpectralNoise(HSITransform):
    """
    Add Gaussian noise to each spectral band independently.
    """

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, patch):
        noise = torch.randn_like(patch) * self.std + self.mean
        return patch + noise


class HSISpectralShift(HSITransform):
    """
    Shift spectral bands by a certain number of places.
    """

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, patch):
        return torch.roll(patch, shifts=self.shift, dims=0)


class HSIRandomSpectralDrop(HSITransform):
    """
    Randomly drop out entire spectral bands.
    """

    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, patch):
        for band in range(patch.shape[0]):
            if torch.rand(1) < self.drop_prob:
                patch[band, :, :] = 0  # Setting the entire band to 0
        return patch
