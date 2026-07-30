import torch.nn.functional as F
from torch import nn


class FixLength1D(nn.Module):
    """
    Crop long waveforms and pad short waveforms with zeros.
    """

    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        current_length = x.shape[-1]

        if current_length > self.target_length:
            x = x[..., :self.target_length]

        elif current_length < self.target_length:
            padding = self.target_length - current_length
            x = F.pad(x, (0, padding))

        return x
