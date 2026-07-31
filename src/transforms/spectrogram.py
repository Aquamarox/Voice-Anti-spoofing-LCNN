import torch
import torchaudio
from torch import nn


class PowerSpectrogram(nn.Module):
    """
    Convert a batch of waveforms into power spectrograms.
    """

    def __init__(
        self,
        n_fft=512,
        win_length=320,
        hop_length=160,
    ):
        super().__init__()

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
            center=False,
        )

    def forward(self, x):
        spectrogram = self.spectrogram(x)

        # [B, frequency, time] -> [B, 1, frequency, time]
        return spectrogram.unsqueeze(1)


class LogTransform(nn.Module):
    """
    Apply a numerically stable logarithm.
    """

    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))