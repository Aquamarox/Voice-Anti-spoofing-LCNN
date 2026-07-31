from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import ASVspoofDataset
from src.datasets.collate import collate_fn
from src.transforms import (
    FixLength1D,
    LogTransform,
    PowerSpectrogram,
)


def main():
    dataset = ASVspoofDataset(
        protocol_path=Path(
            "data/ASVspoof2019_LA/"
            "ASVspoof2019_LA_cm_protocols/"
            "ASVspoof2019.LA.cm.train.trn.txt"
        ),
        audio_dir=Path(
            "data/ASVspoof2019_LA/"
            "ASVspoof2019_LA_train/flac"
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))

    frontend = torch.nn.Sequential(
        PowerSpectrogram(
            n_fft=512,
            win_length=320,
            hop_length=160,
        ),
        FixLength1D(target_length=750),
        LogTransform(eps=1e-10),
    )

    features = frontend(batch["data_object"])

    print("Waveform batch shape:", batch["data_object"].shape)
    print("Spectrogram shape:", features.shape)
    print("Minimum value:", features.min().item())
    print("Maximum value:", features.max().item())

    assert features.shape == (4, 1, 257, 750)
    assert torch.isfinite(features).all()

    print("\nSpectrogram smoke test passed.")


if __name__ == "__main__":
    main()