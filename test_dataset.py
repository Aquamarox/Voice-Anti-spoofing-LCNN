from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets import ASVspoofDataset
from src.datasets.collate import collate_fn
from src.transforms import FixLength1D


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
        instance_transforms={
            "data_object": FixLength1D(target_length=64600)
        },
    )

    print("Dataset size:", len(dataset))
    print("Class counts:", dataset.class_counts)

    item = dataset[0]

    print("\nSingle item:")
    print("Keys:", item.keys())
    print("Waveform shape:", item["data_object"].shape)
    print("Label:", item["labels"])
    print("Utterance ID:", item["utterance_id"])

    assert len(dataset) > 0
    assert item["data_object"].shape == (1, 64600)
    assert item["labels"] in (0, 1)
    assert isinstance(item["utterance_id"], str)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))

    print("\nBatch:")
    print("Waveform shape:", batch["data_object"].shape)
    print("Labels:", batch["labels"])
    print("Utterance IDs:", batch["utterance_id"])

    assert batch["data_object"].shape == (4, 64600)
    assert batch["labels"].shape == (4,)
    assert len(batch["utterance_id"]) == 4

    print("\nDataset smoke test passed.")


if __name__ == "__main__":
    main()
