from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    """
    Dataset for the ASVspoof 2019 Logical Access task.

    Labels:
        spoof: 0
        bonafide: 1
    """

    def __init__(self, protocol_path, audio_dir, *args, **kwargs):
        protocol_path = Path(protocol_path)
        audio_dir = Path(audio_dir)

        index = self._create_index(protocol_path, audio_dir)

        self.class_counts = {
            "spoof": sum(item["label"] == 0 for item in index),
            "bonafide": sum(item["label"] == 1 for item in index),
        }

        super().__init__(index, *args, **kwargs)

    def _create_index(self, protocol_path, audio_dir):
        index = []

        with open(protocol_path, "r") as protocol:
            for line in protocol:
                parts = line.strip().split()

                utterance_id = parts[1]
                label = parts[-1]

                path = audio_dir / f"{utterance_id}.flac"

                index.append(
                    {
                        "path": str(path),
                        "label": 1 if label == "bonafide" else 0,
                        "utterance_id": utterance_id,
                    }
                )

        return index

    def __getitem__(self, index):
        item = self._index[index]

        waveform = self.load_object(item["path"])

        result = {
            "data_object": waveform,
            "labels": item["label"],
            "utterance_id": item["utterance_id"],
        }

        return self.preprocess_data(result)

    def load_object(self, path):
        waveform, sample_rate = torchaudio.load(path)

        if sample_rate != 16000:
            raise ValueError(
                f"Expected sample rate 16000, got {sample_rate}"
            )

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform
