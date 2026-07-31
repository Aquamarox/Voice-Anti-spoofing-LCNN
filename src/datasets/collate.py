import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Combine individual dataset items into a batch.
    """

    result_batch = {}

    data_objects = [
        item["data_object"] for item in dataset_items
    ]

    if data_objects[0].ndim == 1:
        # Audio recordings may have different lengths.
        result_batch["data_object"] = pad_sequence(
            data_objects,
            batch_first=True,
            padding_value=0.0,
        )
    else:
        # Keep the original template example working.
        result_batch["data_object"] = torch.vstack(
            data_objects
        )

    result_batch["labels"] = torch.tensor(
        [item["labels"] for item in dataset_items]
    )

    if "utterance_id" in dataset_items[0]:
        result_batch["utterance_id"] = [
            item["utterance_id"] for item in dataset_items
        ]

    return result_batch