import torch
from torch import nn


class AntiSpoofingLoss(nn.Module):
    """
    Cross-entropy loss for binary anti-spoofing classification.

    Labels:
        spoof: 0
        bonafide: 1
    """

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **batch,
    ):
        loss = self.cross_entropy(logits, labels)

        return {"loss": loss}