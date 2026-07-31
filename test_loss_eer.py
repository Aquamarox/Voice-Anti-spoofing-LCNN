import numpy as np
import torch

from src.loss import AntiSpoofingLoss
from src.metrics import (
    compute_bonafide_scores,
    compute_eer,
)


def test_loss():
    criterion = AntiSpoofingLoss()

    logits = torch.tensor(
        [
            [3.0, -1.0],
            [-2.0, 4.0],
        ],
        requires_grad=True,
    )

    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    result = criterion(
        logits=logits,
        labels=labels,
    )

    loss = result["loss"]

    print("Loss:", loss.item())

    assert "loss" in result
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_scores():
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [3.0, 1.0],
            [0.0, 4.0],
            [1.0, 3.0],
        ]
    )

    scores = compute_bonafide_scores(logits)

    expected = torch.tensor(
        [-4.0, -2.0, 4.0, 2.0]
    )

    print("Bona fide scores:", scores)

    assert scores.shape == (4,)
    assert torch.equal(scores, expected)


def test_eer():
    bonafide_scores = np.array(
        [4.0, 2.0]
    )

    spoof_scores = np.array(
        [-4.0, -2.0]
    )

    eer, threshold = compute_eer(
        bonafide_scores,
        spoof_scores,
    )

    print("EER:", eer)
    print("Threshold:", threshold)

    assert eer == 0.0
    assert np.isfinite(threshold)


def main():
    torch.manual_seed(42)

    test_loss()
    test_scores()
    test_eer()

    print("\nLoss and EER smoke test passed.")


if __name__ == "__main__":
    main()