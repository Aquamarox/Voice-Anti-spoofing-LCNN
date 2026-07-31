import numpy as np
import torch


def compute_bonafide_scores(logits: torch.Tensor):
    """
    Convert two-class logits into continuous bona fide scores.

    Higher score means stronger support for bona fide.
    """

    if logits.ndim != 2:
        raise ValueError(
            "Expected logits with shape [batch_size, 2], "
            f"got {tuple(logits.shape)}"
        )

    if logits.shape[1] != 2:
        raise ValueError(
            "Expected logits for two classes, "
            f"got {logits.shape[1]} classes"
        )

    spoof_logits = logits[:, 0]
    bonafide_logits = logits[:, 1]

    return bonafide_logits - spoof_logits


def compute_det_curve(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
):
    """
    Compute false rejection and false acceptance curves.
    """

    number_of_scores = (
        bonafide_scores.size + spoof_scores.size
    )

    all_scores = np.concatenate(
        (bonafide_scores, spoof_scores)
    )

    labels = np.concatenate(
        (
            np.ones(bonafide_scores.size),
            np.zeros(spoof_scores.size),
        )
    )

    indices = np.argsort(
        all_scores,
        kind="mergesort",
    )

    labels = labels[indices]

    bonafide_sums = np.cumsum(labels)

    spoof_sums = spoof_scores.size - (
        np.arange(1, number_of_scores + 1)
        - bonafide_sums
    )

    false_rejection_rates = np.concatenate(
        (
            np.atleast_1d(0),
            bonafide_sums / bonafide_scores.size,
        )
    )

    false_acceptance_rates = np.concatenate(
        (
            np.atleast_1d(1),
            spoof_sums / spoof_scores.size,
        )
    )

    thresholds = np.concatenate(
        (
            np.atleast_1d(
                all_scores[indices[0]] - 0.001
            ),
            all_scores[indices],
        )
    )

    return (
        false_rejection_rates,
        false_acceptance_rates,
        thresholds,
    )


def compute_eer(
    bonafide_scores,
    spoof_scores,
):
    """
    Compute equal error rate and its threshold.

    Returns:
        eer: Equal error rate in the range [0, 1].
        threshold: Decision threshold at the EER operating point.
    """

    bonafide_scores = np.asarray(
        bonafide_scores,
        dtype=np.float64,
    )

    spoof_scores = np.asarray(
        spoof_scores,
        dtype=np.float64,
    )

    if bonafide_scores.size == 0:
        raise ValueError("No bona fide scores were provided")

    if spoof_scores.size == 0:
        raise ValueError("No spoof scores were provided")

    all_scores = np.concatenate(
        (bonafide_scores, spoof_scores)
    )

    if not np.isfinite(all_scores).all():
        raise ValueError(
            "Scores contain NaN or infinity"
        )

    (
        false_rejection_rates,
        false_acceptance_rates,
        thresholds,
    ) = compute_det_curve(
        bonafide_scores,
        spoof_scores,
    )

    differences = np.abs(
        false_rejection_rates
        - false_acceptance_rates
    )

    min_index = np.argmin(differences)

    eer = np.mean(
        (
            false_rejection_rates[min_index],
            false_acceptance_rates[min_index],
        )
    )

    threshold = thresholds[min_index]

    return float(eer), float(threshold)