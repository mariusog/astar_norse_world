"""Self-scoring evaluator for competition predictions.

Computes entropy-weighted KL divergence to estimate prediction quality
before submitting to the server. Matches the server scoring formula exactly.

See docs/scoring.md for the full specification.
"""

import numpy as np

from src.constants import (
    PROBABILITY_FLOOR,
    SCORE_DECAY_RATE,
    SCORE_ENTROPY_THRESHOLD,
)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute KL(p || q) per cell across the last axis.

    Args:
        p: Ground truth probabilities, shape (..., C).
        q: Predicted probabilities, shape (..., C).

    Returns:
        KL divergence per cell, shape (...). Non-negative floats.
    """
    q_safe = np.maximum(q, PROBABILITY_FLOOR)
    p_safe = np.maximum(p, PROBABILITY_FLOOR)
    # Renormalize after flooring
    q_safe = q_safe / q_safe.sum(axis=-1, keepdims=True)
    p_safe = p_safe / p_safe.sum(axis=-1, keepdims=True)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)


def entropy(p: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy per cell across the last axis.

    Uses the raw probabilities (no floor) so that truly static cells
    (e.g., permanent ocean) have entropy near zero and are correctly
    excluded from scoring.

    Args:
        p: Probability distributions, shape (..., C).

    Returns:
        Entropy per cell, shape (...). Non-negative floats.
    """
    # Replace zeros with 1.0 before log (log(1)=0, and p*0=0 anyway)
    p_masked = np.where(p > 0, p, 1.0)
    return -np.sum(p * np.log(p_masked), axis=-1)


def score_prediction(ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Score a prediction against ground truth using entropy-weighted KL divergence.

    Static cells (entropy below SCORE_ENTROPY_THRESHOLD) are excluded from
    the weighted average, matching server behavior.

    Args:
        ground_truth: H x W x C probability tensor.
        prediction: H x W x C probability tensor (same shape).

    Returns:
        Dictionary with keys:
            score: Final score on 0-100 scale.
            weighted_kl: Entropy-weighted mean KL divergence.
            num_dynamic_cells: Count of cells above entropy threshold.
            mean_entropy: Mean entropy of dynamic cells.
    """
    _validate_tensor_pair(ground_truth, prediction)

    cell_entropy = entropy(ground_truth)
    cell_kl = kl_divergence(ground_truth, prediction)

    dynamic_mask = cell_entropy >= SCORE_ENTROPY_THRESHOLD
    num_dynamic = int(np.sum(dynamic_mask))

    if num_dynamic == 0:
        return _empty_score_result()

    return _compute_weighted_score(
        cell_entropy[dynamic_mask],
        cell_kl[dynamic_mask],
        num_dynamic,
    )


def _empty_score_result() -> dict[str, float | int]:
    """Return a perfect score result when no dynamic cells exist."""
    return {
        "score": 100.0,
        "weighted_kl": 0.0,
        "num_dynamic_cells": 0,
        "mean_entropy": 0.0,
    }


def _compute_weighted_score(
    dynamic_entropy: np.ndarray,
    dynamic_kl: np.ndarray,
    num_dynamic: int,
) -> dict[str, float | int]:
    """Compute the entropy-weighted KL score for dynamic cells."""
    entropy_sum = np.sum(dynamic_entropy)
    weighted_kl = float(np.sum(dynamic_entropy * dynamic_kl) / entropy_sum)
    mean_ent = float(entropy_sum / num_dynamic)

    raw_score = 100.0 * np.exp(-SCORE_DECAY_RATE * weighted_kl)
    score = float(np.clip(raw_score, 0.0, 100.0))

    return {
        "score": score,
        "weighted_kl": weighted_kl,
        "num_dynamic_cells": num_dynamic,
        "mean_entropy": mean_ent,
    }


def score_against_mc(mc_ground_truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Score a prediction against Monte Carlo ground truth.

    Convenience wrapper around score_prediction for use with locally
    generated MC simulation results.

    Args:
        mc_ground_truth: H x W x C probability tensor from MC runs.
        prediction: H x W x C probability tensor.

    Returns:
        Same dictionary as score_prediction.
    """
    return score_prediction(mc_ground_truth, prediction)


def _validate_tensor_pair(ground_truth: np.ndarray, prediction: np.ndarray) -> None:
    """Validate that two probability tensors are compatible.

    Args:
        ground_truth: First tensor.
        prediction: Second tensor.

    Raises:
        ValueError: If shapes don't match or tensors have wrong dimensions.
    """
    if ground_truth.shape != prediction.shape:
        msg = f"Shape mismatch: ground_truth {ground_truth.shape} vs prediction {prediction.shape}"
        raise ValueError(msg)
    if ground_truth.ndim < 2:
        msg = f"Expected at least 2D tensor, got {ground_truth.ndim}D"
        raise ValueError(msg)
