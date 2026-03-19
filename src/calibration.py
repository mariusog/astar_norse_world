"""Simulation calibration by comparing observed and simulated distributions.

Detects systematic biases in local simulation vs server observations
and computes optimal blend weights.
"""

from __future__ import annotations

import csv
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.constants import (
    CALIBRATION_KL_SCALE,
    NUM_PREDICTION_CLASSES,
    OBSERVATION_WEIGHT,
    PROBABILITY_FLOOR,
)

logger = logging.getLogger(__name__)


def compute_divergence(
    observed_probs: np.ndarray,
    simulated_probs: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Compute KL divergence between observed and simulated.

    Only computes divergence for cells where mask is True.

    Args:
        observed_probs: H x W x 6 from observations.
        simulated_probs: H x W x 6 from simulation.
        mask: H x W boolean mask of cells to evaluate.

    Returns:
        Dict with 'per_cell' (H x W array), 'mean_kl', 'max_kl'.
    """
    h, w = mask.shape
    per_cell = np.zeros((h, w), dtype=np.float64)

    if not mask.any():
        return {"per_cell": per_cell, "mean_kl": 0.0, "max_kl": 0.0}

    obs_safe = np.maximum(observed_probs, PROBABILITY_FLOOR)
    sim_safe = np.maximum(simulated_probs, PROBABILITY_FLOOR)

    # Renormalize to ensure valid distributions
    obs_safe = _normalize(obs_safe)
    sim_safe = _normalize(sim_safe)

    # KL(obs || sim) = sum(obs * log(obs / sim))
    kl = obs_safe * np.log(obs_safe / sim_safe)
    kl_sum = kl.sum(axis=2)

    per_cell[mask] = kl_sum[mask]
    mean_kl = float(per_cell[mask].mean())
    max_kl = float(per_cell[mask].max())

    return {"per_cell": per_cell, "mean_kl": mean_kl, "max_kl": max_kl}


def detect_biases(
    observed_probs: np.ndarray,
    simulated_probs: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, str | float]]:
    """Identify systematic differences per terrain class.

    Compares mean probability for each class across observed cells.

    Args:
        observed_probs: H x W x 6 from observations.
        simulated_probs: H x W x 6 from simulation.
        mask: H x W boolean mask of observed cells.

    Returns:
        List of bias dicts with class_name, obs_mean, sim_mean, delta.
    """
    if not mask.any():
        return []

    class_names = _class_names()
    biases: list[dict[str, str | float]] = []

    for cls_idx in range(NUM_PREDICTION_CLASSES):
        obs_mean = float(observed_probs[mask, cls_idx].mean())
        sim_mean = float(simulated_probs[mask, cls_idx].mean())
        delta = obs_mean - sim_mean

        biases.append(
            {
                "class_name": class_names[cls_idx],
                "obs_mean": round(obs_mean, 4),
                "sim_mean": round(sim_mean, 4),
                "delta": round(delta, 4),
            }
        )

    return biases


def calibrate_weights(
    observed_probs: np.ndarray,
    simulated_probs: np.ndarray,
    mask: np.ndarray,
    num_steps: int = 21,
) -> float:
    """Compute blend weight based on sim-obs agreement.

    When sim and obs agree closely, trust sim more (lower obs weight).
    When they diverge, trust obs more (higher obs weight).
    Uses a sigmoid-like mapping from KL divergence to weight.

    Args:
        observed_probs: H x W x 6 from observations.
        simulated_probs: H x W x 6 from simulation.
        mask: H x W boolean mask of observed cells.
        num_steps: Unused, kept for backward compatibility.

    Returns:
        Observation weight (0.1 to 0.95).
    """
    if not mask.any():
        return OBSERVATION_WEIGHT

    div = compute_divergence(observed_probs, simulated_probs, mask)
    mean_kl = float(div["mean_kl"])

    # Low KL (sim agrees with obs) -> weight ~0.5 (trust both)
    # High KL (sim disagrees) -> weight ~0.95 (trust obs more)
    weight = 0.5 + 0.45 * (1.0 - np.exp(-CALIBRATION_KL_SCALE * mean_kl))
    weight = float(np.clip(weight, 0.1, 0.95))

    logger.info(
        "Calibrated obs weight: %.2f (mean_kl=%.6f)",
        weight,
        mean_kl,
    )
    return weight


def suggest_constant_adjustments(
    biases: list[dict[str, str | float]],
) -> list[str]:
    """Generate human-readable suggestions from detected biases.

    Args:
        biases: Output from detect_biases().

    Returns:
        List of suggestion strings.
    """
    suggestions: list[str] = []
    threshold = 0.05

    for bias in biases:
        delta = float(bias["delta"])
        name = bias["class_name"]
        if abs(delta) < threshold:
            continue
        direction = "over" if delta < 0 else "under"
        suggestions.append(
            f"Sim {direction}-predicts {name} by {abs(delta):.3f}. "
            f"Consider adjusting {name}-related constants."
        )

    if not suggestions:
        suggestions.append("No significant biases detected.")

    return suggestions


def write_calibration_report(
    biases: list[dict[str, str | float]],
    optimal_weight: float,
    divergence: dict[str, float | np.ndarray],
    output_dir: str = "logs",
) -> str:
    """Write calibration results to a CSV log file.

    Args:
        biases: Per-class bias data.
        optimal_weight: Computed optimal observation weight.
        divergence: KL divergence statistics.
        output_dir: Directory for output file.

    Returns:
        Path to the written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"{output_dir}/calibration_{ts}.csv"

    with Path(filepath).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["optimal_obs_weight", f"{optimal_weight:.4f}"])
        writer.writerow(["mean_kl", f"{divergence['mean_kl']:.6f}"])
        writer.writerow(["max_kl", f"{divergence['max_kl']:.6f}"])
        writer.writerow([])
        writer.writerow(["class", "obs_mean", "sim_mean", "delta"])
        for bias in biases:
            writer.writerow(
                [
                    bias["class_name"],
                    bias["obs_mean"],
                    bias["sim_mean"],
                    bias["delta"],
                ]
            )

    logger.info("Calibration report written to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(probs: np.ndarray) -> np.ndarray:
    """Renormalize probability tensor along class axis."""
    sums = probs.sum(axis=2, keepdims=True)
    sums = np.maximum(sums, 1e-10)
    return probs / sums


def _class_names() -> list[str]:
    """Human-readable names for the 6 prediction classes."""
    return [
        "empty",
        "settlement",
        "port",
        "ruin",
        "forest",
        "mountain",
    ]
