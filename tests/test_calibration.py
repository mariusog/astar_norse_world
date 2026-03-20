"""Tests for simulation calibration (T23)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.calibration import (
    calibrate_weights,
    compute_divergence,
    detect_biases,
    suggest_constant_adjustments,
    write_calibration_report,
)
from src.constants import NUM_PREDICTION_CLASSES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uniform_probs() -> np.ndarray:
    """5x5x6 uniform probability tensor."""
    return np.full((5, 5, NUM_PREDICTION_CLASSES), 1.0 / 6.0)


@pytest.fixture
def biased_probs() -> np.ndarray:
    """5x5x6 tensor biased toward class 0."""
    p = np.full((5, 5, NUM_PREDICTION_CLASSES), 0.02)
    p[:, :, 0] = 0.9
    return p


@pytest.fixture
def full_mask() -> np.ndarray:
    """5x5 all-True mask."""
    return np.ones((5, 5), dtype=bool)


@pytest.fixture
def empty_mask() -> np.ndarray:
    """5x5 all-False mask."""
    return np.zeros((5, 5), dtype=bool)


# ---------------------------------------------------------------------------
# compute_divergence
# ---------------------------------------------------------------------------


class TestComputeDivergence:
    """Tests for KL divergence computation."""

    def test_identical_distributions_zero_kl(
        self,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        result = compute_divergence(uniform_probs, uniform_probs, full_mask)
        assert result["mean_kl"] == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_positive_kl(
        self,
        uniform_probs: np.ndarray,
        biased_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        result = compute_divergence(biased_probs, uniform_probs, full_mask)
        assert result["mean_kl"] > 0.0

    def test_empty_mask_returns_zeros(
        self,
        uniform_probs: np.ndarray,
        empty_mask: np.ndarray,
    ) -> None:
        result = compute_divergence(uniform_probs, uniform_probs, empty_mask)
        assert result["mean_kl"] == 0.0
        assert result["max_kl"] == 0.0

    def test_per_cell_shape(
        self,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        result = compute_divergence(uniform_probs, uniform_probs, full_mask)
        assert result["per_cell"].shape == (5, 5)

    def test_partial_mask(
        self,
        uniform_probs: np.ndarray,
        biased_probs: np.ndarray,
    ) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        result = compute_divergence(biased_probs, uniform_probs, mask)
        assert result["per_cell"][2, 2] > 0
        assert result["per_cell"][0, 0] == 0.0


# ---------------------------------------------------------------------------
# detect_biases
# ---------------------------------------------------------------------------


class TestDetectBiases:
    """Tests for systematic bias detection."""

    def test_returns_one_entry_per_class(
        self,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        biases = detect_biases(uniform_probs, uniform_probs, full_mask)
        assert len(biases) == NUM_PREDICTION_CLASSES

    def test_uniform_has_zero_delta(
        self,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        biases = detect_biases(uniform_probs, uniform_probs, full_mask)
        for b in biases:
            assert abs(float(b["delta"])) < 1e-6

    def test_detects_class_bias(
        self,
        biased_probs: np.ndarray,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        biases = detect_biases(biased_probs, uniform_probs, full_mask)
        # Class 0 should show positive delta (obs > sim)
        class0 = biases[0]
        assert float(class0["delta"]) > 0.5

    def test_empty_mask_returns_empty(
        self,
        uniform_probs: np.ndarray,
        empty_mask: np.ndarray,
    ) -> None:
        biases = detect_biases(uniform_probs, uniform_probs, empty_mask)
        assert biases == []


# ---------------------------------------------------------------------------
# calibrate_weights
# ---------------------------------------------------------------------------


class TestCalibrateWeights:
    """Tests for optimal weight calibration."""

    def test_identical_returns_default(
        self,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        # When obs==sim, weight doesn't matter, returns some value
        w = calibrate_weights(uniform_probs, uniform_probs, full_mask)
        assert 0.0 <= w <= 1.0

    def test_optimal_weight_favors_observation(
        self,
        biased_probs: np.ndarray,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        # When obs is biased, optimal weight should be 1.0 (obs matches obs)
        w = calibrate_weights(biased_probs, uniform_probs, full_mask)
        assert w >= 0.9

    def test_empty_mask_returns_default(
        self,
        uniform_probs: np.ndarray,
        empty_mask: np.ndarray,
    ) -> None:
        from src.constants import OBSERVATION_WEIGHT

        w = calibrate_weights(uniform_probs, uniform_probs, empty_mask)
        assert w == OBSERVATION_WEIGHT

    def test_returns_float(
        self,
        uniform_probs: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        w = calibrate_weights(uniform_probs, uniform_probs, full_mask)
        assert isinstance(w, float)


# ---------------------------------------------------------------------------
# suggest_constant_adjustments
# ---------------------------------------------------------------------------


class TestSuggestAdjustments:
    """Tests for bias suggestion generation."""

    def test_no_biases_returns_no_significant(self) -> None:
        biases = [{"class_name": "empty", "obs_mean": 0.17, "sim_mean": 0.16, "delta": 0.01}]
        suggestions = suggest_constant_adjustments(biases)
        assert len(suggestions) == 1
        assert "No significant" in suggestions[0]

    def test_detects_over_prediction(self) -> None:
        biases = [{"class_name": "settlement", "obs_mean": 0.1, "sim_mean": 0.2, "delta": -0.1}]
        suggestions = suggest_constant_adjustments(biases)
        assert any("over-predicts" in s for s in suggestions)

    def test_detects_under_prediction(self) -> None:
        biases = [{"class_name": "forest", "obs_mean": 0.3, "sim_mean": 0.1, "delta": 0.2}]
        suggestions = suggest_constant_adjustments(biases)
        assert any("under-predicts" in s for s in suggestions)


# ---------------------------------------------------------------------------
# write_calibration_report
# ---------------------------------------------------------------------------


class TestWriteCalibrationReport:
    """Tests for calibration report writing."""

    def test_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            biases = [{"class_name": "empty", "obs_mean": 0.17, "sim_mean": 0.16, "delta": 0.01}]
            divergence = {"mean_kl": 0.01, "max_kl": 0.05, "per_cell": np.zeros((5, 5))}
            path = write_calibration_report(
                biases,
                0.8,
                divergence,
                output_dir=tmpdir,
            )
            assert os.path.exists(path)

    def test_file_contains_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            biases = [{"class_name": "empty", "obs_mean": 0.17, "sim_mean": 0.16, "delta": 0.01}]
            divergence = {"mean_kl": 0.01, "max_kl": 0.05, "per_cell": np.zeros((5, 5))}
            path = write_calibration_report(
                biases,
                0.8,
                divergence,
                output_dir=tmpdir,
            )
            with open(path) as f:
                content = f.read()
            assert "metric" in content
            assert "optimal_obs_weight" in content
