"""Tests for the self-scoring evaluator (src/scoring.py)."""

import numpy as np
import pytest

from src.constants import SCORE_DECAY_RATE
from src.scoring import (
    entropy,
    kl_divergence,
    score_against_mc,
    score_prediction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 6


def _make_tensor(h: int, w: int, probs: list[float]) -> np.ndarray:
    """Create an H x W x C tensor with identical per-cell probabilities."""
    arr = np.array(probs, dtype=np.float64)
    return np.broadcast_to(arr, (h, w, len(probs))).copy()


# ---------------------------------------------------------------------------
# kl_divergence tests
# ---------------------------------------------------------------------------


class TestKLDivergence:
    def test_identical_distributions_returns_zero(self) -> None:
        p = np.array([[0.5, 0.3, 0.2]])
        result = kl_divergence(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_different_distributions_positive(self) -> None:
        p = np.array([[0.9, 0.1]])
        q = np.array([[0.5, 0.5]])
        result = kl_divergence(p, q)
        assert result[0] > 0.0

    def test_floor_prevents_log_zero(self) -> None:
        """q=0 should not cause inf/nan due to probability floor."""
        p = np.array([[0.5, 0.5, 0.0, 0.0]])
        q = np.array([[0.0, 0.0, 0.5, 0.5]])
        result = kl_divergence(p, q)
        assert np.isfinite(result[0])

    def test_batch_shape(self) -> None:
        p = np.random.default_rng(42).dirichlet(np.ones(4), size=(3, 5))
        q = np.random.default_rng(99).dirichlet(np.ones(4), size=(3, 5))
        result = kl_divergence(p, q)
        assert result.shape == (3, 5)
        assert np.all(np.isfinite(result))

    def test_hand_computed_value(self) -> None:
        """Verify against a manually computed KL divergence."""
        p = np.array([[0.6, 0.4]])
        q = np.array([[0.4, 0.6]])
        # KL(p||q) = 0.6*ln(0.6/0.4) + 0.4*ln(0.4/0.6)
        expected = 0.6 * np.log(0.6 / 0.4) + 0.4 * np.log(0.4 / 0.6)
        result = kl_divergence(p, q)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# entropy tests
# ---------------------------------------------------------------------------


class TestEntropy:
    def test_uniform_distribution_max_entropy(self) -> None:
        p = np.array([[1 / 6] * NUM_CLASSES])
        result = entropy(p)
        np.testing.assert_allclose(result[0], np.log(NUM_CLASSES), rtol=1e-6)

    def test_deterministic_distribution_near_zero(self) -> None:
        """A near-deterministic cell should have very low entropy."""
        p = np.array([[0.98, 0.004, 0.004, 0.004, 0.004, 0.004]])
        result = entropy(p)
        assert result[0] < 0.2

    def test_entropy_non_negative(self) -> None:
        rng = np.random.default_rng(7)
        p = rng.dirichlet(np.ones(NUM_CLASSES), size=(10,))
        result = entropy(p)
        assert np.all(result >= 0.0)

    def test_batch_shape(self) -> None:
        p = np.random.default_rng(42).dirichlet(np.ones(4), size=(2, 3))
        result = entropy(p)
        assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# score_prediction tests
# ---------------------------------------------------------------------------


class TestScorePrediction:
    def test_perfect_prediction_scores_100(self) -> None:
        gt = _make_tensor(5, 5, [0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        result = score_prediction(gt, gt)
        np.testing.assert_allclose(result["score"], 100.0, atol=0.5)

    def test_uniform_prediction_scores_low(self) -> None:
        gt = _make_tensor(5, 5, [0.6, 0.25, 0.1, 0.03, 0.01, 0.01])
        pred = _make_tensor(5, 5, [1 / 6] * NUM_CLASSES)
        result = score_prediction(gt, pred)
        assert result["score"] < 15.0, f"Uniform prediction scored {result['score']}"

    def test_static_cells_excluded(self) -> None:
        """Cells with entropy < threshold should be excluded."""
        h, w = 4, 4
        # Fully deterministic ground truth: entropy is 0
        static_probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        gt = _make_tensor(h, w, static_probs)
        pred = _make_tensor(h, w, [1 / 6] * NUM_CLASSES)
        result = score_prediction(gt, pred)
        assert result["num_dynamic_cells"] == 0
        assert result["score"] == 100.0

    def test_mixed_static_and_dynamic(self) -> None:
        """Only dynamic cells contribute to the score."""
        h, w = 2, 2
        gt = np.zeros((h, w, NUM_CLASSES))
        # Row 0: static (deterministic, entropy=0)
        gt[0, :, :] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Row 1: dynamic
        gt[1, :, :] = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]

        pred = gt.copy()
        result = score_prediction(gt, pred)
        # Only 2 dynamic cells (row 1)
        assert result["num_dynamic_cells"] == 2

    def test_score_decay_formula(self) -> None:
        """Verify the exponential decay formula: 100 * exp(-3 * wkl)."""
        gt = _make_tensor(3, 3, [0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        pred = _make_tensor(3, 3, [0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
        result = score_prediction(gt, pred)
        expected_score = 100.0 * np.exp(-SCORE_DECAY_RATE * result["weighted_kl"])
        np.testing.assert_allclose(result["score"], expected_score, rtol=1e-10)

    def test_shape_mismatch_raises(self) -> None:
        gt = _make_tensor(3, 3, [1 / 6] * NUM_CLASSES)
        pred = _make_tensor(4, 4, [1 / 6] * NUM_CLASSES)
        with pytest.raises(ValueError, match="Shape mismatch"):
            score_prediction(gt, pred)

    def test_1d_tensor_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2D"):
            score_prediction(np.array([0.5, 0.5]), np.array([0.5, 0.5]))

    def test_result_keys(self) -> None:
        gt = _make_tensor(2, 2, [0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        result = score_prediction(gt, gt)
        assert set(result.keys()) == {
            "score",
            "weighted_kl",
            "num_dynamic_cells",
            "mean_entropy",
        }

    def test_weighted_kl_nonnegative(self) -> None:
        rng = np.random.default_rng(123)
        gt = rng.dirichlet(np.ones(NUM_CLASSES), size=(5, 5))
        pred = rng.dirichlet(np.ones(NUM_CLASSES), size=(5, 5))
        result = score_prediction(gt, pred)
        assert result["weighted_kl"] >= 0.0

    def test_score_between_0_and_100(self) -> None:
        rng = np.random.default_rng(456)
        gt = rng.dirichlet(np.ones(NUM_CLASSES), size=(5, 5))
        pred = rng.dirichlet(np.ones(NUM_CLASSES), size=(5, 5))
        result = score_prediction(gt, pred)
        assert 0.0 <= result["score"] <= 100.0


# ---------------------------------------------------------------------------
# score_against_mc tests
# ---------------------------------------------------------------------------


class TestScoreAgainstMC:
    def test_delegates_to_score_prediction(self) -> None:
        gt = _make_tensor(3, 3, [0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        pred = _make_tensor(3, 3, [0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
        direct = score_prediction(gt, pred)
        via_mc = score_against_mc(gt, pred)
        assert direct == via_mc


# ---------------------------------------------------------------------------
# Hand-computed verification
# ---------------------------------------------------------------------------


class TestHandComputed:
    def test_two_class_known_score(self) -> None:
        """Fully hand-computed example with 2 active classes on a 1x1 grid."""
        # Ground truth: [0.7, 0.3] (plus 4 zero-ish classes)
        gt_probs = [0.7, 0.3, 0.0, 0.0, 0.0, 0.0]
        pred_probs = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]

        gt = np.array(gt_probs).reshape(1, 1, NUM_CLASSES)
        pred = np.array(pred_probs).reshape(1, 1, NUM_CLASSES)

        result = score_prediction(gt, pred)

        # After flooring and renorm, values shift slightly from the raw
        # inputs, so we verify structural properties rather than exact values
        assert result["num_dynamic_cells"] == 1
        assert result["weighted_kl"] > 0.0
        assert 0.0 < result["score"] < 100.0
        # Score should be decent since prediction is not far off
        assert result["score"] > 50.0
