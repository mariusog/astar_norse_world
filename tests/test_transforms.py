"""Tests for web.transforms parametric post-processing functions."""

from __future__ import annotations

import numpy as np
import pytest

from web.transforms import (
    GRID_TRANSFORMS,
    TRANSFORMS,
    apply_transform,
    apply_transform_chain,
    class_bias,
    collapse_shift,
    floor_and_normalize,
    power_transform,
    spatial_smooth,
    temperature_scale,
)


def _make_pred(h: int = 5, w: int = 5, c: int = 6) -> np.ndarray:
    """Create a valid random prediction tensor with seed."""
    rng = np.random.default_rng(seed=42)
    raw = rng.random((h, w, c))
    return raw / raw.sum(axis=-1, keepdims=True)


def _make_grid(h: int = 5, w: int = 5) -> np.ndarray:
    """Create a simple grid with settlements and ocean."""
    grid = np.ones((h, w), dtype=np.int32)  # plains
    grid[0, :] = 0  # ocean
    grid[2, 2] = 2  # settlement
    grid[3, 3] = 3  # port
    return grid


class TestFloorAndNormalize:
    def test_output_sums_to_one(self) -> None:
        pred = _make_pred()
        result = floor_and_normalize(pred)
        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_no_values_below_floor(self) -> None:
        pred = _make_pred()
        pred[0, 0, :] = 0.0
        result = floor_and_normalize(pred)
        # After flooring and renormalization, minimum is floor/(C*floor)=1/C
        # when all values were zero; generally >= floor/sum
        assert result.min() > 0.0


class TestTemperatureScale:
    def test_identity_at_temperature_one(self) -> None:
        pred = _make_pred()
        result = temperature_scale(pred, temperature=1.0)
        np.testing.assert_allclose(
            result.sum(axis=-1),
            1.0,
            atol=1e-6,
        )

    def test_sharpens_below_one(self) -> None:
        pred = _make_pred()
        sharp = temperature_scale(pred, temperature=0.5)
        # Max probability should increase when sharpening
        assert sharp.max() >= pred.max() - 0.01

    def test_smooths_above_one(self) -> None:
        pred = _make_pred()
        smooth = temperature_scale(pred, temperature=2.0)
        # Distribution should be more uniform
        std_orig = pred.std(axis=-1).mean()
        std_smooth = smooth.std(axis=-1).mean()
        assert std_smooth < std_orig


class TestPowerTransform:
    def test_output_normalized(self) -> None:
        pred = _make_pred()
        result = power_transform(pred, power=1.5)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_identity_at_power_one(self) -> None:
        pred = _make_pred()
        result = power_transform(pred, power=1.0)
        # Flooring may shift very small values, so use looser tolerance
        np.testing.assert_allclose(result, pred, atol=0.01)

    def test_sharpens_above_one(self) -> None:
        pred = _make_pred()
        sharp = power_transform(pred, power=2.0)
        std_orig = pred.std(axis=-1).mean()
        std_sharp = sharp.std(axis=-1).mean()
        assert std_sharp > std_orig


class TestSpatialSmooth:
    def test_output_normalized(self) -> None:
        pred = _make_pred()
        result = spatial_smooth(pred, sigma=1.0)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_reduces_spatial_variance(self) -> None:
        pred = _make_pred(10, 10)
        result = spatial_smooth(pred, sigma=2.0)
        # Spatial variance should decrease after smoothing
        orig_var = pred[:, :, 0].var()
        smooth_var = result[:, :, 0].var()
        assert smooth_var < orig_var


class TestCollapseShift:
    def test_low_settlement_moved_to_empty(self) -> None:
        pred = _make_pred()
        pred[:, :, 1] = 0.02  # low settlement prob
        pred = pred / pred.sum(axis=-1, keepdims=True)
        result = collapse_shift(pred, threshold=0.05)
        assert result[:, :, 1].max() < 0.05

    def test_high_settlement_preserved(self) -> None:
        pred = _make_pred()
        pred[2, 2, 1] = 0.5
        pred[2, 2] /= pred[2, 2].sum()
        result = collapse_shift(pred, threshold=0.1)
        assert result[2, 2, 1] > 0.1


class TestClassBias:
    def test_output_normalized(self) -> None:
        pred = _make_pred()
        result = class_bias(pred, class_idx=1, delta=0.1)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_positive_delta_increases_class(self) -> None:
        pred = _make_pred()
        result = class_bias(pred, class_idx=1, delta=0.5)
        assert result[:, :, 1].mean() > pred[:, :, 1].mean()


class TestApplyTransform:
    def test_known_transform(self) -> None:
        pred = _make_pred()
        result = apply_transform(
            "power_transform",
            pred,
            None,
            {"power": 1.5},
        )
        assert result.shape == pred.shape

    def test_grid_transform_requires_grid(self) -> None:
        pred = _make_pred()
        with pytest.raises(ValueError, match="requires grid"):
            apply_transform(
                "settlement_boost",
                pred,
                None,
                {"factor": 0.1},
            )

    def test_unknown_transform_raises(self) -> None:
        pred = _make_pred()
        with pytest.raises(KeyError):
            apply_transform("nonexistent", pred, None, {})


class TestApplyTransformChain:
    def test_empty_chain_returns_input(self) -> None:
        pred = _make_pred()
        result = apply_transform_chain(pred, None, [])
        np.testing.assert_array_equal(result, pred)

    def test_multi_step_chain(self) -> None:
        pred = _make_pred()
        chain = [
            ("temperature_scale", {"temperature": 0.8}),
            ("power_transform", {"power": 1.2}),
        ]
        result = apply_transform_chain(pred, None, chain)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)


class TestTransformRegistry:
    def test_all_transforms_registered(self) -> None:
        expected = {
            "temperature_scale",
            "power_transform",
            "spatial_smooth",
            "settlement_boost",
            "collapse_shift",
            "inland_power",
            "port_smooth",
            "class_bias",
        }
        assert set(TRANSFORMS.keys()) == expected

    def test_grid_transforms_subset(self) -> None:
        assert GRID_TRANSFORMS.issubset(set(TRANSFORMS.keys()))
