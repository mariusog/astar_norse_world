"""Tests for web.models strategy registry."""

import numpy as np
import pytest

from web.models import (
    STRATEGIES,
    Strategy,
    apply_static_and_normalize,
    get_strategy,
    list_strategies,
    register,
)


def test_register_adds_strategy():
    """register() adds a strategy to the global registry."""
    name = "_test_reg"
    register(name, "test desc", lambda g, d: np.zeros((*g.shape, 6)))
    assert name in STRATEGIES
    assert STRATEGIES[name].description == "test desc"
    del STRATEGIES[name]


def test_list_strategies_returns_all():
    """list_strategies() returns all registered strategies."""
    result = list_strategies()
    assert len(result) >= 6
    names = {s.name for s in result}
    assert "flat_priors" in names
    assert "xgboost" in names


def test_get_strategy_found():
    """get_strategy() returns the requested strategy."""
    s = get_strategy("flat_priors")
    assert isinstance(s, Strategy)
    assert s.name == "flat_priors"


def test_get_strategy_not_found():
    """get_strategy() raises KeyError for unknown strategy."""
    with pytest.raises(KeyError):
        get_strategy("nonexistent_strategy")


def testapply_static_and_normalize_ocean():
    """Static overrides set ocean cells to near-certain class 0."""
    from src.terrain import InternalTerrain

    grid = np.array([[InternalTerrain.OCEAN, InternalTerrain.PLAINS]])
    tensor = np.ones((1, 2, 6)) / 6.0
    result = apply_static_and_normalize(tensor, grid)
    assert result.shape == (1, 2, 6)
    assert result[0, 0, 0] > 0.9
    np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-6)


def testapply_static_and_normalize_mountain():
    """Static overrides set mountain cells to near-certain class 5."""
    from src.terrain import InternalTerrain

    grid = np.array([[InternalTerrain.MOUNTAIN]])
    tensor = np.ones((1, 1, 6)) / 6.0
    result = apply_static_and_normalize(tensor, grid)
    assert result[0, 0, 5] > 0.9


def test_all_registered_strategies_have_predict_fn():
    """Every registered strategy has a callable predict_fn."""
    for s in list_strategies():
        assert callable(s.predict_fn)


def test_strategy_names_are_expected():
    """Verify all 6 expected strategies are registered."""
    expected = {
        "flat_priors",
        "distance_priors",
        "feature_lookup",
        "xgboost",
        "xgboost_survive",
        "xgboost_collapse",
    }
    actual = {s.name for s in list_strategies()}
    assert expected <= actual
