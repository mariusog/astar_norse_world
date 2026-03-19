"""Shared test fixtures and configuration for Norse world simulator."""

import numpy as np
import pytest

from src.map_generator import generate_map
from src.settlement import Settlement
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_settlement() -> callable:
    """Factory for creating test settlements with sensible defaults."""

    def _make(**overrides: object) -> Settlement:
        defaults = {
            "x": 5,
            "y": 5,
            "owner_id": 0,
            "population": 50,
            "food": 100,
            "wealth": 0,
            "defense": 10,
            "tech_level": 1,
            "is_port": False,
            "has_longship": False,
        }
        defaults.update(overrides)
        return Settlement(**defaults)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def small_map() -> tuple[np.ndarray, list[Settlement]]:
    """A deterministic 20x20 map for fast tests."""
    return generate_map(seed=42, width=20, height=20)


@pytest.fixture
def plains_grid() -> np.ndarray:
    """A 10x10 grid of plains with ocean border -- no mountains or forests."""
    grid = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    return grid


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
