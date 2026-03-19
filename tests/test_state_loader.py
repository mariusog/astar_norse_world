"""Tests for state_loader -- parsing server JSON into internal types."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import INITIAL_FOOD, INITIAL_POPULATION
from src.state_loader import (
    _map_terrain_code,
    _parse_grid,
    _parse_settlements,
    _validate_dimensions,
    load_initial_state,
    load_round,
)
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# _map_terrain_code
# ---------------------------------------------------------------------------


def test_map_terrain_code_all_known() -> None:
    """All 8 known server codes map correctly."""
    expected = {
        0: InternalTerrain.PLAINS,       # Server "Empty"
        1: InternalTerrain.SETTLEMENT,
        2: InternalTerrain.PORT,
        3: InternalTerrain.RUIN,
        4: InternalTerrain.FOREST,
        5: InternalTerrain.MOUNTAIN,
        10: InternalTerrain.OCEAN,
        11: InternalTerrain.PLAINS,
    }
    for code, terrain in expected.items():
        assert _map_terrain_code(code) == terrain


def test_map_terrain_code_unknown_raises() -> None:
    """Unknown code raises ValueError."""
    with pytest.raises(ValueError, match="Unknown"):
        _map_terrain_code(99)


# ---------------------------------------------------------------------------
# _parse_grid
# ---------------------------------------------------------------------------


def test_parse_grid_correct_shape() -> None:
    """Grid is parsed to correct shape and dtype."""
    grid_data = [[10, 11, 4], [5, 11, 10]]
    grid = _parse_grid(grid_data)
    assert grid.shape == (2, 3)
    assert grid.dtype == np.int8


def test_parse_grid_values() -> None:
    """Grid values map correctly from server codes."""
    grid_data = [[10, 11], [4, 5]]
    grid = _parse_grid(grid_data)
    assert grid[0, 0] == InternalTerrain.OCEAN
    assert grid[0, 1] == InternalTerrain.PLAINS
    assert grid[1, 0] == InternalTerrain.FOREST
    assert grid[1, 1] == InternalTerrain.MOUNTAIN


def test_parse_grid_empty_raises() -> None:
    """Empty grid data raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        _parse_grid([])
    with pytest.raises(ValueError, match="empty"):
        _parse_grid([[]])


# ---------------------------------------------------------------------------
# _parse_settlements
# ---------------------------------------------------------------------------


def test_parse_settlements_minimal() -> None:
    """Settlement with only x/y uses defaults."""
    data = [{"x": 5, "y": 10}]
    result = _parse_settlements(data)
    assert len(result) == 1
    s = result[0]
    assert s.x == 5
    assert s.y == 10
    assert s.owner_id == 0
    assert s.population == INITIAL_POPULATION
    assert s.food == INITIAL_FOOD
    assert s.is_port is False


def test_parse_settlements_full_fields() -> None:
    """Settlement with all fields populated."""
    data = [
        {
            "x": 3,
            "y": 7,
            "owner_id": 2,
            "population": 80,
            "food": 200,
            "wealth": 50,
            "defense": 20,
            "tech_level": 3,
            "has_port": True,
            "has_longship": True,
        }
    ]
    result = _parse_settlements(data)
    s = result[0]
    assert s.owner_id == 2
    assert s.population == 80
    assert s.is_port is True
    assert s.has_longship is True
    assert s.tech_level == 3


def test_parse_settlements_empty_list() -> None:
    """Empty settlements list returns empty."""
    assert _parse_settlements([]) == []


# ---------------------------------------------------------------------------
# _validate_dimensions
# ---------------------------------------------------------------------------


def test_validate_dimensions_correct() -> None:
    """Matching dimensions pass without error."""
    grid = np.zeros((10, 20), dtype=np.int8)
    _validate_dimensions(grid, 20, 10, 0)  # w=20, h=10


def test_validate_dimensions_width_mismatch() -> None:
    """Width mismatch raises ValueError."""
    grid = np.zeros((10, 15), dtype=np.int8)
    with pytest.raises(ValueError, match="width"):
        _validate_dimensions(grid, 20, 10, 0)


def test_validate_dimensions_height_mismatch() -> None:
    """Height mismatch raises ValueError."""
    grid = np.zeros((10, 20), dtype=np.int8)
    with pytest.raises(ValueError, match="height"):
        _validate_dimensions(grid, 20, 15, 0)


def test_validate_dimensions_none_skips() -> None:
    """None expected dimensions are skipped."""
    grid = np.zeros((10, 20), dtype=np.int8)
    _validate_dimensions(grid, None, None, 0)  # no error


# ---------------------------------------------------------------------------
# load_initial_state
# ---------------------------------------------------------------------------


def test_load_initial_state_roundtrip() -> None:
    """Full state JSON parses into grid and settlements."""
    state = {
        "grid": [[10, 11, 4], [5, 1, 11]],
        "settlements": [{"x": 1, "y": 1, "has_port": False}],
    }
    grid, settlements = load_initial_state(state)
    assert grid.shape == (2, 3)
    assert len(settlements) == 1
    assert settlements[0].x == 1


def test_load_initial_state_no_settlements() -> None:
    """State with no settlements key still works."""
    state = {"grid": [[10, 11], [11, 10]]}
    grid, settlements = load_initial_state(state)
    assert grid.shape == (2, 2)
    assert settlements == []


# ---------------------------------------------------------------------------
# load_round
# ---------------------------------------------------------------------------


def test_load_round_multiple_seeds() -> None:
    """Round with multiple seeds loads all."""
    round_json = {
        "map_width": 3,
        "map_height": 2,
        "initial_states": [
            {"grid": [[10, 11, 4], [5, 11, 10]], "settlements": []},
            {"grid": [[11, 11, 11], [10, 10, 10]], "settlements": [{"x": 0, "y": 0}]},
        ],
    }
    results = load_round(round_json)
    assert len(results) == 2
    assert results[0][0].shape == (2, 3)
    assert len(results[1][1]) == 1


def test_load_round_dimension_mismatch_raises() -> None:
    """Wrong grid dimensions raise ValueError."""
    round_json = {
        "map_width": 10,
        "map_height": 2,
        "initial_states": [
            {"grid": [[10, 11, 4], [5, 11, 10]], "settlements": []},
        ],
    }
    with pytest.raises(ValueError, match="width"):
        load_round(round_json)
