"""Tests for historical round data collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.round_collector import (
    _filter_completed_rounds,
    _parse_server_grid,
    _round_already_captured,
    collect_all_rounds,
)
from src.terrain import SERVER_TO_INTERNAL, InternalTerrain


class MockClient:
    """Mock API client for testing round collection."""

    def __init__(
        self,
        rounds: list[dict[str, Any]] | None = None,
        round_detail: dict[str, Any] | None = None,
        analysis_data: dict[str, Any] | None = None,
    ) -> None:
        self._rounds = rounds or []
        self._round_detail = round_detail or {}
        self._analysis_data = analysis_data or {}
        self.list_rounds_calls = 0
        self.get_round_calls: list[int] = []
        self.analysis_calls: list[tuple[int, int]] = []

    def list_rounds(self) -> list[dict[str, Any]]:
        self.list_rounds_calls += 1
        return self._rounds

    def get_round(self, round_id: int) -> dict[str, Any]:
        self.get_round_calls.append(round_id)
        return self._round_detail

    def analysis(self, round_id: int, seed_index: int) -> dict[str, Any]:
        self.analysis_calls.append((round_id, seed_index))
        return self._analysis_data


def _make_grid_data(height: int, width: int, val: int = 11) -> list[list[int]]:
    """Create server-format grid data."""
    return [[val] * width for _ in range(height)]


class TestParseServerGrid:
    """Tests for _parse_server_grid."""

    def test_empty_grid_returns_empty_array(self) -> None:
        result = _parse_server_grid([])
        assert result.size == 0

    def test_maps_ocean_code_correctly(self) -> None:
        grid_data = [[10, 10], [10, 10]]
        result = _parse_server_grid(grid_data)
        assert np.all(result == InternalTerrain.OCEAN)

    def test_maps_plains_code_correctly(self) -> None:
        grid_data = [[11]]
        result = _parse_server_grid(grid_data)
        assert result[0, 0] == InternalTerrain.PLAINS

    def test_maps_settlement_code(self) -> None:
        grid_data = [[1]]
        result = _parse_server_grid(grid_data)
        assert result[0, 0] == InternalTerrain.SETTLEMENT

    def test_maps_all_server_codes(self) -> None:
        for server_code, internal_code in SERVER_TO_INTERNAL.items():
            grid_data = [[server_code]]
            result = _parse_server_grid(grid_data)
            assert result[0, 0] == internal_code

    def test_unknown_code_defaults_to_plains(self) -> None:
        grid_data = [[99]]
        result = _parse_server_grid(grid_data)
        assert result[0, 0] == InternalTerrain.PLAINS

    def test_preserves_grid_dimensions(self) -> None:
        grid_data = _make_grid_data(3, 5)
        result = _parse_server_grid(grid_data)
        assert result.shape == (3, 5)


class TestFilterCompletedRounds:
    """Tests for _filter_completed_rounds."""

    def test_filters_completed_only(self) -> None:
        rounds = [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "active"},
            {"id": 3, "status": "completed"},
        ]
        result = _filter_completed_rounds(rounds)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 3

    def test_empty_list_returns_empty(self) -> None:
        assert _filter_completed_rounds([]) == []


class TestRoundAlreadyCaptured:
    """Tests for _round_already_captured."""

    def test_missing_directory_returns_false(self, tmp_path: Path) -> None:
        assert not _round_already_captured(tmp_path, 1)

    def test_complete_data_returns_true(self, tmp_path: Path) -> None:
        round_dir = tmp_path / "1"
        for i in range(5):
            seed_dir = round_dir / f"seed_{i}"
            seed_dir.mkdir(parents=True)
            np.save(seed_dir / "ground_truth.npy", np.zeros((3, 3)))
        assert _round_already_captured(tmp_path, 1)

    def test_partial_data_returns_false(self, tmp_path: Path) -> None:
        round_dir = tmp_path / "1"
        # Only create 3 of 5 seeds
        for i in range(3):
            seed_dir = round_dir / f"seed_{i}"
            seed_dir.mkdir(parents=True)
            np.save(seed_dir / "ground_truth.npy", np.zeros((3, 3)))
        assert not _round_already_captured(tmp_path, 1)


class TestCollectAllRounds:
    """Tests for collect_all_rounds."""

    def test_skips_non_completed_rounds(self, tmp_path: Path) -> None:
        client = MockClient(
            rounds=[{"id": 1, "status": "active"}],
        )
        result = collect_all_rounds(client, tmp_path)
        assert result == []
        assert len(client.get_round_calls) == 0

    def test_collects_completed_round(self, tmp_path: Path) -> None:
        grid_data = _make_grid_data(3, 3)
        initial_states = [{"grid": grid_data, "settlements": []} for _ in range(5)]
        client = MockClient(
            rounds=[{"id": 1, "status": "completed"}],
            round_detail={"initial_states": initial_states},
            analysis_data={"grid": grid_data},
        )
        result = collect_all_rounds(client, tmp_path)
        assert result == [1]
        assert len(client.analysis_calls) == 5

    def test_idempotent_skips_existing(self, tmp_path: Path) -> None:
        # Pre-populate data for round 1
        rounds_dir = tmp_path / "rounds"
        for i in range(5):
            seed_dir = rounds_dir / "1" / f"seed_{i}"
            seed_dir.mkdir(parents=True)
            np.save(seed_dir / "ground_truth.npy", np.zeros((3, 3)))

        client = MockClient(
            rounds=[{"id": 1, "status": "completed"}],
        )
        result = collect_all_rounds(client, tmp_path)
        assert result == []
        assert len(client.get_round_calls) == 0

    def test_saves_round_json(self, tmp_path: Path) -> None:
        grid_data = _make_grid_data(3, 3)
        initial_states = [{"grid": grid_data, "settlements": []} for _ in range(5)]
        client = MockClient(
            rounds=[{"id": 2, "status": "completed"}],
            round_detail={"initial_states": initial_states},
            analysis_data={"grid": grid_data},
        )
        collect_all_rounds(client, tmp_path)
        round_json = tmp_path / "rounds" / "2" / "round.json"
        assert round_json.exists()
        data = json.loads(round_json.read_text())
        assert data["id"] == 2
