"""Tests for web.data loading helpers."""

import json

import pytest

from web.data import get_round, get_round_scores, list_rounds


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a minimal data directory with one round."""
    rd = tmp_path / "abc123"
    rd.mkdir()
    round_data = {
        "round_number": 1,
        "id": "abc123",
        "status": "completed",
        "event_date": "2026-01-01",
        "map_width": 40,
        "map_height": 40,
        "initial_states": [{}, {}],
    }
    (rd / "round.json").write_text(json.dumps(round_data))
    seed = rd / "seed_0"
    seed.mkdir()
    meta = {"score": 72.5, "weighted_kl": 0.05, "num_dynamic_cells": 500}
    (seed / "analysis_meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_list_rounds_returns_metadata(mock_data_dir):
    """list_rounds returns round metadata from disk."""
    rounds = list_rounds(str(mock_data_dir))
    assert len(rounds) == 1
    rd = rounds[0]
    assert rd["round_number"] == 1
    assert rd["status"] == "completed"
    assert rd["seed_count"] == 2


def test_list_rounds_empty(tmp_path):
    """list_rounds returns empty for non-existent dir."""
    result = list_rounds(str(tmp_path / "nope"))
    assert result == []


def test_get_round_found(mock_data_dir):
    """get_round returns full round data by number."""
    data = get_round(1, str(mock_data_dir))
    assert data is not None
    assert data["round_number"] == 1


def test_get_round_not_found(mock_data_dir):
    """get_round returns None for unknown round."""
    assert get_round(99, str(mock_data_dir)) is None


def test_get_round_scores(mock_data_dir):
    """get_round_scores loads per-seed analysis metadata."""
    from pathlib import Path

    rd = Path(mock_data_dir) / "abc123"
    scores = get_round_scores(rd)
    assert len(scores) == 1
    assert scores[0]["score"] == 72.5
    assert scores[0]["seed_index"] == 0


def test_get_round_scores_no_seeds(tmp_path):
    """get_round_scores returns empty when no seed data."""
    scores = get_round_scores(tmp_path)
    assert scores == []


def test_list_rounds_sorted(tmp_path):
    """list_rounds returns rounds sorted by round_number."""
    for rn, name in [(3, "zzz"), (1, "aaa"), (2, "bbb")]:
        rd = tmp_path / name
        rd.mkdir()
        data = {
            "round_number": rn,
            "id": name,
            "status": "completed",
            "event_date": "",
            "map_width": 40,
            "map_height": 40,
            "initial_states": [],
        }
        (rd / "round.json").write_text(json.dumps(data))
    rounds = list_rounds(str(tmp_path))
    assert [r["round_number"] for r in rounds] == [1, 2, 3]
