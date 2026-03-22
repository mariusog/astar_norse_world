"""Tests for web.backtest engine."""

import json

import pytest

from web.backtest import (
    BacktestResult,
    _discover_rounds,
    _get_round_number,
    _load_one_result,
    get_leaderboard,
    load_results,
    save_result,
)


@pytest.fixture
def tmp_results_dir(monkeypatch, tmp_path):
    """Redirect backtest results to a temp directory."""
    monkeypatch.setattr("web.backtest.RESULTS_DIR", str(tmp_path))
    return tmp_path


def test_save_and_load_result(tmp_results_dir):
    """save_result persists JSON, load_results reads it back."""
    result = BacktestResult(
        strategy_name="test_strat",
        scores={1: [70.0, 72.0], 2: [65.0, 68.0]},
        avg_score=68.75,
        timestamp="20260320_120000",
    )
    save_result(result)
    files = list(tmp_results_dir.glob("*.json"))
    assert len(files) == 1

    loaded = load_results()
    assert len(loaded) == 1
    assert loaded[0].strategy_name == "test_strat"
    assert loaded[0].avg_score == 68.75
    assert loaded[0].scores[1] == [70.0, 72.0]


def test_get_leaderboard_empty(tmp_results_dir):
    """get_leaderboard returns empty list when no results exist."""
    lb = get_leaderboard()
    assert lb == []


def test_get_leaderboard_sorted(tmp_results_dir):
    """get_leaderboard returns strategies sorted by avg_score descending."""
    for name, score in [("low", 50.0), ("high", 80.0), ("mid", 65.0)]:
        r = BacktestResult(
            strategy_name=name,
            scores={1: [score]},
            avg_score=score,
            timestamp=f"20260320_{name}",
        )
        save_result(r)
    lb = get_leaderboard()
    assert len(lb) == 3
    assert lb[0]["strategy"] == "high"
    assert lb[1]["strategy"] == "mid"
    assert lb[2]["strategy"] == "low"


def test_get_leaderboard_keeps_best(tmp_results_dir):
    """get_leaderboard keeps only the best result per strategy."""
    for score in [50.0, 70.0, 60.0]:
        r = BacktestResult(
            strategy_name="same",
            scores={1: [score]},
            avg_score=score,
            timestamp=f"ts_{score}",
        )
        save_result(r)
    lb = get_leaderboard()
    assert len(lb) == 1
    assert lb[0]["avg_score"] == 70.0


def test_discover_rounds_empty(tmp_path):
    """_discover_rounds returns empty for non-existent dir."""
    result = _discover_rounds(str(tmp_path / "nonexistent"))
    assert result == []


def test_discover_rounds_finds_dirs(tmp_path):
    """_discover_rounds finds directories with round.json."""
    rd = tmp_path / "round1"
    rd.mkdir()
    (rd / "round.json").write_text(json.dumps({"round_number": 1}))
    result = _discover_rounds(str(tmp_path))
    assert len(result) == 1


def test_get_round_number(tmp_path):
    """_get_round_number reads round_number from JSON."""
    (tmp_path / "round.json").write_text(json.dumps({"round_number": 3}))
    assert _get_round_number(tmp_path) == 3


def test_load_one_result(tmp_path):
    """_load_one_result parses a result JSON file."""
    data = {
        "strategy_name": "test",
        "scores": {"1": [80.0], "2": [75.0]},
        "avg_score": 77.5,
        "timestamp": "20260320",
    }
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))
    r = _load_one_result(path)
    assert r.strategy_name == "test"
    assert r.avg_score == 77.5
    assert r.scores[1] == [80.0]


def test_backtest_result_defaults():
    """BacktestResult has sensible defaults."""
    r = BacktestResult(strategy_name="x")
    assert r.scores == {}
    assert r.avg_score == 0.0
    assert r.timestamp == ""
