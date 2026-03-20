"""Backtest engine for evaluating prediction strategies.

Runs strict leave-one-out (LOO) evaluation: for each target round,
trains using only OTHER rounds and scores against the target's GT.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.scoring import score_prediction
from src.terrain import SERVER_TO_INTERNAL
from web.models import get_strategy

logger = logging.getLogger(__name__)

RESULTS_DIR = "data/backtest_results"
DEFAULT_DATA_DIR = "data/rounds"


@dataclass
class BacktestResult:
    """Result of a LOO backtest run."""

    strategy_name: str
    scores: dict[int, list[float]] = field(default_factory=dict)
    avg_score: float = 0.0
    timestamp: str = ""


def run_loo_backtest(
    strategy_name: str,
    data_dir: str = DEFAULT_DATA_DIR,
) -> BacktestResult:
    """Run strict LOO evaluation of a strategy across all rounds.

    For each round: build prediction using OTHER rounds, score against GT.
    """
    round_dirs = _discover_rounds(data_dir)
    result = BacktestResult(
        strategy_name=strategy_name,
        timestamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
    )
    all_scores: list[float] = []

    for round_dir in round_dirs:
        round_number = _get_round_number(round_dir)
        logger.info("LOO backtest: round %d (excluded from training)", round_number)
        seed_scores = _evaluate_round_loo(
            strategy_name,
            round_dir,
            data_dir,
        )
        if seed_scores:
            result.scores[round_number] = seed_scores
            all_scores.extend(seed_scores)

    result.avg_score = float(np.mean(all_scores)) if all_scores else 0.0
    save_result(result)
    return result


def run_single_round(
    strategy_name: str,
    round_number: int,
    data_dir: str = DEFAULT_DATA_DIR,
) -> list[float]:
    """Score a strategy on one specific round using LOO."""
    round_dirs = _discover_rounds(data_dir)
    target_dir = _find_round_by_number(round_dirs, round_number)
    if target_dir is None:
        logger.warning("Round %d not found", round_number)
        return []
    return _evaluate_round_loo(strategy_name, target_dir, data_dir)


def save_result(result: BacktestResult) -> None:
    """Persist backtest result as JSON."""
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{result.strategy_name}_{result.timestamp}.json"
    data = {
        "strategy_name": result.strategy_name,
        "scores": {str(k): v for k, v in result.scores.items()},
        "avg_score": result.avg_score,
        "timestamp": result.timestamp,
    }
    with open(out_dir / filename, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved backtest result to %s", filename)


def load_results() -> list[BacktestResult]:
    """Load all saved backtest results from disk."""
    out_dir = Path(RESULTS_DIR)
    if not out_dir.exists():
        return []
    results = []
    for path in sorted(out_dir.glob("*.json")):
        results.append(_load_one_result(path))
    return results


def get_leaderboard() -> list[dict[str, Any]]:
    """Return strategies sorted by avg_score descending."""
    results = load_results()
    best: dict[str, BacktestResult] = {}
    for r in results:
        if r.strategy_name not in best or r.avg_score > best[r.strategy_name].avg_score:
            best[r.strategy_name] = r
    entries = [
        {
            "strategy": r.strategy_name,
            "avg_score": round(r.avg_score, 2),
            "num_rounds": len(r.scores),
            "timestamp": r.timestamp,
            "per_round": {k: round(np.mean(v), 2) for k, v in r.scores.items()},
        }
        for r in best.values()
    ]
    return sorted(entries, key=lambda x: x["avg_score"], reverse=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_rounds(data_dir: str) -> list[Path]:
    """Find all round directories with data."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return [d for d in sorted(data_path.iterdir()) if d.is_dir() and (d / "round.json").exists()]


def _get_round_number(round_dir: Path) -> int:
    """Read round_number from round.json."""
    with open(round_dir / "round.json") as f:
        return json.load(f).get("round_number", 0)


def _find_round_by_number(
    round_dirs: list[Path],
    target: int,
) -> Path | None:
    """Find round directory by round number."""
    for d in round_dirs:
        if _get_round_number(d) == target:
            return d
    return None


def _evaluate_round_loo(
    strategy_name: str,
    target_dir: Path,
    data_dir: str,
) -> list[float]:
    """Evaluate one round with LOO: exclude target from training data."""
    strategy = get_strategy(strategy_name)
    target_name = target_dir.name
    seeds = _load_seeds(target_dir)
    scores: list[float] = []

    # Create temp data_dir view excluding target round
    loo_dir = _make_loo_data_dir(data_dir, target_name)

    for grid, gt, _settlements in seeds:
        prediction = strategy.predict_fn(grid, loo_dir)
        result = score_prediction(gt, prediction)
        scores.append(result["score"])
        logger.debug("Seed score: %.2f", result["score"])

    return scores


def _load_seeds(
    round_dir: Path,
) -> list[tuple[np.ndarray, np.ndarray, list[dict]]]:
    """Load grid, GT, and settlements for all seeds in a round."""
    rjson = round_dir / "round.json"
    with open(rjson) as f:
        rd = json.load(f)
    results = []
    for seed_idx, state in enumerate(rd.get("initial_states", [])):
        gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
        if not gt_path.exists():
            continue
        grid_raw = np.array(state["grid"])
        internal = np.vectorize(lambda v: SERVER_TO_INTERNAL.get(v, 1))(grid_raw)
        gt = np.load(gt_path)
        results.append((internal, gt, state.get("settlements", [])))
    return results


def _make_loo_data_dir(data_dir: str, exclude_name: str) -> str:
    """Return a data_dir string; strategies use _collect_training_data.

    We use symlinks to create a temp dir excluding the target round.
    Falls back to passing data_dir with explicit exclude in strategy.
    """
    import tempfile

    data_path = Path(data_dir)
    tmp = Path(tempfile.mkdtemp(prefix="loo_"))
    for rd in data_path.iterdir():
        if rd.is_dir() and rd.name != exclude_name:
            (tmp / rd.name).symlink_to(rd.resolve())
    return str(tmp)


def _load_one_result(path: Path) -> BacktestResult:
    """Load a single result file."""
    with open(path) as f:
        data = json.load(f)
    result = BacktestResult(
        strategy_name=data["strategy_name"],
        avg_score=data.get("avg_score", 0.0),
        timestamp=data.get("timestamp", ""),
    )
    for k, v in data.get("scores", {}).items():
        result.scores[int(k)] = v
    return result
