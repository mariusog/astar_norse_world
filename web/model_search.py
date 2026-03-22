"""Automated model search via parametric transforms on base predictions.

Generates dozens of strategy variants by applying transform chains to a
cached XGBoost base prediction, then ranks them by LOO backtest score.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from src.scoring import score_prediction
from web.transforms import apply_transform_chain, floor_and_normalize

# Note: web.backtest imports are deferred to function scope to avoid
# circular import: model_search -> backtest -> models -> strategies -> model_search

DEFAULT_DATA_DIR = "data/rounds"

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for one parametric strategy variant."""

    name: str
    base: str
    transforms: list[tuple[str, dict]]


@dataclass
class SearchProgress:
    """Tracks progress of a model search run."""

    total: int = 0
    completed: int = 0
    results: list[dict] = field(default_factory=list)
    running: bool = False
    error: str | None = None


# Global search state for progress tracking
_search_state = SearchProgress()
_search_lock = threading.Lock()


def get_search_progress() -> SearchProgress:
    """Return current search progress (thread-safe read)."""
    with _search_lock:
        return SearchProgress(
            total=_search_state.total,
            completed=_search_state.completed,
            results=list(_search_state.results),
            running=_search_state.running,
            error=_search_state.error,
        )


def generate_search_strategies() -> list[SearchConfig]:
    """Generate parametric strategy configs for automated search."""
    configs: list[SearchConfig] = []

    # Temperature scaling variants
    for t in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
        configs.append(
            SearchConfig(
                name=f"temperature_{t}",
                base="xgboost",
                transforms=[("temperature_scale", {"temperature": t})],
            )
        )

    # Power transform variants
    for p in [0.5, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 2.0]:
        configs.append(
            SearchConfig(
                name=f"power_{p}",
                base="xgboost",
                transforms=[("power_transform", {"power": p})],
            )
        )

    # Spatial smoothing
    for s in [0.3, 0.5, 0.7, 1.0, 1.5]:
        configs.append(
            SearchConfig(
                name=f"spatial_smooth_{s}",
                base="xgboost",
                transforms=[("spatial_smooth", {"sigma": s})],
            )
        )

    # Temperature + smoothing combinations
    for t in [0.8, 0.9, 1.1]:
        for s in [0.3, 0.5]:
            configs.append(
                SearchConfig(
                    name=f"temp_{t}_smooth_{s}",
                    base="xgboost",
                    transforms=[
                        ("temperature_scale", {"temperature": t}),
                        ("spatial_smooth", {"sigma": s}),
                    ],
                )
            )

    # Settlement boost variants
    for f in [0.05, 0.1, 0.2, 0.3]:
        configs.append(
            SearchConfig(
                name=f"settle_boost_{f}",
                base="xgboost",
                transforms=[("settlement_boost", {"factor": f})],
            )
        )

    # Collapse threshold variants
    for th in [0.05, 0.1, 0.15, 0.2, 0.3]:
        configs.append(
            SearchConfig(
                name=f"collapse_threshold_{th}",
                base="xgboost",
                transforms=[("collapse_shift", {"threshold": th})],
            )
        )

    # Inland power variants
    for p in [0.5, 0.7, 0.9, 1.1]:
        configs.append(
            SearchConfig(
                name=f"inland_power_{p}",
                base="xgboost",
                transforms=[("inland_power", {"power": p})],
            )
        )

    # Port smoothing variants
    for w in [0.05, 0.1, 0.2]:
        configs.append(
            SearchConfig(
                name=f"port_smooth_{w}",
                base="xgboost",
                transforms=[("port_smooth", {"weight": w})],
            )
        )

    # Power + collapse combos
    for p in [0.8, 1.2]:
        for th in [0.1, 0.2]:
            configs.append(
                SearchConfig(
                    name=f"power_{p}_collapse_{th}",
                    base="xgboost",
                    transforms=[
                        ("power_transform", {"power": p}),
                        ("collapse_shift", {"threshold": th}),
                    ],
                )
            )

    # Settlement boost + smoothing combos
    for f in [0.1, 0.2]:
        for s in [0.3, 0.5]:
            configs.append(
                SearchConfig(
                    name=f"settle_{f}_smooth_{s}",
                    base="xgboost",
                    transforms=[
                        ("settlement_boost", {"factor": f}),
                        ("spatial_smooth", {"sigma": s}),
                    ],
                )
            )

    # Class bias for settlement and ruin classes
    for delta in [-0.05, 0.05, 0.1]:
        configs.append(
            SearchConfig(
                name=f"settle_bias_{delta}",
                base="xgboost",
                transforms=[("class_bias", {"class_idx": 1, "delta": delta})],
            )
        )

    return configs


def run_search(
    data_dir: str = DEFAULT_DATA_DIR,
    configs: list[SearchConfig] | None = None,
) -> list[dict]:
    """Run LOO backtest for each config using cached base predictions.

    Trains XGBoost ONCE, caches base predictions, then applies transforms.
    This is the key optimization: transforms are <1s vs 30-60s for retraining.

    Returns:
        List of result dicts sorted by avg_score descending.
    """
    global _search_state
    if configs is None:
        configs = generate_search_strategies()

    with _search_lock:
        _search_state = SearchProgress(
            total=len(configs),
            completed=0,
            running=True,
        )

    try:
        results = _run_search_inner(configs, data_dir)
    except Exception as e:
        with _search_lock:
            _search_state.running = False
            _search_state.error = str(e)
        raise
    else:
        with _search_lock:
            _search_state.running = False
        return results


def _run_search_inner(
    configs: list[SearchConfig],
    data_dir: str,
) -> list[dict]:
    """Inner search loop: cache base, apply transforms, score."""
    from web.backtest import _discover_rounds, _load_seeds, _make_loo_data_dir
    from web.strategies import predict_xgboost

    round_dirs = _discover_rounds(data_dir)
    all_results: list[dict] = []

    for config in configs:
        logger.info("Search: evaluating %s", config.name)
        all_scores: list[float] = []
        round_scores: dict[int, list[float]] = {}

        for round_dir in round_dirs:
            from web.backtest import _get_round_number

            rn = _get_round_number(round_dir)
            seeds = _load_seeds(round_dir)
            loo_dir = _make_loo_data_dir(data_dir, round_dir.name)
            seed_scores: list[float] = []

            for grid, gt, _setts in seeds:
                base_pred = predict_xgboost(grid, loo_dir)
                transformed = apply_transform_chain(
                    base_pred,
                    grid,
                    config.transforms,
                )
                final = floor_and_normalize(transformed)
                result = score_prediction(gt, final)
                seed_scores.append(result["score"])

            if seed_scores:
                round_scores[rn] = seed_scores
                all_scores.extend(seed_scores)

        avg = float(np.mean(all_scores)) if all_scores else 0.0
        entry = {
            "strategy": config.name,
            "avg_score": round(avg, 2),
            "num_rounds": len(round_scores),
            "per_round": {k: round(float(np.mean(v)), 2) for k, v in round_scores.items()},
        }
        all_results.append(entry)

        # Save as backtest result for leaderboard integration
        from web.backtest import BacktestResult, save_result

        bt_result = BacktestResult(
            strategy_name=config.name,
            scores=round_scores,
            avg_score=avg,
            timestamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
        )
        save_result(bt_result)

        with _search_lock:
            _search_state.completed += 1
            _search_state.results.append(entry)

    all_results.sort(key=lambda x: x["avg_score"], reverse=True)
    return all_results
