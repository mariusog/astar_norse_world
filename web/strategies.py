"""Strategy implementations for prediction models.

Contains the actual predict_fn callables for each strategy.
Imported by web.models for registration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from src.constants import NUM_PREDICTION_CLASSES
from src.terrain import SERVER_TO_INTERNAL, InternalTerrain

logger = logging.getLogger(__name__)


def load_round_data(
    round_dir: Path,
) -> list[tuple[np.ndarray, np.ndarray, list[dict]]]:
    """Load all seeds from a round directory.

    Returns list of (internal_grid, ground_truth, settlements) tuples.
    """
    rjson = round_dir / "round.json"
    if not rjson.exists():
        return []
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


def collect_training_data(
    data_dir: str,
    exclude_round: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray, list[dict]]]:
    """Collect all round data, optionally excluding one round."""
    data_path = Path(data_dir)
    all_data = []
    for round_dir in sorted(data_path.iterdir()):
        if not round_dir.is_dir():
            continue
        if exclude_round and round_dir.name == exclude_round:
            continue
        all_data.extend(load_round_data(round_dir))
    return all_data


def predict_flat_priors(grid: np.ndarray, data_dir: str) -> np.ndarray:
    """Flat terrain-type priors from all available rounds."""
    from src.predictor_v2 import build_priors_from_rounds
    from web.models import apply_static_and_normalize

    priors = build_priors_from_rounds(data_dir)
    h, w = grid.shape
    tensor = np.zeros((h, w, NUM_PREDICTION_CLASSES), dtype=np.float64)
    for t_val, prior_vec in priors.items():
        mask = grid == t_val
        if mask.any():
            tensor[mask] = prior_vec
    return apply_static_and_normalize(tensor, grid)


def predict_distance_priors(grid: np.ndarray, data_dir: str) -> np.ndarray:
    """Flat priors blended with settlement distance model."""
    from src.position_priors import build_distance_model, predict_from_position
    from src.predictor_v2 import DEFAULT_TERRAIN_PRIORS
    from web.models import apply_static_and_normalize

    dist_model = build_distance_model(data_dir)
    settlements: list[dict] = []
    for row, col in np.argwhere(
        (grid == InternalTerrain.SETTLEMENT) | (grid == InternalTerrain.PORT)
    ):
        settlements.append({"x": int(col), "y": int(row)})
    tensor = predict_from_position(
        grid,
        settlements,
        DEFAULT_TERRAIN_PRIORS,
        dist_model,
    )
    return apply_static_and_normalize(tensor, grid)


def predict_feature_lookup(grid: np.ndarray, data_dir: str) -> np.ndarray:
    """Feature-conditioned lookup from historical data."""
    from src.features import compute_feature_grid

    features = compute_feature_grid(grid)
    training = collect_training_data(data_dir)
    if not training:
        return predict_flat_priors(grid, data_dir)
    return _feature_lookup_core(grid, features, training)


def _feature_lookup_core(
    grid: np.ndarray,
    features: dict[str, np.ndarray],
    training: list[tuple[np.ndarray, np.ndarray, list[dict]]],
) -> np.ndarray:
    """Build feature-conditioned priors from training data."""
    from src.features import compute_feature_grid
    from web.models import apply_static_and_normalize

    accum: dict[tuple, np.ndarray] = {}
    counts: dict[tuple, int] = {}
    for t_grid, gt, _setts in training:
        t_feats = compute_feature_grid(t_grid)
        _accumulate(t_grid, gt, t_feats, accum, counts)
    h, w = grid.shape
    tensor = np.zeros((h, w, NUM_PREDICTION_CLASSES), dtype=np.float64)
    _fill(tensor, grid, features, accum, counts)
    return apply_static_and_normalize(tensor, grid)


def _accumulate(
    grid: np.ndarray,
    gt: np.ndarray,
    feats: dict,
    accum: dict,
    counts: dict,
) -> None:
    """Accumulate GT keyed by (terrain, dist_band, coastal)."""
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            key = _cell_key(grid, feats, r, c)
            if key not in accum:
                accum[key] = np.zeros(NUM_PREDICTION_CLASSES)
                counts[key] = 0
            accum[key] += gt[r, c]
            counts[key] += 1


def _fill(
    tensor: np.ndarray,
    grid: np.ndarray,
    features: dict,
    accum: dict,
    counts: dict,
) -> None:
    """Fill tensor from accumulated feature priors."""
    uniform = np.full(NUM_PREDICTION_CLASSES, 1.0 / NUM_PREDICTION_CLASSES)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            key = _cell_key(grid, features, r, c)
            if key in accum and counts[key] > 0:
                tensor[r, c] = accum[key] / counts[key]
            else:
                tensor[r, c] = uniform


def _cell_key(grid: np.ndarray, feats: dict, r: int, c: int) -> tuple:
    """Compute feature key for a cell."""
    t = int(grid[r, c])
    d = min(int(feats["settlement_dist"][r, c]), 5)
    coast = bool(feats["coastal_mask"][r, c])
    return (t, d, coast)


def predict_xgboost(grid: np.ndarray, data_dir: str) -> np.ndarray:
    """XGBoost model trained on all available rounds."""
    return _xgboost_core(grid, data_dir, exclude_rounds=set())


def predict_xgboost_survive(grid: np.ndarray, data_dir: str) -> np.ndarray:
    """XGBoost trained excluding collapse rounds (3, 4)."""
    return _xgboost_core(grid, data_dir, exclude_rounds={3, 4})


def predict_xgboost_collapse(grid: np.ndarray, data_dir: str) -> np.ndarray:
    """XGBoost trained excluding survive rounds (1, 2, 5)."""
    return _xgboost_core(grid, data_dir, exclude_rounds={1, 2, 5})


def _xgboost_core(
    grid: np.ndarray,
    data_dir: str,
    exclude_rounds: set[int],
) -> np.ndarray:
    """Core XGBoost training and prediction."""

    from src.features import compute_feature_grid

    training = collect_training_data(data_dir)
    if exclude_rounds:
        training = _filter_rounds(data_dir, exclude_rounds)
    if not training:
        return predict_flat_priors(grid, data_dir)
    X, y = _build_features(training)
    model = _train(X, y)
    return _predict_grid(model, grid, compute_feature_grid(grid))


def _filter_rounds(
    data_dir: str,
    exclude: set[int],
) -> list[tuple[np.ndarray, np.ndarray, list[dict]]]:
    """Collect training data excluding specific round numbers."""
    data_path = Path(data_dir)
    result: list[tuple[np.ndarray, np.ndarray, list[dict]]] = []
    for round_dir in sorted(data_path.iterdir()):
        if not round_dir.is_dir():
            continue
        rjson = round_dir / "round.json"
        if not rjson.exists():
            continue
        with open(rjson) as f:
            rd = json.load(f)
        if rd.get("round_number", 0) not in exclude:
            result.extend(load_round_data(round_dir))
    return result


def _build_features(
    training: list[tuple[np.ndarray, np.ndarray, list[dict]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and label vector."""
    from src.features import compute_feature_grid

    X_parts, y_parts = [], []
    for t_grid, gt, _setts in training:
        feats = compute_feature_grid(t_grid)
        h, w = t_grid.shape
        for r in range(h):
            for c in range(w):
                X_parts.append(_cell_features(t_grid, feats, r, c))
                y_parts.append(gt[r, c])
    return np.array(X_parts), np.array(y_parts)


def _cell_features(grid: np.ndarray, feats: dict, r: int, c: int) -> list[int]:
    """Extract feature vector for one cell."""
    return [
        int(grid[r, c]),
        int(feats["settlement_dist"][r, c]),
        int(feats["ocean_dist"][r, c]),
        int(feats["forest_density"][r, c]),
        int(feats["coastal_mask"][r, c]),
    ]


def _train(X: np.ndarray, y: np.ndarray) -> object:
    """Train XGBoost multi-output regressor."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        verbosity=0,
        multi_strategy="multi_output_tree",
    )
    model.fit(X, y)
    return model


def _predict_grid(
    model: object,
    grid: np.ndarray,
    features: dict[str, np.ndarray],
) -> np.ndarray:
    """Predict full grid using trained XGBoost model."""
    from web.models import apply_static_and_normalize

    h, w = grid.shape
    X_pred = []
    for r in range(h):
        for c in range(w):
            X_pred.append(_cell_features(grid, features, r, c))
    preds = model.predict(np.array(X_pred))
    tensor = np.maximum(preds.reshape(h, w, NUM_PREDICTION_CLASSES), 0.0)
    return apply_static_and_normalize(tensor, grid)
