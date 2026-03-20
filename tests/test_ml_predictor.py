"""Tests for XGBoost per-cell probability predictor."""

import json

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES, PROBABILITY_FLOOR
from src.terrain import InternalTerrain

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_grid() -> np.ndarray:
    """10x10 grid with diverse terrain types for feature testing."""
    grid = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    grid[3, 3] = InternalTerrain.SETTLEMENT
    grid[3, 4] = InternalTerrain.PORT
    grid[5, 5] = InternalTerrain.FOREST
    grid[5, 6] = InternalTerrain.FOREST
    grid[7, 7] = InternalTerrain.MOUNTAIN
    grid[4, 3] = InternalTerrain.RUIN
    return grid


@pytest.fixture
def sample_data_dir(tmp_path: object) -> object:
    """Create a minimal data directory with two rounds of fake data."""
    import pathlib

    base = pathlib.Path(str(tmp_path)) / "rounds"
    for rnum in range(2):
        rdir = base / f"round_{rnum}"
        for sidx in range(2):
            sdir = rdir / f"seed_{sidx}"
            sdir.mkdir(parents=True)
            grid = _make_deterministic_grid(seed=rnum * 10 + sidx)
            gt = _make_deterministic_gt(grid)
            np.save(sdir / "initial_grid.npy", grid)
            np.save(sdir / "ground_truth.npy", gt)
            meta = {"seed_index": sidx, "score": None}
            (sdir / "analysis_meta.json").write_text(json.dumps(meta))
    return base


def _make_deterministic_grid(seed: int) -> np.ndarray:
    """Build a small 10x10 grid with seeded terrain."""
    rng = np.random.default_rng(seed)
    grid = np.full((10, 10), InternalTerrain.PLAINS, dtype=np.int8)
    grid[0, :] = InternalTerrain.OCEAN
    grid[-1, :] = InternalTerrain.OCEAN
    grid[:, 0] = InternalTerrain.OCEAN
    grid[:, -1] = InternalTerrain.OCEAN
    for _ in range(3):
        y, x = rng.integers(2, 8), rng.integers(2, 8)
        grid[y, x] = InternalTerrain.FOREST
    grid[4, 4] = InternalTerrain.SETTLEMENT
    grid[6, 6] = InternalTerrain.MOUNTAIN
    return grid


def _make_deterministic_gt(grid: np.ndarray) -> np.ndarray:
    """Build a simple GT based on terrain type (one-hot of pred class)."""
    h, w = grid.shape
    gt = np.full((h, w, NUM_PREDICTION_CLASSES), PROBABILITY_FLOOR)
    for y in range(h):
        for x in range(w):
            cls = InternalTerrain(grid[y, x]).to_prediction_class()
            gt[y, x, cls] = 0.9
    gt = gt / gt.sum(axis=-1, keepdims=True)
    return gt


# ---------------------------------------------------------------------------
# Tests: extract_cell_features
# ---------------------------------------------------------------------------


def test_extract_cell_features_shape(mixed_grid: np.ndarray) -> None:
    """Feature matrix has correct shape (H*W, n_features)."""
    from src.ml_predictor import NUM_FEATURES, extract_cell_features

    features = extract_cell_features(mixed_grid)
    h, w = mixed_grid.shape
    assert features.shape == (h * w, NUM_FEATURES)


def test_extract_cell_features_dtype(mixed_grid: np.ndarray) -> None:
    """Feature matrix is float32."""
    from src.ml_predictor import extract_cell_features

    features = extract_cell_features(mixed_grid)
    assert features.dtype == np.float32


def test_extract_cell_features_onehot_sums(mixed_grid: np.ndarray) -> None:
    """One-hot terrain columns sum to 1 per row."""
    from src.ml_predictor import NUM_TERRAIN_TYPES, extract_cell_features

    features = extract_cell_features(mixed_grid)
    onehot_sums = features[:, :NUM_TERRAIN_TYPES].sum(axis=1)
    np.testing.assert_allclose(onehot_sums, 1.0, atol=1e-6)


def test_extract_cell_features_ocean_cell_is_type_zero(
    mixed_grid: np.ndarray,
) -> None:
    """Ocean cell (0,0) has one-hot[0]=1."""
    from src.ml_predictor import extract_cell_features

    features = extract_cell_features(mixed_grid)
    assert features[0, 0] == 1.0  # ocean = InternalTerrain(0)


# ---------------------------------------------------------------------------
# Tests: build_training_data
# ---------------------------------------------------------------------------


def test_build_training_data_shape(sample_data_dir: object) -> None:
    """Training data has correct shapes from 2 rounds x 2 seeds."""
    from src.ml_predictor import NUM_FEATURES, build_training_data

    x, y = build_training_data(str(sample_data_dir))
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == NUM_FEATURES
    assert y.shape[1] == NUM_PREDICTION_CLASSES
    assert x.shape[0] == 4 * 100  # 4 seeds x 10x10 grid


def test_build_training_data_exclude_round(sample_data_dir: object) -> None:
    """Excluding a round reduces sample count."""
    from src.ml_predictor import build_training_data

    x_full, _ = build_training_data(str(sample_data_dir))
    x_excl, _ = build_training_data(str(sample_data_dir), exclude_round="round_0")
    assert x_excl.shape[0] < x_full.shape[0]
    assert x_excl.shape[0] == 2 * 100  # 2 seeds from round_1


# ---------------------------------------------------------------------------
# Tests: train_model
# ---------------------------------------------------------------------------


def test_train_model_returns_fitted(sample_data_dir: object) -> None:
    """Model trains without error and has predict method."""
    from src.ml_predictor import build_training_data, train_model

    x, y = build_training_data(str(sample_data_dir))
    model = train_model(x, y, seed=42)
    assert hasattr(model, "predict")


def test_train_model_deterministic(sample_data_dir: object) -> None:
    """Same seed produces identical predictions."""
    from src.ml_predictor import build_training_data, train_model

    x, y = build_training_data(str(sample_data_dir))
    m1 = train_model(x, y, seed=99)
    m2 = train_model(x, y, seed=99)
    pred1 = m1.predict(x[:10])
    pred2 = m2.predict(x[:10])
    np.testing.assert_allclose(pred1, pred2)


# ---------------------------------------------------------------------------
# Tests: predict_grid
# ---------------------------------------------------------------------------


def test_predict_grid_shape(mixed_grid: np.ndarray, sample_data_dir: object) -> None:
    """Predictions have shape H x W x 6."""
    from src.ml_predictor import build_training_data, predict_grid, train_model

    x, y = build_training_data(str(sample_data_dir))
    model = train_model(x, y, seed=42)
    pred = predict_grid(mixed_grid, model)
    assert pred.shape == (10, 10, NUM_PREDICTION_CLASSES)


def test_predict_grid_probabilities_sum_to_one(
    mixed_grid: np.ndarray,
    sample_data_dir: object,
) -> None:
    """Each cell's predicted probabilities sum to 1."""
    from src.ml_predictor import build_training_data, predict_grid, train_model

    x, y = build_training_data(str(sample_data_dir))
    model = train_model(x, y, seed=42)
    pred = predict_grid(mixed_grid, model)
    sums = pred.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_predict_grid_floor_respected(
    mixed_grid: np.ndarray,
    sample_data_dir: object,
) -> None:
    """No prediction is below the probability floor."""
    from src.ml_predictor import build_training_data, predict_grid, train_model

    x, y = build_training_data(str(sample_data_dir))
    model = train_model(x, y, seed=42)
    pred = predict_grid(mixed_grid, model)
    assert pred.min() >= PROBABILITY_FLOOR - 1e-7


def test_predict_grid_ocean_override(
    mixed_grid: np.ndarray,
    sample_data_dir: object,
) -> None:
    """Ocean cells get high probability for class 0 (EMPTY)."""
    from src.ml_predictor import build_training_data, predict_grid, train_model

    x, y = build_training_data(str(sample_data_dir))
    model = train_model(x, y, seed=42)
    pred = predict_grid(mixed_grid, model)
    ocean_pred = pred[0, 0]  # top-left is ocean
    assert ocean_pred[0] >= 0.95


def test_predict_grid_mountain_override(
    mixed_grid: np.ndarray,
    sample_data_dir: object,
) -> None:
    """Mountain cells get high probability for class 5 (MOUNTAIN)."""
    from src.ml_predictor import build_training_data, predict_grid, train_model

    x, y = build_training_data(str(sample_data_dir))
    model = train_model(x, y, seed=42)
    pred = predict_grid(mixed_grid, model)
    mtn_pred = pred[7, 7]  # mountain cell
    assert mtn_pred[5] >= 0.95


# ---------------------------------------------------------------------------
# Tests: _floor_and_normalize
# ---------------------------------------------------------------------------


def test_floor_and_normalize_handles_negatives() -> None:
    """Negative raw outputs get floored and normalized properly."""
    from src.ml_predictor import _floor_and_normalize

    raw = np.array([[-0.5, 0.0, 0.3, 0.0, 0.0, 1.2]])
    result = _floor_and_normalize(raw)
    assert result.min() >= PROBABILITY_FLOOR - 1e-7
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: loo_evaluate
# ---------------------------------------------------------------------------


def test_loo_evaluate_returns_scores(sample_data_dir: object) -> None:
    """LOO evaluation returns per-round and average scores."""
    from src.ml_predictor import loo_evaluate

    results = loo_evaluate(str(sample_data_dir), seed=42)
    assert "avg" in results
    assert results["avg"] > 0
    assert len(results) == 3  # 2 rounds + avg
