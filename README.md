# Astar Norse World

Terrain prediction agent for the [NM i AI 2026](https://app.ainm.no) A* Norse World challenge. Predicts 6-class terrain probabilities on a 40x40 grid in a Norse settlement simulation, using adaptive observation budgets and Bayesian updates.

## How It Works

1. **Probe** (5 queries) -- detect regime (survive/aggressive/collapse) from settlement survival rates
2. **Train** -- XGBoost on 19 spatial features per cell (terrain, distance, density, neighborhood)
3. **Observe** (45 queries) -- 15x15 viewports targeted at settlement clusters
4. **Predict** -- XGBoost + Bayesian Dirichlet update from per-terrain observation marginals
5. **Submit** -- power transform, floor, normalize, validate, submit per seed

## Quick Start

```bash
pip install -e ".[dev]"
export ASTAR_TOKEN="your_jwt_token"

# Submit a round
python -m scripts.submit_v3 --token $ASTAR_TOKEN --regime survive --budget 45

# Capture completed round data for training
python -m scripts.post_round --token $ASTAR_TOKEN

# Run leave-one-out backtest across all historical rounds
python scripts/loo_backtest_v3.py

# Dry run (no actual submission)
python -m scripts.submit_v3 --token $ASTAR_TOKEN --dry-run
```

## Project Structure

```
src/                # Core prediction and simulation modules
  constants.py      # All tuning parameters (regime weights, thresholds, etc.)
  ml_predictor.py   # XGBoost model training and prediction
  features.py       # Spatial feature engineering (19 features per cell)
  pipeline.py       # End-to-end prediction pipeline
scripts/            # Submission pipeline, backtesting, data capture
  submit_v3.py      # Main submission script with regime-adaptive strategies
  loo_backtest_v3.py # Leave-one-out cross-validation
  post_round.py     # Download round ground truth after completion
tests/              # 480+ unit tests
data/rounds/        # Historical round ground truth (20+ rounds)
docs/               # Strategy documents, benchmarks, analysis
web/                # Flask dashboard for exploration and backtesting
```

## Development

```bash
# Run tests (fast, excludes slow tests)
python -m pytest tests/ -m "not slow" -q --tb=line

# Lint and format
ruff check .
ruff format .

# Type check
mypy src/ --ignore-missing-imports
```

## Configuration

All tuning parameters live in `src/constants.py`. Regime-specific settings (ensemble weights, power transforms, Dirichlet concentrations) are in `scripts/submit_v3.py`. See `docs/strategy.md` for the full parameter table.

## Requirements

- Python 3.11+
- numpy, scipy, scikit-learn, xgboost, joblib
- FastAPI, uvicorn (for the web dashboard)
- requests, httpx (API client)

## License

MIT
