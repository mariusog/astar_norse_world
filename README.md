# Astar Norse World

Terrain prediction agent for the [NM i AI 2026](https://app.ainm.no) A* Norse World challenge. Predicts 6-class terrain probabilities on a 40x40 grid in a Norse settlement simulation, using adaptive observation budgets and Bayesian updates.

## Competition

**Challenge:** Predict 6-class terrain probability distributions on a 40x40 grid in a Norse settlement simulation. Each round, you get 50 stochastic queries to observe the world, then submit probability predictions for all 1,600 cells across 5 seeds.

**Scoring:** `100 * exp(-3 * entropy_weighted_KL)` -- entropy-weighted KL divergence concentrates scoring on uncertain cells where the outcome varies, not static terrain like ocean or mountains.

## Solution

### Pipeline

1. **Probe** (5 queries) -- detect regime (survive/aggressive/collapse) from settlement survival rates
2. **Train** -- XGBoost on 19 spatial features per cell (terrain, distance, density, neighborhood)
3. **Observe** (45 queries) -- 15x15 viewports targeted at settlement clusters
4. **Predict** -- XGBoost + Bayesian Dirichlet update from per-terrain observation marginals
5. **Submit** -- power transform, floor, normalize, validate, submit per seed

### Key Design Decisions

**XGBoost with spatial features** -- 19 features per cell capturing terrain type, settlement distance/density, forest density, coastal proximity, and neighborhood composition at radius 2 and 4. Chosen over deep learning because the spatial features are sufficient and the model is interpretable. LOO baseline: 72.3 vs 67.0 for flat priors.

**Per-terrain equilibrium shift** -- The most impactful technique. Instead of using noisy per-cell observations (only 1-2 draws each), we pool all observations by terrain type to get 150-300+ samples per class. This produces stable marginal distributions that automatically adapt to round dynamics -- collapse rounds shift toward empty, aggressive rounds shift toward settlements. No explicit regime detection needed.

**Regime-specific concentration parameters** -- Controls how much to trust observations vs the XGBoost model. Survive rounds (concentration=500) trust XGBoost heavily; aggressive rounds (concentration=10) shift heavily from observations because settlement expansion is unpredictable.

**Adaptive viewport placement** -- Viewports are placed at settlement density peaks (box-blur over settlement mask). Dense areas provide the most information -- more cells observed, more reliable per-terrain marginals.

### What Worked and What Didn't

| Decision | Outcome |
|----------|---------|
| Per-terrain aggregation of observations | +5-6 points over XGBoost alone |
| Regime-specific training data splits | Helped for aggressive/collapse, hurt when misclassified |
| Per-cell observation blending | Noise dominated with only 1-2 draws per cell -- abandoned |
| Low probability floor (0.01) | Essential -- 0.03 floor dropped score to 65.9 |
| Regime detection from 5 probes | Only 37.5% accurate -- R13 misclassification cost 42.6 points |
| Static terrain overrides (ocean/mountain at 99%) | Free points on known cells |

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
