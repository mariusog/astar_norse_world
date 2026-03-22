# Astar Norse World

Predicts 6-class terrain probabilities on a 40x40 Norse settlement simulation grid for the [NM i AI 2026](https://app.ainm.no) competition.

## Quick Start

```bash
pip install -e .
export ASTAR_TOKEN="your_jwt_token"
python -m scripts.submit_v3 --token $ASTAR_TOKEN --regime survive --budget 45
```

## Usage

```bash
# Probe a round (5 queries) to detect regime, then submit with 45 observation queries
python -m scripts.submit_v3 --token $ASTAR_TOKEN --regime aggressive --budget 45

# Capture completed round data for training
python -m scripts.post_round --token $ASTAR_TOKEN

# Run leave-one-out backtest across all historical rounds
python scripts/loo_backtest_v3.py

# Dry run (no actual submission)
python -m scripts.submit_v3 --token $ASTAR_TOKEN --dry-run
```

## How It Works

1. **Probe** (5 queries) — detect regime (survive/aggressive/collapse) from settlement survival rates
2. **Train** — XGBoost on 19 spatial features per cell (terrain, distance, density, neighborhood)
3. **Observe** (45 queries) — 15x15 viewports on settlement clusters
4. **Predict** — XGBoost + Bayesian Dirichlet update from per-terrain observation marginals
5. **Submit** — power transform, floor, normalize, validate, submit per seed

## Project Structure

```
src/              # Core prediction and simulation modules
scripts/          # Submission pipeline, backtesting, data capture
tests/            # 480+ unit tests (pytest)
data/rounds/      # Historical round ground truth (20+ rounds)
docs/             # Strategy, benchmarks, analysis reports
web/              # Flask dashboard for exploration and backtesting
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ --ignore=tests/_legacy -q -m "not slow"

# Lint and format
ruff check src/ scripts/ tests/
ruff format src/ scripts/ tests/
```

## Configuration

All tuning parameters are in `src/constants.py`. Regime-specific settings (ensemble weights, power transforms, Dirichlet concentrations) are in `scripts/submit_v3.py`. See `docs/strategy.md` for the full parameter table.

## License

MIT
