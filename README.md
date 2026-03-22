# Astar Norse World — NM i AI 2026

Terrain probability predictor for the Norse World simulation challenge at [app.ainm.no](https://app.ainm.no).

Predicts 6-class terrain probability distributions (Empty, Settlement, Port, Ruin, Forest, Mountain) on a 40x40 grid after a 50-year Norse settlement simulation, using XGBoost with spatial features and Bayesian observation updating.

## Quick Start

```bash
# Install
pip install -e .

# Capture round data
python -m src.round_collector --token $ASTAR_TOKEN

# Submit predictions for active round
python -m scripts.submit_v3 --token $ASTAR_TOKEN --regime survive --budget 45

# Run LOO backtest
python scripts/loo_backtest_v3.py
```

## Architecture

1. **Probe** (5 queries): Detect regime from settlement survival rates
2. **Train**: XGBoost on 19 spatial features (terrain, distance, density, neighborhood context)
3. **Observe** (45 queries): 15x15 viewports targeting settlement clusters
4. **Predict**: XGBoost + Bayesian Dirichlet update from observations
5. **Post-process**: Power transform, spatial smoothing, floor + normalize
6. **Submit**: Validated 40x40x6 probability tensor per seed

## Project Structure

```
src/                    # Core modules
  ml_predictor.py       # XGBoost per-cell classifier
  features.py           # Spatial feature extraction
  observation.py        # Viewport observation aggregator
  simulation.py         # Norse world simulation engine
  terrain.py            # Terrain types and mappings
  predictor_protocol.py # GridPredictor protocol
  transforms.py         # Probability transforms
scripts/                # Submission and analysis
  submit_v3.py          # Main submission pipeline
  loo_backtest_v3.py    # Leave-one-out validation
  post_round.py         # Post-round data capture
tests/                  # Test suite (480+ tests)
data/rounds/            # Historical round data
docs/                   # Strategy and analysis docs
web/                    # Flask dashboard
```

## Tests

```bash
python -m pytest tests/ --ignore=tests/_legacy -q -m "not slow"
```

## License

MIT
