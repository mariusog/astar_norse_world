# Competition Strategy

**Updated**: 2026-03-22 | **Rounds**: R1-R22 | **Best scores**: R19=87.6 (collapse), R15=87.0 (aggressive), R17=84.7 (aggressive)

## Architecture

```
Initial Grid → 5 Probe Queries (regime detection from settlement survival)
             → Force regime via --regime flag (skip probes if confident)
             → XGBoost Training (all data for survive, regime-specific otherwise)
             → 19-feature model (terrain one-hot + spatial + neighborhood context)
             → Conditional Ensemble + Power Transform
             → Bayesian Dirichlet Update (per-terrain equilibrium from observations)
             → Regime Transforms (spatial_smooth / collapse_shift)
             → 45 Observation Queries (settlement clusters + tiling)
             → Floor + Normalize → Validation → Submit
```

## Regime Detection (Safe-Default)

| Condition | Regime | Rationale |
|-----------|--------|-----------|
| < 3 settlements checked | survive | Insufficient data, safest default |
| > 35% survival rate | aggressive | Strong expansion signal |
| < 5% survival AND >= 5 checked | deep_collapse | High confidence collapse |
| Everything else | survive | Covers survive + partial_collapse |

**Best practice**: Run 5 probes first, then force regime via `--regime` flag to save 5 queries for observations.

## Per-Regime Parameters

| Regime | XGBoost Wt | Power | Dirichlet Conc | Transform | Training Data |
|--------|-----------|-------|----------------|-----------|---------------|
| survive | 0.9 | 0.95 | 500 (no shift) | spatial_smooth(0.3) | All rounds |
| deep_collapse | 0.7 | 1.0 | 30 | collapse_shift(0.3) | R3,4,8,9,10,13,19 |
| aggressive | 1.0 | 1.0 | 10 (heavy shift) | none | R6,7,11,12,15,17,18 |
| partial_collapse | 0.9 | 1.05 | 500 (no shift) | none | All rounds |

**Key insight**: XGBoost alone scores 88+ on survive rounds. Observation shifting hurts survive (conc=500 effectively disables it). For aggressive/collapse, observations are critical (+20-30 pts).

## Scoring

```
KL(p||q) = sum(p_i * log(p_i / q_i))    per cell
entropy(cell) = -sum(p_i * log(p_i))
weighted_kl = sum(entropy * KL) / sum(entropy)    (dynamic cells only)
score = 100 * exp(-3 * weighted_kl)
```

The optimal predictor under KL(p||q) loss is q* = E[p], the posterior mean. The entire optimization reduces to estimating E[p] well from limited observations.

## Score History

| Round | Regime | Score | Notes |
|-------|--------|-------|-------|
| R15 | aggressive | **87.0** | Best aggressive |
| R16 | survive | 79.0 | Power 0.9 + eq shift hurt |
| R17 | aggressive | 84.7 | Solid |
| R18 | aggressive | 70.8 | Below average |
| R19 | deep_collapse | **87.6** | Best collapse, Dirichlet update worked |
| R21 | survive | 82.3 | First with conc=500 fix |
| R22 | survive | *pending* | New features + regime fix |

## Submission Checklist

```bash
# 1. Capture any new completed rounds
python -m scripts.post_round --token $TOKEN

# 2. Probe with 5 queries to detect regime
# (run the inline probe script, check survival rate)

# 3. Submit with forced regime and 45 obs budget
python -m scripts.submit_v3 --token $TOKEN --regime survive --budget 45

# 4. If validation fails, DO NOT use --force. Fix the bug first.
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/submit_v3.py` | Main submission pipeline |
| `scripts/loo_backtest_v3.py` | LOO backtest for offline validation |
| `src/ml_predictor.py` | XGBoost per-cell classifier (19 features) |
| `src/features.py` | Spatial feature extraction (7 functions) |
| `src/predictor_protocol.py` | GridPredictor protocol + adapters |
| `src/prediction_validator.py` | Pre-submission sanity checks |
| `src/observation.py` | ObservationStore with disk persistence |
| `src/scoring.py` | KL divergence scoring (matches server) |
| `src/simulation.py` | 50-year lifecycle simulator (phases in sim_phases.py) |
| `src/terrain.py` | Terrain types, mappings, map_server_codes() |
| `src/transforms.py` | Post-processing transforms (power, smooth, floor) |
| `web/transforms.py` | Parametric post-processing transforms |
