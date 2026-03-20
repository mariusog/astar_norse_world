# Prediction Quality Benchmark

Ground truth: 500 MC runs | Predictor: 50 MC runs | Map: 20x20

| Seed | Strategy | Score | W-KL | Dyn Cells | Time(s) |
|------|----------|-------|------|-----------|---------|
| 42 | pure_sim | 95.8 | 0.0143 | 400 | 2.96 |
| 42 | sim+obs | 97.3 | 0.0091 | 400 | 2.98 |
| 123 | pure_sim | 98.4 | 0.0054 | 400 | 0.79 |
| 123 | sim+obs | 99.1 | 0.0029 | 400 | 0.81 |
| 256 | pure_sim | 97.7 | 0.0079 | 400 | 1.15 |
| 256 | sim+obs | 98.5 | 0.0051 | 400 | 1.15 |
| 789 | pure_sim | 100.0 | 0.0000 | 400 | 0.28 |
| 789 | sim+obs | 100.0 | 0.0000 | 400 | 0.29 |
| 1024 | pure_sim | 97.0 | 0.0101 | 400 | 1.98 |
| 1024 | sim+obs | 97.9 | 0.0069 | 400 | 1.99 |

**Avg pure sim**: 97.8 | **Avg sim+obs**: 98.6
