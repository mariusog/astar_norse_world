"""Tests for ObservationStore -- viewport observation aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import NUM_PREDICTION_CLASSES
from src.observation import ObservationStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> ObservationStore:
    """10x10 map with 2 seeds."""
    return ObservationStore(height=10, width=10, num_seeds=2)


# ---------------------------------------------------------------------------
# add_observation
# ---------------------------------------------------------------------------


def test_add_observation_updates_counts(store: ObservationStore) -> None:
    """Single observation increments counts at correct positions."""
    patch = np.array([[0, 1], [2, 3]])  # 2x2 terrain codes
    store.add_observation(0, viewport_x=3, viewport_y=5, grid_patch=patch)

    obs = store.observation_count(0)
    assert obs[5, 3] == 1
    assert obs[5, 4] == 1
    assert obs[6, 3] == 1
    assert obs[6, 4] == 1
    # Unobserved cell
    assert obs[0, 0] == 0


def test_add_observation_out_of_bounds_ignored(store: ObservationStore) -> None:
    """Viewport extending beyond map edges is clipped."""
    patch = np.array([[0, 1], [2, 3]])  # 2x2
    # Place at bottom-right corner, partially out of bounds
    store.add_observation(0, viewport_x=9, viewport_y=9, grid_patch=patch)
    obs = store.observation_count(0)
    assert obs[9, 9] == 1  # in bounds
    # Total observed should be 1 (only one cell in bounds)
    assert obs.sum() == 1


# ---------------------------------------------------------------------------
# Overlapping observations
# ---------------------------------------------------------------------------


def test_overlapping_observations_accumulate(store: ObservationStore) -> None:
    """Two overlapping observations accumulate counts."""
    patch1 = np.array([[0]])  # terrain class 0
    patch2 = np.array([[1]])  # terrain class 1
    store.add_observation(0, viewport_x=5, viewport_y=5, grid_patch=patch1)
    store.add_observation(0, viewport_x=5, viewport_y=5, grid_patch=patch2)

    obs = store.observation_count(0)
    assert obs[5, 5] == 2


# ---------------------------------------------------------------------------
# get_observed_probs
# ---------------------------------------------------------------------------


def test_get_observed_probs_unobserved_is_nan(store: ObservationStore) -> None:
    """Unobserved cells return NaN probabilities."""
    probs = store.get_observed_probs(0)
    assert probs.shape == (10, 10, NUM_PREDICTION_CLASSES)
    assert np.all(np.isnan(probs))


def test_get_observed_probs_single_observation(store: ObservationStore) -> None:
    """Single observation with small alpha smoothing stays sharp."""
    patch = np.array([[0]])  # terrain class 0
    store.add_observation(0, viewport_x=3, viewport_y=3, grid_patch=patch)

    probs = store.get_observed_probs(0)
    cell_probs = probs[3, 3]
    assert not np.any(np.isnan(cell_probs))
    np.testing.assert_allclose(cell_probs.sum(), 1.0, atol=1e-9)
    assert cell_probs[0] > cell_probs[1]  # observed class is dominant
    # With small alpha (0.01), class 0 gets ~(1+0.01)/(1+6*0.01) ≈ 0.95
    assert cell_probs[0] > 0.9


def test_get_observed_probs_multiple_observations(store: ObservationStore) -> None:
    """Multiple same-cell observations refine probabilities."""
    for _ in range(3):
        store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=np.array([[0]]))
    store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=np.array([[1]]))

    probs = store.get_observed_probs(0)
    cell_probs = probs[0, 0]
    np.testing.assert_allclose(cell_probs.sum(), 1.0, atol=1e-9)
    # Class 0 (3 obs) should dominate, class 1 (1 obs) second
    assert cell_probs[0] > cell_probs[1] > cell_probs[2]


def test_get_observed_probs_no_zeros(store: ObservationStore) -> None:
    """Laplace smoothing ensures no zero probabilities."""
    store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=np.array([[0]]))
    probs = store.get_observed_probs(0)
    cell_probs = probs[0, 0]
    assert np.all(cell_probs > 0)


# ---------------------------------------------------------------------------
# get_coverage_mask
# ---------------------------------------------------------------------------


def test_get_coverage_mask_empty(store: ObservationStore) -> None:
    """No observations means no coverage."""
    mask = store.get_coverage_mask(0)
    assert mask.shape == (10, 10)
    assert not np.any(mask)


def test_get_coverage_mask_after_observation(store: ObservationStore) -> None:
    """Coverage mask reflects observed cells."""
    patch = np.array([[0, 1, 2]])  # 1x3
    store.add_observation(0, viewport_x=2, viewport_y=4, grid_patch=patch)
    mask = store.get_coverage_mask(0)
    assert mask[4, 2]
    assert mask[4, 3]
    assert mask[4, 4]
    assert not mask[0, 0]


# ---------------------------------------------------------------------------
# observation_count
# ---------------------------------------------------------------------------


def test_observation_count_unknown_seed(store: ObservationStore) -> None:
    """Unknown seed returns zero counts."""
    obs = store.observation_count(99)
    assert obs.shape == (10, 10)
    assert obs.sum() == 0


# ---------------------------------------------------------------------------
# coverage_fraction
# ---------------------------------------------------------------------------


def test_coverage_fraction_empty(store: ObservationStore) -> None:
    """No observations gives 0% coverage."""
    assert store.coverage_fraction(0) == 0.0


def test_coverage_fraction_partial(store: ObservationStore) -> None:
    """Partial coverage is computed correctly."""
    patch = np.zeros((5, 5), dtype=np.int32)  # 25 cells on a 10x10 map
    store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=patch)
    frac = store.coverage_fraction(0)
    np.testing.assert_allclose(frac, 25.0 / 100.0, atol=1e-9)


# ---------------------------------------------------------------------------
# max_cell_observations
# ---------------------------------------------------------------------------


def test_max_cell_observations_empty(store: ObservationStore) -> None:
    """No observations returns 0."""
    assert store.max_cell_observations(0) == 0


def test_max_cell_observations_after_queries(store: ObservationStore) -> None:
    """Multiple queries track max observation count."""
    store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=np.array([[0]]))
    store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=np.array([[1]]))
    assert store.max_cell_observations(0) == 2


# ---------------------------------------------------------------------------
# Seed isolation
# ---------------------------------------------------------------------------


def test_seeds_are_isolated(store: ObservationStore) -> None:
    """Observations for different seeds don't cross-contaminate."""
    store.add_observation(0, viewport_x=0, viewport_y=0, grid_patch=np.array([[0]]))
    store.add_observation(1, viewport_x=5, viewport_y=5, grid_patch=np.array([[3]]))

    mask0 = store.get_coverage_mask(0)
    mask1 = store.get_coverage_mask(1)
    assert mask0[0, 0] and not mask0[5, 5]
    assert mask1[5, 5] and not mask1[0, 0]


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(store: ObservationStore, tmp_path: object) -> None:
    """Observations survive save/load cycle."""
    import pathlib

    path = pathlib.Path(str(tmp_path)) / "obs.npz"

    # Add observations to both seeds
    store.add_observation(0, viewport_x=1, viewport_y=2, grid_patch=np.array([[0, 1], [3, 4]]))
    store.add_observation(1, viewport_x=5, viewport_y=5, grid_patch=np.array([[2]]))

    # Save and reload
    store.save_to_disk(path)
    loaded = ObservationStore.load_from_disk(path)

    # Coverage masks should match
    np.testing.assert_array_equal(store.get_coverage_mask(0), loaded.get_coverage_mask(0))
    np.testing.assert_array_equal(store.get_coverage_mask(1), loaded.get_coverage_mask(1))

    # Observed probs should match
    orig_probs = store.get_observed_probs(0)
    loaded_probs = loaded.get_observed_probs(0)
    observed = ~np.isnan(orig_probs[:, :, 0])
    np.testing.assert_allclose(orig_probs[observed], loaded_probs[observed])


def test_load_from_nonexistent_raises(tmp_path: object) -> None:
    """Loading a missing file raises FileNotFoundError."""
    import pathlib

    path = pathlib.Path(str(tmp_path)) / "missing.npz"
    with pytest.raises(FileNotFoundError):
        ObservationStore.load_from_disk(path)
