"""Historical round data collector for competition analysis.

Fetches completed rounds from the competition server and saves
ground truth, initial states, and analysis metadata for offline
use by the prior builder.

CLI: python -m src.round_collector --token <JWT>
Also accepts ASTAR_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.constants import NUM_SEEDS
from src.terrain import server_grid_to_internal

logger = logging.getLogger(__name__)


class RoundClient(Protocol):
    """Minimal protocol for round collection API calls."""

    def list_rounds(self) -> list[dict[str, Any]]: ...
    def get_round(self, round_id: int) -> dict[str, Any]: ...
    def analysis(self, round_id: int, seed_index: int) -> dict[str, Any]: ...


def collect_all_rounds(
    client: RoundClient,
    data_dir: str | Path,
) -> list[int]:
    """Fetch all completed rounds and save ground truth data.

    Idempotent: skips rounds where ground_truth.npy already exists
    for all seeds.

    Args:
        client: API client implementing RoundClient protocol.
        data_dir: Base directory for saved data (e.g. "data").

    Returns:
        List of newly collected round IDs.
    """
    data_path = Path(data_dir)
    rounds_dir = data_path / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = client.list_rounds()
    completed = _filter_completed_rounds(all_rounds)
    logger.info("Found %d completed rounds", len(completed))

    newly_collected: list[int] = []
    for round_info in completed:
        round_id = round_info["id"]
        if _round_already_captured(rounds_dir, round_id):
            logger.info("Round %d already captured, skipping", round_id)
            continue

        _collect_single_round(client, round_info, rounds_dir)
        newly_collected.append(round_id)

    logger.info("Collected %d new rounds", len(newly_collected))
    return newly_collected


def _filter_completed_rounds(
    rounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only rounds with status 'completed'."""
    return [r for r in rounds if r.get("status") == "completed"]


def _round_already_captured(rounds_dir: Path, round_id: int) -> bool:
    """Check if all seeds' ground truth files exist for a round."""
    round_dir = rounds_dir / str(round_id)
    if not round_dir.exists():
        return False
    for seed_idx in range(NUM_SEEDS):
        gt_path = round_dir / f"seed_{seed_idx}" / "ground_truth.npy"
        if not gt_path.exists():
            return False
    return True


def _collect_single_round(
    client: RoundClient,
    round_info: dict[str, Any],
    rounds_dir: Path,
) -> None:
    """Fetch and save all data for one round."""
    round_id = round_info["id"]
    round_dir = rounds_dir / str(round_id)
    round_dir.mkdir(parents=True, exist_ok=True)

    # Save round metadata
    _save_json(round_dir / "round.json", round_info)
    logger.info("Collecting round %d", round_id)

    round_detail = client.get_round(round_id)
    initial_states = round_detail.get("initial_states", [])

    for seed_idx in range(min(NUM_SEEDS, len(initial_states))):
        _collect_seed_data(
            client,
            round_id,
            seed_idx,
            initial_states[seed_idx],
            round_dir,
        )


def _collect_seed_data(
    client: RoundClient,
    round_id: int,
    seed_idx: int,
    initial_state: dict[str, Any],
    round_dir: Path,
) -> None:
    """Fetch and save data for a single seed within a round."""
    seed_dir = round_dir / f"seed_{seed_idx}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Save initial grid
    grid = _parse_server_grid(initial_state.get("grid", []))
    np.save(seed_dir / "initial_grid.npy", grid)

    # Save settlements
    settlements = initial_state.get("settlements", [])
    _save_json(seed_dir / "initial_settlements.json", settlements)

    # Fetch and save ground truth from analysis endpoint
    try:
        analysis = client.analysis(round_id, seed_idx)
        gt_grid = _parse_server_grid(analysis.get("grid", []))
        np.save(seed_dir / "ground_truth.npy", gt_grid)

        meta = {
            "round_id": round_id,
            "seed_index": seed_idx,
            "height": gt_grid.shape[0],
            "width": gt_grid.shape[1],
        }
        _save_json(seed_dir / "analysis_meta.json", meta)
        logger.info("  Seed %d: saved GT %s", seed_idx, gt_grid.shape)
    except Exception:
        logger.exception("  Seed %d: failed to fetch analysis", seed_idx)


def _parse_server_grid(grid_data: list[list[int]]) -> np.ndarray:
    """Convert server grid (list of lists) to InternalTerrain array."""
    if not grid_data:
        return np.array([], dtype=np.int8)
    return server_grid_to_internal(grid_data)


def _save_json(path: Path, data: Any) -> None:
    """Save data as JSON to the given path."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_client_from_token(token: str) -> RoundClient:
    """Build an API client from a JWT token.

    Imports AstarClient lazily to avoid hard dependency when
    api_client.py is not yet available.
    """
    from src.api_client import AstarClient  # type: ignore[import-untyped]

    return AstarClient(token=token)  # type: ignore[return-value]


def main() -> None:
    """CLI entry point for round collection."""
    parser = argparse.ArgumentParser(
        description="Collect historical round data from competition server",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("ASTAR_TOKEN", ""),
        help="JWT auth token (or set ASTAR_TOKEN env var)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (default: data)",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("Token required via --token or ASTAR_TOKEN env var")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    client = _build_client_from_token(args.token)
    collected = collect_all_rounds(client, args.data_dir)
    logger.info("Done. Collected rounds: %s", collected)


if __name__ == "__main__":
    main()
