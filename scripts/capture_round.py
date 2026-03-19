"""Capture all available data from a competition round.

For completed rounds, fetches ground truth via the analysis endpoint —
this is the most valuable data for calibrating our local simulation.

For active rounds, uses viewport queries to observe final states.

Usage:
    python scripts/capture_round.py --token <JWT> [--round-id <ID>]

If --round-id is omitted, lists all rounds so you can pick one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.api_client import APIError, AstarClient, BudgetExhaustedError
from src.constants import DEFAULT_MAP_HEIGHT, DEFAULT_MAP_WIDTH, NUM_SEEDS
from src.query_strategy import QueryPlanner, Viewport
from src.state_loader import load_round

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def list_rounds(client: AstarClient) -> None:
    """Print all available rounds."""
    _print_round_table(client)
    _print_team_results(client)


def _print_round_table(client: AstarClient) -> None:
    """Print the table of all rounds."""
    rounds = client.list_rounds()
    print(f"\n{'ID':<40} {'#':<4} {'Status':<12} {'Seeds':<6} {'Size'}")
    print("-" * 80)
    for r in rounds:
        size = f"{r.get('map_width', '?')}x{r.get('map_height', '?')}"
        print(
            f"{r['id']:<40} {r.get('round_number', '?'):<4} "
            f"{r.get('status', '?'):<12} {r.get('seeds_count', '?'):<6} {size}"
        )
    print()


def _print_team_results(client: AstarClient) -> None:
    """Print the team's results if available."""
    try:
        my_rounds = client.my_rounds()
        if not my_rounds:
            return
        print("Your team's results:")
        print(f"  {'#':<4} {'Score':<10} {'Rank':<8} {'Submitted':<10} {'Queries'}")
        print("  " + "-" * 50)
        for r in my_rounds:
            _print_team_round_row(r)
        print()
    except APIError:
        pass  # Not on a team or auth issue


def _print_team_round_row(r: dict) -> None:
    """Print one row of the team results table."""
    score = r.get("round_score")
    score_str = f"{score:.1f}" if score is not None else "\u2014"
    rank = r.get("rank")
    rank_str = f"{rank}/{r.get('total_teams', '?')}" if rank else "\u2014"
    print(
        f"  {r.get('round_number', '?'):<4} {score_str:<10} "
        f"{rank_str:<8} {r.get('seeds_submitted', 0):<10} "
        f"{r.get('queries_used', 0)}/{r.get('queries_max', 50)}"
    )


def capture_round(client: AstarClient, round_id: str) -> None:
    """Fetch and save all data from a round."""
    out_dir = Path("data/rounds") / round_id
    out_dir.mkdir(parents=True, exist_ok=True)

    round_data = _fetch_and_save_round(client, round_id, out_dir)
    states = _save_initial_states(round_data, out_dir)

    status = round_data.get("status", "unknown")
    ground_truth_captured = _maybe_capture_ground_truth(
        client,
        round_id,
        round_data,
        status,
        out_dir,
    )
    _capture_predictions(client, round_id, out_dir)
    _maybe_capture_viewports(
        client,
        round_id,
        round_data,
        status,
        states,
        out_dir,
    )
    _write_summary(client, round_id, round_data, out_dir, ground_truth_captured)


def _fetch_and_save_round(
    client: AstarClient,
    round_id: str,
    out_dir: Path,
) -> dict:
    """Fetch round details from the server and persist them."""
    logger.info("Fetching round details for %s", round_id)
    round_data = client.get_round(round_id)
    (out_dir / "round.json").write_text(json.dumps(round_data, indent=2, default=str))
    width = round_data.get("map_width", DEFAULT_MAP_WIDTH)
    height = round_data.get("map_height", DEFAULT_MAP_HEIGHT)
    seeds_count = round_data.get("seeds_count", NUM_SEEDS)
    status = round_data.get("status", "unknown")
    logger.info("Round status: %s, %dx%d, %d seeds", status, width, height, seeds_count)
    return round_data


def _save_initial_states(round_data: dict, out_dir: Path) -> list:
    """Parse and save initial states for each seed."""
    states = load_round(round_data)
    for idx, (grid, settlements) in enumerate(states):
        seed_dir = out_dir / f"seed_{idx}"
        seed_dir.mkdir(exist_ok=True)
        np.save(seed_dir / "initial_grid.npy", grid)
        settle_data = [
            {"x": s.x, "y": s.y, "owner_id": s.owner_id, "is_port": s.is_port} for s in settlements
        ]
        (seed_dir / "initial_settlements.json").write_text(json.dumps(settle_data, indent=2))
    logger.info("Saved %d initial states", len(states))
    return states


def _maybe_capture_ground_truth(
    client: AstarClient,
    round_id: str,
    round_data: dict,
    status: str,
    out_dir: Path,
) -> bool:
    """Capture ground truth if the round is completed/scoring."""
    if status not in ("completed", "scoring"):
        return False
    seeds_count = round_data.get("seeds_count", NUM_SEEDS)
    return _capture_ground_truth(client, round_id, seeds_count, out_dir)


def _maybe_capture_viewports(
    client: AstarClient,
    round_id: str,
    round_data: dict,
    status: str,
    states: list,
    out_dir: Path,
) -> None:
    """Use viewport queries if the round is still active."""
    if status != "active":
        return
    width = round_data.get("map_width", DEFAULT_MAP_WIDTH)
    height = round_data.get("map_height", DEFAULT_MAP_HEIGHT)
    seeds_count = round_data.get("seeds_count", NUM_SEEDS)
    _capture_viewports(client, round_id, seeds_count, width, height, states, out_dir)


def _capture_ground_truth(
    client: AstarClient,
    round_id: str,
    seeds_count: int,
    out_dir: Path,
) -> bool:
    """Fetch ground truth from the analysis endpoint."""
    logger.info("Fetching ground truth from analysis endpoint")
    captured = 0
    for seed_idx in range(seeds_count):
        try:
            analysis = client.analysis(round_id, seed_idx)
            seed_dir = out_dir / f"seed_{seed_idx}"
            seed_dir.mkdir(exist_ok=True)
            _save_analysis_artifacts(analysis, seed_idx, seed_dir)
            captured += 1
        except APIError as e:
            logger.warning("Analysis failed for seed %d: %s", seed_idx, e)

    logger.info("Captured ground truth for %d/%d seeds", captured, seeds_count)
    return captured > 0


def _save_analysis_artifacts(
    analysis: dict,
    seed_idx: int,
    seed_dir: Path,
) -> None:
    """Persist ground truth, prediction, and metadata from an analysis response."""
    if "ground_truth" in analysis:
        gt = np.array(analysis["ground_truth"], dtype=np.float64)
        np.save(seed_dir / "ground_truth.npy", gt)
        logger.info(
            "Seed %d: ground truth saved (shape %s), score=%.1f",
            seed_idx,
            gt.shape,
            analysis.get("score") or 0,
        )

    if "prediction" in analysis:
        pred = np.array(analysis["prediction"], dtype=np.float64)
        np.save(seed_dir / "our_prediction.npy", pred)

    meta = {
        "seed_index": seed_idx,
        "score": analysis.get("score"),
        "width": analysis.get("width"),
        "height": analysis.get("height"),
    }
    (seed_dir / "analysis_meta.json").write_text(json.dumps(meta, indent=2))


def _capture_predictions(
    client: AstarClient,
    round_id: str,
    out_dir: Path,
) -> None:
    """Fetch our submitted predictions."""
    try:
        preds = client.my_predictions(round_id)
        if preds:
            (out_dir / "my_predictions.json").write_text(json.dumps(preds, indent=2, default=str))
            logger.info("Saved %d prediction summaries", len(preds))
    except APIError as e:
        logger.info("Could not fetch predictions: %s", e)


def _capture_viewports(
    client: AstarClient,
    round_id: str,
    seeds_count: int,
    width: int,
    height: int,
    states: list,
    out_dir: Path,
) -> None:
    """Use viewport queries to observe final states (active rounds)."""
    logger.info("Starting viewport queries (budget: 50)")
    planner = QueryPlanner(width, height)
    observations: dict[int, list[dict]] = {i: [] for i in range(seeds_count)}

    for seed_idx in range(seeds_count):
        budget_exhausted = _query_seed_viewports(
            client,
            round_id,
            seed_idx,
            states[seed_idx][0],
            planner,
            observations,
        )
        if budget_exhausted:
            break

    _save_observations(observations, out_dir)


def _query_seed_viewports(
    client: AstarClient,
    round_id: str,
    seed_idx: int,
    initial_grid: np.ndarray,
    planner: QueryPlanner,
    observations: dict[int, list[dict]],
) -> bool:
    """Query all viewports for one seed. Returns True if budget exhausted."""
    viewports = planner.plan_initial_queries(seed_idx, initial_grid)
    for vp in viewports:
        try:
            obs = _execute_viewport_query(client, round_id, vp)
            observations[seed_idx].append(obs)
            remaining = client.queries_remaining(round_id)
            logger.info(
                "Seed %d: queried (%d,%d) %dx%d -- %d remaining",
                seed_idx,
                vp.viewport_x,
                vp.viewport_y,
                vp.viewport_w,
                vp.viewport_h,
                remaining,
            )
        except BudgetExhaustedError:
            logger.warning("Budget exhausted at seed %d", seed_idx)
            return True
        except APIError as e:
            logger.warning("Query failed for seed %d: %s", seed_idx, e)
    return False


def _execute_viewport_query(
    client: AstarClient,
    round_id: str,
    vp: Viewport,
) -> dict:
    """Execute a single viewport query and return the observation dict."""
    result = client.query(
        round_id,
        vp.seed_index,
        vp.viewport_x,
        vp.viewport_y,
        vp.viewport_w,
        vp.viewport_h,
    )
    return {
        "seed_index": vp.seed_index,
        "viewport": {
            "x": vp.viewport_x,
            "y": vp.viewport_y,
            "w": vp.viewport_w,
            "h": vp.viewport_h,
        },
        "grid": result.get("grid"),
        "settlements": result.get("settlements"),
    }


def _save_observations(
    observations: dict[int, list[dict]],
    out_dir: Path,
) -> None:
    """Persist collected viewport observations to disk."""
    for seed_idx, obs_list in observations.items():
        if obs_list:
            seed_dir = out_dir / f"seed_{seed_idx}"
            seed_dir.mkdir(exist_ok=True)
            (seed_dir / "observations.json").write_text(
                json.dumps(obs_list, indent=2, default=str),
            )
    total_obs = sum(len(v) for v in observations.values())
    logger.info("Saved %d observations", total_obs)


def _write_summary(
    client: AstarClient,
    round_id: str,
    round_data: dict,
    out_dir: Path,
    has_ground_truth: bool,
) -> None:
    """Write capture summary."""
    summary = {
        "round_id": round_id,
        "round_number": round_data.get("round_number"),
        "status": round_data.get("status"),
        "map_width": round_data.get("map_width"),
        "map_height": round_data.get("map_height"),
        "seeds_count": round_data.get("seeds_count"),
        "has_ground_truth": has_ground_truth,
        "queries_used": client.query_count(round_id),
    }
    (out_dir / "capture_summary.json").write_text(json.dumps(summary, indent=2))

    status = round_data.get("status", "?")
    seeds = round_data.get("seeds_count", "?")
    w = round_data.get("map_width", "?")
    h = round_data.get("map_height", "?")
    print("\nCapture complete!")
    print(f"  Round #{round_data.get('round_number')}: {round_id}")
    print(f"  Status: {status}, {w}x{h}, {seeds} seeds")
    print(f"  Ground truth: {'YES' if has_ground_truth else 'no'}")
    print(f"  Queries used: {client.query_count(round_id)}")
    print(f"  Output: {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture data from a competition round")
    parser.add_argument("--token", required=True, help="JWT auth token")
    parser.add_argument(
        "--round-id",
        help="Round ID to capture (omit to list rounds)",
    )
    args = parser.parse_args()

    client = AstarClient(args.token)

    if not args.round_id:
        list_rounds(client)
        print("Rerun with --round-id <ID> to capture a specific round.")
        sys.exit(0)

    capture_round(client, args.round_id)


if __name__ == "__main__":
    main()
