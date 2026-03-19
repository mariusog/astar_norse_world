"""CLI entry point: python -m src.pipeline --token <JWT>."""

from __future__ import annotations

import argparse
import logging
import sys

from src.api_client import AstarClient
from src.constants import DEFAULT_MC_RUNS
from src.pipeline import CompetitionPipeline
from src.pipeline_types import PipelineResult


def main() -> int:
    """Run the competition pipeline from the command line."""
    args = _parse_args()
    _setup_logging(args.verbose)

    client = AstarClient(token=args.token)
    pipeline = CompetitionPipeline(
        client=client,
        num_mc_runs=args.mc_runs,
    )
    result = pipeline.run(round_id=args.round_id)

    _print_results(result)
    return 0 if all(s.submitted for s in result.seed_results) else 1


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Astar Island competition pipeline",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="JWT authentication token",
    )
    parser.add_argument(
        "--round-id",
        default=None,
        help="Specific round ID (default: active)",
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=DEFAULT_MC_RUNS,
        help=f"Monte Carlo simulation runs (default: {DEFAULT_MC_RUNS})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def _setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _print_results(result: PipelineResult) -> None:
    """Print pipeline results summary."""
    logging.getLogger(__name__).info("Pipeline finished: %s", result)


if __name__ == "__main__":
    sys.exit(main())
