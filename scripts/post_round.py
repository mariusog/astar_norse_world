"""Post-round automation: capture data, run backtest, print summary.

CLI: python -m scripts.post_round --token <JWT>
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def capture_rounds(token: str) -> list[Any]:
    """Fetch new completed rounds from server."""
    from src.api_client import AstarClient
    from src.round_collector import collect_all_rounds

    return collect_all_rounds(AstarClient(token=token), str(DATA_DIR))


def git_commit_new_data() -> bool:
    """Stage and commit any new round data files."""
    rounds_dir = DATA_DIR / "rounds"
    if not rounds_dir.exists():
        return False
    result = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", str(rounds_dir)],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if not result.stdout.strip():
        logger.info("No new data files to commit")
        return False
    subprocess.run(  # noqa: S603
        ["git", "add", str(rounds_dir)],  # noqa: S607
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Capture new round data from server"],  # noqa: S607
        cwd=str(PROJECT_ROOT),
        check=True,
    )
    logger.info("Committed new round data")
    return True


def run_backtest() -> dict[str, Any] | None:
    """Run LOO backtest and return results summary."""
    from scripts.backtest import run_backtest as _run_backtest

    results = _run_backtest(loo=True, obs_per_cell=3, seed=42)
    if not results:
        return None
    import numpy as np

    scores = [r["avg_score"] for r in results]
    return {
        "num_rounds": len(results),
        "avg_score": float(np.mean(scores)),
        "per_round": [{"round": r["round_number"], "score": r["avg_score"]} for r in results],
    }


def print_summary(collected: list[Any], bt: dict[str, Any] | None) -> None:
    """Log concise post-round summary."""
    logger.info("=== Post-Round Summary ===")
    logger.info("Rounds captured: %d", len(collected))
    if collected:
        logger.info("  New round IDs: %s", collected)
    if bt:
        logger.info("Backtest (LOO, 3 obs/cell): avg=%.1f/100", bt["avg_score"])
        for pr in bt["per_round"]:
            logger.info("  R%s: %.1f", pr["round"], pr["score"])
    else:
        logger.info("Backtest: no results (insufficient data)")
    logger.info("===========================")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Post-round automation")
    parser.add_argument("--token", default=os.environ.get("ASTAR_TOKEN", ""))
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--skip-commit", action="store_true")
    args = parser.parse_args()

    collected: list[Any] = []
    if not args.skip_capture:
        if not args.token:
            parser.error("Token required via --token or ASTAR_TOKEN env var")
        logger.info("Step 1: Capturing rounds...")
        collected = capture_rounds(args.token)
    else:
        logger.info("Step 1: Skipping capture (offline mode)")

    if not args.skip_commit and collected:
        logger.info("Step 2: Committing new data...")
        git_commit_new_data()

    logger.info("Step 3: Running LOO backtest...")
    bt = run_backtest()
    print_summary(collected, bt)


if __name__ == "__main__":
    main()
