"""Data loading helpers for the web dashboard.

Reads round metadata, scores, and submission data from data/rounds/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "data/rounds"


def list_rounds(data_dir: str = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    """List all rounds with metadata from disk.

    Returns list of dicts with round_number, id, status, event_date,
    map dimensions, and seed count.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    rounds = []
    for round_dir in sorted(data_path.iterdir()):
        if not round_dir.is_dir():
            continue
        meta = _load_round_meta(round_dir)
        if meta:
            rounds.append(meta)
    return sorted(rounds, key=lambda r: r.get("round_number", 0))


def get_round(
    round_number: int,
    data_dir: str = DEFAULT_DATA_DIR,
) -> dict[str, Any] | None:
    """Get full round data by round number."""
    for rd in Path(data_dir).iterdir():
        if not rd.is_dir():
            continue
        rjson = rd / "round.json"
        if not rjson.exists():
            continue
        with open(rjson) as f:
            data = json.load(f)
        if data.get("round_number") == round_number:
            return data
    return None


def get_round_scores(
    round_dir: Path,
) -> list[dict[str, Any]]:
    """Load per-seed analysis/score data from a round directory."""
    scores = []
    for seed_idx in range(10):
        meta_path = round_dir / f"seed_{seed_idx}" / "analysis_meta.json"
        if not meta_path.exists():
            break
        with open(meta_path) as f:
            meta = json.load(f)
        meta["seed_index"] = seed_idx
        scores.append(meta)
    return scores


def _load_round_meta(round_dir: Path) -> dict[str, Any] | None:
    """Load round metadata from round.json."""
    rjson = round_dir / "round.json"
    if not rjson.exists():
        return None
    with open(rjson) as f:
        data = json.load(f)
    seed_count = len(data.get("initial_states", []))
    return {
        "round_number": data.get("round_number", 0),
        "id": data.get("id", round_dir.name),
        "status": data.get("status", "unknown"),
        "event_date": data.get("event_date", ""),
        "map_width": data.get("map_width", 0),
        "map_height": data.get("map_height", 0),
        "seed_count": seed_count,
        "dir_name": round_dir.name,
    }
