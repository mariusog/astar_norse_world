"""Data loading helpers for the web dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"

# InternalTerrain enum values
INTERNAL_TERRAIN_NAMES: dict[int, str] = {
    0: "Ocean",
    1: "Plains",
    2: "Settlement",
    3: "Port",
    4: "Ruin",
    5: "Forest",
    6: "Mountain",
}

# Prediction class names (6 classes)
PRED_CLASS_NAMES: dict[int, str] = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
}

TERRAIN_COLORS: dict[int, str] = {
    0: "#1a3a5c",  # Ocean
    1: "#2d5a1e",  # Plains
    2: "#d4a017",  # Settlement
    3: "#4a90d9",  # Port
    4: "#8b4513",  # Ruin
    5: "#0d7a0d",  # Forest
    6: "#696969",  # Mountain
}

# Colors for prediction classes (GT argmax view)
PRED_COLORS: dict[int, str] = {
    0: "#2d5a1e",  # Empty (green)
    1: "#d4a017",  # Settlement (gold)
    2: "#4a90d9",  # Port (blue)
    3: "#8b4513",  # Ruin (brown)
    4: "#0d7a0d",  # Forest (green)
    5: "#696969",  # Mountain (gray)
}


def list_rounds() -> list[dict[str, Any]]:
    """List all rounds with metadata."""
    rounds = []
    if not DATA_DIR.exists():
        return rounds
    for rdir in sorted(DATA_DIR.iterdir()):
        meta_path = rdir / "round.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        seed_dirs = [d for d in rdir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
        rounds.append(
            {
                "id": meta.get("id", rdir.name),
                "round_number": meta.get("round_number"),
                "status": meta.get("status", "unknown"),
                "round_weight": meta.get("round_weight"),
                "seeds_count": len(seed_dirs),
                "event_date": meta.get("event_date"),
            }
        )
    rounds.sort(key=lambda r: r.get("round_number", 0))
    return rounds


def get_round_detail(round_id: str) -> dict[str, Any] | None:
    """Get detailed round info including per-seed GT class distributions."""
    rdir = DATA_DIR / round_id
    meta_path = rdir / "round.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    seeds_info = []
    for i in range(meta.get("seeds_count", 5)):
        sdir = rdir / f"seed_{i}"
        gt_path = sdir / "ground_truth.npy"
        seed_data: dict[str, Any] = {"seed_idx": i, "has_gt": gt_path.exists()}
        if gt_path.exists():
            gt = np.load(gt_path)
            argmax = np.argmax(gt, axis=2)
            unique, counts = np.unique(argmax, return_counts=True)
            seed_data["class_distribution"] = {
                PRED_CLASS_NAMES.get(int(u), str(u)): int(c)
                for u, c in zip(unique, counts, strict=True)
            }
        seeds_info.append(seed_data)
    return {**meta, "seeds": seeds_info}


def load_initial_grid(round_id: str, seed_idx: int) -> np.ndarray | None:
    """Load initial grid as (40,40) InternalTerrain array."""
    path = DATA_DIR / round_id / f"seed_{seed_idx}" / "initial_grid.npy"
    if not path.exists():
        return None
    return np.load(path)


def load_ground_truth(round_id: str, seed_idx: int) -> np.ndarray | None:
    """Load ground truth as (40,40,6) probability tensor."""
    path = DATA_DIR / round_id / f"seed_{seed_idx}" / "ground_truth.npy"
    if not path.exists():
        return None
    return np.load(path)


def grid_to_colored_cells(
    grid: np.ndarray,
    color_map: dict[int, str],
) -> list[list[dict[str, Any]]]:
    """Convert grid to list of rows with color and label info."""
    rows = []
    h, w = grid.shape
    for r in range(h):
        row = []
        for c in range(w):
            val = int(grid[r, c])
            row.append(
                {
                    "val": val,
                    "color": color_map.get(val, "#333333"),
                }
            )
        rows.append(row)
    return rows


def entropy_heatmap(gt: np.ndarray) -> list[list[dict[str, Any]]]:
    """Convert GT probabilities to entropy heatmap cells."""
    h, w, _ = gt.shape
    # Compute per-cell entropy
    eps = 1e-10
    ent = -np.sum(gt * np.log(gt + eps), axis=2)
    max_ent = np.log(gt.shape[2])  # max possible entropy
    rows = []
    for r in range(h):
        row = []
        for c in range(w):
            norm = float(ent[r, c] / max_ent) if max_ent > 0 else 0.0
            # Blue (low entropy) to red (high entropy)
            red = int(255 * norm)
            blue = int(255 * (1 - norm))
            color = f"rgb({red},0,{blue})"
            row.append({"val": round(float(ent[r, c]), 3), "color": color})
        rows.append(row)
    return rows
