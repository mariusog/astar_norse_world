"""FastAPI web dashboard for Astar Norse World competition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.data import (
    INTERNAL_TERRAIN_NAMES,
    PRED_CLASS_NAMES,
    PRED_COLORS,
    TERRAIN_COLORS,
    entropy_heatmap,
    get_round_detail,
    grid_to_colored_cells,
    list_rounds,
    load_ground_truth,
    load_initial_grid,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Norse World Dashboard")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Dashboard home with round summary table."""
    rounds = list_rounds()
    return templates.TemplateResponse("index.html", {"request": request, "rounds": rounds})


@app.get("/explorer/{round_id}", response_class=HTMLResponse)
async def explorer_page(request: Request, round_id: str) -> HTMLResponse:
    """Map explorer for a specific round."""
    detail = get_round_detail(round_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Round not found")
    all_rounds = list_rounds()
    return templates.TemplateResponse(
        "explorer.html",
        {
            "request": request,
            "round": detail,
            "round_id": round_id,
            "all_rounds": all_rounds,
            "terrain_names": INTERNAL_TERRAIN_NAMES,
            "terrain_colors": TERRAIN_COLORS,
            "pred_names": PRED_CLASS_NAMES,
            "pred_colors": PRED_COLORS,
        },
    )


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request) -> HTMLResponse:
    """Backtest page placeholder."""
    return templates.TemplateResponse(
        "placeholder.html",
        {
            "request": request,
            "title": "Backtest",
            "message": "Backtest coming soon.",
        },
    )


@app.get("/research", response_class=HTMLResponse)
async def research_page(request: Request) -> HTMLResponse:
    """Model research page placeholder."""
    return templates.TemplateResponse(
        "placeholder.html",
        {
            "request": request,
            "title": "Research",
            "message": "Research coming soon.",
        },
    )


# ---------------------------------------------------------------------------
# HTMX partials
# ---------------------------------------------------------------------------


@app.get("/htmx/grid/{round_id}/{seed_idx}/{view}", response_class=HTMLResponse)
async def htmx_grid(
    request: Request,
    round_id: str,
    seed_idx: int,
    view: str = "initial",
) -> HTMLResponse:
    """Return grid HTML partial for HTMX swap."""
    cells, legend, counts = _build_grid_data(round_id, seed_idx, view)
    if cells is None:
        return HTMLResponse("<p class='error'>Data not found</p>", status_code=404)
    return templates.TemplateResponse(
        "partials/grid.html",
        {
            "request": request,
            "cells": cells,
            "legend": legend,
            "counts": counts,
            "round_id": round_id,
            "seed_idx": seed_idx,
            "view": view,
        },
    )


@app.get("/htmx/cell/{round_id}/{seed_idx}/{row}/{col}", response_class=HTMLResponse)
async def htmx_cell_detail(
    request: Request,
    round_id: str,
    seed_idx: int,
    row: int,
    col: int,
) -> HTMLResponse:
    """Return GT probability distribution for a single cell."""
    gt = load_ground_truth(round_id, seed_idx)
    grid = load_initial_grid(round_id, seed_idx)
    if gt is None or grid is None:
        return HTMLResponse("<p class='error'>Data not found</p>", status_code=404)
    probs = gt[row, col]
    terrain_val = int(grid[row, col])
    dist = [
        {"name": PRED_CLASS_NAMES[i], "prob": round(float(probs[i]), 4)} for i in range(len(probs))
    ]
    return templates.TemplateResponse(
        "partials/cell_detail.html",
        {
            "request": request,
            "row": row,
            "col": col,
            "terrain": INTERNAL_TERRAIN_NAMES.get(terrain_val, "Unknown"),
            "distribution": dist,
        },
    )


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/rounds")
async def api_rounds() -> list[dict[str, Any]]:
    """List all rounds with metadata."""
    return list_rounds()


@app.get("/api/rounds/{round_id}")
async def api_round_detail(round_id: str) -> dict[str, Any]:
    """Round detail with per-seed GT class distributions."""
    detail = get_round_detail(round_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Round not found")
    return detail


@app.get("/api/rounds/{round_id}/seed/{seed_idx}/grid")
async def api_seed_grid(round_id: str, seed_idx: int) -> JSONResponse:
    """Return initial grid as 2D array."""
    grid = load_initial_grid(round_id, seed_idx)
    if grid is None:
        raise HTTPException(status_code=404, detail="Grid not found")
    return JSONResponse(content={"grid": grid.tolist()})


@app.get("/api/rounds/{round_id}/seed/{seed_idx}/gt")
async def api_seed_gt(round_id: str, seed_idx: int) -> JSONResponse:
    """Return GT argmax grid and entropy grid."""
    gt = load_ground_truth(round_id, seed_idx)
    if gt is None:
        raise HTTPException(status_code=404, detail="GT not found")
    argmax = np.argmax(gt, axis=2)
    eps = 1e-10
    ent = -np.sum(gt * np.log(gt + eps), axis=2)
    return JSONResponse(
        content={
            "argmax": argmax.tolist(),
            "entropy": np.round(ent, 4).tolist(),
        }
    )


@app.get("/api/scores")
async def api_scores() -> dict[str, Any]:
    """Submission history placeholder."""
    return {"message": "Score tracking not yet implemented", "scores": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_grid_data(
    round_id: str,
    seed_idx: int,
    view: str,
) -> tuple[
    list[list[dict[str, Any]]] | None,
    list[dict[str, str]] | None,
    dict[str, int] | None,
]:
    """Build grid cells, legend, and counts for a given view."""
    if view == "initial":
        grid = load_initial_grid(round_id, seed_idx)
        if grid is None:
            return None, None, None
        cells = grid_to_colored_cells(grid, TERRAIN_COLORS)
        legend = [
            {"name": n, "color": TERRAIN_COLORS[v]} for v, n in INTERNAL_TERRAIN_NAMES.items()
        ]
        unique, cnt = np.unique(grid, return_counts=True)
        counts = {
            INTERNAL_TERRAIN_NAMES.get(int(u), str(u)): int(c)
            for u, c in zip(unique, cnt, strict=True)
        }
        return cells, legend, counts

    gt = load_ground_truth(round_id, seed_idx)
    if gt is None:
        return None, None, None

    if view == "gt_argmax":
        argmax = np.argmax(gt, axis=2)
        cells = grid_to_colored_cells(argmax, PRED_COLORS)
        legend = [{"name": n, "color": PRED_COLORS[v]} for v, n in PRED_CLASS_NAMES.items()]
        unique, cnt = np.unique(argmax, return_counts=True)
        counts = {
            PRED_CLASS_NAMES.get(int(u), str(u)): int(c) for u, c in zip(unique, cnt, strict=True)
        }
        return cells, legend, counts

    if view == "entropy":
        cells = entropy_heatmap(gt)
        legend = [
            {"name": "Low entropy", "color": "rgb(0,0,255)"},
            {"name": "High entropy", "color": "rgb(255,0,0)"},
        ]
        return cells, legend, {}

    return None, None, None
