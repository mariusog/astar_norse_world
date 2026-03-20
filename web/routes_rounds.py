"""Route handlers for the Round Dashboard and Submission page (T303).

Include this router in the main app:
    from web.routes_rounds import router
    app.include_router(router)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/rounds", tags=["rounds"])
templates = Jinja2Templates(directory="web/templates")

# -- Hardcoded round data for initial dashboard --

ROUNDS: list[dict[str, Any]] = [
    {
        "number": 1,
        "id": "a1b2c3d4",
        "weight": 0.512,
        "status": "completed",
        "our_score": 72.4,
        "rank": 15,
        "seeds_submitted": 5,
        "queries_used": 48,
        "seed_scores": [74.1, 70.2, 73.8, 68.5, 75.4],
        "regime": "growth",
        "gt_distribution": {"survive": 62, "collapse": 18, "expand": 12, "static": 8},
        "initial_settlements": 6,
    },
    {
        "number": 2,
        "id": "e5f6g7h8",
        "weight": 0.734,
        "status": "completed",
        "our_score": 68.9,
        "rank": 22,
        "seeds_submitted": 5,
        "queries_used": 50,
        "seed_scores": [71.0, 66.3, 69.8, 64.2, 73.2],
        "regime": "conflict",
        "gt_distribution": {"survive": 45, "collapse": 30, "expand": 8, "static": 17},
        "initial_settlements": 7,
    },
    {
        "number": 3,
        "id": "i9j0k1l2",
        "weight": 0.891,
        "status": "completed",
        "our_score": 75.1,
        "rank": 11,
        "seeds_submitted": 5,
        "queries_used": 45,
        "seed_scores": [77.3, 73.8, 76.0, 71.5, 76.9],
        "regime": "growth",
        "gt_distribution": {"survive": 58, "collapse": 15, "expand": 18, "static": 9},
        "initial_settlements": 5,
    },
    {
        "number": 4,
        "id": "m3n4o5p6",
        "weight": 1.023,
        "status": "completed",
        "our_score": 81.3,
        "rank": 7,
        "seeds_submitted": 5,
        "queries_used": 50,
        "seed_scores": [83.1, 79.8, 82.4, 78.0, 83.2],
        "regime": "trade",
        "gt_distribution": {"survive": 55, "collapse": 10, "expand": 22, "static": 13},
        "initial_settlements": 4,
    },
    {
        "number": 5,
        "id": "q7r8s9t0",
        "weight": 1.156,
        "status": "completed",
        "our_score": 78.6,
        "rank": 9,
        "seeds_submitted": 5,
        "queries_used": 49,
        "seed_scores": [80.2, 76.9, 79.5, 74.8, 81.6],
        "regime": "growth",
        "gt_distribution": {"survive": 60, "collapse": 14, "expand": 16, "static": 10},
        "initial_settlements": 6,
    },
    {
        "number": 6,
        "id": "u1v2w3x4",
        "weight": 1.289,
        "status": "completed",
        "our_score": 83.2,
        "rank": 5,
        "seeds_submitted": 5,
        "queries_used": 47,
        "seed_scores": [85.0, 81.4, 84.1, 79.6, 85.9],
        "regime": "trade",
        "gt_distribution": {"survive": 52, "collapse": 12, "expand": 24, "static": 12},
        "initial_settlements": 5,
    },
    {
        "number": 7,
        "id": "y5z6a7b8",
        "weight": 1.378,
        "status": "scoring",
        "our_score": None,
        "rank": None,
        "seeds_submitted": 5,
        "queries_used": 50,
        "seed_scores": [],
        "regime": "unknown",
        "gt_distribution": {},
        "initial_settlements": 7,
    },
    {
        "number": 8,
        "id": "c9d0e1f2",
        "weight": 1.478,
        "status": "active",
        "our_score": None,
        "rank": None,
        "seeds_submitted": 0,
        "queries_used": 0,
        "seed_scores": [],
        "regime": "unknown",
        "gt_distribution": {},
        "initial_settlements": 6,
    },
]


@router.get("/", response_class=HTMLResponse)
async def rounds_page(request: Request) -> HTMLResponse:
    """Render the full rounds dashboard."""
    active_round = next((r for r in ROUNDS if r["status"] == "active"), None)
    return templates.TemplateResponse(
        "rounds.html",
        {
            "request": request,
            "rounds": ROUNDS,
            "active_round": active_round,
            "total_query_budget": 50,
        },
    )


@router.get("/detail/{round_number}", response_class=HTMLResponse)
async def round_detail(request: Request, round_number: int) -> HTMLResponse:
    """Return expandable detail panel for a round (HTMX)."""
    rd = next((r for r in ROUNDS if r["number"] == round_number), None)
    if rd is None:
        return HTMLResponse(content="<div>Round not found</div>", status_code=404)
    return templates.TemplateResponse(
        "components/round_detail.html",
        {"request": request, "round": rd},
    )


@router.post("/probe/{round_number}", response_class=HTMLResponse)
async def run_probe(request: Request, round_number: int) -> HTMLResponse:
    """Stub: run a 5-query probe on the active round (HTMX POST)."""
    return HTMLResponse(
        content=(
            '<div class="action-status running">'
            f"Probe started for R{round_number} (5 queries)..."
            "</div>"
        )
    )


@router.post("/submit/{round_number}", response_class=HTMLResponse)
async def full_submit(request: Request, round_number: int) -> HTMLResponse:
    """Stub: submit predictions for all seeds (HTMX POST)."""
    return HTMLResponse(
        content=(
            '<div class="action-status running">'
            f"Full submission started for R{round_number}..."
            "</div>"
        )
    )


@router.post("/dryrun/{round_number}", response_class=HTMLResponse)
async def dry_run(request: Request, round_number: int) -> HTMLResponse:
    """Stub: run local dry-run without using queries (HTMX POST)."""
    return HTMLResponse(
        content=(
            '<div class="action-status info">'
            f"Dry run started for R{round_number} (no queries used)..."
            "</div>"
        )
    )
