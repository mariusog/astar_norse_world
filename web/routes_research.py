"""Research page routes for model comparison and backtesting.

Provides the research dashboard with real model leaderboard,
per-round scores, HTMX-powered backtest triggering, and
automated model search endpoints.
"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from web.backtest import get_leaderboard, run_loo_backtest
from web.data import list_rounds
from web.model_search import get_search_progress, run_search
from web.models import list_strategies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])
templates = Jinja2Templates(directory="web/templates")


@router.get("/", response_class=HTMLResponse)
async def research_page(request: Request) -> HTMLResponse:
    """Research dashboard with model leaderboard."""
    strategies = list_strategies()
    leaderboard = get_leaderboard()
    return templates.TemplateResponse(
        request,
        "research.html",
        context={
            "strategies": strategies,
            "leaderboard": leaderboard,
        },
    )


@router.post("/backtest/{strategy_name}", response_class=HTMLResponse)
async def run_backtest(request: Request, strategy_name: str) -> HTMLResponse:
    """Trigger LOO backtest for a strategy, return HTMX partial."""
    logger.info("Starting LOO backtest for %s", strategy_name)
    result = run_loo_backtest(strategy_name)
    leaderboard = get_leaderboard()
    return templates.TemplateResponse(
        request,
        "partials/backtest_result.html",
        context={
            "result": result,
            "leaderboard": leaderboard,
        },
    )


@router.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request) -> HTMLResponse:
    """Dedicated backtest page with comparison view."""
    strategies = list_strategies()
    leaderboard = get_leaderboard()
    rounds = list_rounds()
    round_numbers = sorted(r["round_number"] for r in rounds)
    return templates.TemplateResponse(
        request,
        "backtest.html",
        context={
            "strategies": strategies,
            "leaderboard": leaderboard,
            "round_numbers": round_numbers,
        },
    )


@router.get("/model/{strategy_name}", response_class=HTMLResponse)
async def model_detail(request: Request, strategy_name: str) -> HTMLResponse:
    """Show per-round scores for a specific strategy."""
    leaderboard = get_leaderboard()
    entry = next((e for e in leaderboard if e["strategy"] == strategy_name), None)
    strategies = list_strategies()
    return templates.TemplateResponse(
        request,
        "partials/model_detail.html",
        context={
            "strategy_name": strategy_name,
            "entry": entry,
            "strategies": strategies,
        },
    )


# ---------------------------------------------------------------------------
# Model search API endpoints
# ---------------------------------------------------------------------------


@router.post("/api/search", response_class=JSONResponse)
async def api_start_search() -> JSONResponse:
    """Launch the full model search in a background thread.

    Trains XGBoost once, then applies all parametric transforms.
    Progress available via GET /research/api/search/status.
    """
    progress = get_search_progress()
    if progress.running:
        return JSONResponse(
            {"error": "Search already running", "progress": _progress_dict(progress)},
            status_code=409,
        )

    def _run_in_background() -> None:
        try:
            run_search()
        except Exception:
            logger.exception("Model search failed")

    thread = threading.Thread(target=_run_in_background, daemon=True)
    thread.start()
    logger.info("Model search started in background thread")
    return JSONResponse({"status": "started"})


@router.get("/api/search/status", response_class=JSONResponse)
async def api_search_status() -> JSONResponse:
    """Return current model search progress."""
    progress = get_search_progress()
    return JSONResponse(_progress_dict(progress))


def _progress_dict(progress: object) -> dict:
    """Convert SearchProgress to a JSON-serializable dict."""
    return {
        "total": progress.total,
        "completed": progress.completed,
        "running": progress.running,
        "error": progress.error,
        "results": progress.results,
    }
