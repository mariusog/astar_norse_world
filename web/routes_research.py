"""Research page routes for model comparison and backtesting.

Provides the research dashboard with real model leaderboard,
per-round scores, and HTMX-powered backtest triggering.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from web.backtest import get_leaderboard, run_loo_backtest
from web.data import list_rounds
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
        "research.html",
        {
            "request": request,
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
        "partials/backtest_result.html",
        {
            "request": request,
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
        "backtest.html",
        {
            "request": request,
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
        "partials/model_detail.html",
        {
            "request": request,
            "strategy_name": strategy_name,
            "entry": entry,
            "strategies": strategies,
        },
    )
