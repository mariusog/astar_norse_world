"""FastAPI application for the Astar Norse World dashboard.

Serves the web UI with real model execution, backtesting,
and round data from disk.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from web.backtest import get_leaderboard, run_loo_backtest
from web.models import list_strategies
from web.routes_research import router as research_router
from web.routes_rounds import router as rounds_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Astar Norse World Dashboard")

# Mount routers
app.include_router(research_router)
app.include_router(rounds_router)

templates = Jinja2Templates(directory="web/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Dashboard home page."""
    strategies = list_strategies()
    leaderboard = get_leaderboard()
    return templates.TemplateResponse(
        request,
        "index.html",
        context={
            "strategies": strategies,
            "leaderboard": leaderboard,
        },
    )


@app.post("/api/backtest", response_class=JSONResponse)
async def api_backtest(request: Request) -> JSONResponse:
    """JSON endpoint to trigger a backtest."""
    body = await request.json()
    strategy_name = body.get("strategy_name", "")
    if not strategy_name:
        return JSONResponse(
            {"error": "strategy_name required"},
            status_code=400,
        )
    try:
        result = run_loo_backtest(strategy_name)
        return JSONResponse(
            {
                "strategy_name": result.strategy_name,
                "avg_score": result.avg_score,
                "scores": {str(k): v for k, v in result.scores.items()},
                "timestamp": result.timestamp,
            }
        )
    except KeyError:
        return JSONResponse(
            {"error": f"Unknown strategy: {strategy_name}"},
            status_code=404,
        )


@app.get("/api/leaderboard", response_class=JSONResponse)
async def api_leaderboard() -> JSONResponse:
    """JSON endpoint for model leaderboard."""
    return JSONResponse(get_leaderboard())
