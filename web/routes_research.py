"""Route handlers for the Model Research Dashboard (T302).

Include this router in the main app:
    from web.routes_research import router
    app.include_router(router)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/research", tags=["research"])
templates = Jinja2Templates(directory="web/templates")

# -- Hardcoded model strategies for initial dashboard --

MODELS: list[dict[str, Any]] = [
    {
        "id": "M01",
        "name": "Flat Terrain Priors",
        "description": "Baseline model using uniform priors across all terrain types.",
        "params": {"prior_type": "flat"},
        "round_scores": [62.1, 58.3, 65.0, 60.2, 57.8, 63.4, 59.9],
        "runtime_ms": 120,
        "status": "stable",
    },
    {
        "id": "M02",
        "name": "Distance-Weighted Priors",
        "description": "Weights priors by distance bins from nearest settlement.",
        "params": {"prior_type": "distance", "bins": 5},
        "round_scores": [71.2, 68.5, 74.1, 70.3, 66.9, 72.8, 69.4],
        "runtime_ms": 180,
        "status": "stable",
    },
    {
        "id": "M03",
        "name": "Feature Lookup",
        "description": "Lookup table keyed on (terrain, distance, density).",
        "params": {"features": ["terrain", "distance", "density"]},
        "round_scores": [76.4, 73.1, 78.9, 75.0, 72.3, 77.5, 74.8],
        "runtime_ms": 250,
        "status": "stable",
    },
    {
        "id": "M04",
        "name": "XGBoost Classifier",
        "description": "Gradient-boosted classifier using ml_predictor module.",
        "params": {"n_estimators": 200, "max_depth": 6, "lr": 0.1},
        "round_scores": [82.1, 79.4, 85.3, 81.0, 78.6, 83.7, 80.2],
        "runtime_ms": 1400,
        "status": "best",
    },
    {
        "id": "M05",
        "name": "Regime-Adaptive XGBoost",
        "description": "XGBoost with probe-based regime detection and adaptive weights.",
        "params": {"n_estimators": 200, "max_depth": 6, "regime": True},
        "round_scores": [80.5, 77.8, 83.9, 79.2, 76.4, 82.1, 78.9],
        "runtime_ms": 2100,
        "status": "testing",
    },
    {
        "id": "M06",
        "name": "Survive-Only Priors",
        "description": "Always predicts survive for every settlement.",
        "params": {"strategy": "always_survive"},
        "round_scores": [45.2, 42.1, 48.3, 44.0, 41.5, 46.8, 43.6],
        "runtime_ms": 50,
        "status": "baseline",
    },
    {
        "id": "M07",
        "name": "Collapse-Only Priors",
        "description": "Always predicts collapse for every settlement.",
        "params": {"strategy": "always_collapse"},
        "round_scores": [38.7, 35.4, 41.2, 37.5, 34.8, 40.1, 36.9],
        "runtime_ms": 50,
        "status": "baseline",
    },
]


def _enrich_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add computed fields (avg_score, delta) to each model dict."""
    enriched = []
    for m in models:
        avg = sum(m["round_scores"]) / len(m["round_scores"])
        enriched.append({**m, "avg_score": round(avg, 1)})
    enriched.sort(key=lambda x: x["avg_score"], reverse=True)
    best = enriched[0]["avg_score"]
    for m in enriched:
        m["delta"] = round(m["avg_score"] - best, 1)
    return enriched


@router.get("/", response_class=HTMLResponse)
async def research_page(request: Request) -> HTMLResponse:
    """Render the full research dashboard."""
    enriched = _enrich_models(MODELS)
    return templates.TemplateResponse(
        "research.html",
        {
            "request": request,
            "models": enriched,
            "selected": enriched[0],
        },
    )


@router.get("/model/{model_id}", response_class=HTMLResponse)
async def model_detail(request: Request, model_id: str) -> HTMLResponse:
    """Return the detail panel fragment for a specific model (HTMX)."""
    enriched = _enrich_models(MODELS)
    selected = next((m for m in enriched if m["id"] == model_id), enriched[0])
    return templates.TemplateResponse(
        "components/model_detail.html",
        {"request": request, "selected": selected},
    )


@router.post("/backtest/{model_id}", response_class=HTMLResponse)
async def run_backtest(request: Request, model_id: str) -> HTMLResponse:
    """Stub: trigger a backtest run for a model (HTMX POST)."""
    enriched = _enrich_models(MODELS)
    selected = next((m for m in enriched if m["id"] == model_id), enriched[0])
    return HTMLResponse(
        content=(
            f'<div class="backtest-status running">Backtest queued for {selected["name"]}...</div>'
        )
    )
