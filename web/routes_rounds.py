"""Rounds page routes showing real round data and scores.

Loads round metadata from data/rounds/ and displays submission
scores from analysis_meta.json files.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from web.data import get_round_scores, list_rounds

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rounds", tags=["rounds"])
templates = Jinja2Templates(directory="web/templates")

DATA_DIR = "data/rounds"


@router.get("/", response_class=HTMLResponse)
async def rounds_page(request: Request) -> HTMLResponse:
    """Rounds overview with real data from disk."""
    rounds = list_rounds(DATA_DIR)
    for rd in rounds:
        round_dir = Path(DATA_DIR) / rd["dir_name"]
        rd["scores"] = get_round_scores(round_dir)
    return templates.TemplateResponse(
        request,
        "rounds.html",
        context={"rounds": rounds},
    )


@router.get("/{round_number}", response_class=HTMLResponse)
async def round_detail(request: Request, round_number: int) -> HTMLResponse:
    """Detail view for a single round."""
    rounds = list_rounds(DATA_DIR)
    rd = next((r for r in rounds if r["round_number"] == round_number), None)
    scores: list = []
    if rd:
        round_dir = Path(DATA_DIR) / rd["dir_name"]
        scores = get_round_scores(round_dir)
    return templates.TemplateResponse(
        request,
        "round_detail.html",
        context={
            "round": rd,
            "scores": scores,
            "round_number": round_number,
        },
    )
