"""Tests for web.app FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from web.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_home_page(client):
    """GET / returns 200 with HTML."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Norse World" in resp.text


def test_api_leaderboard(client):
    """GET /api/leaderboard returns JSON list."""
    resp = client.get("/api/leaderboard")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_api_backtest_missing_strategy(client):
    """POST /api/backtest with empty body returns 400."""
    resp = client.post("/api/backtest", json={})
    assert resp.status_code == 400


def test_api_backtest_unknown_strategy(client):
    """POST /api/backtest with unknown strategy returns 404."""
    resp = client.post("/api/backtest", json={"strategy_name": "fake"})
    assert resp.status_code == 404


def test_rounds_page(client):
    """GET /rounds/ returns 200."""
    resp = client.get("/rounds/")
    assert resp.status_code == 200
    assert "Rounds" in resp.text


def test_research_page(client):
    """GET /research/ returns 200."""
    resp = client.get("/research/")
    assert resp.status_code == 200
    assert "Research" in resp.text


def test_backtest_page(client):
    """GET /research/backtest returns 200."""
    resp = client.get("/research/backtest")
    assert resp.status_code == 200
    assert "Backtest" in resp.text
