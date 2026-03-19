"""Tests for AstarClient -- API client for competition server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.api_client import (
    APIError,
    AstarClient,
    AuthError,
    BudgetExhaustedError,
    _apply_probability_floor,
    _validate_viewport,
)
from src.constants import (
    TOTAL_QUERY_BUDGET,
    VIEWPORT_MAX_SIZE,
    VIEWPORT_MIN_SIZE,
)

DUMMY_TOKEN = "test-jwt-token"  # noqa: S105


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> AstarClient:
    """Client with a dummy token."""
    return AstarClient(token=DUMMY_TOKEN, base_url="https://test.api")


@pytest.fixture
def mock_response() -> MagicMock:
    """Factory for mock HTTP responses."""

    def _make(status_code: int = 200, json_data: object = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = ""
        return resp

    return _make


# ---------------------------------------------------------------------------
# Auth setup
# ---------------------------------------------------------------------------


def test_init_sets_cookie_and_bearer() -> None:
    """Constructor sets both auth cookie and bearer header."""
    tok = "abc123"
    c = AstarClient(token=tok)
    assert c._session.cookies.get("access_token") == tok
    assert c._session.headers["Authorization"] == f"Bearer {tok}"


# ---------------------------------------------------------------------------
# list_rounds
# ---------------------------------------------------------------------------


def test_list_rounds_returns_list(client: AstarClient, mock_response: MagicMock) -> None:
    """list_rounds returns the JSON list from server."""
    rounds = [{"id": "r1", "status": "active"}]
    with patch.object(client._session, "request", return_value=mock_response(200, rounds)):
        result = client.list_rounds()
    assert result == rounds


# ---------------------------------------------------------------------------
# get_round
# ---------------------------------------------------------------------------


def test_get_round_returns_detail(client: AstarClient, mock_response: MagicMock) -> None:
    """get_round returns round details dict."""
    detail = {"id": "r1", "map_width": 40, "map_height": 40}
    with patch.object(client._session, "request", return_value=mock_response(200, detail)):
        result = client.get_round("r1")
    assert result["map_width"] == 40


# ---------------------------------------------------------------------------
# get_active_round
# ---------------------------------------------------------------------------


def test_get_active_round_finds_active(client: AstarClient, mock_response: MagicMock) -> None:
    """get_active_round returns the active round."""
    rounds = [
        {"id": "r1", "status": "closed"},
        {"id": "r2", "status": "active"},
    ]
    with patch.object(client._session, "request", return_value=mock_response(200, rounds)):
        result = client.get_active_round()
    assert result is not None
    assert result["id"] == "r2"


def test_get_active_round_returns_none_when_no_active(
    client: AstarClient, mock_response: MagicMock
) -> None:
    """get_active_round returns None when no active round."""
    rounds = [{"id": "r1", "status": "closed"}]
    with patch.object(client._session, "request", return_value=mock_response(200, rounds)):
        result = client.get_active_round()
    assert result is None


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


def test_query_sends_correct_body(client: AstarClient, mock_response: MagicMock) -> None:
    """query sends the right POST body and tracks budget."""
    resp_data = {"grid": [[0]], "settlements": []}
    with patch.object(
        client._session, "request", return_value=mock_response(200, resp_data)
    ) as mock_req:
        result = client.query("r1", 0, 10, 5, 15, 15)

    assert result == resp_data
    assert client.query_count("r1") == 1
    call_kwargs = mock_req.call_args
    assert call_kwargs[1]["json"]["viewport_x"] == 10


def test_query_budget_exhausted_raises(client: AstarClient) -> None:
    """query raises BudgetExhaustedError when budget is used up."""
    client._query_counts["r1"] = TOTAL_QUERY_BUDGET
    with pytest.raises(BudgetExhaustedError):
        client.query("r1", 0, 0, 0, 10, 10)


def test_query_invalid_viewport_raises(client: AstarClient) -> None:
    """query raises ValueError for out-of-range viewport."""
    with pytest.raises(ValueError, match="viewport_w"):
        client.query("r1", 0, 0, 0, 3, 10)
    with pytest.raises(ValueError, match="viewport_h"):
        client.query("r1", 0, 0, 0, 10, 20)


def test_queries_remaining(client: AstarClient) -> None:
    """queries_remaining reports correctly."""
    assert client.queries_remaining("r1") == TOTAL_QUERY_BUDGET
    client._query_counts["r1"] = 10
    assert client.queries_remaining("r1") == TOTAL_QUERY_BUDGET - 10


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_applies_floor_and_sends(client: AstarClient, mock_response: MagicMock) -> None:
    """submit applies probability floor before sending."""
    pred = np.zeros((2, 2, 6))
    pred[:, :, 0] = 1.0  # all mass on class 0
    resp_data = {"score": 50.0}
    with patch.object(
        client._session, "request", return_value=mock_response(200, resp_data)
    ) as mock_req:
        result = client.submit("r1", 0, pred)

    assert result["score"] == 50.0
    sent_pred = mock_req.call_args[1]["json"]["prediction"]
    # Check that no probability is exactly zero (floor was applied)
    for row in sent_pred:
        for cell in row:
            for p in cell:
                assert p > 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_auth_error_on_401(client: AstarClient, mock_response: MagicMock) -> None:
    """401 raises AuthError."""
    with (
        patch.object(client._session, "request", return_value=mock_response(401)),
        pytest.raises(AuthError),
    ):
        client.list_rounds()


def test_auth_error_on_403(client: AstarClient, mock_response: MagicMock) -> None:
    """403 raises AuthError."""
    with (
        patch.object(client._session, "request", return_value=mock_response(403)),
        pytest.raises(AuthError),
    ):
        client.list_rounds()


def test_api_error_on_500(client: AstarClient, mock_response: MagicMock) -> None:
    """500 raises APIError."""
    with (
        patch.object(client._session, "request", return_value=mock_response(500)),
        pytest.raises(APIError),
    ):
        client.list_rounds()


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


def test_retry_on_connection_error(client: AstarClient, mock_response: MagicMock) -> None:
    """Transient errors trigger retry; succeeds on second attempt."""
    import requests as req

    good_resp = mock_response(200, {"ok": True})
    with (
        patch.object(
            client._session,
            "request",
            side_effect=[req.ConnectionError("down"), good_resp],
        ),
        patch("src.api_client.time.sleep"),
    ):
        result = client.list_rounds()
    assert result == {"ok": True}


def test_retry_exhausted_raises(client: AstarClient) -> None:
    """All retries exhausted raises APIError."""
    import requests as req

    with (
        patch.object(
            client._session,
            "request",
            side_effect=req.ConnectionError("down"),
        ),
        patch("src.api_client.time.sleep"),
        pytest.raises(APIError, match="retries"),
    ):
        client.list_rounds()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_validate_viewport_valid() -> None:
    """Valid viewport dimensions pass without error."""
    _validate_viewport(VIEWPORT_MIN_SIZE, VIEWPORT_MAX_SIZE)
    _validate_viewport(10, 10)


def test_validate_viewport_too_small() -> None:
    """Below minimum raises ValueError."""
    with pytest.raises(ValueError):
        _validate_viewport(4, 10)


def test_validate_viewport_too_large() -> None:
    """Above maximum raises ValueError."""
    with pytest.raises(ValueError):
        _validate_viewport(10, 16)


def test_apply_probability_floor() -> None:
    """Floor clamps zeros and renormalizes to sum to 1."""
    pred = np.zeros((2, 2, 6))
    pred[:, :, 0] = 1.0
    result = _apply_probability_floor(pred)
    # No value is exactly zero
    assert np.all(result > 0.0)
    # Sums to 1
    sums = result.sum(axis=2)
    np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    # Original dominant class retains highest probability
    assert np.all(result[:, :, 0] > result[:, :, 1])
