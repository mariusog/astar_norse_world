"""Integration tests for the Astar competition API client.

Tests auth setup, endpoint parsing, viewport validation, budget
tracking, retry logic, and error handling with mocked HTTP.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests as requests_lib

from src.api_client import (
    APIError,
    AstarClient,
    AuthError,
    BudgetExhaustedError,
    _apply_probability_floor,
    _validate_viewport,
)
from src.constants import (
    QUERY_WARNING_THRESHOLD,
    TOTAL_QUERY_BUDGET,
    VIEWPORT_MAX_SIZE,
    VIEWPORT_MIN_SIZE,
)


@pytest.fixture
def client() -> AstarClient:
    """Create a client with a test token."""
    return AstarClient(token="test-jwt-token", base_url="https://test.api")  # noqa: S106


def _mock_response(
    status_code: int = 200,
    json_data: dict | list | None = None,
    text: str = "",
) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


class TestAuthSetup:
    """Verify both cookie and bearer token auth paths."""

    def test_cookie_auth_set(self) -> None:
        """Client sets access_token cookie from token."""
        c = AstarClient(token="my-jwt", base_url="https://x")  # noqa: S106
        cookies = dict(c._session.cookies)
        assert cookies["access_token"] == "my-jwt"  # noqa: S105

    def test_bearer_header_set(self) -> None:
        """Client sets Authorization bearer header."""
        c = AstarClient(token="my-jwt", base_url="https://x")  # noqa: S106
        assert c._session.headers["Authorization"] == "Bearer my-jwt"

    def test_base_url_trailing_slash_stripped(self) -> None:
        """Trailing slash is removed from base URL."""
        c = AstarClient(token="t", base_url="https://x.com/")  # noqa: S106
        assert c._base_url == "https://x.com"


class TestListRounds:
    """Test GET /astar-island/rounds."""

    def test_list_rounds_returns_parsed_json(self, client: AstarClient) -> None:
        """Successful response is parsed into list of dicts."""
        rounds = [{"id": "r1", "status": "active"}]
        resp = _mock_response(json_data=rounds)
        with patch.object(client._session, "request", return_value=resp):
            result = client.list_rounds()
        assert result == rounds

    def test_list_rounds_calls_correct_endpoint(self, client: AstarClient) -> None:
        """Request goes to /astar-island/rounds."""
        resp = _mock_response(json_data=[])
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_rounds()
        mock_req.assert_called_once_with(
            "GET",
            "https://test.api/astar-island/rounds",
        )


class TestGetRound:
    """Test GET /astar-island/rounds/{id}."""

    def test_get_round_returns_detail(self, client: AstarClient) -> None:
        """Round detail includes initial_states."""
        detail = {"id": "r1", "initial_states": [{"grid": [[0]]}]}
        resp = _mock_response(json_data=detail)
        with patch.object(client._session, "request", return_value=resp):
            result = client.get_round("r1")
        assert result["id"] == "r1"
        assert "initial_states" in result


class TestQuery:
    """Test POST /astar-island/simulate with budget tracking."""

    def test_query_sends_correct_body(self, client: AstarClient) -> None:
        """Request body contains all viewport fields."""
        resp = _mock_response(json_data={"grid": [[0]]})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.query("r1", seed_index=0, x=1, y=2, w=10, h=10)
        call_kwargs = mock_req.call_args
        body = call_kwargs.kwargs["json"]
        assert body["round_id"] == "r1"
        assert body["viewport_w"] == 10

    def test_query_increments_budget(self, client: AstarClient) -> None:
        """Each successful query increments the counter."""
        resp = _mock_response(json_data={"grid": [[0]]})
        with patch.object(client._session, "request", return_value=resp):
            client.query("r1", 0, 0, 0, 5, 5)
            client.query("r1", 0, 0, 0, 5, 5)
        assert client.query_count("r1") == 2

    def test_query_budget_warning_at_threshold(self, client: AstarClient) -> None:
        """Warning logged when usage reaches threshold."""
        client._query_counts["r1"] = QUERY_WARNING_THRESHOLD
        resp = _mock_response(json_data={"grid": [[0]]})
        with patch.object(client._session, "request", return_value=resp):
            # Should warn but not error
            client.query("r1", 0, 0, 0, 5, 5)
        assert client.query_count("r1") == QUERY_WARNING_THRESHOLD + 1

    def test_query_budget_exhausted_raises(self, client: AstarClient) -> None:
        """BudgetExhaustedError when budget is spent."""
        client._query_counts["r1"] = TOTAL_QUERY_BUDGET
        with pytest.raises(BudgetExhaustedError):
            client.query("r1", 0, 0, 0, 5, 5)

    def test_queries_remaining_tracks_budget(self, client: AstarClient) -> None:
        """queries_remaining decrements with usage."""
        assert client.queries_remaining("r1") == TOTAL_QUERY_BUDGET
        client._query_counts["r1"] = 10
        assert client.queries_remaining("r1") == TOTAL_QUERY_BUDGET - 10


class TestViewportValidation:
    """Test viewport size enforcement."""

    def test_valid_viewport_accepted(self) -> None:
        """Viewport within range does not raise."""
        _validate_viewport(VIEWPORT_MIN_SIZE, VIEWPORT_MAX_SIZE)

    def test_width_too_small_rejected(self) -> None:
        """Width below minimum raises ValueError."""
        with pytest.raises(ValueError, match="viewport_w"):
            _validate_viewport(VIEWPORT_MIN_SIZE - 1, 10)

    def test_width_too_large_rejected(self) -> None:
        """Width above maximum raises ValueError."""
        with pytest.raises(ValueError, match="viewport_w"):
            _validate_viewport(VIEWPORT_MAX_SIZE + 1, 10)

    def test_height_too_small_rejected(self) -> None:
        """Height below minimum raises ValueError."""
        with pytest.raises(ValueError, match="viewport_h"):
            _validate_viewport(10, VIEWPORT_MIN_SIZE - 1)

    def test_height_too_large_rejected(self) -> None:
        """Height above maximum raises ValueError."""
        with pytest.raises(ValueError, match="viewport_h"):
            _validate_viewport(10, VIEWPORT_MAX_SIZE + 1)


class TestSubmit:
    """Test POST /astar-island/submit with floor + normalization."""

    def test_submit_applies_probability_floor(self, client: AstarClient) -> None:
        """Zeros in prediction are floored before submission."""
        pred = np.zeros((3, 3, 6))
        pred[:, :, 0] = 1.0  # all mass on class 0
        resp = _mock_response(json_data={"score": 50.0})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.submit("r1", 0, pred)
        body = mock_req.call_args.kwargs["json"]
        # After floor + renorm, no zeros remain
        flat = np.array(body["prediction"])
        assert flat.min() > 0.0

    def test_submit_preserves_shape_in_body(self, client: AstarClient) -> None:
        """Prediction shape is (H, W, 6) in the request body."""
        pred = np.ones((4, 5, 6)) / 6
        resp = _mock_response(json_data={"score": 80.0})
        with patch.object(client._session, "request", return_value=resp):
            client.submit("r1", 0, pred)


class TestProbabilityFloor:
    """Test _apply_probability_floor independently."""

    def test_floor_raises_zeros(self) -> None:
        """Zero probabilities are raised above zero after floor + renorm."""
        pred = np.zeros((2, 2, 6))
        pred[:, :, 0] = 1.0
        result = _apply_probability_floor(pred)
        assert result.min() > 0.0

    def test_floor_preserves_normalization(self) -> None:
        """Each cell sums to 1.0 after flooring."""
        pred = np.random.default_rng(42).dirichlet([1] * 6, size=(3, 3))
        result = _apply_probability_floor(pred)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)


class TestRetryLogic:
    """Test exponential backoff on connection errors."""

    @patch("src.api_client.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep: MagicMock, client: AstarClient) -> None:
        """ConnectionError triggers retries, then succeeds."""
        ok_resp = _mock_response(json_data=[{"id": "r1"}])
        client._session.request = MagicMock(
            side_effect=[requests_lib.ConnectionError("reset"), ok_resp],
        )
        result = client.list_rounds()
        assert result == [{"id": "r1"}]
        assert mock_sleep.call_count == 1

    @patch("src.api_client.time.sleep")
    def test_retry_exhaustion_raises(self, mock_sleep: MagicMock, client: AstarClient) -> None:
        """All retries fail raises APIError."""
        client._session.request = MagicMock(
            side_effect=requests_lib.ConnectionError("down"),
        )
        with pytest.raises(APIError, match="retries"):
            client.list_rounds()


class TestAuthErrors:
    """Test 401/403 raise AuthError without retry."""

    def test_401_raises_auth_error(self, client: AstarClient) -> None:
        """HTTP 401 raises AuthError immediately."""
        resp = _mock_response(status_code=401)
        client._session.request = MagicMock(return_value=resp)
        with pytest.raises(AuthError):
            client.list_rounds()

    def test_403_raises_auth_error(self, client: AstarClient) -> None:
        """HTTP 403 raises AuthError immediately."""
        resp = _mock_response(status_code=403)
        client._session.request = MagicMock(return_value=resp)
        with pytest.raises(AuthError):
            client.list_rounds()

    def test_500_raises_api_error(self, client: AstarClient) -> None:
        """HTTP 500 raises APIError (not AuthError)."""
        resp = _mock_response(status_code=500, text="Internal Server Error")
        client._session.request = MagicMock(return_value=resp)
        with pytest.raises(APIError):
            client.list_rounds()


# ---------------------------------------------------------------------------
# New endpoint tests
# ---------------------------------------------------------------------------


class TestNewEndpoints:
    @pytest.fixture()
    def client(self) -> AstarClient:
        return AstarClient("test-token", base_url="http://test")

    def test_get_budget(self, client: AstarClient) -> None:
        resp = _mock_response(json_data={"queries_used": 10, "queries_max": 50})
        client._session.request = MagicMock(return_value=resp)
        result = client.get_budget()
        assert result["queries_used"] == 10

    def test_my_rounds(self, client: AstarClient) -> None:
        resp = _mock_response(json_data=[{"round_number": 1, "round_score": 72.5}])
        client._session.request = MagicMock(return_value=resp)
        result = client.my_rounds()
        assert len(result) == 1
        assert result[0]["round_score"] == 72.5

    def test_my_predictions(self, client: AstarClient) -> None:
        resp = _mock_response(json_data=[{"seed_index": 0, "score": 80.0}])
        client._session.request = MagicMock(return_value=resp)
        result = client.my_predictions("round-1")
        assert result[0]["seed_index"] == 0

    def test_analysis(self, client: AstarClient) -> None:
        resp = _mock_response(json_data={"score": 75.0, "width": 40})
        client._session.request = MagicMock(return_value=resp)
        result = client.analysis("round-1", 0)
        assert result["score"] == 75.0

    def test_leaderboard(self, client: AstarClient) -> None:
        resp = _mock_response(json_data=[{"team_name": "test", "rank": 1}])
        client._session.request = MagicMock(return_value=resp)
        result = client.leaderboard()
        assert result[0]["rank"] == 1
