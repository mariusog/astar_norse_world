"""HTTP client for the Astar Island competition API.

Wraps requests.Session with JWT auth, query budget tracking,
typed exceptions, and exponential backoff retry.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import requests

from src.constants import (
    API_BASE_URL,
    API_RETRY_BACKOFF,
    API_RETRY_MAX,
    PROBABILITY_FLOOR,
    QUERY_WARNING_THRESHOLD,
    TOTAL_QUERY_BUDGET,
    VIEWPORT_MAX_SIZE,
    VIEWPORT_MIN_SIZE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(APIError):
    """Authentication failed (401/403)."""


class BudgetExhaustedError(APIError):
    """Query budget exceeded for the current round."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AstarClient:
    """Client for the Astar Island competition server.

    Handles JWT authentication, query budget tracking, and
    automatic retry with exponential backoff.
    """

    def __init__(self, token: str, base_url: str = API_BASE_URL) -> None:
        self._session = requests.Session()
        self._session.cookies.set("access_token", token)
        self._session.headers["Authorization"] = f"Bearer {token}"
        self._base_url = base_url.rstrip("/")
        self._query_counts: dict[str, int] = {}

    # -- Public API ---------------------------------------------------------

    def list_rounds(self) -> list[dict[str, Any]]:
        """GET /astar-island/rounds -- list all rounds."""
        return self._get("/astar-island/rounds")

    def get_round(self, round_id: str) -> dict[str, Any]:
        """GET /astar-island/rounds/{round_id} -- round details."""
        return self._get(f"/astar-island/rounds/{round_id}")

    def get_active_round(self) -> dict[str, Any] | None:
        """Find the currently active round, or None."""
        rounds = self.list_rounds()
        return next(
            (r for r in rounds if r.get("status") == "active"),
            None,
        )

    def query(
        self,
        round_id: str,
        seed_index: int,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> dict[str, Any]:
        """POST /astar-island/simulate -- query a viewport.

        Validates viewport dimensions and tracks query budget.
        Raises BudgetExhaustedError if budget would be exceeded.
        """
        _validate_viewport(w, h)
        self._check_budget(round_id)

        body = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }
        result = self._post("/astar-island/simulate", json=body)
        self._increment_budget(round_id)
        return result

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: np.ndarray,
    ) -> dict[str, Any]:
        """POST /astar-island/submit -- submit a prediction tensor.

        Applies probability floor and renormalization automatically.
        prediction shape: (height, width, 6)
        """
        safe = _apply_probability_floor(prediction)
        body = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": safe.tolist(),
        }
        return self._post("/astar-island/submit", json=body)

    def query_count(self, round_id: str) -> int:
        """Return how many queries have been used for this round."""
        return self._query_counts.get(round_id, 0)

    def queries_remaining(self, round_id: str) -> int:
        """Return how many queries remain for this round."""
        return TOTAL_QUERY_BUDGET - self.query_count(round_id)

    # -- Budget tracking ----------------------------------------------------

    def _check_budget(self, round_id: str) -> None:
        """Raise if budget exhausted; warn if near limit."""
        used = self.query_count(round_id)
        if used >= TOTAL_QUERY_BUDGET:
            raise BudgetExhaustedError(
                f"Query budget exhausted: {used}/{TOTAL_QUERY_BUDGET}",
            )
        if used >= QUERY_WARNING_THRESHOLD:
            logger.warning(
                "Query budget low: %d/%d used for round %s",
                used,
                TOTAL_QUERY_BUDGET,
                round_id,
            )

    def _increment_budget(self, round_id: str) -> None:
        """Record a successful query."""
        self._query_counts[round_id] = self.query_count(round_id) + 1
        used = self._query_counts[round_id]
        logger.info(
            "Query %d/%d used for round %s",
            used,
            TOTAL_QUERY_BUDGET,
            round_id,
        )

    # -- HTTP helpers -------------------------------------------------------

    def _get(self, path: str) -> Any:
        """GET with retry and error handling."""
        return self._request("GET", path)

    def _post(self, path: str, json: dict[str, Any]) -> Any:
        """POST with retry and error handling."""
        return self._request("POST", path, json=json)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Execute HTTP request with exponential backoff retry."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(API_RETRY_MAX):
            try:
                resp = self._session.request(method, url, **kwargs)
                _raise_for_status(resp)
                return resp.json()
            except (AuthError, BudgetExhaustedError):
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                delay = API_RETRY_BACKOFF * (2**attempt)
                logger.warning(
                    "Transient error on %s %s (attempt %d/%d), retrying in %.1fs: %s",
                    method,
                    path,
                    attempt + 1,
                    API_RETRY_MAX,
                    delay,
                    exc,
                )
                time.sleep(delay)
            except APIError:
                raise

        raise APIError(
            f"Request failed after {API_RETRY_MAX} retries: {last_exc}",
        )


# ---------------------------------------------------------------------------
# Helpers (module-level, not methods)
# ---------------------------------------------------------------------------


def _validate_viewport(w: int, h: int) -> None:
    """Raise ValueError if viewport dimensions out of range."""
    if not (VIEWPORT_MIN_SIZE <= w <= VIEWPORT_MAX_SIZE):
        msg = f"viewport_w={w} out of range [{VIEWPORT_MIN_SIZE}, {VIEWPORT_MAX_SIZE}]"
        raise ValueError(msg)
    if not (VIEWPORT_MIN_SIZE <= h <= VIEWPORT_MAX_SIZE):
        msg = f"viewport_h={h} out of range [{VIEWPORT_MIN_SIZE}, {VIEWPORT_MAX_SIZE}]"
        raise ValueError(msg)


def _apply_probability_floor(prediction: np.ndarray) -> np.ndarray:
    """Clamp minimum probabilities and renormalize.

    Prevents infinite KL divergence from zero probabilities.
    """
    safe = np.maximum(prediction, PROBABILITY_FLOOR)
    sums = safe.sum(axis=2, keepdims=True)
    return safe / sums


def _raise_for_status(resp: requests.Response) -> None:
    """Raise typed exceptions based on HTTP status code."""
    if resp.status_code in (401, 403):
        raise AuthError(
            f"Authentication failed: {resp.status_code}",
            status_code=resp.status_code,
        )
    if resp.status_code == 429:
        raise BudgetExhaustedError(
            "Rate limited by server",
            status_code=429,
        )
    if resp.status_code >= 400:
        raise APIError(
            f"API error {resp.status_code}: {resp.text}",
            status_code=resp.status_code,
        )
