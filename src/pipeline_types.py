"""Data types for the competition pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeedResult:
    """Result of processing one seed."""

    seed_index: int
    queries_used: int = 0
    submitted: bool = False
    score: float | None = None
    error: str | None = None


@dataclass
class PipelineResult:
    """Result of running the full pipeline."""

    round_id: str
    seed_results: list[SeedResult] = field(default_factory=list)
    total_queries: int = 0
    elapsed_seconds: float = 0.0
