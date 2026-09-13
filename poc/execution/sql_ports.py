"""Ports supplied by an application or a benchmark to the SQL strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

from poc.execution.sql_contracts import Answer, ArchitectureOptions


class PhaseTimeout(TimeoutError):
    pass


class RequestTimeout(TimeoutError):
    pass


class ToolBudgetExceeded(RuntimeError):
    pass


def failure_details(
    exc: BaseException,
    *,
    scope: str,
    stage: str,
    threshold: dict[str, Any],
    consumption: dict[str, Any],
    stage_index: int | None = None,
) -> dict[str, Any]:
    """Preserve the innermost scope without persisting provider exception bodies."""
    existing = getattr(exc, "failure_details", None)
    if existing is not None:
        return existing
    details = {
        "type": type(exc).__name__,
        "scope": scope,
        "stage": stage,
        "stage_index": stage_index,
        "threshold": threshold,
        "consumption": consumption,
    }
    cast(Any, exc).failure_details = details
    return details


class TrialState(Protocol):
    options: ArchitectureOptions
    started: float
    phase: str
    phases: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    request_attempts: int
    request_responses: int
    usage_complete: bool
    unreported_token_reserve: int
    exhausted_phase: str | None
    recovery: dict[str, Any] | None
    answer_source: str | None
    review_decision: dict[str, Any] | None
    emit: Callable[[dict[str, Any]], None]

    def candidate(self, answer: Answer, source: str) -> dict[str, Any]: ...


class TaskEnvironment(Protocol):
    @property
    def schema(self) -> dict[str, list[str]]: ...

    @property
    def state(self) -> TrialState: ...

    def query(self, sql: str) -> dict[str, Any]: ...

    def submit_candidate(self, values: dict[str, Any]) -> dict[str, Any]: ...

    def fork(self, max_calls: int) -> TaskEnvironment: ...

    def absorb(self, child: TaskEnvironment) -> None: ...

    def close(self) -> None: ...
