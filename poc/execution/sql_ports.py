"""Ports supplied by an application or a benchmark to the SQL strategies."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from poc.execution.sql_contracts import Answer, ArchitectureOptions


class PhaseTimeout(TimeoutError):
    pass


class RequestTimeout(TimeoutError):
    pass


class ToolBudgetExceeded(RuntimeError):
    pass


class TrialState(Protocol):
    options: ArchitectureOptions
    started: float
    phase: str
    phases: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    request_attempts: int
    request_responses: int
    usage_complete: bool
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
