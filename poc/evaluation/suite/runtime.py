"""Trial-local observation and explicit candidate checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from opentelemetry import trace

from poc.evaluation.suite.diagnostics import diagnose
from poc.evaluation.suite.models import Answer, ArchitectureOptions, TaskInput


class PhaseTimeout(TimeoutError):
    pass


class RequestTimeout(TimeoutError):
    pass


class ToolBudgetExceeded(RuntimeError):
    pass


class TrialState:
    def __init__(
        self,
        task: TaskInput,
        options: ArchitectureOptions | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.task = task
        self.options = options or ArchitectureOptions()
        self.emit: Callable[[dict[str, Any]], None] = emit or (lambda event: None)
        self.started = perf_counter()
        self.phase = "adapter"
        self.phases: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.request_attempts = 0
        self.request_responses = 0
        self.usage_complete = True
        self.exhausted_phase: str | None = None

    def candidate(self, answer: Answer, source: str) -> dict[str, Any]:
        # Copies prevent later reviewer mutation from erasing the checkpoint.
        record: dict[str, Any] = {
            "answer": answer.model_dump(),
            "diagnostics": diagnose(self.task, answer),
            "phase": self.phase,
            "source": source,
            "ts": datetime.now(UTC).isoformat(),
            "elapsed_seconds": perf_counter() - self.started,
        }
        self.candidates.append(record)
        self.emit({"candidate": record})
        with trace.get_tracer(__name__).start_as_current_span("evaluation.candidate") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("evaluation.phase", self.phase)
            span.set_attribute("evaluation.candidate.source", source)
            span.set_attribute("input.value", json.dumps(record["answer"]))
            span.set_attribute(
                "evaluation.candidate.valid", record["diagnostics"]["answer_valid"] is True
            )
            span.set_attribute(
                "evaluation.candidate.feasible", record["diagnostics"]["feasible"] is True
            )
        return record

    def feedback(self, answer: Answer, *, constraints: bool) -> dict[str, Any]:
        diagnostic = diagnose(self.task, answer)
        if constraints:
            return diagnostic
        return {
            "answer_valid": diagnostic["answer_valid"],
            "violations": diagnostic["violations"] if diagnostic["answer_valid"] is False else [],
        }

    def best_candidate(self) -> dict[str, Any] | None:
        # Diagnostic only. The runner never substitutes this for the official final answer.
        return max(
            self.candidates,
            key=lambda c: (
                c["diagnostics"].get("feasible") is True,
                c["diagnostics"].get("scores", {}).get("fraction_correct", 0),
                c["diagnostics"].get("answer_valid") is True,
            ),
            default=None,
        )
