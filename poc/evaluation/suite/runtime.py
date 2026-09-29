"""Trial-local observation and explicit candidate checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from opentelemetry import trace

from poc.evaluation.suite.diagnostics import diagnose
from poc.evaluation.suite.models import Answer, ArchitectureOptions, ModelSpec, TaskInput
from poc.execution.sql_ports import (
    PhaseTimeout as PhaseTimeout,
)
from poc.execution.sql_ports import (
    RequestTimeout as RequestTimeout,
)
from poc.execution.sql_ports import (
    ToolBudgetExceeded as ToolBudgetExceeded,
)


class TrialState:
    def __init__(
        self,
        task: TaskInput,
        options: ArchitectureOptions | None = None,
        emit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.task = task
        self.model_spec: ModelSpec | None = None
        self.options = options or ArchitectureOptions()
        self.emit: Callable[[dict[str, Any]], None] = emit or (lambda event: None)
        self.started = perf_counter()
        self.phase = "adapter"
        self.phases: list[dict[str, Any]] = []
        self.candidates: list[dict[str, Any]] = []
        self.diagnostic_candidates: list[dict[str, Any]] = []
        self.request_attempts = 0
        self.request_responses = 0
        self.usage_complete = True
        self.unreported_token_reserve = 0
        self.exhausted_phase: str | None = None
        self.recovery: dict[str, Any] | None = None
        self.answer_source: str | None = None
        self.review_decision: dict[str, Any] | None = None

    def candidate(self, answer: Answer, source: str) -> dict[str, Any]:
        # Copies prevent later reviewer mutation from erasing the checkpoint.
        record: dict[str, Any] = {
            "answer": answer.model_dump(),
            "phase": self.phase,
            "source": source,
            "ts": datetime.now(UTC).isoformat(),
            "elapsed_seconds": perf_counter() - self.started,
        }
        self.candidates.append(record)
        observed = {**deepcopy(record), "diagnostics": diagnose(self.task, answer)}
        self.diagnostic_candidates.append(observed)
        self.emit({"candidate": observed})
        with trace.get_tracer(__name__).start_as_current_span("evaluation.candidate") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("evaluation.phase", self.phase)
            span.set_attribute("evaluation.candidate.source", source)
            span.set_attribute("input.value", json.dumps(record["answer"]))
            span.set_attribute(
                "evaluation.candidate.valid", observed["diagnostics"]["answer_valid"] is True
            )
            span.set_attribute(
                "evaluation.candidate.feasible", observed["diagnostics"]["feasible"] is True
            )
        return record

    def best_candidate(self) -> dict[str, Any] | None:
        # Diagnostic only. The runner never substitutes this for the official final answer.
        return max(
            self.diagnostic_candidates,
            key=lambda c: (
                c["diagnostics"].get("feasible") is True,
                c["diagnostics"].get("scores", {}).get("fraction_correct", 0),
                c["diagnostics"].get("answer_valid") is True,
            ),
            default=None,
        )
