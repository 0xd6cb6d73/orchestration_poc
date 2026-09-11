from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from poc.execution.ooda_graph import OODAHarness, WorkerState
from poc.models import AgentBackend, AgentInstance, TaskSpec


@dataclass(frozen=True)
class AgentExecutionRequest:
    task: TaskSpec
    workflow_id: str
    workflow_revision: int
    run_id: str
    agent: AgentInstance
    attempt_id: str
    inputs: dict[str, Any]
    input_artifacts: list[str]

    def worker_state(self) -> WorkerState:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "workflow_revision": self.workflow_revision,
            "task_id": self.task.id,
            "goal": self.task.goal,
            "output_schema": self.task.output_schema,
            "acceptance_criteria": self.task.acceptance_criteria,
            "inputs": self.inputs,
            "input_artifacts": self.input_artifacts,
            "agent": self.agent.model_dump(mode="json"),
            "attempt_id": self.attempt_id,
            "cycle": 0,
            "tool_calls": 0,
            "validations": 0,
            "working": {},
            "evidence_artifacts": [],
        }


class AgentExecutorNotRegistered(LookupError):
    pass


@runtime_checkable
class AgentExecutor(Protocol):
    """Framework-neutral worker execution boundary above the LangGraph layer."""

    @property
    def backend(self) -> str: ...

    def execute(self, request: AgentExecutionRequest) -> dict[str, Any]: ...


class AgentExecutorRegistry:
    """Selects worker implementations without coupling schedulers to frameworks."""

    def __init__(self) -> None:
        self._executors: dict[str, AgentExecutor] = {}

    def register(self, executor: AgentExecutor, *, replace: bool = False) -> None:
        backend = str(executor.backend)
        if backend in self._executors and not replace:
            raise ValueError(f"executor already registered for {backend!r}")
        self._executors[backend] = executor

    def get(self, backend: str) -> AgentExecutor:
        normalized = str(backend)
        try:
            return self._executors[normalized]
        except KeyError as exc:
            raise AgentExecutorNotRegistered(
                f"no agent executor registered for {normalized!r}"
            ) from exc

    @property
    def backends(self) -> frozenset[str]:
        return frozenset(self._executors)


class CustomPythonAgentExecutor:
    """Adapter around the existing custom Python/LangGraph worker implementation."""

    backend: str = AgentBackend.CUSTOM_PYTHON

    def __init__(self, ooda: OODAHarness):
        self.ooda = ooda

    def execute(self, request: AgentExecutionRequest) -> dict[str, Any]:
        return self.ooda.graph.invoke(request.worker_state())
