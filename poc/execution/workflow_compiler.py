from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Annotated, Any, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import Command

from poc.execution.worker_adapter import WorkerAdapter
from poc.models import AgentInstance, TaskSpec, WorkflowSpec


def merge_task_results(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"conflicting workflow result writes: {sorted(overlap)}")
    return {**left, **right}


class WorkflowState(TypedDict, total=False):
    run_id: str
    results: Annotated[dict[str, Any], merge_task_results]


class TaskSnapshot(Protocol):
    interrupts: Sequence[Any]


class WorkflowSnapshot(Protocol):
    values: dict[str, Any]
    tasks: Sequence[TaskSnapshot]
    next: Sequence[str]


class WorkflowGraph(Protocol):
    def invoke(
        self,
        input: WorkflowState | Command[Any] | None,
        config: dict[str, Any],
    ) -> dict[str, Any]: ...

    def get_state(self, config: dict[str, Any]) -> WorkflowSnapshot: ...


class WorkflowCompiler:
    def __init__(self, adapter: WorkerAdapter, role_lookup: Callable[[str], AgentInstance]):
        self.adapter = adapter
        self.role_lookup = role_lookup

    def compile(self, spec: WorkflowSpec, checkpointer: Any) -> WorkflowGraph:
        # LangGraph exposes partially unknown internal generic parameters to Pyright.
        graph = cast(Any, StateGraph(WorkflowState))
        task_ids = {task.id for task in spec.tasks}
        depended_on = {dependency for task in spec.tasks for dependency in task.depends_on}
        for task in spec.tasks:
            graph.add_node(task.id, self._node(spec, task))
        roots = [task.id for task in spec.tasks if not task.depends_on]
        for root in roots:
            graph.add_edge(START, root)
        for task in spec.tasks:
            if task.depends_on:
                graph.add_edge(task.depends_on, task.id)
        for leaf in task_ids - depended_on:
            graph.add_edge(leaf, END)
        return cast(WorkflowGraph, graph.compile(checkpointer=checkpointer))

    def _node(
        self, spec: WorkflowSpec, task: TaskSpec
    ) -> Callable[[WorkflowState], dict[str, Any]]:
        def run_task(state: WorkflowState) -> dict[str, Any]:
            results = state.get("results", {})
            predecessor_results = {dep: results[dep] for dep in task.depends_on}
            for dep, result in predecessor_results.items():
                if result["outcome"] != "succeeded":
                    blocked: dict[str, Any] = {
                        "task_id": task.id,
                        "agent_instance_id": "not-started",
                        "attempt_id": "not-started",
                        "outcome": "failed",
                        "output_schema": task.output_schema,
                        "result": {},
                        "output_artifact": None,
                        "evidence_artifacts": result.get("evidence_artifacts", []),
                        "acceptance_checks": [],
                        "completion_summary": f"Blocked by failed predecessor {dep}.",
                    }
                    return {"results": {task.id: blocked}}
            inputs = dict(task.static_inputs)
            input_artifacts: list[str] = []
            for result in predecessor_results.values():
                input_artifacts.extend(result.get("evidence_artifacts", []))
            for binding in task.input_bindings:
                value: Any = predecessor_results[binding.source_task]["result"]
                for component in binding.field.split("."):
                    value = value[component]
                inputs[binding.target] = value
            output = self.adapter.execute(
                task=task,
                workflow_id=spec.workflow_id,
                workflow_revision=spec.revision,
                run_id=spec.run_id,
                plan_version=spec.approved_plan_version,
                owner=self.role_lookup(spec.owner),
                inputs=inputs,
                input_artifacts=list(dict.fromkeys(input_artifacts)),
            )
            if "worker_result" not in output:
                # A nested interrupt bubbles through this wrapper to the root graph.
                return {}
            return {"results": {task.id: output["worker_result"]}}

        return run_task
