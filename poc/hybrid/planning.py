"""Orchestrator-authored decomposition plans for the hybrid_v2 strategy.

Plans are data, never authority. Parsing and validation are hard: a plan that
fails validation cannot be approved, and one bounded repair attempt is granted
to the orchestrator before the run fails.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from poc.models import new_id, utc_now

MIN_PLAN_TASKS = 2
MAX_PLAN_TASKS = 8


class PlanValidationError(ValueError):
    """Raised when an orchestrator plan or decision fails hard validation."""

    def __init__(self, problems: Sequence[str]):
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems) or "invalid orchestrator plan")


class PlannedTask(BaseModel):
    """One small, well-defined unit of work inside an orchestrator plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    objective: str
    local_scope: str
    allowed_tables: tuple[str, ...] = ()
    out_of_scope: tuple[str, ...] = ()
    definition_of_done: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    output_schema: str = "Answer"
    evidence_requirements: tuple[str, ...] = ()
    budgets: dict[str, int] = Field(default_factory=dict)


class OrchestratorPlan(BaseModel):
    """A validated decomposition of the parent objective into scoped tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_id: str = Field(default_factory=lambda: new_id("oplan"))
    version: int = 1
    parent_objective: str
    tasks: tuple[PlannedTask, ...]
    integration_definition_of_done: tuple[str, ...]
    round_intent: str = ""
    created_at: str = Field(default_factory=utc_now)


class OrchestratorDecision(BaseModel):
    """One feedback-loop verdict over a critiqued task output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Literal["accept", "revise", "add_task", "escalate"]
    rationale: str
    revision: PlannedTask | None = None
    added_task: PlannedTask | None = None


def _normalize_criteria(raw: Any) -> tuple[str, ...] | str:
    """Accept a single string as one criterion; otherwise require a list."""
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        items = cast(list[object], raw)
        if all(isinstance(item, str) for item in items):
            return tuple(cast(str, item) for item in items)
    return "definition_of_done entries must be strings"


def _parse_task(raw: Any, problems: list[str], index: int) -> PlannedTask | None:
    if not isinstance(raw, Mapping):
        problems.append(f"task {index} must be an object")
        return None
    fields: dict[str, Any] = dict(cast(Mapping[str, Any], raw))
    if "task_id" not in fields:
        fields["task_id"] = f"t{index + 1}"
    for field in (
        "definition_of_done",
        "out_of_scope",
        "dependencies",
        "evidence_requirements",
        "allowed_tables",
    ):
        if field in fields:
            normalized = _normalize_criteria(fields[field])
            if isinstance(normalized, str):
                problems.append(f"task {index + 1}: {normalized}")
                return None
            fields[field] = normalized
    try:
        return PlannedTask.model_validate(fields)
    except ValueError as exc:
        problems.append(f"task {index + 1}: {exc}")
        return None


def plan_problems(
    plan: OrchestratorPlan,
    *,
    min_tasks: int = MIN_PLAN_TASKS,
    max_tasks: int = MAX_PLAN_TASKS,
) -> list[str]:
    problems: list[str] = []
    if not plan.parent_objective.strip():
        problems.append("parent_objective must not be empty")
    if not min_tasks <= len(plan.tasks) <= max_tasks:
        problems.append(
            f"plan must contain between {min_tasks} and {max_tasks} tasks, got {len(plan.tasks)}"
        )
    ids = [task.task_id for task in plan.tasks]
    if any(not task_id.strip() for task_id in ids):
        problems.append("task ids must not be empty")
    duplicates = {task_id for task_id in ids if ids.count(task_id) > 1}
    if duplicates:
        problems.append(f"duplicate task ids: {sorted(duplicates)}")
    known = set(ids)
    for task in plan.tasks:
        if not task.objective.strip():
            problems.append(f"task {task.task_id}: objective must not be empty")
        if not task.local_scope.strip():
            problems.append(f"task {task.task_id}: local_scope must not be empty")
        done = [item for item in task.definition_of_done if item.strip()]
        if not done:
            problems.append(f"task {task.task_id}: definition_of_done needs at least one entry")
        if len(set(task.allowed_tables)) != len(task.allowed_tables):
            problems.append(f"task {task.task_id}: duplicate allowed_tables")
        if any(not table.strip() for table in task.allowed_tables):
            problems.append(f"task {task.task_id}: allowed_tables must not contain empty names")
        if any(
            key not in {"requests", "tool_calls", "total_tokens", "seconds"}
            or type(value) is not int
            or value <= 0
            for key, value in task.budgets.items()
        ):
            problems.append(
                f"task {task.task_id}: budgets require positive integer requests, tool_calls, total_tokens, or seconds"
            )
        if not task.output_schema.strip():
            problems.append(f"task {task.task_id}: output_schema must not be empty")
        for dependency in task.dependencies:
            if dependency == task.task_id:
                problems.append(f"task {task.task_id}: cannot depend on itself")
            elif dependency not in known:
                problems.append(f"task {task.task_id}: unknown dependency {dependency!r}")
    if _cycle(plan.tasks):
        problems.append("dependency graph must be acyclic")
    integration = [item for item in plan.integration_definition_of_done if item.strip()]
    if not integration:
        problems.append("integration_definition_of_done needs at least one entry")
    return problems


def _cycle(tasks: tuple[PlannedTask, ...]) -> bool:
    edges = {task.task_id: set(task.dependencies) for task in tasks}
    resolved: set[str] = set()
    while edges:
        ready = [task_id for task_id, deps in edges.items() if deps <= resolved]
        if not ready:
            return True
        for task_id in ready:
            del edges[task_id]
            resolved.add(task_id)
    return False


def parse_plan(
    values: Mapping[str, Any],
    *,
    min_tasks: int = MIN_PLAN_TASKS,
    max_tasks: int = MAX_PLAN_TASKS,
) -> OrchestratorPlan:
    """Parse and hard-validate an orchestrator plan from model JSON."""
    problems: list[str] = []
    tasks_value = values.get("tasks")
    tasks_raw: list[object] = []
    if isinstance(tasks_value, list):
        tasks_raw = list(cast(list[object], tasks_value))
    else:
        problems.append("tasks must be a list")
    tasks: list[PlannedTask] = []
    for index, raw in enumerate(tasks_raw):
        task = _parse_task(raw, problems, index)
        if task is not None:
            tasks.append(task)
    integration_raw = values.get("integration_definition_of_done")
    integration: tuple[str, ...] = ()
    if integration_raw is None:
        problems.append("integration_definition_of_done is required")
    else:
        normalized = _normalize_criteria(integration_raw)
        if isinstance(normalized, str):
            problems.append(normalized)
        else:
            integration = normalized
    plan = OrchestratorPlan(
        parent_objective=str(values.get("parent_objective", "")),
        tasks=tuple(tasks),
        integration_definition_of_done=integration,
        round_intent=str(values.get("round_intent", "")),
    )
    problems.extend(plan_problems(plan, min_tasks=min_tasks, max_tasks=max_tasks))
    if problems:
        raise PlanValidationError(problems)
    return plan


def parse_decision(values: Mapping[str, Any]) -> OrchestratorDecision:
    """Parse one feedback-loop decision from orchestrator model output."""
    try:
        return OrchestratorDecision.model_validate(dict(values))
    except ValueError as exc:
        raise PlanValidationError([f"invalid orchestrator decision: {exc}"]) from exc


def decision_problems(
    decision: OrchestratorDecision,
    plan: OrchestratorPlan,
    *,
    pending_task_id: str | None = None,
) -> list[str]:
    problems: list[str] = []
    if not decision.rationale.strip():
        problems.append("decision rationale must not be empty")
    if decision.revision is not None and decision.decision != "revise":
        problems.append("only revise decisions may carry a revision")
    if decision.added_task is not None and decision.decision != "add_task":
        problems.append("only add_task decisions may carry an added task")
    if decision.decision == "revise":
        if decision.revision is None:
            problems.append("revise decision requires a revised task")
        elif pending_task_id is not None and decision.revision.task_id != pending_task_id:
            problems.append("revision must target the critiqued task")
        elif all(task.task_id != decision.revision.task_id for task in plan.tasks):
            problems.append(f"unknown revision target {decision.revision.task_id!r}")
    if decision.decision == "add_task":
        if decision.added_task is None:
            problems.append("add_task decision requires an added task")
        elif any(task.task_id == decision.added_task.task_id for task in plan.tasks):
            problems.append(f"task id {decision.added_task.task_id!r} already exists")
        elif any(
            dependency not in {task.task_id for task in plan.tasks}
            for dependency in decision.added_task.dependencies
        ):
            problems.append("added task references an unknown dependency")
    return problems


def validate_decision(
    decision: OrchestratorDecision,
    plan: OrchestratorPlan,
    *,
    pending_task_id: str | None = None,
) -> OrchestratorDecision:
    problems = decision_problems(decision, plan, pending_task_id=pending_task_id)
    if problems:
        raise PlanValidationError(problems)
    return decision


def wave_order(plan: OrchestratorPlan) -> tuple[tuple[PlannedTask, ...], ...]:
    """Group tasks into dependency waves; wave k may run once wave k-1 is accepted."""
    remaining: list[PlannedTask] = list(plan.tasks)
    done: set[str] = set()
    waves: list[tuple[PlannedTask, ...]] = []
    while remaining:
        ready = tuple(task for task in remaining if all(dep in done for dep in task.dependencies))
        if not ready:
            raise PlanValidationError(["plan dependencies are not satisfiable"])
        waves.append(ready)
        done.update(task.task_id for task in ready)
        remaining = [task for task in remaining if task not in ready]
    return tuple(waves)


def apply_decision(plan: OrchestratorPlan, decision: OrchestratorDecision) -> OrchestratorPlan:
    """Return the plan after one revise or add_task decision."""
    if decision.decision == "revise" and decision.revision is not None:
        replacement = decision.revision
        tasks = tuple(
            replacement if task.task_id == replacement.task_id else task for task in plan.tasks
        )
        return plan.model_copy(update={"version": plan.version + 1, "tasks": tasks})
    if decision.decision == "add_task" and decision.added_task is not None:
        return plan.model_copy(
            update={"version": plan.version + 1, "tasks": (*plan.tasks, decision.added_task)}
        )
    return plan
