from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class Tier(StrEnum):
    MAIN = "main"
    SUB = "sub_orchestrator"
    WORKER = "worker"


class RunStatus(StrEnum):
    PROPOSED = "proposed"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Outcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionMode(StrEnum):
    """How an authorized sub-orchestrator distributes its work."""

    HIERARCHICAL_DAG = "hierarchical_dag"
    BOARD_CLAIM = "board_claim"
    MANAGED_POOL = "managed_pool"
    SPECULATIVE = "speculative"


class OwnershipType(StrEnum):
    DAG = "dag"
    CLAIM = "claim"
    ASSIGNMENT = "assignment"
    SPECULATION = "speculation"


class EffectPolicy(StrEnum):
    PURE_ONLY = "pure_only"
    READ_ONLY = "read_only"
    STAGED_EFFECTS = "staged_effects"


class ExecutionPolicy(BaseModel):
    """Immutable authority envelope supplied to an execution strategy.

    Strategy-specific settings live in ``options`` so adding a scheduler does not
    require changing this common contract.  Security-relevant common limits remain
    typed and are validated here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ExecutionMode = ExecutionMode.HIERARCHICAL_DAG
    max_workers: int = Field(default=4, ge=1)
    allowed_roles: frozenset[str] = Field(default_factory=frozenset)
    max_task_attempts: int = Field(default=3, ge=1)
    tool_policy_id: str = "default"
    budget_id: str = "default"
    lease_seconds: int = Field(default=30, ge=1)
    speculative_fanout: int = Field(default=1, ge=1)
    effect_policy: EffectPolicy = EffectPolicy.READ_ONLY
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_mode_limits(self) -> "ExecutionPolicy":
        if not self.allowed_roles:
            raise ValueError("allowed_roles must not be empty")
        if self.speculative_fanout > self.max_workers:
            raise ValueError("speculative_fanout cannot exceed max_workers")
        if self.mode != ExecutionMode.SPECULATIVE and self.speculative_fanout != 1:
            raise ValueError("speculative_fanout is only valid in speculative mode")
        return self


class ExecutionHandle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    run_id: str
    owner_suborchestrator_id: str
    mode: ExecutionMode = ExecutionMode.HIERARCHICAL_DAG


class WorkerGrant(BaseModel):
    """Non-secret worker authority metadata safe for model context and telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grant_id: str = Field(default_factory=lambda: new_id("grant"))
    run_id: str
    plan_id: str
    plan_version: int
    suborchestrator_id: str
    execution_id: str
    execution_mode: ExecutionMode
    task_id: str
    task_revision: int = 1
    attempt_id: str
    worker_instance_id: str
    role_id: str
    role_version: int = 1
    ownership_type: OwnershipType
    ownership_generation: int = Field(ge=1)
    ownership_token_fingerprint: str
    allowed_tools: frozenset[str] = Field(default_factory=frozenset)
    tool_policy_id: str
    artifact_scope: str
    budget: dict[str, int] = Field(default_factory=dict)
    expires_at: str | None = None


class OwnershipCredential(BaseModel):
    """Runtime-only fenced credential; never place this object in model state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_id: str
    ownership_type: OwnershipType
    ownership_id: str
    generation: int = Field(ge=1)
    token: str = Field(exclude=True, repr=False)


class PlanArea(BaseModel):
    id: str
    goal: str
    owner_role: str
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.HIERARCHICAL_DAG
    allowed_execution_modes: frozenset[ExecutionMode] = Field(
        default_factory=lambda: frozenset({ExecutionMode.HIERARCHICAL_DAG})
    )

    @model_validator(mode="after")
    def selected_mode_is_authorized(self) -> "PlanArea":
        if self.execution_mode not in self.allowed_execution_modes:
            raise ValueError("execution_mode must be included in allowed_execution_modes")
        return self


class MissionPlan(BaseModel):
    plan_id: str
    run_id: str
    version: int = 1
    status: Literal["proposed", "approved", "superseded"] = "proposed"
    objective: str
    constraints: list[str]
    permitted_sources: list[str]
    permitted_tools: list[str]
    areas: list[PlanArea]
    budgets: dict[str, int] = Field(default_factory=lambda: {"max_workers": 4, "ooda_cycles": 3, "tool_calls": 2})
    completion_criteria: list[str]
    created_at: str = Field(default_factory=utc_now)


class PlanEdit(BaseModel):
    constraints: list[str] | None = None
    permitted_sources: list[str] | None = None
    permitted_tools: list[str] | None = None


class ApprovalRequest(BaseModel):
    plan_version: int
    edits: PlanEdit | None = None


class RunCreate(BaseModel):
    objective: str = (
        "Investigate a checkout latency regression using the supplied metrics, logs, "
        "and deployment records. Produce an evidence-backed report identifying likely "
        "causes and useful follow-up tests. Do not modify any live system."
    )


class MessageEnvelope(BaseModel):
    message_id: str = Field(default_factory=lambda: new_id("msg"))
    run_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    correlation_id: str | None = None
    causation_id: str | None = None
    event_seq: int | None = None
    plan_version: int
    workflow_revision: int | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    assignment_id: str | None = None
    payload: dict[str, Any]
    payload_ref: str | None = None
    trace_context: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
    delivered_at: str | None = None
    consumed_at: str | None = None


class RoleSpec(BaseModel):
    role_id: str
    version: int = 1
    tier: Tier
    system_prompt: str
    provider: str = "deterministic"
    api_type: str = "local"
    model: str = "fixture-reasoner-v1"
    provider_options: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    output_schemas: list[str] = Field(default_factory=list)
    execution_limits: dict[str, int] = Field(default_factory=dict)
    allowed_child_roles: list[str] = Field(default_factory=list)


class AgentInstance(BaseModel):
    agent_instance_id: str
    run_id: str
    parent_agent_id: str | None
    tier: Tier
    role_id: str
    role_version: int
    plan_version: int
    status: str = "active"
    created_at: str = Field(default_factory=utc_now)


class InputBinding(BaseModel):
    source_task: str
    field: str
    target: str


class TaskSpec(BaseModel):
    id: str
    role: str
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    input_bindings: list[InputBinding] = Field(default_factory=list)
    output_schema: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    static_inputs: dict[str, Any] = Field(default_factory=dict)


class WorkflowSpec(BaseModel):
    workflow_id: str
    run_id: str
    revision: int = 1
    owner: str
    approved_plan_version: int
    authorized_worker_roles: list[str]
    max_workers: int = 4
    execution_mode: ExecutionMode = ExecutionMode.HIERARCHICAL_DAG
    execution_options: dict[str, Any] = Field(default_factory=dict)
    tasks: list[TaskSpec]

    @model_validator(mode="after")
    def validate_dag(self) -> "WorkflowSpec":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            if task.role not in self.authorized_worker_roles:
                raise ValueError(f"role {task.role!r} is not authorized")
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"task {task.id!r} has unknown dependencies: {sorted(missing)}")
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id!r} depends on itself")
        visiting: set[str] = set()
        visited: set[str] = set()
        deps = {task.id: task.depends_on for task in self.tasks}

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("workflow must be acyclic")
            if node in visited:
                return
            visiting.add(node)
            for predecessor in deps[node]:
                visit(predecessor)
            visiting.remove(node)
            visited.add(node)

        for task_id in ids:
            visit(task_id)
        return self


class ValidationRequest(BaseModel):
    request_id: str
    workflow_id: str
    task_id: str
    agent_instance_id: str
    question: str
    evidence_artifacts: list[str]


class ValidationResolution(BaseModel):
    request_id: str
    approved: bool
    response: str
    evidence_artifacts: list[str] = Field(default_factory=list)


class WorkerResult(BaseModel):
    task_id: str
    agent_instance_id: str
    attempt_id: str
    outcome: Outcome
    output_schema: str
    result: dict[str, Any] = Field(default_factory=dict)
    output_artifact: str | None = None
    evidence_artifacts: list[str] = Field(default_factory=list)
    acceptance_checks: list[dict[str, Any]] = Field(default_factory=list)
    completion_summary: str


class ArtifactRecord(BaseModel):
    artifact_id: str
    run_id: str
    media_type: str
    sha256: str
    path: str
    producer_task_id: str | None = None
    created_at: str = Field(default_factory=utc_now)


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    event_type: str
    actor_id: str | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now)
