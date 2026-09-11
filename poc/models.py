from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


class AgentBackend(StrEnum):
    """Built-in worker runtime identifiers; registries also accept extension IDs."""

    CUSTOM_PYTHON = "custom_python"
    PYDANTIC_AI = "pydantic_ai"
    SEMANTIC_PYDANTIC_AI = "semantic_pydantic_ai"


def is_pydantic_ai_backend(backend: str) -> bool:
    return backend in {
        AgentBackend.PYDANTIC_AI,
        AgentBackend.SEMANTIC_PYDANTIC_AI,
    }


class SwarmStrategy(StrEnum):
    """Versioned policy bundles used inside the board-claim swarm backend."""

    BOARD = "board"
    HYBRID_V1 = "hybrid_v1"


class DependencyRequirement(StrEnum):
    """Required authority state for consuming an artifact dependency."""

    CANDIDATE_PUBLISHED = "candidate_published"
    ARTIFACT_VERIFIED = "artifact_verified"
    DELIVERABLE_ACCEPTED = "deliverable_accepted"


class HybridConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    proposals_per_round: int = Field(default=3, ge=2, le=12)
    max_collaboration_rounds: int = Field(default=2, ge=1, le=8)
    max_critics_per_candidate: int = Field(default=2, ge=1, le=8)
    initial_candidate_visibility: Literal["sealed_to_round"] = "sealed_to_round"


class SwarmPolicySet(BaseModel):
    """Resolved, immutable policy versions for one approved run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: SwarmStrategy = SwarmStrategy.BOARD
    allocation_policy: str = "compatibility_fifo_v1"
    context_policy: str = "compatibility_context_v1"
    communication_policy: str = "supervisor_only_v1"
    collaboration_policy: str = "none_v1"
    acceptance_policy: str = "worker_result_v1"
    completion_policy: str = "workflow_complete_v1"


class AgentRuntimeConfig(BaseModel):
    """Per-run worker implementation and model selection.

    ``options`` is deliberately framework-owned so additional executors can add
    configuration without widening the common orchestration contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = Field(default=AgentBackend.CUSTOM_PYTHON, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_pydantic_ai_model(self) -> AgentRuntimeConfig:
        if not is_pydantic_ai_backend(self.backend):
            return self
        provider_from_model = (
            self.model.split(":", 1)[0] if self.model and ":" in self.model else None
        )
        if self.model is None:
            raise ValueError(f"{self.backend} requires a model")
        if self.provider is None and provider_from_model is None:
            raise ValueError(f"{self.backend} requires a provider or provider:model identifier")
        return self


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
    swarm_strategy: SwarmStrategy = SwarmStrategy.BOARD
    policy_set: SwarmPolicySet = Field(default_factory=SwarmPolicySet)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    agent_backend: str = Field(default=AgentBackend.CUSTOM_PYTHON, min_length=1)
    agent_provider: str | None = Field(default=None, min_length=1)
    agent_model: str | None = Field(default=None, min_length=1)
    agent_options: dict[str, Any] = Field(default_factory=dict)
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
    def validate_mode_limits(self) -> ExecutionPolicy:
        if not self.allowed_roles:
            raise ValueError("allowed_roles must not be empty")
        if self.speculative_fanout > self.max_workers:
            raise ValueError("speculative_fanout cannot exceed max_workers")
        if self.mode != ExecutionMode.SPECULATIVE and self.speculative_fanout != 1:
            raise ValueError("speculative_fanout is only valid in speculative mode")
        if (
            self.swarm_strategy == SwarmStrategy.HYBRID_V1
            and self.mode != ExecutionMode.BOARD_CLAIM
        ):
            raise ValueError("hybrid_v1 is a strategy inside board_claim mode")
        if self.policy_set.strategy != self.swarm_strategy:
            raise ValueError("policy_set strategy must match swarm_strategy")
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
    def selected_mode_is_authorized(self) -> PlanArea:
        if self.execution_mode not in self.allowed_execution_modes:
            raise ValueError("execution_mode must be included in allowed_execution_modes")
        return self


class MissionPlan(BaseModel):
    plan_id: str
    run_id: str
    version: int = 1
    status: Literal["proposed", "approved", "superseded"] = "proposed"
    objective: str
    agent_runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    execution_mode: ExecutionMode = ExecutionMode.HIERARCHICAL_DAG
    execution_options: dict[str, Any] = Field(default_factory=dict)
    swarm_strategy: SwarmStrategy = SwarmStrategy.BOARD
    policy_set: SwarmPolicySet = Field(default_factory=SwarmPolicySet)
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    constraints: list[str]
    permitted_sources: list[str]
    permitted_tools: list[str]
    areas: list[PlanArea]
    budgets: dict[str, int] = Field(
        default_factory=lambda: {"max_workers": 4, "ooda_cycles": 3, "tool_calls": 2}
    )
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
    agent_runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    execution_mode: ExecutionMode = ExecutionMode.HIERARCHICAL_DAG
    execution_options: dict[str, Any] = Field(default_factory=dict)
    swarm_strategy: SwarmStrategy = SwarmStrategy.BOARD
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    @model_validator(mode="after")
    def validate_swarm_strategy(self) -> RunCreate:
        if self.swarm_strategy == SwarmStrategy.HYBRID_V1 and (
            self.execution_mode != ExecutionMode.BOARD_CLAIM
        ):
            raise ValueError("hybrid_v1 requires board_claim execution mode")
        return self


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
    agent_backend: str = Field(default=AgentBackend.CUSTOM_PYTHON, min_length=1)
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
    agent_backend: str = Field(default=AgentBackend.CUSTOM_PYTHON, min_length=1)
    agent_provider: str | None = Field(default=None, min_length=1)
    agent_model: str | None = Field(default=None, min_length=1)
    status: str = "active"
    created_at: str = Field(default_factory=utc_now)


class InputBinding(BaseModel):
    source_task: str
    field: str
    target: str


class TaskSpec(BaseModel):
    id: str
    role: str
    agent_backend: str | None = Field(default=None, min_length=1)
    agent_provider: str | None = Field(default=None, min_length=1)
    agent_model: str | None = Field(default=None, min_length=1)
    agent_options: dict[str, Any] = Field(default_factory=dict)
    goal: str
    depends_on: list[str] = Field(default_factory=list)
    input_bindings: list[InputBinding] = Field(default_factory=list[InputBinding])
    output_schema: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    static_inputs: dict[str, Any] = Field(default_factory=dict)
    static_artifact_refs: list[str] = Field(default_factory=list)
    artifact_requirements: dict[str, DependencyRequirement] = Field(default_factory=dict)
    domain_id: str | None = None
    team_id: str | None = None
    collaboration_round_id: str | None = None
    candidate_id: str | None = None
    artifact_visibility: str = "run_wide"

    @model_validator(mode="after")
    def validate_artifact_requirements(self) -> TaskSpec:
        unknown = set(self.artifact_requirements) - set(self.static_artifact_refs)
        if unknown:
            raise ValueError(
                f"artifact requirements reference non-input artifacts: {sorted(unknown)}"
            )
        return self


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
    def validate_dag(self) -> WorkflowSpec:
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
    acceptance_checks: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    completion_summary: str


class ArtifactRecord(BaseModel):
    artifact_id: str
    run_id: str
    media_type: str
    sha256: str
    path: str
    producer_task_id: str | None = None
    producer_attempt_id: str | None = None
    visibility: str = "run_wide"
    visibility_ref: str | None = None
    access_labels: frozenset[str] = Field(default_factory=frozenset)
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
