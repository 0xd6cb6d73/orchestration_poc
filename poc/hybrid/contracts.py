from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from poc.models import new_id, utc_now


class VisibilityPolicy(StrEnum):
    PRIVATE_TO_ATTEMPT = "private_to_attempt"
    SEALED_TO_ROUND = "sealed_to_round"
    RELEASED_TO_TEAM = "released_to_team"
    DOMAIN = "domain"
    RUN_WIDE = "run_wide"


class CollaborationPhase(StrEnum):
    FRAMED = "framed"
    COLLECTING_SEALED_PROPOSALS = "collecting_sealed_proposals"
    RELEASED = "released"
    CLUSTERED = "clustered"
    CRITIQUED = "critiqued"
    TESTED = "tested"
    RECOMBINED = "recombined"
    VERIFIED_AND_SCORED = "verified_and_scored"
    COMPLETE = "complete"
    REFRAME = "reframe"


class GoalContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    goal_contract_id: str = Field(default_factory=lambda: new_id("goal"))
    version: int = 1
    run_id: str
    plan_version: int
    parent_objective: str
    local_scope: str
    global_invariants: tuple[str, ...]
    input_artifact_refs: tuple[str, ...] = ()
    output_schema: str
    evidence_requirements: tuple[str, ...] = ()
    definition_of_done: tuple[str, ...] = ()
    budgets: dict[str, int] = Field(default_factory=dict)
    permissions: tuple[str, ...] = ()
    created_at: str = Field(default_factory=utc_now)


class ControlSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    artifact_refs: tuple[str, ...] = ()
    discovered_dependencies: tuple[str, ...] = ()
    exceptions: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    conclusions: tuple[str, ...] = ()


class ContextManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_manifest_id: str = Field(default_factory=lambda: new_id("context"))
    version: int = 1
    run_id: str
    plan_version: int
    agent_instance_id: str
    task_id: str
    attempt_id: str
    context_policy_version: str
    goal_contract_id: str
    available_artifact_refs: tuple[str, ...] = ()
    retrieved_artifact_refs: tuple[str, ...] = ()
    included_artifact_refs: tuple[str, ...] = ()
    included_message_ids: tuple[str, ...] = ()
    retrieval_decisions: tuple[str, ...] = ()
    redaction_version: str = "none_v1"
    created_at: str = Field(default_factory=utc_now)


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provenance_id: str = Field(default_factory=lambda: new_id("provenance"))
    run_id: str
    artifact_id: str
    producer_agent_id: str
    task_id: str
    attempt_id: str
    input_artifact_refs: tuple[str, ...] = ()
    tool_operation_ids: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    role_id: str
    role_version: int
    prompt_fingerprint: str
    agent_backend: str
    provider: str | None = None
    model: str | None = None
    context_manifest_id: str
    created_at: str = Field(default_factory=utc_now)


class TeamSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    team_id: str = Field(default_factory=lambda: new_id("team"))
    version: int = 1
    run_id: str
    domain_id: str
    steward_agent_id: str
    member_agent_ids: tuple[str, ...]
    allowed_message_types: frozenset[str] = frozenset(
        {"DELEGATE", "PROGRESS", "PUBLISH", "REQUEST", "CHALLENGE", "COMPLETE"}
    )
    permitted_edges: frozenset[str] = frozenset()
    subscriptions: tuple[str, ...] = ()
    expires_at: str | None = None
    message_budget: int = Field(default=30, ge=1)
    communication_policy_version: str = "scoped_team_v1"
    created_at: str = Field(default_factory=utc_now)


class CapabilityProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(default_factory=lambda: new_id("capability"))
    version: int = 1
    role_id: str
    role_version: int
    agent_backend: str
    provider: str | None = None
    model: str | None = None
    task_classes: frozenset[str]
    tools: frozenset[str]
    preference: int = 0


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    route_decision_id: str = Field(default_factory=lambda: new_id("route"))
    run_id: str
    task_id: str
    offer_generation: int = Field(ge=1)
    policy_version: str
    selected_profile_id: str
    considered_profile_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    created_at: str = Field(default_factory=utc_now)


class CollaborationRound(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    round_id: str = Field(default_factory=lambda: new_id("round"))
    run_id: str
    domain_id: str
    steward_agent_id: str
    team_id: str
    round_number: int = Field(default=1, ge=1)
    phase: CollaborationPhase = CollaborationPhase.FRAMED
    expected_proposals: int = Field(ge=2)
    submitted_proposals: int = Field(default=0, ge=0)
    policy_version: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)


class EvaluationVector(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    validity: int = Field(ge=0, le=5)
    evidence: int = Field(ge=0, le=5)
    usefulness: int = Field(ge=0, le=5)
    novelty: int = Field(ge=0, le=5)
    constraint_satisfaction: int = Field(ge=0, le=5)


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(default_factory=lambda: new_id("candidate"))
    version: int = 1
    round_id: str
    author_agent_id: str
    hypothesis_key: str
    artifact_id: str
    blackboard_record_id: str
    evidence_refs: tuple[str, ...]
    visibility: VisibilityPolicy = VisibilityPolicy.SEALED_TO_ROUND
    state: Literal["proposed", "viable", "refuted", "selected"] = "proposed"
    evaluation: EvaluationVector | None = None
    parent_candidate_ids: tuple[str, ...] = ()
    created_at: str = Field(default_factory=utc_now)


class CandidateCluster(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster_id: str = Field(default_factory=lambda: new_id("cluster"))
    version: int = 1
    round_id: str
    label: str
    candidate_ids: tuple[str, ...]
    assignment_policy_version: str = "hypothesis_key_v1"
    rationale: str
    created_at: str = Field(default_factory=utc_now)


class Challenge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    challenge_id: str = Field(default_factory=lambda: new_id("challenge"))
    round_id: str
    candidate_id: str
    critic_agent_id: str
    assumption: str
    artifact_id: str
    created_at: str = Field(default_factory=utc_now)


class TestResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    test_result_id: str = Field(default_factory=lambda: new_id("test"))
    round_id: str
    candidate_id: str
    tester_agent_id: str
    passed: bool
    findings: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    created_at: str = Field(default_factory=utc_now)


class VerificationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verification_id: str = Field(default_factory=lambda: new_id("verification"))
    run_id: str
    subject_artifact_id: str
    producer_agent_id: str
    claim_or_output_schema: str
    required_checks: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    independence_policy: str = "separate_worker_v1"
    rubric_version: str = "evidence_rubric_v1"
    created_at: str = Field(default_factory=utc_now)


class VerificationVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict_id: str = Field(default_factory=lambda: new_id("verdict"))
    verification_id: str
    verdict: Literal["pass", "fail", "inconclusive"]
    checks_performed: tuple[str, ...]
    findings: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    verifier_agent_id: str
    created_at: str = Field(default_factory=utc_now)


class AcceptanceDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    acceptance_decision_id: str = Field(default_factory=lambda: new_id("acceptance"))
    run_id: str
    subject_artifact_id: str
    verdict_refs: tuple[str, ...]
    acceptance_policy_version: str
    permitted_downstream_uses: tuple[str, ...]
    accepted: bool
    created_at: str = Field(default_factory=utc_now)


class DeliveryDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    delivery_decision_id: str = Field(default_factory=lambda: new_id("delivery"))
    run_id: str
    artifact_id: str
    plan_version: int
    completion_policy_version: str
    permitted: bool
    checks: dict[str, bool]
    reasons: tuple[str, ...]
    created_at: str = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def permitted_only_when_checks_pass(self) -> DeliveryDecision:
        if self.permitted != all(self.checks.values()):
            raise ValueError("permitted must match the aggregate delivery checks")
        return self
