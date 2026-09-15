"""Public SQL-agent inputs, model configuration and execution limits (no grading data)."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskInput(StrictModel):
    """The complete public interface; references never cross the adapter boundary."""

    prompt: str
    tables: dict[str, list[dict[str, Any]]]


class Budget(StrictModel):
    requests: int = Field(default=40, ge=1)
    tool_calls: int = Field(default=80, ge=1)
    total_tokens: int = Field(default=100000, ge=1)
    seconds: float = Field(default=300, gt=0)
    request_timeout_seconds: float | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    finalization_seconds: float = Field(default=0, ge=0)
    finalization_tokens: int = Field(default=0, ge=0)
    finalization_requests: int = Field(default=0, ge=0)
    draft_fraction: float = Field(default=1, gt=0, le=1)

    @model_validator(mode="after")
    def valid_reserves(self) -> Budget:
        if self.finalization_seconds >= self.seconds:
            raise ValueError("finalization_seconds must be less than seconds")
        if self.finalization_tokens >= self.total_tokens:
            raise ValueError("finalization_tokens must be less than total_tokens")
        if self.finalization_requests >= self.requests:
            raise ValueError("finalization_requests must be less than requests")
        return self


class ModelSpec(StrictModel):
    name: str
    model_class: Literal["7B", "27B", "100B", "frontier-flash", "baseline"]
    model: str
    base_url_env: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    settings: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0})
    context_window: int | None = Field(default=None, gt=0)
    endpoint_context_window: int | None = Field(default=None, gt=0)
    completion_limit: int | None = Field(default=None, gt=0)
    role_profiles: dict[str, RoleBudgetProfile] = Field(default_factory=dict)
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)


PhaseName: TypeAlias = Literal[
    "solve",
    "draft",
    "review",
    "finalize",
    "plan",
    "proposal",
    "critique",
    "verify",
    "reconcile",
    "orchestrate",
    "task",
    "integrate",
]

HYBRID_V2_POOL_ROLES = frozenset({"orchestrate", "task", "critique", "integrate", "verify"})


class RoleBudgetProfile(StrictModel):
    """Measured allowances, including provider reasoning tokens; no implicit escalation."""

    version: str = "role-budget-v1"
    measurement_source: str = Field(min_length=1)
    max_output_tokens: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)


class ArchitectureOptions(StrictModel):
    """Explicit interventions; baseline strategies receive no additional assistance."""

    phase_models: dict[PhaseName, ModelSpec] = Field(default_factory=dict[PhaseName, ModelSpec])
    candidate_submission: bool = False
    output_validation: bool = False
    constraint_feedback: bool = False
    output_retries: int = Field(default=1, ge=0)
    review_failure_policy: Literal["fail", "return_submitted_draft"] = "fail"
    review_protocol: Literal["replace", "decision-v1"] = "replace"
    team_policy: Literal["legacy-v1", "bounded-v1", "reliable-v2", "concurrent-v1"] = "reliable-v2"
    artifact_contract: Literal["legacy-v1", "public-v1"] = "public-v1"
    role_profiles: dict[PhaseName, RoleBudgetProfile] = Field(
        default_factory=dict[PhaseName, RoleBudgetProfile]
    )
    transport_policy: Literal["strict-v1", "json-normalize-v1"] = "strict-v1"
    provider_retries: int = Field(default=0, ge=0, le=3)
    return_reserve_seconds: float = Field(default=0.05, ge=0)
    hybrid_proposal_quorum: int = Field(default=2, ge=1, le=2)
    hybrid_plan_fanout: int = Field(default=2, ge=1, le=3)
    hybrid_max_decision_rounds: int = Field(default=3, ge=1, le=8)
    hybrid_pool_weights: dict[str, float] | None = None
    speculative_failure_policy: Literal["fail", "return_first_submitted"] = "fail"
    stage_weights: list[float] | None = None

    @model_validator(mode="after")
    def valid_stage_weights(self) -> ArchitectureOptions:
        if self.stage_weights is not None and (
            not self.stage_weights
            or any(not 0 < w <= 1 for w in self.stage_weights)
            or abs(sum(self.stage_weights) - 1) > 1e-6
        ):
            raise ValueError("stage_weights must be positive and sum to one")
        if self.hybrid_pool_weights is not None and (
            set(self.hybrid_pool_weights) != set(HYBRID_V2_POOL_ROLES)
            or any(not 0 < w <= 1 for w in self.hybrid_pool_weights.values())
            or abs(sum(self.hybrid_pool_weights.values()) - 1) > 1e-6
        ):
            raise ValueError(
                "hybrid_pool_weights must cover every hybrid_v2 role with positive values summing to one"
            )
        return self


class Answer(StrictModel):
    values: dict[str, Any]


ModelSpec.model_rebuild()
