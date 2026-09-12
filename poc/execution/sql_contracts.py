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
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)


PhaseName: TypeAlias = Literal["solve", "draft", "review", "finalize"]


class ArchitectureOptions(StrictModel):
    """Explicit interventions; baseline strategies receive no additional assistance."""

    phase_models: dict[PhaseName, ModelSpec] = Field(default_factory=dict[PhaseName, ModelSpec])
    candidate_submission: bool = False
    output_validation: bool = False
    constraint_feedback: bool = False
    output_retries: int = Field(default=1, ge=0)
    review_failure_policy: Literal["fail", "return_submitted_draft"] = "fail"
    review_protocol: Literal["replace", "decision-v1"] = "replace"


class Answer(StrictModel):
    values: dict[str, Any]
