from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskInput(StrictModel):
    """The complete public interface; references never cross the adapter boundary."""

    prompt: str
    tables: dict[str, list[dict[str, Any]]]


class TaskCase(StrictModel):
    id: str
    family: str
    version: str = "1"
    seed: int
    split: Literal["dev", "test"]
    difficulty: Literal["standard", "hard", "stress"]
    input: TaskInput
    expected: dict[str, Any]

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


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


class ArchitectureOptions(StrictModel):
    """Explicit interventions; baseline strategies receive no additional assistance."""

    candidate_submission: bool = False
    output_validation: bool = False
    constraint_feedback: bool = False
    output_retries: int = Field(default=1, ge=0)


class ModelSpec(StrictModel):
    name: str
    model_class: Literal["7B", "27B", "100B", "frontier-flash", "baseline"]
    model: str
    base_url_env: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    settings: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0})
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)


class Matrix(StrictModel):
    models: list[ModelSpec] = Field(min_length=1)
    strategies: list[str] = Field(default_factory=lambda: ["single", "review"], min_length=1)
    families: list[str] = Field(default_factory=lambda: ["scheduling", "routing"], min_length=1)
    seeds: list[int] = Field(default_factory=lambda: list(range(10)), min_length=1)
    split: Literal["dev", "test"] = "dev"
    difficulty: Literal["standard", "hard", "stress"] = "standard"
    repetitions: int = Field(default=3, ge=1)
    budget: Budget = Field(default_factory=Budget)
    order_seed: int = 1729
    max_concurrency: int = Field(default=4, ge=1)
    architecture_options: dict[str, ArchitectureOptions] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_axes(self) -> Matrix:
        for axis in ([m.name for m in self.models], self.strategies, self.families, self.seeds):
            if len(axis) != len(set(axis)):
                raise ValueError("matrix axes must contain unique values")
        if set(self.architecture_options) - set(self.strategies):
            raise ValueError("architecture_options must refer to configured strategies")
        return self


class Answer(StrictModel):
    values: dict[str, Any]


def grade(case: TaskCase, answer: Answer) -> dict[str, float]:
    """Exact typed values, with partial credit; unknown keys penalize precision."""

    def canonical(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    correct = sum(
        key in answer.values and canonical(answer.values[key]) == canonical(value)
        for key, value in case.expected.items()
    )
    denominator = len(set(case.expected) | set(answer.values))
    return {
        "exact": float(canonical(case.expected) == canonical(answer.values)),
        "fraction_correct": correct / max(1, denominator),
    }
