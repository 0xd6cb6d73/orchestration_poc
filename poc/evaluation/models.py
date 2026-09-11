from __future__ import annotations

from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from poc.models import AgentBackend, Outcome


class AgentEvaluationInput(BaseModel):
    """One framework-neutral worker task to execute during an experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    inputs: dict[str, Any] = Field(default_factory=dict[str, Any])
    acceptance_criteria: list[str] = Field(default_factory=list[str])


class AgentEvaluationExpected(BaseModel):
    """Golden output and trajectory constraints for an evaluation case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: Outcome = Outcome.SUCCEEDED
    result_contains: dict[str, Any] = Field(default_factory=dict[str, Any])
    required_tools: frozenset[str] = Field(default_factory=frozenset[str])
    forbidden_tools: frozenset[str] = Field(default_factory=frozenset[str])
    tool_arguments_contain: dict[str, dict[str, Any]] = Field(
        default_factory=dict[str, dict[str, Any]]
    )
    max_tool_calls: int | None = Field(default=None, ge=0)


class AgentToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str
    arguments: dict[str, Any]
    status: str


class AgentEvaluationOutput(BaseModel):
    """Normalized result returned to Pydantic Evals for grading."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    provider: str
    model: str
    system_prompt_sha256: str
    allowed_tools: list[str]
    outcome: Outcome
    result: dict[str, Any] = Field(default_factory=dict[str, Any])
    completion_summary: str = ""
    tool_calls: list[AgentToolCall] = Field(default_factory=list[AgentToolCall])
    evidence_artifacts: list[str] = Field(default_factory=list[str])
    error: str | None = None


class AgentEvaluationVariant(BaseModel):
    """Prompt, model, and tool configuration evaluated as one experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    agent_backend: str = Field(default=AgentBackend.CUSTOM_PYTHON, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    system_prompt: str | None = None
    allowed_tools_by_role: dict[str, frozenset[str]] = Field(
        default_factory=dict[str, frozenset[str]]
    )
    execution_limits: dict[str, int] = Field(default_factory=dict[str, int])

    def experiment_metadata(self) -> dict[str, Any]:
        return {
            "variant": self.name,
            "agent_backend": self.agent_backend,
            "provider": self.provider,
            "model": self.model,
            "system_prompt_sha256": (
                sha256(self.system_prompt.encode()).hexdigest() if self.system_prompt else None
            ),
            "allowed_tools_by_role": {
                role: sorted(tools) for role, tools in self.allowed_tools_by_role.items()
            },
            "execution_limits": self.execution_limits,
        }
