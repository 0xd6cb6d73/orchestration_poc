from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from poc.models import (
    AgentBackend,
    ExecutionMode,
    HybridConfig,
    SwarmStrategy,
    is_pydantic_ai_backend,
)


class OrchestrationBenchmarkVariant(BaseModel):
    """One orchestration/runtime combination to exercise against a workload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    execution_mode: ExecutionMode
    swarm_strategy: SwarmStrategy = SwarmStrategy.BOARD
    agent_backend: str = Field(default=AgentBackend.CUSTOM_PYTHON, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    model_settings: dict[str, object] = Field(default_factory=dict[str, object])
    execution_options: dict[str, object] = Field(default_factory=dict[str, object])
    hybrid: HybridConfig = Field(default_factory=HybridConfig)

    @model_validator(mode="after")
    def validate_runtime(self) -> OrchestrationBenchmarkVariant:
        if is_pydantic_ai_backend(self.agent_backend):
            provider_from_model = (
                self.model.split(":", 1)[0] if self.model and ":" in self.model else None
            )
            if self.model is None:
                raise ValueError(f"{self.agent_backend} benchmark variants require a model")
            if self.provider is None and provider_from_model is None:
                raise ValueError(
                    f"{self.agent_backend} benchmark variants require a provider or provider:model"
                )
        if (
            self.swarm_strategy == SwarmStrategy.HYBRID_V1
            and self.execution_mode != ExecutionMode.BOARD_CLAIM
        ):
            raise ValueError("hybrid_v1 requires board_claim execution mode")
        return self


class OrchestrationWorkload(BaseModel):
    """A multi-agent workload and its deterministic, inspectable quality gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    required_report_concepts: dict[str, tuple[str, ...]]
    minimum_workflows: int = Field(ge=1)
    minimum_tasks: int = Field(ge=1)
    minimum_workers: int = Field(ge=1)
    required_tools: frozenset[str] = Field(default_factory=frozenset[str])


class TokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class OrchestrationBenchmarkTrial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workload: str
    variant: str
    repetition: int = Field(ge=1)
    execution_mode: ExecutionMode
    swarm_strategy: SwarmStrategy
    agent_backend: str
    provider: str | None
    model: str | None
    status: str
    passed: bool
    elapsed_seconds: float = Field(ge=0)
    quality_checks: dict[str, bool]
    workflow_count: int = Field(ge=0)
    task_count: int = Field(ge=0)
    successful_task_count: int = Field(ge=0)
    worker_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)
    failed_tool_call_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    final_report_sha256: str | None = None
    final_report: str | None = None
    error: str | None = None


class OrchestrationBenchmarkSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str
    execution_mode: ExecutionMode
    swarm_strategy: SwarmStrategy
    trials: int = Field(ge=1)
    completed: int = Field(ge=0)
    passed: int = Field(ge=0)
    completion_rate: float = Field(ge=0, le=1)
    pass_rate: float = Field(ge=0, le=1)
    mean_elapsed_seconds: float = Field(ge=0)
    p50_elapsed_seconds: float = Field(ge=0)
    p95_elapsed_seconds: float = Field(ge=0)
    mean_tool_calls: float = Field(ge=0)
    mean_workers: float = Field(ge=0)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)


class OrchestrationBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    workload: OrchestrationWorkload
    variants: list[OrchestrationBenchmarkVariant]
    trials: list[OrchestrationBenchmarkTrial]
    summaries: list[OrchestrationBenchmarkSummary]
