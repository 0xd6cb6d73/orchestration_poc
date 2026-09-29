from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from poc.execution.sql_contracts import (
    Answer as Answer,
)
from poc.execution.sql_contracts import (
    ArchitectureOptions as ArchitectureOptions,
)
from poc.execution.sql_contracts import (
    Budget as Budget,
)
from poc.execution.sql_contracts import (
    ModelSpec as ModelSpec,
)
from poc.execution.sql_contracts import (
    StrictModel,
)
from poc.execution.sql_contracts import (
    TaskInput as TaskInput,
)


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
        for strategy, options in self.architecture_options.items():
            method = strategy.removesuffix("-json")
            team_phases = {
                "hierarchical_dag": {"plan", "solve"},
                "board_claim": {"plan", "solve"},
                "managed_pool": {"plan", "solve"},
                "speculative": {"proposal", "reconcile"},
                "hybrid_v1": {"proposal", "critique", "verify"},
                "hybrid_v2": {"orchestrate", "task", "critique", "integrate", "verify"},
                "hybrid_v2_1": {"orchestrate", "task", "critique", "integrate", "verify"},
                "hybrid_v2_2": {"orchestrate", "task", "critique", "integrate", "verify"},
            }
            allowed_phases = (
                {"draft", "review", "finalize"}
                if strategy in {"review", "review-json"}
                else {"solve", "finalize"}
            )
            if method in team_phases:
                allowed_phases = team_phases[method] | {"solve"}
                if method in {"hybrid_v2", "hybrid_v2_1", "hybrid_v2_2"}:
                    if options.stage_weights is not None:
                        raise ValueError(
                            "hybrid_v2 uses hybrid_pool_weights, not positional stage_weights"
                        )
                else:
                    expected = 5 if method == "hybrid_v1" else 3 if method == "speculative" else 2
                    if options.team_policy == "concurrent-v1":
                        from poc.execution.sql_concurrency import CONCURRENT_WEIGHTS

                        expected = len(CONCURRENT_WEIGHTS[method])
                    if options.stage_weights is not None and len(options.stage_weights) != expected:
                        raise ValueError("stage_weights length must match the strategy stage count")
            elif options.stage_weights is not None:
                raise ValueError("stage_weights require a team strategy")
            if options.speculative_failure_policy != "fail" and method != "speculative":
                raise ValueError("speculative failure policy requires speculative execution")
            if (
                options.team_policy == "legacy-v1"
                and (
                    options.stage_weights is not None
                    or set(options.phase_models) - {"solve"}
                    or options.speculative_failure_policy != "fail"
                )
                and method in team_phases
            ):
                raise ValueError("legacy team policy only supports the solve model binding")
            if set(options.phase_models) - allowed_phases:
                raise ValueError("phase_models contains phases unused by this strategy")
            if (
                options.review_failure_policy != "fail" or options.review_protocol != "replace"
            ) and strategy not in {
                "review",
                "review-json",
            }:
                raise ValueError("review options require review or review-json")
        return self


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
