from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

from pydantic_evals import Case, Dataset

from poc.evaluation.evaluators import AgentEvaluator, default_evaluators
from poc.evaluation.models import (
    AgentEvaluationExpected,
    AgentEvaluationInput,
    AgentEvaluationOutput,
)

AgentEvaluationCase: TypeAlias = Case[
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationExpected,
]
AgentEvaluationDataset: TypeAlias = Dataset[
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationExpected,
]
AgentEvaluationDatasetFactory: TypeAlias = Callable[[], AgentEvaluationDataset]


def create_agent_dataset(
    name: str,
    cases: Sequence[AgentEvaluationCase],
    *,
    evaluators: Sequence[AgentEvaluator] | None = None,
) -> AgentEvaluationDataset:
    """Create an extensible dataset with the standard agent quality gates."""
    return Dataset(
        name=name,
        cases=cases,
        evaluators=list(evaluators) if evaluators is not None else default_evaluators(),
    )


def incident_worker_dataset() -> AgentEvaluationDataset:
    """Deterministic golden cases covering output quality and tool trajectory."""
    return create_agent_dataset(
        "incident-workers",
        [
            Case(
                name="manifest-timezone",
                inputs=AgentEvaluationInput(
                    role_id="manifest_reader",
                    goal="Read only the fixture timezone declaration.",
                    output_schema="ManifestFact",
                    acceptance_criteria=["timezone statement is explicit"],
                ),
                metadata=AgentEvaluationExpected(
                    result_contains={"fact": "timezone-less fixture timestamps are UTC"},
                    required_tools=frozenset({"read_manifest"}),
                    forbidden_tools=frozenset({"write_artifact"}),
                    max_tool_calls=1,
                ),
            ),
            Case(
                name="deployment-match",
                inputs=AgentEvaluationInput(
                    role_id="deployment_matcher",
                    goal="Match the checkout deployment to the incident window.",
                    output_schema="DeploymentMatch",
                    acceptance_criteria=["deployment proximity in minutes is present"],
                ),
                metadata=AgentEvaluationExpected(
                    result_contains={
                        "deployment_match": {
                            "count": 1,
                            "matches": [{"version": "2026.04.17.2", "minutes_before_incident": 5}],
                        }
                    },
                    required_tools=frozenset({"read_deployment_record"}),
                    forbidden_tools=frozenset({"write_artifact"}),
                    tool_arguments_contain={
                        "read_deployment_record": {
                            "incident_start": "2026-04-17T13:00:00Z",
                            "lookback_minutes": 30,
                        }
                    },
                    max_tool_calls=1,
                ),
            ),
        ],
    )


class AgentEvaluationDatasetRegistry:
    """Named dataset extension point used by the CLI and automation."""

    def __init__(self) -> None:
        self._factories: dict[str, AgentEvaluationDatasetFactory] = {}

    def register(
        self,
        name: str,
        factory: AgentEvaluationDatasetFactory,
        *,
        replace: bool = False,
    ) -> None:
        if name in self._factories and not replace:
            raise ValueError(f"evaluation dataset already registered: {name!r}")
        self._factories[name] = factory

    def get(self, name: str) -> AgentEvaluationDataset:
        try:
            return self._factories[name]()
        except KeyError as exc:
            raise ValueError(f"unknown evaluation dataset: {name!r}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_dataset_registry() -> AgentEvaluationDatasetRegistry:
    registry = AgentEvaluationDatasetRegistry()
    registry.register("incident-workers", incident_worker_dataset)
    return registry
