from pathlib import Path

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case

from poc.evaluation import (
    AgentEvaluationExpected,
    AgentEvaluationInput,
    AgentEvaluationRunner,
    AgentEvaluationVariant,
    OrchestrationBenchmarkRunner,
    OrchestrationBenchmarkVariant,
    create_agent_dataset,
    incident_worker_dataset,
    load_report,
    report_passed,
    save_report,
)
from poc.models import AgentBackend, ExecutionMode, RoleSpec


@pytest.mark.asyncio
async def test_offline_agent_evaluation_grades_results_and_tool_trajectory(
    tmp_path: Path,
) -> None:
    runner = AgentEvaluationRunner(incident_worker_dataset(), work_dir=tmp_path / "runs")

    report = await runner.evaluate(
        AgentEvaluationVariant(name="custom-baseline"),
        progress=False,
    )

    assert report_passed(report)
    assert len(report.cases) == 2
    assert all(
        len(case.assertions) == 7 and all(assertion.value for assertion in case.assertions.values())
        for case in report.cases
    )
    destination = tmp_path / "reports" / "baseline.json"
    save_report(report, destination)
    loaded = load_report(destination)
    assert loaded.name == "custom-baseline"
    assert loaded.experiment_metadata == report.experiment_metadata


async def _manifest_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "execute_allowed_tool",
                    {"tool_name": "read_manifest", "arguments": {}},
                )
            ]
        )
    return ModelResponse(
        parts=[
            ToolCallPart(
                info.output_tools[0].name,
                {
                    "result": {
                        "fact": "timezone-less fixture timestamps are UTC",
                        "manifest": {
                            "fixture": "incident_example",
                            "timestamp_timezone": "UTC",
                            "note": "Fixture timestamps are UTC.",
                        },
                    },
                    "completion_summary": "Read the manifest through the approved tool.",
                },
            )
        ]
    )


@pytest.mark.asyncio
async def test_evaluation_variant_overrides_prompt_model_and_tools(tmp_path: Path) -> None:
    configured_roles: list[RoleSpec] = []
    test_model = FunctionModel(_manifest_model)

    def model_factory(role: RoleSpec) -> FunctionModel:
        configured_roles.append(role)
        return test_model

    dataset = create_agent_dataset(
        "manifest-only",
        [
            Case(
                name="manifest",
                inputs=AgentEvaluationInput(
                    role_id="manifest_reader",
                    goal="Read the timezone declaration.",
                    output_schema="ManifestFact",
                ),
                metadata=AgentEvaluationExpected(
                    result_contains={"fact": "timezone-less fixture timestamps are UTC"},
                    required_tools=frozenset({"read_manifest"}),
                    max_tool_calls=1,
                ),
            )
        ],
    )
    variant = AgentEvaluationVariant(
        name="prompt-model-tools-v2",
        agent_backend=AgentBackend.PYDANTIC_AI,
        provider="test-provider",
        model="test-model-v2",
        system_prompt="Use only explicit evidence from the approved manifest tool.",
        allowed_tools_by_role={"manifest_reader": frozenset({"read_manifest"})},
        execution_limits={"max_tool_calls": 1},
    )
    runner = AgentEvaluationRunner(
        dataset,
        work_dir=tmp_path / "runs",
        pydantic_model_factory=model_factory,
    )

    report = await runner.evaluate(variant, progress=False)

    assert report_passed(report)
    assert len(configured_roles) == 1
    configured = configured_roles[0]
    assert configured.provider == "test-provider"
    assert configured.model == "test-model-v2"
    assert configured.system_prompt == variant.system_prompt
    assert configured.allowed_tools == ["read_manifest"]
    assert configured.execution_limits["max_tool_calls"] == 1
    assert report.experiment_metadata is not None
    assert report.experiment_metadata["system_prompt_sha256"] is not None
    assert "system_prompt" not in report.experiment_metadata
    assert (
        report.cases[0].output.system_prompt_sha256
        == report.experiment_metadata["system_prompt_sha256"]
    )
    assert report.cases[0].output.allowed_tools == ["read_manifest"]


@pytest.mark.asyncio
async def test_orchestration_benchmark_runs_complete_nontrivial_workload(
    tmp_path: Path,
) -> None:
    runner = OrchestrationBenchmarkRunner(work_dir=tmp_path / "benchmark-runs")

    report = await runner.evaluate(
        [
            OrchestrationBenchmarkVariant(
                name="managed-pool-offline",
                execution_mode=ExecutionMode.MANAGED_POOL,
            )
        ]
    )

    assert len(report.trials) == 1
    trial = report.trials[0]
    assert trial.passed
    assert trial.execution_mode == ExecutionMode.MANAGED_POOL
    assert trial.workflow_count >= 4
    assert trial.task_count >= 12
    assert trial.worker_count >= 12
    assert trial.tool_call_count >= 14
    assert trial.final_report is None
    assert trial.final_report_sha256 is not None
    assert all(trial.quality_checks.values())
    assert report.summaries[0].completion_rate == 1
    assert report.summaries[0].pass_rate == 1
