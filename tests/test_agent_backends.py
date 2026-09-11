from pathlib import Path

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.control.runtime import Runtime
from poc.control.spawn_policy import SpawnDenied
from poc.execution.agent_executor import AgentExecutionRequest, AgentExecutorRegistry
from poc.execution.pydantic_ai_executor import model_from_role
from poc.execution.workflow_compiler import WorkflowBindingError
from poc.models import (
    AgentBackend,
    ApprovalRequest,
    ExecutionMode,
    ExecutionPolicy,
    InputBinding,
    RoleSpec,
    RunCreate,
    TaskSpec,
    Tier,
    WorkflowSpec,
)


async def _pydantic_agent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
                        "fact": "produced by Pydantic AI",
                        "manifest": {
                            "fixture": "incident_example",
                            "timestamp_timezone": "UTC",
                            "note": "Fixture timestamps are UTC.",
                        },
                    },
                    "completion_summary": "Produced a typed Pydantic AI result.",
                },
            )
        ]
    )


class _ExtensionExecutor:
    backend = "extension_runtime"

    def execute(self, request: AgentExecutionRequest) -> dict[str, object]:
        return {"task_id": request.task.id}


def _output_model(*outputs: dict[str, object]) -> FunctionModel:
    remaining = iter(outputs)

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "result": next(remaining),
                        "completion_summary": "Produced a schema-valid result.",
                    },
                )
            ]
        )

    return FunctionModel(respond)


def test_executor_registry_accepts_extension_backend_ids() -> None:
    registry = AgentExecutorRegistry()

    registry.register(_ExtensionExecutor())

    assert registry.get("extension_runtime").backend == "extension_runtime"


def test_openai_compatible_model_suffix_is_not_treated_as_provider_prefix() -> None:
    role = RoleSpec(
        role_id="openrouter-worker",
        tier=Tier.WORKER,
        system_prompt="Return a typed result.",
        provider="openai",
        model="google/gemma-4-26b-a4b-it:free",
    )

    assert model_from_role(role) == "openai:google/gemma-4-26b-a4b-it:free"

    prefixed = role.model_copy(update={"model": "openai:google/gemma-4-26b-a4b-it:free"})
    assert model_from_role(prefixed) == "openai:google/gemma-4-26b-a4b-it:free"


@pytest.mark.asyncio
async def test_workflow_can_mix_custom_and_pydantic_ai_agents(tmp_path: Path) -> None:
    test_model = FunctionModel(_pydantic_agent_model)
    configured_roles: list[RoleSpec] = []

    def model_factory(role: RoleSpec) -> FunctionModel:
        configured_roles.append(role)
        return test_model

    runtime = Runtime(tmp_path / "mixed", pydantic_model_factory=model_factory)
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="metrics_supervisor",
            plan_version=plan.version,
            stable_key="mixed-backend-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-mixed-backends",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["manifest_reader"],
            tasks=[
                TaskSpec(
                    id="custom-manifest",
                    role="manifest_reader",
                    agent_backend=AgentBackend.CUSTOM_PYTHON,
                    goal="Read the manifest with the custom Python agent.",
                    output_schema="ManifestFact",
                ),
                TaskSpec(
                    id="pydantic-manifest",
                    role="manifest_reader",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Produce a manifest fact with a Pydantic AI agent.",
                    output_schema="ManifestFact",
                ),
            ],
        )

        result = await runtime.runner.start(workflow)

        assert result["results"]["custom-manifest"]["outcome"] == "succeeded"
        assert result["results"]["pydantic-manifest"]["outcome"] == "succeeded"
        assert result["results"]["pydantic-manifest"]["result"] == {
            "fact": "produced by Pydantic AI",
            "manifest": {
                "fixture": "incident_example",
                "timestamp_timezone": "UTC",
                "note": "Fixture timestamps are UTC.",
            },
        }
        assert configured_roles[-1].provider == "test-provider"
        assert configured_roles[-1].model == "test-model"
        worker_backends = {
            agent["agent_backend"]
            for agent in runtime.db.list_agents(run["run_id"])
            if agent["tier"] == "worker"
        }
        assert worker_backends == {
            AgentBackend.CUSTOM_PYTHON,
            AgentBackend.PYDANTIC_AI,
        }
        pydantic_worker = next(
            agent
            for agent in runtime.db.list_agents(run["run_id"])
            if agent["agent_backend"] == AgentBackend.PYDANTIC_AI
        )
        assert pydantic_worker["agent_provider"] == "test-provider"
        assert pydantic_worker["agent_model"] == "test-model"
        pydantic_completion = next(
            event
            for event in runtime.db.events(run["run_id"])
            if event["event_type"] == "worker.completed"
            and event["data"].get("agent_backend") == AgentBackend.PYDANTIC_AI
        )
        assert pydantic_completion["data"]["outcome"] == "succeeded"
        assert pydantic_completion["data"]["tool_calls"] == 1
        assert any(
            event["event_type"] == "tool.executed"
            and event["data"]["tool"] == "read_manifest"
            and event["actor_id"] == pydantic_completion["actor_id"]
            for event in runtime.db.events(run["run_id"])
        )
        with pytest.raises(SpawnDenied, match="cannot change agent backend"):
            runtime.spawns.spawn(
                run_id=run["run_id"],
                parent=supervisor,
                child_role="manifest_reader",
                plan_version=plan.version,
                stable_key=f"{workflow.workflow_id}-r1-pydantic-manifest",
                agent_backend=AgentBackend.CUSTOM_PYTHON,
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_swarm_policy_selects_materialized_worker_backend(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path / "scheduler")
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="metrics_supervisor",
            plan_version=plan.version,
            stable_key="scheduler-backend-supervisor",
        )
        policy = ExecutionPolicy(
            mode=ExecutionMode.BOARD_CLAIM,
            agent_backend=AgentBackend.PYDANTIC_AI,
            max_workers=1,
            allowed_roles=frozenset({"manifest_reader"}),
        )
        handle = await runtime.submit_execution(
            run_id=run["run_id"],
            owner_suborchestrator_id=supervisor.agent_instance_id,
            goal_ref="pydantic-ai-board",
            policy=policy,
        )

        workers = runtime.capacity.reconcile(handle, {"manifest_reader": 1})

        assert len(workers) == 1
        assert workers[0].agent_backend == AgentBackend.PYDANTIC_AI
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_pydantic_output_contract_rejects_flat_pattern_result(tmp_path: Path) -> None:
    async def flat_pattern_result(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "result": {
                            "counts": {"cache_miss": 2},
                            "total": 2,
                            "interpretation": "flat tool response",
                        },
                        "completion_summary": "Returned the tool response directly.",
                    },
                )
            ]
        )

    runtime = Runtime(
        tmp_path / "invalid-output",
        pydantic_model_factory=lambda role: FunctionModel(flat_pattern_result),
    )
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="evidence_supervisor",
            plan_version=plan.version,
            stable_key="invalid-output-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-invalid-output",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["log_pattern_counter"],
            tasks=[
                TaskSpec(
                    id="count",
                    role="log_pattern_counter",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Count patterns.",
                    output_schema="PatternCount",
                )
            ],
        )

        result = await runtime.runner.start(workflow)

        worker = result["results"]["count"]
        assert worker["outcome"] == "failed"
        assert "output retries" in worker["completion_summary"]
        assert not any(
            event["event_type"] == "artifact.created"
            and event["data"].get("producer_task_id") == "count"
            for event in runtime.db.events(run["run_id"])
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_pydantic_tools_are_hidden_when_budget_is_spent(tmp_path: Path) -> None:
    async def use_every_available_tool(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        if info.function_tools:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_allowed_tool",
                        {
                            "tool_name": "read_metric_slice",
                            "arguments": {
                                "start": "2026-04-17T12:00:00Z",
                                "end": "2026-04-17T13:09:00Z",
                            },
                        },
                    )
                ]
            )
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "result": {
                            "baseline": {
                                "start": "2026-04-17T12:50:00Z",
                                "end": "2026-04-17T13:00:00Z",
                                "service": "checkout",
                            },
                            "incident": {
                                "start": "2026-04-17T13:00:00",
                                "end": "2026-04-17T13:10:00Z",
                                "service": "checkout",
                            },
                            "sample_count": 70,
                            "timezone_note": "The incident boundary needs UTC validation.",
                        },
                        "completion_summary": "Stopped after the bounded evidence reads.",
                    },
                )
            ]
        )

    runtime = Runtime(
        tmp_path / "bounded-tools",
        pydantic_model_factory=lambda role: FunctionModel(use_every_available_tool),
    )
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="metrics_supervisor",
            plan_version=plan.version,
            stable_key="bounded-tools-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-bounded-tools",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["window_selector"],
            tasks=[
                TaskSpec(
                    id="windows",
                    role="window_selector",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Select comparable windows.",
                    output_schema="WindowSelection",
                )
            ],
        )

        result = await runtime.runner.start(workflow)

        assert result["results"]["windows"]["outcome"] == "succeeded"
        completion = next(
            event
            for event in runtime.db.events(run["run_id"])
            if event["event_type"] == "worker.completed"
        )
        assert completion["data"]["tool_calls"] == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_pydantic_percentile_result_binds_into_downstream_task(tmp_path: Path) -> None:
    model = _output_model(
        {
            "result": {
                "percentile": 95,
                "value": 401,
                "units": "ms",
                "sample_count": 10,
                "method": "linear interpolation",
            },
            "window": {
                "start": "2026-04-17T13:00:00Z",
                "end": "2026-04-17T13:10:00Z",
                "service": "checkout",
            },
        },
        {
            "content": {
                "baseline_p95_ms": 302,
                "incident_p95_ms": 401,
                "absolute_increase_ms": 99,
                "ratio": 1.33,
                "claim": "Checkout latency increased.",
            },
            "published": {
                "artifact_id": "artifact-test-comparison",
                "sha256": "test-sha256",
                "media_type": "application/json",
            },
        },
    )
    runtime = Runtime(tmp_path / "percentile-binding", pydantic_model_factory=lambda role: model)
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="metrics_supervisor",
            plan_version=plan.version,
            stable_key="percentile-binding-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-percentile-binding",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["percentile_calculator", "metric_comparator"],
            tasks=[
                TaskSpec(
                    id="percentile",
                    role="percentile_calculator",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Return a percentile.",
                    output_schema="PercentileResult",
                ),
                TaskSpec(
                    id="compare",
                    role="metric_comparator",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Consume the percentile.",
                    depends_on=["percentile"],
                    input_bindings=[
                        InputBinding(source_task="percentile", field="result", target="baseline")
                    ],
                    output_schema="MetricComparison",
                ),
            ],
        )

        result = await runtime.runner.start(workflow)

        assert result["results"]["percentile"]["outcome"] == "succeeded"
        assert result["results"]["compare"]["outcome"] == "succeeded"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_pydantic_pattern_count_binds_into_deployment_task(tmp_path: Path) -> None:
    pattern_counts = {
        "counts": {"cache_miss": 6, "payment_timeout": 2},
        "total": 8,
        "interpretation": "cache_miss is incident-correlated",
    }
    model = _output_model(
        {"pattern_counts": pattern_counts},
        {
            "deployment_match": {
                "matches": [
                    {
                        "deployment_id": "deploy-checkout",
                        "service": "checkout",
                        "version": "2026.04.17.2",
                        "timestamp": "2026-04-17T12:55:00Z",
                        "change": "Enabled pricing-cache-v2",
                        "minutes_before_incident": 5,
                    }
                ],
                "count": 1,
            },
            "pattern_counts": pattern_counts,
        },
    )
    runtime = Runtime(tmp_path / "pattern-binding", pydantic_model_factory=lambda role: model)
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="evidence_supervisor",
            plan_version=plan.version,
            stable_key="pattern-binding-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-pattern-binding",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["log_pattern_counter", "deployment_matcher"],
            tasks=[
                TaskSpec(
                    id="count",
                    role="log_pattern_counter",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Return pattern counts.",
                    output_schema="PatternCount",
                ),
                TaskSpec(
                    id="deployment",
                    role="deployment_matcher",
                    agent_backend=AgentBackend.PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Consume pattern counts.",
                    depends_on=["count"],
                    input_bindings=[
                        InputBinding(
                            source_task="count",
                            field="pattern_counts",
                            target="pattern_counts",
                        )
                    ],
                    output_schema="DeploymentMatch",
                ),
            ],
        )

        result = await runtime.runner.start(workflow)

        assert result["results"]["count"]["outcome"] == "succeeded"
        assert result["results"]["deployment"]["outcome"] == "succeeded"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_workflow_rejects_binding_missing_from_output_contract(tmp_path: Path) -> None:
    runtime = Runtime(tmp_path / "invalid-binding")
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="metrics_supervisor",
            plan_version=plan.version,
            stable_key="invalid-binding-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-invalid-binding",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["window_selector", "percentile_calculator"],
            tasks=[
                TaskSpec(
                    id="windows",
                    role="window_selector",
                    goal="Return windows.",
                    output_schema="WindowSelection",
                ),
                TaskSpec(
                    id="percentile",
                    role="percentile_calculator",
                    goal="Consume a window.",
                    depends_on=["windows"],
                    input_bindings=[
                        InputBinding(
                            source_task="windows",
                            field="missing_window",
                            target="window",
                        )
                    ],
                    output_schema="PercentileResult",
                ),
            ],
        )

        with pytest.raises(WorkflowBindingError, match="missing_window"):
            await runtime.runner.start(workflow)
    finally:
        await runtime.close()
