from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.control.runtime import Runtime
from poc.control.spawn_policy import SpawnDenied
from poc.execution.agent_executor import AgentExecutionRequest, AgentExecutorRegistry
from poc.execution.pydantic_ai_executor import model_from_role
from poc.execution.workflow_compiler import WorkflowBindingError
from poc.models import (
    AgentBackend,
    AgentRuntimeConfig,
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


def _publishing_output_model(content: Mapping[str, Any]) -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "execute_allowed_tool"
        ]
        if not tool_returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "execute_allowed_tool",
                        {
                            "tool_name": "write_artifact",
                            "arguments": {
                                "content": content,
                                "media_type": "application/json",
                            },
                        },
                    )
                ]
            )
        publication = cast(dict[str, object], tool_returns[-1].content)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "result": {"content": content, "published": publication},
                        "completion_summary": "Produced a candidate assessment.",
                    },
                )
            ]
        )

    return FunctionModel(respond)


def _semantic_review_model(
    criteria: list[str],
    *,
    internally_consistent: bool = True,
    contradictions: list[str] | None = None,
) -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "internally_consistent": internally_consistent,
                        "criteria": [
                            {
                                "criterion": criterion,
                                "status": "satisfied",
                                "evidence_refs": [],
                                "rationale": "The supplied result and evidence satisfy it.",
                            }
                            for criterion in criteria
                        ],
                        "contradictions": contradictions or [],
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


def test_semantic_pydantic_backend_requires_model_configuration() -> None:
    with pytest.raises(ValueError, match="semantic_pydantic_ai requires a model"):
        AgentRuntimeConfig(backend=AgentBackend.SEMANTIC_PYDANTIC_AI)

    configured = AgentRuntimeConfig(
        backend=AgentBackend.SEMANTIC_PYDANTIC_AI,
        provider="openai",
        model="gpt-5",
    )
    assert configured.backend == AgentBackend.SEMANTIC_PYDANTIC_AI


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


@pytest.mark.asyncio
async def test_semantic_pydantic_backend_accepts_independently_reviewed_result(
    tmp_path: Path,
) -> None:
    criterion = "timezone statement is explicit"
    runtime = Runtime(
        tmp_path / "semantic-valid",
        pydantic_model_factory=lambda role: _output_model(
            {
                "fact": "timezone-less fixture timestamps are UTC",
                "manifest": {
                    "fixture": "incident_example",
                    "timestamp_timezone": "UTC",
                    "note": "Fixture timestamps are UTC.",
                },
            }
        ),
        semantic_pydantic_model_factory=lambda role: _semantic_review_model([criterion]),
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
            stable_key="semantic-valid-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-semantic-valid",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["manifest_reader"],
            tasks=[
                TaskSpec(
                    id="manifest",
                    role="manifest_reader",
                    agent_backend=AgentBackend.SEMANTIC_PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Read the fixture timezone declaration.",
                    output_schema="ManifestFact",
                    acceptance_criteria=[criterion],
                )
            ],
        )

        result = await runtime.runner.start(workflow)

        worker = result["results"]["manifest"]
        assert worker["outcome"] == "succeeded"
        assert worker["acceptance_checks"] == [
            {
                "criterion": criterion,
                "passed": True,
                "semantic_status": "satisfied",
                "evidence_refs": [],
                "rationale": "The supplied result and evidence satisfy it.",
            }
        ]
        semantic_event = next(
            event
            for event in runtime.db.events(run["run_id"])
            if event["event_type"] == "agent.semantic_validation_completed"
        )
        assert semantic_event["data"]["valid"] is True
        assert semantic_event["data"]["stage"] == "model_review"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_semantic_rejection_is_returned_to_worker_for_revision(tmp_path: Path) -> None:
    criterion = "timezone fact matches the supplied manifest"
    primary_outputs = iter(
        [
            {
                "fact": "timezone-less fixture timestamps are local time",
                "manifest": {
                    "fixture": "incident_example",
                    "timestamp_timezone": "UTC",
                    "note": "Fixture timestamps are UTC.",
                },
            },
            {
                "fact": "timezone-less fixture timestamps are UTC",
                "manifest": {
                    "fixture": "incident_example",
                    "timestamp_timezone": "UTC",
                    "note": "Fixture timestamps are UTC.",
                },
            },
        ]
    )
    worker_messages: list[list[ModelMessage]] = []

    async def revising_worker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        worker_messages.append(messages)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "result": next(primary_outputs),
                        "completion_summary": "Reassessed the timezone fact.",
                    },
                )
            ]
        )

    review_payloads: list[dict[str, Any]] = [
        {
            "internally_consistent": True,
            "criteria": [
                {
                    "criterion": criterion,
                    "status": "violated",
                    "evidence_refs": [],
                    "rationale": "The fact conflicts with the UTC manifest value.",
                }
            ],
            "contradictions": [],
        },
        {
            "internally_consistent": True,
            "criteria": [
                {
                    "criterion": criterion,
                    "status": "satisfied",
                    "evidence_refs": [],
                    "rationale": "The revised fact matches the UTC manifest value.",
                }
            ],
            "contradictions": [],
        },
    ]
    reviews: Iterator[dict[str, Any]] = iter(review_payloads)

    async def reviewing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, next(reviews))])

    worker_model = FunctionModel(revising_worker)
    reviewer_model = FunctionModel(reviewing_model)
    runtime = Runtime(
        tmp_path / "semantic-revision",
        pydantic_model_factory=lambda role: worker_model,
        semantic_pydantic_model_factory=lambda role: reviewer_model,
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
            stable_key="semantic-revision-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-semantic-revision",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["manifest_reader"],
            tasks=[
                TaskSpec(
                    id="manifest",
                    role="manifest_reader",
                    agent_backend=AgentBackend.SEMANTIC_PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Report the fixture timezone.",
                    output_schema="ManifestFact",
                    acceptance_criteria=[criterion],
                )
            ],
        )

        result = await runtime.runner.start(workflow)

        worker = result["results"]["manifest"]
        assert worker["outcome"] == "succeeded"
        assert worker["result"]["fact"] == "timezone-less fixture timestamps are UTC"
        assert len(worker_messages) == 2
        assert "semantic_validation_issues" in str(worker_messages[1])
        events = runtime.db.events(run["run_id"])
        revision = next(
            event for event in events if event["event_type"] == "agent.output_revision_requested"
        )
        assert "criterion" in revision["data"]["reason"]
        semantic_events = [
            event
            for event in events
            if event["event_type"] == "agent.semantic_validation_completed"
        ]
        assert [event["data"]["valid"] for event in semantic_events] == [False, True]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_semantic_pydantic_backend_rejects_internally_inconsistent_critique(
    tmp_path: Path,
) -> None:
    acceptance = "assumption and evidence check are explicit"
    consistency = (
        "The supported verdict, evidence checks, caveat, and completion summary are mutually "
        "consistent."
    )
    evidence_strength = (
        "The verdict distinguishes evidence that contradicts the claim from evidence that is "
        "merely insufficient to establish causation."
    )
    draft = {
        "hypothesis_key": "cache_queueing",
        "claim": "A cold cache is a plausible cause of the latency regression.",
        "confidence": "medium",
        "alternatives": ["payment provider latency"],
    }
    critique = {
        "hypothesis_key": "cache_queueing",
        "claim": draft["claim"],
        "supported": False,
        "checks": ["The cache evidence supports the candidate."],
        "caveat": "The evidence supports the candidate.",
    }
    runtime = Runtime(
        tmp_path / "semantic-invalid",
        pydantic_model_factory=lambda role: _publishing_output_model(critique),
        semantic_pydantic_model_factory=lambda role: _semantic_review_model(
            [acceptance, consistency, evidence_strength],
            internally_consistent=False,
            contradictions=["supported=false conflicts with the affirmative evidence checks"],
        ),
    )
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="reporting_supervisor",
            plan_version=plan.version,
            stable_key="semantic-invalid-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-semantic-invalid",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["claim_checker"],
            tasks=[
                TaskSpec(
                    id="critique",
                    role="claim_checker",
                    agent_backend=AgentBackend.SEMANTIC_PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Challenge one candidate against the original evidence.",
                    output_schema="CheckedClaim",
                    acceptance_criteria=[acceptance],
                    static_inputs={
                        "draft": draft,
                        "hypothesis_key": "cache_queueing",
                    },
                )
            ],
        )

        result = await runtime.runner.start(workflow)

        worker = result["results"]["critique"]
        assert worker["outcome"] == "failed"
        assert worker["output_artifact"] is None
        assert "internally inconsistent" in worker["completion_summary"]
        semantic_event = next(
            event
            for event in runtime.db.events(run["run_id"])
            if event["event_type"] == "agent.semantic_validation_completed"
        )
        assert semantic_event["data"]["valid"] is False
        assert semantic_event["data"]["contradictions"] == [
            "supported=false conflicts with the affirmative evidence checks"
        ]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_semantic_pydantic_backend_rejects_changed_hypothesis_focus(tmp_path: Path) -> None:
    semantic_reviewer_called = False

    def semantic_factory(role: RoleSpec) -> FunctionModel:
        nonlocal semantic_reviewer_called
        semantic_reviewer_called = True
        return _semantic_review_model([])

    draft = {
        "hypothesis_key": "checkout_latency_increase",
        "claim": "Checkout latency increased.",
        "confidence": "high",
        "alternatives": ["cache queueing"],
    }
    runtime = Runtime(
        tmp_path / "semantic-focus-drift",
        pydantic_model_factory=lambda role: _publishing_output_model(draft),
        semantic_pydantic_model_factory=semantic_factory,
    )
    try:
        run, _ = runtime.create_run(RunCreate())
        plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
        main = runtime.db.get_agent(run["main_agent_id"])
        assert main is not None
        supervisor = runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=main,
            child_role="reporting_supervisor",
            plan_version=plan.version,
            stable_key="semantic-focus-supervisor",
        )
        workflow = WorkflowSpec(
            workflow_id=f"{run['run_id']}-semantic-focus",
            run_id=run["run_id"],
            owner=supervisor.agent_instance_id,
            approved_plan_version=plan.version,
            authorized_worker_roles=["claim_drafter"],
            tasks=[
                TaskSpec(
                    id="draft",
                    role="claim_drafter",
                    agent_backend=AgentBackend.SEMANTIC_PYDANTIC_AI,
                    agent_provider="test-provider",
                    agent_model="test-model",
                    goal="Draft the assigned payment-timeout hypothesis.",
                    output_schema="DraftClaim",
                    static_inputs={"hypothesis_focus": "payment_timeout"},
                )
            ],
        )

        result = await runtime.runner.start(workflow)

        worker = result["results"]["draft"]
        assert worker["outcome"] == "failed"
        assert "does not match assigned focus" in worker["completion_summary"]
        assert semantic_reviewer_called is False
        semantic_event = next(
            event
            for event in runtime.db.events(run["run_id"])
            if event["event_type"] == "agent.semantic_validation_completed"
        )
        assert semantic_event["data"]["stage"] == "deterministic"
    finally:
        await runtime.close()
