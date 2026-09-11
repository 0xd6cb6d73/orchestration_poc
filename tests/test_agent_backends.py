from pathlib import Path

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.control.runtime import Runtime
from poc.control.spawn_policy import SpawnDenied
from poc.execution.agent_executor import AgentExecutionRequest, AgentExecutorRegistry
from poc.models import (
    AgentBackend,
    ApprovalRequest,
    ExecutionMode,
    ExecutionPolicy,
    RunCreate,
    TaskSpec,
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
                    "result": {"fact": "produced by Pydantic AI"},
                    "completion_summary": "Produced a typed Pydantic AI result.",
                },
            )
        ]
    )


class _ExtensionExecutor:
    backend = "extension_runtime"

    def execute(self, request: AgentExecutionRequest) -> dict[str, object]:
        return {"task_id": request.task.id}


def test_executor_registry_accepts_extension_backend_ids() -> None:
    registry = AgentExecutorRegistry()

    registry.register(_ExtensionExecutor())

    assert registry.get("extension_runtime").backend == "extension_runtime"


@pytest.mark.asyncio
async def test_workflow_can_mix_custom_and_pydantic_ai_agents(tmp_path: Path) -> None:
    test_model = FunctionModel(_pydantic_agent_model)
    runtime = Runtime(tmp_path / "mixed", pydantic_model_factory=lambda role: test_model)
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
                    goal="Produce a manifest fact with a Pydantic AI agent.",
                    output_schema="ManifestFact",
                ),
            ],
        )

        result = await runtime.runner.start(workflow)

        assert result["results"]["custom-manifest"]["outcome"] == "succeeded"
        assert result["results"]["pydantic-manifest"]["outcome"] == "succeeded"
        assert result["results"]["pydantic-manifest"]["result"] == {
            "fact": "produced by Pydantic AI"
        }
        worker_backends = {
            agent["agent_backend"]
            for agent in runtime.db.list_agents(run["run_id"])
            if agent["tier"] == "worker"
        }
        assert worker_backends == {
            AgentBackend.CUSTOM_PYTHON,
            AgentBackend.PYDANTIC_AI,
        }
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
