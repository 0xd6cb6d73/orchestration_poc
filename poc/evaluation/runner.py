from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from pydantic import BaseModel, TypeAdapter
from pydantic_evals.reporting import EvaluationReport

from poc.control.runtime import Runtime
from poc.evaluation.datasets import AgentEvaluationDataset
from poc.evaluation.models import (
    AgentEvaluationExpected,
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationVariant,
    AgentToolCall,
)
from poc.execution.pydantic_ai_executor import PydanticModelFactory
from poc.models import (
    ApprovalRequest,
    RoleSpec,
    RunCreate,
    TaskSpec,
    Tier,
    WorkerResult,
    WorkflowSpec,
)
from poc.roles.registry import RoleRegistry

AgentEvaluationReport = EvaluationReport[
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationExpected,
]
AgentEvaluationReportAdapter = TypeAdapter(AgentEvaluationReport)


class _WorkflowOutput(BaseModel):
    results: dict[str, WorkerResult]


class _OperationRecord(BaseModel):
    kind: str
    status: str
    request: str


class AgentEvaluationRunner:
    """Runs isolated agent experiments through the public orchestration boundary."""

    def __init__(
        self,
        dataset: AgentEvaluationDataset,
        *,
        fixture_root: str | Path | None = None,
        work_dir: str | Path | None = None,
        pydantic_model_factory: PydanticModelFactory | None = None,
    ):
        self.dataset = dataset
        self.fixture_root = Path(fixture_root) if fixture_root is not None else None
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.pydantic_model_factory = pydantic_model_factory

    async def evaluate(
        self,
        variant: AgentEvaluationVariant,
        *,
        repeat: int = 1,
        max_concurrency: int | None = 1,
        progress: bool = True,
    ) -> AgentEvaluationReport:
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        evaluated_roles = {case.inputs.role_id for case in self.dataset.cases}
        unknown_tool_roles = set(variant.allowed_tools_by_role) - evaluated_roles
        if unknown_tool_roles:
            raise ValueError(
                f"tool overrides target roles absent from the dataset: {sorted(unknown_tool_roles)}"
            )
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)

        async def task(case: AgentEvaluationInput) -> AgentEvaluationOutput:
            return await self._run_case(case, variant)

        return await self.dataset.evaluate(
            task,
            name=variant.name,
            task_name="orchestrated_agent",
            metadata=variant.experiment_metadata(),
            repeat=repeat,
            max_concurrency=max_concurrency,
            progress=progress,
        )

    async def _run_case(
        self,
        case: AgentEvaluationInput,
        variant: AgentEvaluationVariant,
    ) -> AgentEvaluationOutput:
        parent_dir = str(self.work_dir) if self.work_dir is not None else None
        with TemporaryDirectory(prefix="agent-eval-", dir=parent_dir) as data_dir:
            roles = RoleRegistry()
            role = _variant_role(roles.get(case.role_id), case.role_id, variant)
            roles.register(role, replace=True)
            runtime = Runtime(
                data_dir,
                fixture_root=self.fixture_root,
                roles=roles,
                pydantic_model_factory=self.pydantic_model_factory,
            )
            try:
                result = await _execute_case(runtime, roles, case, variant)
                tool_calls = _tool_calls(runtime, result.agent_instance_id, result.task_id)
                return AgentEvaluationOutput(
                    backend=variant.agent_backend,
                    provider=role.provider,
                    model=role.model,
                    system_prompt_sha256=sha256(role.system_prompt.encode()).hexdigest(),
                    allowed_tools=sorted(role.allowed_tools),
                    outcome=result.outcome,
                    result=result.result,
                    completion_summary=result.completion_summary,
                    tool_calls=tool_calls,
                    evidence_artifacts=result.evidence_artifacts,
                    error=result.completion_summary if result.outcome != "succeeded" else None,
                )
            finally:
                await runtime.close()


async def _execute_case(
    runtime: Runtime,
    roles: RoleRegistry,
    case: AgentEvaluationInput,
    variant: AgentEvaluationVariant,
) -> WorkerResult:
    run, _ = runtime.create_run(RunCreate(objective=f"Evaluation: {case.goal}"))
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    if main is None:
        raise RuntimeError("evaluation run did not create its main orchestrator")
    supervisor_role = _supervisor_for(roles, case.role_id)
    supervisor = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=main,
        child_role=supervisor_role,
        plan_version=plan.version,
        stable_key="evaluation-supervisor",
    )
    workflow = WorkflowSpec(
        workflow_id=f"{run['run_id']}-evaluation",
        run_id=run["run_id"],
        owner=supervisor.agent_instance_id,
        approved_plan_version=plan.version,
        authorized_worker_roles=[case.role_id],
        max_workers=1,
        tasks=[
            TaskSpec(
                id="evaluation-task",
                role=case.role_id,
                agent_backend=variant.agent_backend,
                goal=case.goal,
                output_schema=case.output_schema,
                static_inputs=case.inputs,
                acceptance_criteria=case.acceptance_criteria,
            )
        ],
    )
    raw_result = await runtime.runner.start(workflow)
    result = _WorkflowOutput.model_validate(raw_result)
    try:
        return result.results["evaluation-task"]
    except KeyError as exc:
        raise RuntimeError("evaluation workflow did not return its worker result") from exc


def _variant_role(
    base: RoleSpec,
    role_id: str,
    variant: AgentEvaluationVariant,
) -> RoleSpec:
    return RoleSpec.model_validate(
        {
            **base.model_dump(),
            "agent_backend": variant.agent_backend,
            "provider": variant.provider if variant.provider is not None else base.provider,
            "model": variant.model if variant.model is not None else base.model,
            "system_prompt": (
                variant.system_prompt if variant.system_prompt is not None else base.system_prompt
            ),
            "allowed_tools": sorted(
                variant.allowed_tools_by_role.get(role_id, frozenset(base.allowed_tools))
            ),
            "execution_limits": {**base.execution_limits, **variant.execution_limits},
        }
    )


def _supervisor_for(roles: RoleRegistry, role_id: str) -> str:
    for role in roles.all():
        if role.tier == Tier.SUB and role_id in role.allowed_child_roles:
            return role.role_id
    raise ValueError(f"no sub-orchestrator can spawn evaluation role {role_id!r}")


def _tool_calls(runtime: Runtime, agent_id: str, task_id: str) -> list[AgentToolCall]:
    rows = runtime.db.conn.execute(
        "SELECT kind,status,request FROM operations "
        "WHERE run_id=(SELECT run_id FROM agents WHERE agent_instance_id=?) AND task_id=? "
        "ORDER BY started_at,operation_id",
        (agent_id, task_id),
    ).fetchall()
    calls: list[AgentToolCall] = []
    for row in rows:
        operation = _OperationRecord.model_validate(dict(row))
        arguments = TypeAdapter(dict[str, Any]).validate_python(json.loads(operation.request))
        calls.append(
            AgentToolCall(
                tool_name=operation.kind,
                arguments=arguments,
                status=operation.status,
            )
        )
    return calls


def save_report(report: AgentEvaluationReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(AgentEvaluationReportAdapter.dump_json(report, indent=2))


def load_report(path: str | Path) -> AgentEvaluationReport:
    return AgentEvaluationReportAdapter.validate_json(Path(path).read_bytes())


def report_passed(report: AgentEvaluationReport) -> bool:
    if not report.cases or report.failures or report.report_evaluator_failures:
        return False
    return all(
        not case.evaluator_failures
        and all(assertion.value for assertion in case.assertions.values())
        for case in report.cases
    )
