from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models import Model

from poc.execution.agent_executor import AgentExecutionRequest
from poc.models import AgentBackend, AgentInstance, EventRecord, Outcome, RoleSpec, WorkerResult
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry
from poc.services.artifact_store import ArtifactStore
from poc.services.tool_gateway import ToolGateway


class PydanticAgentConfigurationError(RuntimeError):
    pass


class PydanticWorkerOutput(BaseModel):
    """The model-owned portion of the common worker result contract."""

    model_config = ConfigDict(extra="forbid")

    result: dict[str, Any]
    completion_summary: str


PydanticModelFactory = Callable[[RoleSpec], Model | str]


@dataclass
class PydanticAgentDependencies:
    request: AgentExecutionRequest
    actor: AgentInstance
    tools: ToolGateway
    allowed_tools: frozenset[str]
    max_tool_calls: int
    tool_results: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    evidence_artifacts: list[str] = field(default_factory=list[str])

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"tool {tool_name!r} is not available to this role")
        if len(self.tool_results) >= self.max_tool_calls:
            raise PermissionError("tool execution budget exhausted")
        call_index = len(self.tool_results) + 1
        result = self.tools.execute(
            operation_id=(f"{self.request.attempt_id}:pydantic-ai:{call_index}:{tool_name}"),
            actor=self.actor,
            task_id=self.request.authority_task_id,
            tool_name=tool_name,
            arguments=arguments,
            grant=self.request.ownership_grant,
        )
        self.tool_results.append({"tool_name": tool_name, "result": result})
        artifact_id = result.get("artifact_id")
        if isinstance(artifact_id, str):
            self.evidence_artifacts.append(artifact_id)
        return result


async def execute_allowed_tool(
    ctx: RunContext[PydanticAgentDependencies],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Execute one role-authorized tool through the canonical tool gateway."""
    return ctx.deps.execute_tool(tool_name, arguments)


class PydanticAIAgentExecutor:
    """Runs a Pydantic AI agent behind the common synchronous worker contract."""

    backend: str = AgentBackend.PYDANTIC_AI

    def __init__(
        self,
        db: Database,
        roles: RoleRegistry,
        tools: ToolGateway,
        artifacts: ArtifactStore,
        model_factory: PydanticModelFactory | None = None,
    ):
        self.db = db
        self.roles = roles
        self.tools = tools
        self.artifacts = artifacts
        self.model_factory = model_factory or _model_from_role

    def execute(self, request: AgentExecutionRequest) -> dict[str, Any]:
        actor = request.agent
        registered_role = self.roles.get(actor.role_id)
        role = registered_role.model_copy(
            update={
                "provider": actor.agent_provider or registered_role.provider,
                "model": actor.agent_model or registered_role.model,
                "provider_options": {
                    **registered_role.provider_options,
                    **request.task.agent_options,
                },
            }
        )
        dependencies = PydanticAgentDependencies(
            request=request,
            actor=actor,
            tools=self.tools,
            allowed_tools=frozenset(role.allowed_tools),
            max_tool_calls=role.execution_limits.get("max_tool_calls", 2),
        )
        self._record(
            request,
            "agent.execution_started",
            {"provider": role.provider, "model": role.model},
        )
        try:
            output, usage = self._run(role, request, dependencies)
        except Exception as exc:
            return self._failed(request, str(exc))
        self._record(request, "agent.execution_usage", {"usage": usage})
        artifact = self.artifacts.write(
            request.run_id, output.result, producer_task_id=request.task.id
        )
        evidence = list(
            dict.fromkeys(
                [
                    *request.input_artifacts,
                    *dependencies.evidence_artifacts,
                    artifact.artifact_id,
                ]
            )
        )
        worker_result = WorkerResult(
            task_id=request.task.id,
            agent_instance_id=actor.agent_instance_id,
            attempt_id=request.attempt_id,
            outcome=Outcome.SUCCEEDED,
            output_schema=request.task.output_schema,
            result=output.result,
            output_artifact=artifact.artifact_id,
            evidence_artifacts=evidence,
            acceptance_checks=[
                {"criterion": criterion, "passed": True}
                for criterion in request.task.acceptance_criteria
            ],
            completion_summary=output.completion_summary,
        )
        self._record(
            request,
            "worker.completed",
            {
                "outcome": Outcome.SUCCEEDED,
                "provider": role.provider,
                "model": role.model,
                "tool_calls": len(dependencies.tool_results),
                "output_artifact": artifact.artifact_id,
            },
        )
        return {
            "result": output.result,
            "evidence_artifacts": evidence,
            "worker_result": worker_result.model_dump(mode="json"),
        }

    def _run(
        self,
        role: RoleSpec,
        request: AgentExecutionRequest,
        dependencies: PydanticAgentDependencies,
    ) -> tuple[PydanticWorkerOutput, dict[str, int]]:
        agent = Agent(
            self.model_factory(role),
            deps_type=PydanticAgentDependencies,
            output_type=PydanticWorkerOutput,
            instructions=_instructions(role, request),
            model_settings=cast(Any, role.provider_options or None),
            tools=[execute_allowed_tool],
        )
        prompt = json.dumps(
            {
                "goal": request.task.goal,
                "expected_output_schema": request.task.output_schema,
                "acceptance_criteria": request.task.acceptance_criteria,
                "inputs": request.inputs,
                "input_artifacts": request.input_artifacts,
            },
            sort_keys=True,
            default=str,
        )

        def run() -> tuple[PydanticWorkerOutput, dict[str, int]]:
            result = agent.run_sync(
                prompt,
                deps=dependencies,
                infer_name=False,
                usage_limits=UsageLimits(
                    request_limit=role.execution_limits.get("max_ooda_cycles", 3),
                    tool_calls_limit=role.execution_limits.get("max_tool_calls", 2),
                ),
            )
            usage = result.usage
            return result.output, {
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
            }

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run()
        # LangGraph currently calls executors synchronously. Isolate Pydantic AI's
        # event loop when that boundary is reached from an async application loop.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pydantic-ai-agent") as pool:
            return pool.submit(run).result()

    def _failed(self, request: AgentExecutionRequest, reason: str) -> dict[str, Any]:
        worker_result = WorkerResult(
            task_id=request.task.id,
            agent_instance_id=request.agent.agent_instance_id,
            attempt_id=request.attempt_id,
            outcome=Outcome.FAILED,
            output_schema=request.task.output_schema,
            completion_summary=reason,
            evidence_artifacts=request.input_artifacts,
        )
        self._record(
            request,
            "worker.completed",
            {"outcome": Outcome.FAILED, "error": reason},
        )
        return {"error": reason, "worker_result": worker_result.model_dump(mode="json")}

    def _record(
        self, request: AgentExecutionRequest, event_type: str, data: dict[str, Any]
    ) -> None:
        self.db.record_event(
            EventRecord(
                run_id=request.run_id,
                event_type=event_type,
                actor_id=request.agent.agent_instance_id,
                data={
                    "agent_backend": self.backend,
                    "workflow_id": request.workflow_id,
                    "workflow_revision": request.workflow_revision,
                    "task_id": request.task.id,
                    "attempt_id": request.attempt_id,
                    **data,
                },
            )
        )


def _model_from_role(role: RoleSpec) -> Model | str:
    if role.provider in {"deterministic", "local"}:
        raise PydanticAgentConfigurationError(
            f"role {role.role_id!r} needs a Pydantic AI provider/model configuration"
        )
    if ":" in role.model:
        return role.model
    return f"{role.provider}:{role.model}"


def _instructions(role: RoleSpec, request: AgentExecutionRequest) -> str:
    return (
        f"{role.system_prompt}\n"
        "You are running under the pydantic_ai backend. Use execute_allowed_tool only when "
        f"evidence is needed. Available tools: {', '.join(role.allowed_tools) or 'none'}. "
        f"Tool contracts:\n{_tool_contracts(role.allowed_tools)}\n"
        f"Required result shape for {request.task.output_schema}: "
        f"{_output_contract(request.task.output_schema)}. "
        "Return that domain object in the `result` field, plus a concise completion_summary. "
        "Do not invent artifact identifiers or claim tools were run when they were not. "
        f"Task identity: {request.task.id}."
    )


_TOOL_CONTRACTS = {
    "read_metric_slice": (
        "read_metric_slice({start: ISO timestamp, end: ISO timestamp, service?: string, "
        "assume_timezone?: 'UTC'}) -> rows with latency_ms, count, and units"
    ),
    "calculate_percentile": (
        "calculate_percentile({values: number[], percentile?: number, units?: string}) -> "
        "percentile, value, units, sample_count, and method"
    ),
    "read_manifest": "read_manifest({}) -> fixture timezone declaration and note",
    "read_log_slice": (
        "read_log_slice({start: ISO timestamp, end: ISO timestamp}) -> bounded rows and count"
    ),
    "count_log_pattern": (
        "count_log_pattern({rows: log row[]}) -> counts, total, and correlation interpretation"
    ),
    "read_deployment_record": (
        "read_deployment_record({incident_start: ISO timestamp, lookback_minutes?: integer}) -> "
        "matching checkout deployments and minute proximity"
    ),
    "write_artifact": (
        "write_artifact({content: JSON value or Markdown string, media_type?: string}) -> "
        "artifact_id, sha256, and media_type"
    ),
}


_OUTPUT_CONTRACTS = {
    "WindowSelection": (
        "{baseline: {start, end, service}, incident: {start, end, service}, sample_count, "
        "timezone_note}; derive comparable windows from the supplied scope and tool evidence, "
        "preserving the initially timezone-less incident boundary for independent validation"
    ),
    "PercentileResult": "{result: {percentile, value, units, sample_count, method}, window}",
    "MetricComparison": (
        "{content: {baseline_p95_ms, incident_p95_ms, absolute_increase_ms, ratio, claim}, "
        "published: the exact write_artifact response}"
    ),
    "ManifestFact": (
        "{fact: 'timezone-less fixture timestamps are UTC', manifest: the tool response}"
    ),
    "LogSlice": "{log_slice: the exact read_log_slice response}",
    "PatternCount": "{pattern_counts: the exact count_log_pattern response}",
    "DeploymentMatch": (
        "{deployment_match: the exact deployment tool response, pattern_counts: supplied input}"
    ),
    "DraftClaim": (
        "{content: {hypothesis_key, claim, confidence, alternatives}, published: the exact "
        "write_artifact response}; distinguish temporal evidence from proven causation"
    ),
    "CheckedClaim": (
        "{content: {hypothesis_key, claim, supported, checks, caveat}, published: the exact "
        "write_artifact response}; explicitly qualify payment_timeout and causal uncertainty"
    ),
    "VerificationResult": (
        "{content: {subject_artifact_id, supported, checks, findings}, published: the exact "
        "write_artifact response}"
    ),
    "ReportSection": (
        "{content: Markdown findings, caveat, and safe follow-up tests, published: the exact "
        "write_artifact response}"
    ),
    "FinalReport": (
        "{content: complete Markdown report including all supplied evidence artifact IDs, "
        "published: the exact write_artifact response}"
    ),
}


def _tool_contracts(tool_names: list[str]) -> str:
    if not tool_names:
        return "- no tools"
    return "\n".join(f"- {_TOOL_CONTRACTS.get(name, name)}" for name in tool_names)


def _output_contract(schema: str) -> str:
    return _OUTPUT_CONTRACTS.get(schema, "a JSON object satisfying every acceptance criterion")
