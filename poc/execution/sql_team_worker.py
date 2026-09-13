"""Typed, budgeted SQL workers. Policies see public task inputs only."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Any, Literal, cast

from opentelemetry import trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, UsageLimits
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.execution.sql_artifacts import ArtifactContractError, validate_artifact
from poc.execution.sql_contracts import Answer, Budget, PhaseName, StrictModel, TaskInput
from poc.execution.sql_ports import (
    PhaseTimeout,
    RequestTimeout,
    TaskEnvironment,
    ToolBudgetExceeded,
    failure_details,
)
from poc.execution.sql_protocol import ProtocolError, parse_action
from poc.execution.sql_strategy import ContextBoundModel, ObservedModel, resolve_model
from poc.hybrid.contracts import EvaluationVector


class PlanValues(StrictModel):
    plan: str = Field(min_length=1, max_length=6000)


class PlanOutput(StrictModel):
    values: PlanValues


class CritiqueOutput(StrictModel):
    values: EvaluationVector


class VerificationValues(StrictModel):
    answer_supported: bool = Field(strict=True)


class VerificationOutput(StrictModel):
    values: VerificationValues


class SupportedCritiqueValues(StrictModel):
    structural_validity: bool = Field(strict=True)
    feasibility: Literal["supported", "unsupported", "abstain"]
    evidence_refs: list[str]
    reason: str = Field(min_length=1)
    evaluation: EvaluationVector


class SupportedCritiqueOutput(StrictModel):
    values: SupportedCritiqueValues


class SupportedVerificationValues(StrictModel):
    answer_supported: bool = Field(strict=True)
    evidence_refs: list[str]
    reason: str = Field(min_length=1)


class SupportedVerificationOutput(StrictModel):
    values: SupportedVerificationValues


CONTRACTS: dict[str, type[BaseModel]] = {
    "plan": PlanOutput,
    "critique": CritiqueOutput,
    "verify": VerificationOutput,
}
INSTRUCTIONS = {
    "plan": "Your role is planner. Develop a concise plan grounded in the public tables. "
    "Return only the plan, at most 6000 characters. Leave execution of the plan to the solver.",
    "critique": "Your role is critic. Inspect the supplied candidate against the public task "
    "and SQL data. Return the five integer evaluation scores, not a solution or a plan.",
    "verify": "Your role is verifier. Independently check the selected answer using public SQL "
    "data. Return the boolean answer_supported. Do not generate a replacement answer.",
}
WEIGHTS = {
    "hierarchical_dag": [0.2, 0.8],
    "board_claim": [0.2, 0.8],
    "managed_pool": [0.2, 0.8],
    "speculative": [0.35, 0.35, 0.3],
    "hybrid_v1": [0.3, 0.3, 0.12, 0.12, 0.16],
}


class TeamWorker:
    def __init__(
        self,
        method: str,
        task: TaskInput,
        env: TaskEnvironment,
        model: Model | str,
        settings: ModelSettings,
        budget: Budget,
        usage: RunUsage,
        *,
        json_protocol: bool,
    ):
        self.task, self.env, self.model = task, env, model
        self.method = method
        self.settings, self.budget, self.usage = settings, budget, usage
        self.json_protocol = json_protocol
        self.weights = env.state.options.stage_weights or WEIGHTS[method]
        if len(self.weights) != len(WEIGHTS[method]):
            raise ValueError("stage_weights length does not match the strategy")
        self.index = 0
        self.tool_calls = 0
        self.evidence_namespace: str | None = None
        self.reliable = env.state.options.team_policy in {"reliable-v2", "concurrent-v1"}

    async def __call__(self, instruction: str, role: PhaseName = "solve") -> Answer:
        state, usage, budget = self.env.state, self.usage, self.budget
        if self.reliable:
            if self.method == "hybrid_v1" and role in {"critique", "verify"}:
                self.index = max(self.index, 2 if role == "critique" else 4)
            elif self.method == "speculative" and role == "reconcile":
                self.index = max(self.index, 2)
        index = self.index
        self.index += 1
        # Unused allocation carries forward. A stage cannot borrow from future stages.
        cumulative = 1.0 if index == len(self.weights) - 1 else sum(self.weights[: index + 1])
        reserve = (
            min(state.options.return_reserve_seconds, budget.seconds / 10) if self.reliable else 0
        )
        deadline = state.started + (budget.seconds - reserve) * cumulative
        token_limit = max(1, int(budget.total_tokens * cumulative))
        request_limit = max(1, int(budget.requests * cumulative))
        binding = state.options.phase_models.get(role) or state.options.phase_models.get("solve")
        model = resolve_model(binding) if binding else self.model
        settings = cast(ModelSettings, binding.settings) if binding else self.settings
        profile = state.options.role_profiles.get(role) or (
            binding.role_profiles.get(role)
            if binding
            else (
                self.model.role_profiles.get(role)
                if isinstance(self.model, ContextBoundModel)
                else None
            )
        )
        default_output_cap = 16384 if self.reliable or role not in CONTRACTS else 4096
        output_cap = budget.max_output_tokens or (
            profile.max_output_tokens if profile else default_output_cap
        )
        if profile:
            output_cap = min(output_cap, profile.max_output_tokens)
        stage_budget = budget.model_copy(
            update={
                "total_tokens": token_limit,
                "requests": request_limit,
                "tool_calls": max(1, int(budget.tool_calls * cumulative)),
                "request_timeout_seconds": min(
                    budget.request_timeout_seconds or budget.seconds,
                    profile.request_timeout_seconds
                    if profile
                    else (budget.seconds if self.reliable else 60),
                ),
                "max_output_tokens": output_cap,
            }
        )
        state.phase = role
        started, input_tokens, output_tokens = (
            perf_counter(),
            usage.input_tokens,
            usage.output_tokens,
        )
        record: dict[str, Any] = {
            "name": role,
            "stage_index": index,
            "policy": state.options.team_policy,
            "budget_profile": profile.model_dump() if profile else None,
            "max_output_tokens": stage_budget.max_output_tokens,
            "model_binding": binding.model_dump() if binding else None,
            "model": model if isinstance(model, str) else model.model_name,
            "status": "running",
            "start_seconds": started - state.started,
            "token_limit": token_limit,
            "request_limit": request_limit,
            "tool_limit": stage_budget.tool_calls,
            "deadline_seconds": deadline - state.started,
        }
        state.phases.append(record)

        def annotate(exc: BaseException) -> None:
            record["failure"] = failure_details(
                exc,
                scope="phase",
                stage=role,
                stage_index=index,
                threshold={
                    "seconds": deadline - state.started,
                    "tokens": token_limit,
                    "requests": request_limit,
                    "tools": stage_budget.tool_calls,
                },
                consumption={
                    "seconds": perf_counter() - state.started,
                    "tokens": usage.total_tokens,
                    "requests": usage.requests,
                    "tools": self.tool_calls,
                },
            )

        with trace.get_tracer(__name__).start_as_current_span("evaluation.phase") as span:
            span.set_attribute("evaluation.phase", role)
            span.set_attribute("orchestration.stage_index", index)
            try:
                if started >= deadline:
                    raise PhaseTimeout("stage wall-clock allocation exhausted")
                if usage.total_tokens >= token_limit or usage.requests >= request_limit:
                    raise UsageLimitExceeded("stage allocation exhausted")
                async with asyncio.timeout(deadline - started):
                    answer = await self.solve(instruction, role, model, settings, stage_budget)
                if perf_counter() >= deadline:
                    raise PhaseTimeout("stage completed after its return deadline")
                record["status"] = "completed"
                if role not in CONTRACTS:
                    validate_artifact(self.task, answer, state.options.artifact_contract)
                    state.candidate(answer, "output")
                return answer
            except RequestTimeout as exc:
                annotate(exc)
                record["status"] = "request_timeout"
                raise
            except TimeoutError as exc:
                record["status"] = "timeout"
                error = PhaseTimeout("stage wall-clock allocation exhausted")
                annotate(error)
                raise error from exc
            except (UsageLimitExceeded, ToolBudgetExceeded) as exc:
                annotate(exc)
                record["status"] = "budget_exhausted"
                raise
            except BaseException as exc:
                annotate(exc)
                record["status"] = "interrupted"
                raise
            finally:
                if record["status"] != "completed":
                    state.exhausted_phase = role
                record.update(
                    elapsed_seconds=perf_counter() - started,
                    input_tokens=usage.input_tokens - input_tokens,
                    output_tokens=usage.output_tokens - output_tokens,
                    tokens=usage.input_tokens + usage.output_tokens - input_tokens - output_tokens,
                    usage_complete=state.usage_complete,
                )
                span.set_attribute("evaluation.phase.status", record["status"])
                state.emit({"phase": record.copy()})

    async def solve(
        self,
        instruction: str,
        role: str,
        model: Model | str,
        settings: ModelSettings,
        budget: Budget,
    ) -> Answer:
        contract = CONTRACTS.get(role, Answer)
        if self.reliable and role in {"critique", "verify"}:
            contract = (
                SupportedCritiqueOutput if role == "critique" else SupportedVerificationOutput
            )
        instructions = INSTRUCTIONS.get(
            role,
            "Your role is solver. Return the complete task answer "
            "using the requested IDs and value format, not a plan, "
            "example, SQL string, or evaluation scores.",
        )
        instructions += " Use read-only SQLite to inspect the public data as needed."
        if self.reliable and role in {"critique", "verify"}:
            instructions += (
                " Decision protocol supported-v1: distinguish structural validity, supported "
                "feasibility and abstention. Cite SQL evidence_refs returned by your query tool. "
                "Evidence references prove observation provenance, not correctness. Output schema: "
                + json.dumps(contract.model_json_schema())
            )
        prompt = (
            self.task.prompt
            + "\nRole task: "
            + instruction
            + "\nSQL schema: "
            + json.dumps(self.env.schema)
        )
        errors: dict[str, int] = {}
        evidence: set[str] = set()

        def execute_query(sql: str) -> dict[str, Any]:
            """Read-only SQLite query. Bounds are in the task; default 200 rows / 64KB. Paginate."""
            if self.tool_calls >= budget.tool_calls:
                raise ToolBudgetExceeded("stage tool allocation exhausted")
            self.tool_calls += 1
            result = self.env.query(sql)
            if "error" in result:
                key = str(result["error"])
                errors[key] = errors.get(key, 0) + 1
                if errors[key] > self.env.state.options.output_retries + 1:
                    raise UnexpectedModelBehavior("repeated SQL error without progress")
            else:
                errors.clear()
                if self.reliable and role in {"critique", "verify"}:
                    ref = f"query:{role}:{self.evidence_namespace or self.index}:{self.tool_calls}"
                    evidence.add(ref)
                    result = {**result, "evidence_ref": ref}
                    self.env.state.emit(
                        {"evidence": {"ref": ref, "stage": role, "sql": sql, "result": result}}
                    )
            return result

        def check_output(output: BaseModel) -> Answer:
            answer = Answer.model_validate(output.model_dump())
            if role not in CONTRACTS:
                validate_artifact(self.task, answer, self.env.state.options.artifact_contract)
            elif self.reliable and role in {"critique", "verify"}:
                refs = set(answer.values["evidence_refs"])
                positive = (
                    answer.values.get("feasibility") == "supported"
                    if role == "critique"
                    else answer.values["answer_supported"]
                )
                if not refs <= evidence or (positive and not refs):
                    raise ArtifactContractError(
                        "supported-v1: absent or foreign SQL evidence references"
                    )
            return answer

        limits = UsageLimits(
            request_limit=budget.requests,
            total_tokens_limit=budget.total_tokens,
            tool_calls_limit=budget.tool_calls,
        )
        observed = ObservedModel(model, self.env.state, budget, self.usage)
        if not self.json_protocol:

            async def query(sql: str) -> dict[str, Any]:
                """Read-only SQLite query. Bounds are in the task; default 200 rows / 64KB. Paginate."""
                # Pydantic runs synchronous tools in a worker thread. The environment's
                # SQLite connection is owned by this event-loop thread.
                return execute_query(sql)

            agent: Agent[None, Any] = Agent(
                observed,
                output_type=cast(Any, contract),
                tools=[query],
                model_settings=settings,
                retries={"output": self.env.state.options.output_retries},
                instructions=instructions,
                capabilities=[Instrumentation()],
            )

            @agent.output_validator
            async def validate_output(output: Any) -> Any:
                try:
                    check_output(output)
                except ArtifactContractError as exc:
                    raise ModelRetry(str(exc)) from exc
                return output

            result = await agent.run(prompt, usage=self.usage, usage_limits=limits)
            return check_output(result.output)
        json_agent = Agent(
            observed,
            output_type=str,
            model_settings=settings,
            instructions=instructions + " On each turn return exactly one JSON object: "
            '{"sql":"SELECT ..."} for a query, or an output matching this schema: '
            + json.dumps(contract.model_json_schema())
            + " No markdown.",
            capabilities=[Instrumentation()],
        )
        history = None
        invalid = 0
        while True:
            result = await json_agent.run(
                prompt, message_history=history, usage=self.usage, usage_limits=limits
            )
            history = result.all_messages()
            try:
                action = parse_action(result.output, self.env.state.options.transport_policy)
                if set(action) == {"sql"} and isinstance(action["sql"], str):
                    sql = action["sql"]
                else:
                    return check_output(contract.model_validate(action))
            except (ValueError, TypeError) as exc:
                self.env.state.emit(
                    {
                        "protocol_error": {
                            "type": type(exc).__name__,
                            "stage": role,
                            "code": str(exc) if isinstance(exc, ProtocolError) else "role_contract",
                        }
                    }
                )
                invalid += 1
                if invalid > self.env.state.options.output_retries:
                    raise UnexpectedModelBehavior("SQL role output retries exhausted") from None
                prompt = (
                    'Invalid JSON or role output. Return {"sql":"..."} or this schema: '
                    + json.dumps(contract.model_json_schema())
                )
                continue
            prompt = "SQL result: " + json.dumps(execute_query(sql))
