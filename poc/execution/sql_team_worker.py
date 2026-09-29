"""Typed, budgeted SQL workers. Policies see public task inputs only."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, cast

from opentelemetry import trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, UsageLimits
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openrouter import OpenRouterReasoning
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
from poc.hybrid.planning import MAX_PLAN_TASKS, PlannedTask


@dataclass
class ScopedTaskState:
    task: PlannedTask
    started_at: float
    consumed_requests: int = 0
    consumed_tokens: int = 0
    consumed_tools: int = 0


_scoped_task: ContextVar[ScopedTaskState | None] = ContextVar(
    "hybrid_v2_1_scoped_task", default=None
)


@contextmanager
def scoped_task(task: PlannedTask) -> Generator[None]:
    token = _scoped_task.set(ScopedTaskState(task=task, started_at=perf_counter()))
    try:
        yield
    finally:
        _scoped_task.reset(token)


class EvidenceClaim(StrictModel):
    claim: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)


class TaskEvidenceValues(StrictModel):
    proposed_values: dict[str, Any]
    findings: list[EvidenceClaim]
    assumptions: list[str]
    unresolved_questions: list[str]


class TaskEvidenceOutput(StrictModel):
    values: TaskEvidenceValues


class V21Answer(Answer):
    status: Literal["supported", "partial", "inconclusive"]
    claims: list[EvidenceClaim]
    assumptions: list[str]
    unresolved_questions: list[str]


class V21VerificationValues(StrictModel):
    status: Literal["supported", "partial", "inconclusive"]
    claims_supported: bool = Field(strict=True)
    gaps_disclosed: bool = Field(strict=True)
    evidence_refs: list[str]
    reason: str = Field(min_length=1)


class V21VerificationOutput(StrictModel):
    values: V21VerificationValues


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
    "orchestrate": Answer,
    "task": Answer,
}
INSTRUCTIONS = {
    "plan": "Your role is planner. Develop a concise plan grounded in the public tables. "
    "Return only the plan, at most 6000 characters. Leave execution of the plan to the solver.",
    "critique": "Your role is critic. Inspect the supplied candidate against the public task "
    "and SQL data. Return the five integer evaluation scores, not a solution or a plan.",
    "verify": "Your role is verifier. Independently check the selected answer using public SQL "
    "data. Return the boolean answer_supported. Do not generate a replacement answer.",
    "orchestrate": "Your role is orchestrator. Respond with exactly the JSON object requested "
    "in the role task: an orchestrator plan or a decision. Do not perform task work yourself.",
    "task": "Your role is scoped task executor. Complete only the assigned task. Stay strictly "
    "inside the stated local scope and definition of done. Do not solve the whole task.",
    "integrate": "Your role is integrator. Combine the supplied scoped task outputs into the "
    "complete final answer for the original task. Resolve conflicts across outputs.",
}
WEIGHTS = {
    "hierarchical_dag": [0.2, 0.8],
    "board_claim": [0.2, 0.8],
    "managed_pool": [0.2, 0.8],
    "speculative": [0.35, 0.35, 0.3],
    "hybrid_v1": [0.3, 0.3, 0.12, 0.12, 0.16],
}
# Judge roles turn on reasoning in JSON turn loops and burn the whole request
# timeout on hidden reasoning tokens without ever emitting an answer.
JUDGE_ROLES = frozenset({"orchestrate", "critique"})
# OpenRouter's settings contract merges model-agnostic keys (`openrouter_` prefix).
JUDGE_REASONING_LIMIT: OpenRouterReasoning = {"effort": "low"}
JUDGE_REQUEST_TIMEOUT_SECONDS = 180.0
ROLE_POOLS: dict[str, dict[str, float]] = {
    "hybrid_v2": {
        "orchestrate": 0.24,
        "task": 0.3,
        "critique": 0.34,
        "integrate": 0.05,
        "verify": 0.07,
    },
}
POOL_PLANNED: dict[str, int] = {
    "orchestrate": 2,
    "task": MAX_PLAN_TASKS,
    "critique": 4,
    "integrate": 1,
    "verify": 1,
}


def merge_reasoning_limit(settings: ModelSettings, role: str) -> ModelSettings:
    """Cap provider reasoning for judge roles via the OpenRouter reasoning parameter.

    This is a mechanical capability boundary, not prompt content. Non-judge roles
    keep the incoming settings untouched.
    """
    if role not in JUDGE_ROLES:
        return settings
    return cast(ModelSettings, {**settings, "openrouter_reasoning": JUDGE_REASONING_LIMIT})


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
        self.pool_shares: dict[str, float] | None = None
        self.pool_planned: dict[str, int] = {}
        self.consumed = 0.0
        if method in {"hybrid_v2", "hybrid_v2_1", "hybrid_v2_2"}:
            shares = env.state.options.hybrid_pool_weights or ROLE_POOLS["hybrid_v2"]
            self.pool_shares = dict(shares)
            self.pool_planned = dict(POOL_PLANNED)
            self.pool_planned["orchestrate"] = (
                2 + env.state.options.hybrid_max_decision_rounds
                if method == "hybrid_v2_2"
                else env.state.options.hybrid_plan_fanout
            )
            self.pool_planned["critique"] = env.state.options.hybrid_plan_fanout + 2
            self.index = 0
        else:
            self.weights = env.state.options.stage_weights or WEIGHTS[method]
            if len(self.weights) != len(WEIGHTS[method]):
                raise ValueError("stage_weights length does not match the strategy")
            self.index = 0
        self.tool_calls = 0
        self.evidence_namespace: str | None = None
        self.reliable = env.state.options.team_policy in {"reliable-v2", "concurrent-v1"}

    async def __call__(self, instruction: str, role: PhaseName = "solve") -> Answer:
        state, usage, budget = self.env.state, self.usage, self.budget
        if self.pool_shares is not None:
            key = role if role in self.pool_shares else "task"
            planned = max(1, self.pool_planned.get(key, 1))
            share = self.pool_shares[key] / planned
            if key == "verify":
                # The verifier is the final stage; unallocated budget carries forward to it.
                share = max(share, 1.0 - self.consumed)
            self.consumed = min(1.0, self.consumed + share)
            cumulative = self.consumed
            index = self.index
            self.index += 1
        else:
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
        scope_state = (
            _scoped_task.get()
            if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "task"
            else None
        )
        if scope_state is not None and "seconds" in scope_state.task.budgets:
            deadline = min(deadline, scope_state.started_at + scope_state.task.budgets["seconds"])
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
        default_output_cap = (
            16384
            if self.reliable or role in {"integrate", "orchestrate"} or role not in CONTRACTS
            else 4096
        )
        output_cap = budget.max_output_tokens or (
            profile.max_output_tokens if profile else default_output_cap
        )
        if profile:
            output_cap = min(output_cap, profile.max_output_tokens)
        # A judge turn that hangs must be retried in minutes, not after the 600s default.
        timeout_caps: tuple[float, ...] = (
            (JUDGE_REQUEST_TIMEOUT_SECONDS,) if role in JUDGE_ROLES else ()
        )
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
                    *timeout_caps,
                ),
                "max_output_tokens": output_cap,
            }
        )
        if scope_state is not None:
            limits = scope_state.task.budgets
            stage_budget = stage_budget.model_copy(
                update={
                    "requests": min(
                        stage_budget.requests,
                        usage.requests + max(0, limits["requests"] - scope_state.consumed_requests),
                    ),
                    "total_tokens": min(
                        stage_budget.total_tokens,
                        usage.total_tokens
                        + max(0, limits["total_tokens"] - scope_state.consumed_tokens),
                    ),
                    "tool_calls": min(
                        stage_budget.tool_calls,
                        self.tool_calls + max(0, limits["tool_calls"] - scope_state.consumed_tools),
                    ),
                }
            )
            token_limit = stage_budget.total_tokens
            request_limit = stage_budget.requests
        state.phase = role
        task_tools_start = self.tool_calls
        task_requests_start = usage.requests
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
                if role not in CONTRACTS and not (
                    self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "integrate"
                ):
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
                if scope_state is not None:
                    scope_state.consumed_requests += usage.requests - task_requests_start
                    scope_state.consumed_tokens += (usage.input_tokens - input_tokens) + (
                        usage.output_tokens - output_tokens
                    )
                    scope_state.consumed_tools += self.tool_calls - task_tools_start
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
        scope_state = (
            _scoped_task.get()
            if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "task"
            else None
        )
        scope = scope_state.task if scope_state is not None else None
        contract = TaskEvidenceOutput if scope is not None else CONTRACTS.get(role, Answer)
        if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "integrate":
            contract = V21Answer
        if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "verify":
            contract = V21VerificationOutput
        elif self.reliable and role in {"critique", "verify"}:
            contract = (
                SupportedCritiqueOutput if role == "critique" else SupportedVerificationOutput
            )
        instructions = INSTRUCTIONS.get(
            role,
            "Your role is solver. Return the complete task answer "
            "using the requested IDs and value format, not a plan, "
            "example, SQL string, or evaluation scores.",
        )
        if scope is not None:
            instructions += (
                " Return proposed_values only for this task, evidence-backed findings, "
                "explicit assumptions, and unresolved questions. A finding must cite "
                "SQL evidence_ref values from your own queries. If evidence is absent, "
                "leave findings empty and explain the gap in unresolved_questions."
            )
        if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "integrate":
            instructions += (
                " Return final values plus status supported, partial, or inconclusive; "
                "list evidence-backed claims, assumptions, and unresolved questions. "
                "Cite your own SQL evidence_ref for each asserted claim. If coverage "
                "is incomplete, disclose the gap rather than assert a full answer."
            )
        if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "verify":
            instructions += (
                " Independently check each claim and coverage. Distinguish disproved "
                "claims from claims not yet checked. Return claims_supported, "
                "gaps_disclosed, status, reason, and your own SQL evidence_refs."
            )
        if role == "orchestrate":
            # The v2.2 planner receives only the request and a capability summary.
            # Earlier variants also receive the SQL schema summary.
            instructions += (
                " You have no SQL access or SQL schema. Leave table selection, queries, "
                "and factual investigation to specialist workers."
                if self.method == "hybrid_v2_2"
                else " You have no SQL access. The schema summary in this prompt is "
                "authoritative for what tables and columns exist."
            )
        else:
            instructions += " Use read-only SQLite to inspect the public data as needed."
        if self.reliable and (
            role == "critique"
            or (role == "verify" and self.method not in {"hybrid_v2_1", "hybrid_v2_2"})
        ):
            instructions += (
                " Decision protocol supported-v1: distinguish structural validity, supported "
                "feasibility and abstention. Cite SQL evidence_refs returned by your query tool. "
                "Evidence references prove observation provenance, not correctness. Output schema: "
                + json.dumps(contract.model_json_schema())
            )
        visible_schema = (
            {table: self.env.schema[table] for table in scope.allowed_tables}
            if scope is not None
            else self.env.schema
        )
        prompt = self.task.prompt + "\nRole task: " + instruction
        if self.method == "hybrid_v2_2" and role == "orchestrate":
            prompt += (
                "\nSpecialist capabilities: independent read-only SQL investigation, "
                "evidence-backed task reports, integration, and independent verification."
            )
        else:
            prompt += "\nSQL schema: " + json.dumps(visible_schema)
        errors: dict[str, int] = {}
        evidence: set[str] = set()

        def execute_query(sql: str) -> dict[str, Any]:
            """Read-only SQLite query. Bounds are in the task; default 200 rows / 64KB. Paginate."""
            if self.tool_calls >= budget.tool_calls:
                raise ToolBudgetExceeded("stage tool allocation exhausted")
            self.tool_calls += 1
            result = self.env.query(
                sql, allowed_tables=frozenset(scope.allowed_tables) if scope is not None else None
            )
            if "error" in result:
                key = str(result["error"])
                errors[key] = errors.get(key, 0) + 1
                if errors[key] > self.env.state.options.output_retries + 1:
                    raise UnexpectedModelBehavior("repeated SQL error without progress")
            else:
                errors.clear()
                if (
                    (self.reliable and role in {"critique", "verify"})
                    or scope is not None
                    or (
                        self.method in {"hybrid_v2_1", "hybrid_v2_2"}
                        and role in {"integrate", "verify"}
                    )
                ):
                    ref = f"query:{role}:{self.evidence_namespace or self.index}:{self.tool_calls}"
                    evidence.add(ref)
                    result = {**result, "evidence_ref": ref}
                    self.env.state.emit(
                        {"evidence": {"ref": ref, "stage": role, "sql": sql, "result": result}}
                    )
            return result

        def check_output(output: BaseModel) -> Answer:
            answer = (
                V21Answer.model_validate(output.model_dump())
                if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "integrate"
                else Answer.model_validate(output.model_dump())
            )
            if self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "integrate":
                integrated = V21Answer.model_validate(answer)
                refs = {ref for claim in integrated.claims for ref in claim.evidence_refs}
                if not refs <= evidence or (
                    integrated.status == "supported" and not integrated.claims
                ):
                    raise ArtifactContractError("v2.1 integration claims require own SQL evidence")
                if integrated.status != "supported" and not integrated.unresolved_questions:
                    raise ArtifactContractError(
                        "partial or inconclusive answer requires explicit gaps"
                    )
                if integrated.status == "supported" and integrated.unresolved_questions:
                    raise ArtifactContractError(
                        "supported answer cannot retain unresolved questions"
                    )
                if integrated.status == "partial" and (
                    not integrated.values or not integrated.claims
                ):
                    raise ArtifactContractError(
                        "partial answer requires values and supported claims"
                    )
                if integrated.status == "inconclusive" and integrated.values:
                    raise ArtifactContractError("inconclusive answer cannot assert final values")
                if integrated.status in {"supported", "partial"}:
                    validate_artifact(
                        self.task,
                        Answer(values=integrated.values),
                        self.env.state.options.artifact_contract,
                        allow_partial=integrated.status == "partial",
                    )
            elif role not in CONTRACTS:
                validate_artifact(self.task, answer, self.env.state.options.artifact_contract)
            elif scope is not None:
                task_evidence = TaskEvidenceValues.model_validate(answer.values)
                refs = {ref for finding in task_evidence.findings for ref in finding.evidence_refs}
                if (
                    not refs <= evidence
                    or (not task_evidence.findings and not task_evidence.unresolved_questions)
                    or (task_evidence.proposed_values and not task_evidence.findings)
                ):
                    raise ArtifactContractError(
                        "TaskEvidence requires observed SQL references or explicit unresolved questions"
                    )
            elif self.method in {"hybrid_v2_1", "hybrid_v2_2"} and role == "verify":
                verdict = V21VerificationValues.model_validate(answer.values)
                refs = set(verdict.evidence_refs)
                if not refs <= evidence or (verdict.claims_supported and not refs):
                    raise ArtifactContractError("v2.1 verification requires own SQL evidence")
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
        settings = merge_reasoning_limit(settings, role)
        if not self.json_protocol:

            async def query(sql: str) -> dict[str, Any]:
                """Read-only SQLite query. Bounds are in the task; default 200 rows / 64KB. Paginate."""
                # Pydantic runs synchronous tools in a worker thread. The environment's
                # SQLite connection is owned by this event-loop thread.
                return execute_query(sql)

            agent: Agent[None, Any] = Agent(
                observed,
                output_type=cast(Any, contract),
                tools=[] if role == "orchestrate" else [query],
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
        if role == "orchestrate":
            turn_protocol = (
                " On each turn return exactly one JSON object matching this schema: "
                + json.dumps(contract.model_json_schema())
                + " No markdown."
            )
        else:
            turn_protocol = (
                " On each turn return exactly one JSON object: "
                '{"sql":"SELECT ..."} for a query, or an output matching this schema: '
                + json.dumps(contract.model_json_schema())
                + " No markdown."
            )
        json_agent = Agent(
            observed,
            output_type=str,
            model_settings=settings,
            instructions=instructions + turn_protocol,
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
                    if role == "orchestrate":
                        raise ProtocolError("the orchestrator has no SQL access")
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
                if role == "orchestrate":
                    prompt = (
                        "Invalid JSON or role output. Return one JSON object matching "
                        "this schema: " + json.dumps(contract.model_json_schema())
                    )
                else:
                    prompt = (
                        'Invalid JSON or role output. Return {"sql":"..."} or this schema: '
                        + json.dumps(contract.model_json_schema())
                    )
                # Drop the invalid attempts' model responses but keep the gathered
                # evidence: the retry prompt plus prior "SQL result" user messages
                # must survive, or the model loses the data it needs and re-queries.
                history = [
                    message for message in (history or []) if not isinstance(message, ModelResponse)
                ]
                continue
            prompt = "SQL result: " + json.dumps(execute_query(sql))
