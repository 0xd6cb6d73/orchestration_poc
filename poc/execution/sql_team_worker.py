"""Typed, budgeted SQL workers. Policies see public task inputs only."""

from __future__ import annotations

import asyncio
import json
from time import perf_counter
from typing import Any, cast

from opentelemetry import trace
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.execution.sql_contracts import Answer, Budget, PhaseName, StrictModel, TaskInput
from poc.execution.sql_ports import (
    PhaseTimeout,
    RequestTimeout,
    TaskEnvironment,
    ToolBudgetExceeded,
)
from poc.execution.sql_strategy import ObservedModel, resolve_model
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
        self.settings, self.budget, self.usage = settings, budget, usage
        self.json_protocol = json_protocol
        self.weights = env.state.options.stage_weights or WEIGHTS[method]
        if len(self.weights) != len(WEIGHTS[method]):
            raise ValueError("stage_weights length does not match the strategy")
        self.index = 0
        self.tool_calls = 0

    async def __call__(self, instruction: str, role: PhaseName = "solve") -> Answer:
        state, usage, budget = self.env.state, self.usage, self.budget
        index = self.index
        self.index += 1
        # Unused allocation carries forward. A stage cannot borrow from future stages.
        cumulative = 1.0 if index == len(self.weights) - 1 else sum(self.weights[: index + 1])
        deadline = state.started + budget.seconds * cumulative
        token_limit = max(1, int(budget.total_tokens * cumulative))
        request_limit = max(1, int(budget.requests * cumulative))
        binding = state.options.phase_models.get(role) or state.options.phase_models.get("solve")
        model = resolve_model(binding) if binding else self.model
        settings = cast(ModelSettings, binding.settings) if binding else self.settings
        stage_budget = budget.model_copy(
            update={
                "total_tokens": token_limit,
                "requests": request_limit,
                "tool_calls": max(1, int(budget.tool_calls * cumulative)),
                "request_timeout_seconds": budget.request_timeout_seconds or 60,
                "max_output_tokens": budget.max_output_tokens
                or (4096 if role in {"plan", "critique", "verify"} else 16384),
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
            "policy": "sql-team-v2",
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
                record["status"] = "completed"
                if role not in CONTRACTS:
                    state.candidate(answer, "output")
                return answer
            except RequestTimeout:
                record["status"] = "request_timeout"
                raise
            except TimeoutError as exc:
                record["status"] = "timeout"
                raise PhaseTimeout("stage wall-clock allocation exhausted") from exc
            except (UsageLimitExceeded, ToolBudgetExceeded):
                record["status"] = "budget_exhausted"
                raise
            except BaseException:
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
        instructions = INSTRUCTIONS.get(
            role,
            "Your role is solver. Return the complete task answer "
            "using the requested IDs and value format, not a plan, "
            "example, SQL string, or evaluation scores.",
        )
        instructions += " Use read-only SQLite to inspect the public data as needed."
        prompt = (
            self.task.prompt
            + "\nRole task: "
            + instruction
            + "\nSQL schema: "
            + json.dumps(self.env.schema)
        )
        errors: dict[str, int] = {}

        def execute_query(sql: str) -> dict[str, Any]:
            """Read-only SQLite query; 200 rows / 64KB maximum. Paginate with LIMIT/OFFSET."""
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
            return result

        limits = UsageLimits(
            request_limit=budget.requests,
            total_tokens_limit=budget.total_tokens,
            tool_calls_limit=budget.tool_calls,
        )
        observed = ObservedModel(model, self.env.state, budget, self.usage)
        if not self.json_protocol:

            async def query(sql: str) -> dict[str, Any]:
                """Read-only SQLite query; 200 rows / 64KB maximum. Paginate with LIMIT/OFFSET."""
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
            result = await agent.run(prompt, usage=self.usage, usage_limits=limits)
            return Answer.model_validate(result.output.model_dump())
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
                action = json.loads(result.output)
                if (
                    isinstance(action, dict)
                    and set(cast(dict[str, Any], action)) == {"sql"}
                    and isinstance(action["sql"], str)
                ):
                    sql = action["sql"]
                else:
                    return Answer.model_validate(contract.model_validate(action).model_dump())
            except (ValueError, TypeError):
                invalid += 1
                if invalid > self.env.state.options.output_retries:
                    raise UnexpectedModelBehavior("SQL role output retries exhausted") from None
                prompt = (
                    'Invalid JSON or role output. Return {"sql":"..."} or this schema: '
                    + json.dumps(contract.model_json_schema())
                )
                continue
            prompt = "SQL result: " + json.dumps(execute_query(sql))
