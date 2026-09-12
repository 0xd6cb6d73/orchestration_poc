from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any, cast

from opentelemetry import trace
from pydantic_ai import Agent, ModelMessage, ModelResponse, ModelRetry, UsageLimits
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Budget, ModelSpec, TaskInput
from poc.evaluation.suite.runtime import (
    PhaseTimeout,
    RequestTimeout,
    ToolBudgetExceeded,
    TrialState,
)

# Trusted in-process plugins receive only public inputs, a shared environment and usage.
Adapter = Callable[
    [TaskInput, TaskEnvironment, Model | str, ModelSettings, Budget, RunUsage], Awaitable[Answer]
]
ADAPTERS: dict[str, Adapter] = {}


def register_adapter(name: str, adapter: Adapter) -> None:
    if name in ADAPTERS:
        raise ValueError(f"adapter already registered: {name}")
    ADAPTERS[name] = adapter


def resolve_model(spec: ModelSpec) -> Model | str:
    if spec.base_url_env:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        endpoint = os.environ[spec.base_url_env]
        return OpenAIChatModel(
            spec.model,
            provider=OpenAIProvider(
                base_url=endpoint,
                api_key=os.environ[spec.api_key_env],
            ),
        )
    return spec.model


class ObservedModel(WrapperModel):
    def __init__(self, model: Model | str, state: TrialState, budget: Budget):
        super().__init__(cast(Any, model))
        self.state = state
        self.budget = budget

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.state.request_attempts += 1
        settings = dict(model_settings or {})
        if self.budget.max_output_tokens is not None:
            settings["max_tokens"] = min(
                cast(int, settings.get("max_tokens") or self.budget.max_output_tokens),
                self.budget.max_output_tokens,
            )
        try:
            async with asyncio.timeout(self.budget.request_timeout_seconds):
                response = await self.wrapped.request(
                    messages, cast(ModelSettings, settings), model_request_parameters
                )
        except TimeoutError as exc:
            self.state.usage_complete = False
            self.state.exhausted_phase = self.state.phase
            raise RequestTimeout("model request timeout") from exc
        except BaseException:
            self.state.usage_complete = False
            raise
        self.state.request_responses += 1
        if response.usage.input_tokens == 0 and response.usage.output_tokens == 0:
            self.state.usage_complete = False
        return response


def _check_output(env: TaskEnvironment, answer: Answer) -> Answer:
    env.state.candidate(answer, "output")
    if env.state.options.output_validation:
        feedback = env.state.feedback(answer, constraints=False)
        if feedback["answer_valid"] is False:
            raise ModelRetry(json.dumps(feedback))
    return answer


async def _solve_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
    draft: Answer | None = None,
    finalizing: bool = False,
) -> Answer:
    # The same text JSON protocol works even when a provider lacks native function calling.
    agent = Agent(
        ObservedModel(model, env.state, budget),
        output_type=str,
        model_settings=settings,
        capabilities=[Instrumentation()],
        instructions=(
            "Solve the task with the read-only SQLite tool. On EACH turn return exactly one "
            'JSON object: {"sql": "SELECT ..."} to execute a query, or '
            '{"values": {"id": value}} for your COMPLETE final answer. '
            "No markdown. SQLite joins, windows and recursive CTEs are supported. "
            "Queries return at most 200 rows / 64KB; use LIMIT/OFFSET pagination. "
            "Inspect and compute over the tables, check edge cases, never invent results."
        ),
    )
    prompt = task.prompt + "\nSQL schema: " + json.dumps(env.schema)
    if draft is not None:
        prompt += "\nIndependently verify and correct this draft: " + draft.model_dump_json()
        if env.state.options.constraint_feedback and not finalizing:
            prompt += "\nPublic constraint validation: " + json.dumps(
                env.validate_candidate(draft.values)
            )
    if env.state.options.candidate_submission:
        prompt += '\nYou may checkpoint an answer with {"candidate": {"id": value}}.'
    if env.state.options.constraint_feedback:
        prompt += '\nYou may validate a candidate with {"validate": {"id": value}}.'
    if finalizing:
        prompt += "\nFinalization phase: return your final answer now. No tools are available."
    history = None
    invalid_outputs = 0
    while True:
        result = await agent.run(
            prompt,
            message_history=history,
            usage=usage,
            usage_limits=UsageLimits(
                request_limit=budget.requests, total_tokens_limit=budget.total_tokens
            ),
        )
        history = result.all_messages()
        try:
            action = json.loads(result.output)
            if (
                isinstance(action, dict)
                and set(cast(dict[str, Any], action)) == {"sql"}
                and isinstance(action["sql"], str)
            ):
                if finalizing:
                    prompt = "No tools in finalization; return your final values mapping."
                else:
                    prompt = "SQL result: " + json.dumps(env.query(action["sql"]))
            elif (
                isinstance(action, dict)
                and set(cast(dict[str, Any], action)) == {"candidate"}
                and env.state.options.candidate_submission
                and not finalizing
            ):
                prompt = json.dumps(env.submit_candidate(cast(dict[str, Any], action)["candidate"]))
            elif (
                isinstance(action, dict)
                and set(cast(dict[str, Any], action)) == {"validate"}
                and env.state.options.constraint_feedback
                and not finalizing
            ):
                prompt = json.dumps(
                    env.validate_candidate(cast(dict[str, Any], action)["validate"])
                )
            else:
                try:
                    return _check_output(env, Answer.model_validate(action))
                except ModelRetry as exc:
                    invalid_outputs += 1
                    if invalid_outputs > env.state.options.output_retries:
                        raise RuntimeError("output validation retries exhausted") from exc
                    prompt = str(exc)
        except (ValueError, TypeError):
            prompt = 'Invalid action. Return ONLY {"sql":"..."} or {"values":{...}}.'


async def _solve_native(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
    draft: Answer | None = None,
    finalizing: bool = False,
) -> Answer:
    async def query(sql: str) -> dict[str, Any]:
        """Read-only SQLite query; 200 rows and 64KB maximum. Paginate with LIMIT/OFFSET."""
        return env.query(sql)

    async def submit_candidate(values: dict[str, Any]) -> dict[str, Any]:
        """Record a candidate answer without ending the trial."""
        return env.submit_candidate(values)

    async def validate_candidate(values: dict[str, Any]) -> dict[str, Any]:
        """Check a candidate against the public task constraints; consumes one tool call."""
        return env.validate_candidate(values)

    tool_functions: list[Any] = [] if finalizing else [query]
    if not finalizing and env.state.options.candidate_submission:
        tool_functions.append(submit_candidate)
    if not finalizing and env.state.options.constraint_feedback:
        tool_functions.append(validate_candidate)
    agent = Agent(
        ObservedModel(model, env.state, budget),
        output_type=Answer,
        tools=tool_functions,
        retries={"output": env.state.options.output_retries},
        model_settings=settings,
        capabilities=[Instrumentation()],
        instructions=(
            "Solve the task using SQL over the provided schema. SQLite joins, "
            "windows and recursive CTEs are supported. Check all constraints "
            "and return the complete values mapping."
        ),
    )
    prompt = task.prompt + "\nSQL schema: " + json.dumps(env.schema)
    if draft is not None:
        prompt += "\nIndependently verify and correct this draft: " + draft.model_dump_json()
        if env.state.options.constraint_feedback and not finalizing:
            prompt += "\nPublic constraint validation: " + json.dumps(
                env.validate_candidate(draft.values)
            )
    if finalizing:
        prompt += "\nFinalization phase: return your final answer now. No tools are available."

    @agent.output_validator
    async def validate_output(answer: Answer) -> Answer:
        return _check_output(env, answer)

    result = await agent.run(
        prompt,
        usage=usage,
        usage_limits=UsageLimits(
            request_limit=budget.requests,
            tool_calls_limit=budget.tool_calls,
            total_tokens_limit=budget.total_tokens,
        ),
    )
    return result.output


async def _orchestrate(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
    *,
    review: bool,
    json_protocol: bool,
) -> Answer:
    solve = _solve_json if json_protocol else _solve_native
    state = env.state
    deadline = state.started + budget.seconds
    work_deadline = deadline - budget.finalization_seconds
    work_tokens = budget.total_tokens - budget.finalization_tokens
    work_requests = budget.requests - budget.finalization_requests

    async def phase(
        name: str,
        until: float,
        token_limit: int,
        draft: Answer | None = None,
        finalizing: bool = False,
    ) -> Answer:
        state.phase = name
        started, tokens = perf_counter(), usage.total_tokens
        record: dict[str, Any] = {
            "name": name,
            "status": "running",
            "start_seconds": started - state.started,
        }
        state.phases.append(record)
        request_limit = budget.requests if finalizing else work_requests
        if name == "draft":
            request_limit = max(1, int(request_limit * budget.draft_fraction))
        phase_budget = budget.model_copy(
            update={"total_tokens": token_limit, "requests": request_limit}
        )
        with trace.get_tracer(__name__).start_as_current_span("evaluation.phase") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("evaluation.phase", name)
            try:
                if started >= until:
                    raise PhaseTimeout("phase wall-clock budget exhausted")
                if usage.total_tokens >= token_limit:
                    raise UsageLimitExceeded("phase token budget exhausted")
                async with asyncio.timeout(until - started):
                    result = await solve(
                        task, env, model, settings, phase_budget, usage, draft, finalizing
                    )
                record["status"] = "completed"
                return result
            except RequestTimeout:
                record["status"] = "request_timeout"
                state.exhausted_phase = name
                raise
            except TimeoutError as exc:
                record["status"] = "timeout"
                state.exhausted_phase = name
                raise PhaseTimeout("phase wall-clock budget exhausted") from exc
            except (UsageLimitExceeded, ToolBudgetExceeded):
                record["status"] = "budget_exhausted"
                state.exhausted_phase = name
                raise
            except BaseException:
                record["status"] = "interrupted"
                raise
            finally:
                record.update(
                    elapsed_seconds=perf_counter() - started, tokens=usage.total_tokens - tokens
                )
                span.set_attribute("evaluation.phase.status", record["status"])
                state.emit({"phase": record.copy()})

    draft: Answer | None = None
    try:
        if review:
            draft_deadline = state.started + (work_deadline - state.started) * budget.draft_fraction
            try:
                draft = await phase(
                    "draft", draft_deadline, max(1, int(work_tokens * budget.draft_fraction))
                )
            except (PhaseTimeout, UsageLimitExceeded, ToolBudgetExceeded):
                if not state.candidates or budget.draft_fraction == 1:
                    raise
                draft = Answer.model_validate(state.candidates[-1]["answer"])
            return await phase("review", work_deadline, work_tokens, draft)
        return await phase("solve", work_deadline, work_tokens)
    except (PhaseTimeout, UsageLimitExceeded, ToolBudgetExceeded):
        # A reserve enables another MODEL submission, never automatic promotion of a checkpoint.
        if not state.candidates or not (
            budget.finalization_seconds
            or budget.finalization_tokens
            or budget.finalization_requests
        ):
            raise
        if (
            perf_counter() >= deadline
            or usage.total_tokens >= budget.total_tokens
            or usage.requests >= budget.requests
        ):
            raise
        return await phase(
            "finalize",
            deadline,
            budget.total_tokens,
            Answer.model_validate(state.candidates[-1]["answer"]),
            True,
        )


async def single_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    return await _orchestrate(
        task, env, model, settings, budget, usage, review=False, json_protocol=True
    )


async def review_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    return await _orchestrate(
        task, env, model, settings, budget, usage, review=True, json_protocol=True
    )


async def single(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    return await _orchestrate(
        task, env, model, settings, budget, usage, review=False, json_protocol=False
    )


async def review(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    return await _orchestrate(
        task, env, model, settings, budget, usage, review=True, json_protocol=False
    )


async def sql_baseline(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    """Non-LLM oracle checks the environment and grader using independent SQL algorithms."""
    if "invoices" in env.schema:
        sql = """WITH dedup AS (SELECT DISTINCT * FROM payments),
        paid AS (SELECT invoice, SUM(CASE WHEN kind='refund' THEN -amount ELSE amount END) n
                 FROM dedup WHERE status='settled' GROUP BY invoice)
        SELECT i.invoice, i.amount-COALESCE(p.n,0) FROM invoices i
        LEFT JOIN paid p ON p.invoice=i.invoice
        WHERE i.revision=(SELECT MAX(revision) FROM invoices x WHERE x.invoice=i.invoice)"""
    elif "jobs" in env.schema:
        # A topological dynamic program, independent from the generating loop.
        jobs = {r[0]: r[1] for r in _rows(env, "SELECT id,duration FROM jobs")}
        edges = _rows(env, "SELECT job,requires FROM dependencies")
        finished: dict[str, Any] = {}
        while jobs:
            ready = [j for j in jobs if all(p in finished for child, p in edges if child == j)]
            if not ready:
                raise ValueError("dependency cycle")
            for job in ready:
                finished[job] = jobs.pop(job) + max(
                    (finished[p] for child, p in edges if child == job), default=0
                )
        return Answer(values=finished)
    else:
        sql = """WITH RECURSIVE effective(user,group_id) AS (
        SELECT user,group_id FROM memberships UNION
        SELECT e.user,n.parent FROM effective e JOIN nesting n ON n.child=e.group_id)
        SELECT q.id, CASE WHEN
        SUM(CASE WHEN p.effect='allow' THEN 1 ELSE 0 END)>0 AND
        SUM(CASE WHEN p.effect='deny' THEN 1 ELSE 0 END)=0 THEN 'allow' ELSE 'deny' END
        FROM queries q LEFT JOIN effective e ON e.user=q.user
        LEFT JOIN policies p ON p.group_id=e.group_id AND p.resource=q.resource GROUP BY q.id"""
    return Answer(values={str(row[0]): row[1] for row in _rows(env, sql)})


def _rows(env: TaskEnvironment, sql: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    while True:
        result = env.query(f"SELECT * FROM ({sql}) LIMIT 200 OFFSET {len(rows)}")
        if "error" in result:
            raise RuntimeError(result["error"])
        batch = result["rows"]
        rows.extend(batch)
        if len(batch) < 200:
            return rows


register_adapter("single", single)
register_adapter("review", review)
register_adapter("sql-baseline", sql_baseline)

register_adapter("single-json", single_json)
register_adapter("review-json", review_json)
