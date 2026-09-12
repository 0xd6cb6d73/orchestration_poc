"""Reusable single/review SQL-agent strategies; no evaluator or reference access."""

from __future__ import annotations

import asyncio
import json
import os
from time import perf_counter
from typing import Any, cast

from opentelemetry import trace
from pydantic_ai import Agent, ModelMessage, ModelResponse, UsageLimits
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.execution.review import ReviewDecision, review_decision_submission, review_submission
from poc.execution.sql_contracts import Answer, Budget, ModelSpec, PhaseName, TaskInput
from poc.execution.sql_ports import (
    PhaseTimeout,
    RequestTimeout,
    TaskEnvironment,
    ToolBudgetExceeded,
    TrialState,
)


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
    return answer


REVIEW_DECISION_INSTRUCTIONS = """
Review protocol decision-v1: make an explicit decision about the supplied draft.
Use SQL as needed. Your final output must be a decision object, not a bare values mapping:
{"action":"accept","reason":"why you accept","replacement":null} returns the exact draft;
{"action":"revise","reason":"what you changed and why","replacement":{"values":{...}}}
submits your complete replacement;
{"action":"decline","reason":"why you cannot complete review","replacement":null}
retains the original draft and records that review was declined, without endorsing it.
Accept and decline must not include a replacement. Revise must include one.
Do not rewrite or copy the draft when accepting it. No decision guarantees correctness.
"""


def _record_decision(
    env: TaskEnvironment, decision: ReviewDecision[Answer]
) -> ReviewDecision[Answer]:
    env.state.review_decision = decision.model_dump()
    env.state.emit({"review_decision": env.state.review_decision})
    if decision.replacement is not None:
        _check_output(env, decision.replacement)
    return decision


async def _solve_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
    draft: Answer | None = None,
    finalizing: bool = False,
) -> Answer | ReviewDecision[Answer]:
    decision_mode = (
        draft is not None and not finalizing and env.state.options.review_protocol == "decision-v1"
    )
    # The same text JSON protocol works even when a provider lacks native function calling.
    agent = Agent(
        ObservedModel(model, env.state, budget),
        output_type=str,
        model_settings=settings,
        capabilities=[Instrumentation()],
        instructions=(
            "Solve the task with the read-only SQLite tool. On EACH turn return exactly one "
            'JSON object: {"sql": "SELECT ..."} to execute a query, or '
            + (
                'a final {"action":"accept|revise|decline","reason":"...","replacement":null} decision. '
                if decision_mode
                else '{"values": {"id": value}} for your COMPLETE final answer. '
            )
            + "No markdown. SQLite joins, windows and recursive CTEs are supported. "
            "Queries return at most 200 rows / 64KB; use LIMIT/OFFSET pagination. "
            "Inspect and compute over the tables, check edge cases, never invent results."
        ),
    )
    prompt = task.prompt + "\nSQL schema: " + json.dumps(env.schema)
    if draft is not None:
        prompt += (
            "\nReview this submitted draft: "
            if decision_mode
            else "\nIndependently verify and correct this draft: "
        ) + draft.model_dump_json()
    if env.state.options.candidate_submission:
        prompt += '\nYou may checkpoint an answer with {"candidate": {"id": value}}.'
    if finalizing:
        prompt += "\nFinalization phase: return your final answer now. No tools are available."
    if decision_mode:
        prompt += REVIEW_DECISION_INSTRUCTIONS
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
            else:
                if decision_mode:
                    return _record_decision(env, ReviewDecision[Answer].model_validate(action))
                return _check_output(env, Answer.model_validate(action))
        except (ValueError, TypeError):
            if decision_mode:
                invalid_outputs += 1
                if invalid_outputs > env.state.options.output_retries:
                    raise UnexpectedModelBehavior(
                        "review decision output retries exhausted"
                    ) from None
                prompt = "Invalid review protocol output." + REVIEW_DECISION_INSTRUCTIONS
                continue
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
) -> Answer | ReviewDecision[Answer]:
    decision_mode = (
        draft is not None and not finalizing and env.state.options.review_protocol == "decision-v1"
    )

    async def query(sql: str) -> dict[str, Any]:
        """Read-only SQLite query; 200 rows and 64KB maximum. Paginate with LIMIT/OFFSET."""
        return env.query(sql)

    async def submit_candidate(values: dict[str, Any]) -> dict[str, Any]:
        """Record a candidate answer without ending the trial."""
        return env.submit_candidate(values)

    tool_functions: list[Any] = [] if finalizing else [query]
    if not finalizing and env.state.options.candidate_submission:
        tool_functions.append(submit_candidate)
    agent = Agent(
        ObservedModel(model, env.state, budget),
        output_type=ReviewDecision[Answer] if decision_mode else Answer,
        tools=tool_functions,
        retries={"output": env.state.options.output_retries},
        model_settings=settings,
        capabilities=[Instrumentation()],
        instructions=(
            "Solve the task using SQL over the provided schema. SQLite joins, "
            "windows and recursive CTEs are supported. Check all constraints "
            + (
                "and return a review decision."
                if decision_mode
                else "and return the complete values mapping."
            )
        ),
    )
    prompt = task.prompt + "\nSQL schema: " + json.dumps(env.schema)
    if draft is not None:
        prompt += (
            "\nReview this submitted draft: "
            if decision_mode
            else "\nIndependently verify and correct this draft: "
        ) + draft.model_dump_json()
    if finalizing:
        prompt += "\nFinalization phase: return your final answer now. No tools are available."

    if decision_mode:
        prompt += REVIEW_DECISION_INSTRUCTIONS

    @agent.output_validator
    async def validate_output(
        answer: Answer | ReviewDecision[Answer],
    ) -> Answer | ReviewDecision[Answer]:
        if isinstance(answer, ReviewDecision):
            return _record_decision(env, answer)
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
    if env.state.options.output_validation or env.state.options.constraint_feedback:
        raise ValueError("grader feedback is not available to orchestration strategies")
    solve = _solve_json if json_protocol else _solve_native
    state = env.state
    deadline = state.started + budget.seconds
    work_deadline = deadline - budget.finalization_seconds
    work_tokens = budget.total_tokens - budget.finalization_tokens
    work_requests = budget.requests - budget.finalization_requests

    async def phase(
        name: PhaseName,
        until: float,
        token_limit: int,
        draft: Answer | None = None,
        finalizing: bool = False,
    ) -> Answer | ReviewDecision[Answer]:
        state.phase = name
        started, tokens = perf_counter(), usage.total_tokens
        input_tokens, output_tokens = usage.input_tokens, usage.output_tokens
        binding = state.options.phase_models.get(name)
        selected_model = resolve_model(binding) if binding else model
        selected_settings = cast(ModelSettings, binding.settings) if binding else settings
        record: dict[str, Any] = {
            "name": name,
            "model_binding": binding.model_dump() if binding else None,
            "model": binding.model
            if binding
            else (model if isinstance(model, str) else model.model_name),
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
                        task,
                        env,
                        selected_model,
                        selected_settings,
                        phase_budget,
                        usage,
                        draft,
                        finalizing,
                    )
                record["status"] = "completed"
                if isinstance(result, ReviewDecision):
                    record["review_action"] = result.action
                    span.set_attribute("orchestration.review.action", result.action)
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
                    elapsed_seconds=perf_counter() - started,
                    tokens=usage.total_tokens - tokens,
                    input_tokens=usage.input_tokens - input_tokens,
                    output_tokens=usage.output_tokens - output_tokens,
                    usage_complete=state.usage_complete,
                )
                span.set_attribute("evaluation.phase.status", record["status"])
                state.emit({"phase": record.copy()})

    draft: Answer | None = None
    try:
        if review:
            draft_deadline = state.started + (work_deadline - state.started) * budget.draft_fraction
            draft = cast(
                Answer,
                await phase(
                    "draft", draft_deadline, max(1, int(work_tokens * budget.draft_fraction))
                ),
            )

            async def review_output(submission: Answer) -> Any:
                return await phase("review", work_deadline, work_tokens, submission)

            controller = (
                review_decision_submission
                if state.options.review_protocol == "decision-v1"
                else review_submission
            )
            outcome = await controller(
                draft,
                review_output,
                return_draft_on_error=state.options.review_failure_policy
                == "return_submitted_draft",
                recoverable_errors=(
                    PhaseTimeout,
                    RequestTimeout,
                    UsageLimitExceeded,
                    ToolBudgetExceeded,
                    ModelAPIError,
                    UnexpectedModelBehavior,
                ),
            )
            if outcome.source in {"draft_accept", "draft_decline"}:
                state.candidate(outcome.output, outcome.source)
            state.answer_source = outcome.source
            if outcome.reviewer_error:
                state.recovery = {"error": outcome.reviewer_error, "phase": "review"}
                state.emit({"recovery": state.recovery})
            return outcome.output
        return cast(Answer, await phase("solve", work_deadline, work_tokens))
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
        return cast(
            Answer,
            await phase(
                "finalize",
                deadline,
                budget.total_tokens,
                Answer.model_validate(state.candidates[-1]["answer"]),
                True,
            ),
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
