"""Strands native agents and public background-tool execution."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from pydantic_ai.exceptions import UsageLimitExceeded
from strands import Agent, tool  # pyright: ignore[reportUnknownVariableType]
from strands.background_tasks import BackgroundTasksConfig
from strands.models.openai import OpenAIModel
from strands.types.exceptions import EventLoopException

from poc.execution.sql_contracts import Answer
from poc.frameworks.common import RunBridge, TaskScope, initial_prompt, result_answer


class BudgetedOpenAIModel(OpenAIModel):
    """Strands model wrapper for shared accounting at its public stream boundary."""

    def __init__(self, bridge: RunBridge, scope: TaskScope, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bridge = bridge
        self.scope = scope

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        request_id = self.bridge.begin_model_request(
            {"system_prompt": system_prompt, "messages": messages, "tools": tool_specs},
            self.scope,
        )
        chunks: list[Any] = []
        input_tokens: int | None = None
        output_tokens: int | None = None
        responded = False
        try:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                chunks.append(event)
                metadata = event.get("metadata")
                if metadata and "usage" in metadata:
                    usage = metadata["usage"]
                    input_tokens = usage.get("inputTokens")
                    output_tokens = usage.get("outputTokens")
                yield event
            responded = True
        finally:
            self.bridge.record_model_response(self.scope, chunks, request_id)
            self.bridge.finish_model_request(
                request_id, input_tokens, output_tokens, self.scope, responded=responded
            )


def _model(bridge: RunBridge, scope: TaskScope) -> OpenAIModel:
    spec = bridge.model_spec
    if spec is None:
        raise ValueError("Strands strategy requires a configured ModelSpec")
    client_args: dict[str, Any] = {"api_key": os.environ[spec.api_key_env]}
    if spec.base_url_env:
        client_args["base_url"] = os.environ[spec.base_url_env]
    if spec.role_profiles:
        raise ValueError("Strands strategy does not support role_profiles")
    if bridge.budget.request_timeout_seconds is not None:
        client_args["timeout"] = bridge.budget.request_timeout_seconds
    params = dict(bridge.settings)
    params["max_tokens"] = bridge.output_cap()
    return BudgetedOpenAIModel(
        bridge,
        scope,
        client_args=client_args,
        model_id=spec.model,
        params=params,
        stream=False,
        context_window_limit=spec.endpoint_context_window or spec.context_window,
    )


async def _run_agent(bridge: RunBridge, scope: TaskScope) -> str:
    bridge.event("task.started", scope, role=scope.role)

    @tool
    async def query_sql(sql: str) -> str:
        """Execute read-only SQL against the public task tables."""
        return json.dumps(bridge.query(scope, sql))

    @tool
    async def finish_task(output: str, used_result_ids: list[str]) -> str:
        """Propose a final JSON result after all delegated children finish."""
        if scope.active_children:
            return json.dumps({"accepted": False, "pending": sorted(scope.active_children)})
        if not set(used_result_ids) <= scope.completed_children:
            return json.dumps({"accepted": False, "reason": "unknown result ID"})
        try:
            result_answer(output)
        except Exception:
            return json.dumps(
                {"accepted": False, "reason": "output must be a JSON object of answer fields"}
            )
        scope.finished = output
        bridge.event("task.finish_accepted", scope, used_result_ids=used_result_ids)
        return json.dumps({"accepted": True})

    tools: list[Any] = [query_sql, finish_task]
    background: BackgroundTasksConfig | None = None
    if scope.role == "leaf" and bridge.approvals is not None:

        @tool
        async def write_ledger(entry: str) -> str:
            """Write one synthetic ledger entry after exact-call approval."""
            return json.dumps(await bridge.write_ledger(scope, entry))

        tools.append(write_ledger)
    if scope.role != "leaf":

        @tool
        async def delegate(role_key: str, task: str) -> str:
            """Launch an allowed child; its completed result arrives automatically."""
            child = bridge.start_task(role_key, task, scope)
            try:
                output = await _run_agent(bridge, child)
            except Exception as exc:
                bridge.completed(child, "", error=type(exc).__name__)
                return json.dumps(
                    {"task_id": child.id, "status": "failed", "error_category": type(exc).__name__}
                )
            bridge.completed(child, output)
            return json.dumps({"task_id": child.id, "status": "succeeded", "output": output})

        tools.append(delegate)
        background = {"always": ["delegate"], "agentic": [], "wait_for_completion": True}

    agent = Agent(
        model=_model(bridge, scope),
        name=f"{scope.role}_{scope.id[:8]}",
        system_prompt=(
            f"You are a {scope.role}. Solve the given task with your tools. "
            "You may launch independent delegations together. Their initial acknowledgements "
            "are not results; wait for their actual completion. "
            "Call finish_task with JSON output after all children complete, then reply done."
        ),
        tools=tools,
        callback_handler=None,
        background_tasks=background,
    )
    try:
        result = await agent.invoke_async(initial_prompt(bridge, scope))
    except EventLoopException as exc:
        # Strands wraps provider-boundary budget failures in its event-loop exception.
        # Preserve the suite's budget_exhausted trial status for the original error.
        if isinstance(exc.original_exception, UsageLimitExceeded):
            raise exc.original_exception from exc
        raise
    bridge.event("model.run_completed", scope, stop_reason=result.stop_reason)
    if result.stop_reason == "cancelled" or scope.active_children:
        raise RuntimeError("Strands run ended with unfinished child tasks")
    if scope.finished is None:
        raise RuntimeError("Strands run ended without accepted finish_task")
    return scope.finished


async def run(bridge: RunBridge) -> Answer:
    root = bridge.start_task("root", bridge.task.prompt)
    try:
        output = await _run_agent(bridge, root)
        bridge.completed(root, output)
        return result_answer(output)
    except BaseException as exc:
        bridge.completed(root, "", error=type(exc).__name__)
        raise
    finally:
        bridge.close()
