"""Pydantic AI native agents with Harness background result delivery."""

from __future__ import annotations

import json
from typing import Any, cast
from uuid import uuid4

from pydantic_ai import Agent, ModelMessage, ModelResponse, UsageLimits
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import BackgroundTools

from poc.execution.sql_contracts import Answer
from poc.frameworks.common import RunBridge, TaskScope, initial_prompt, result_answer


class TracedModel(WrapperModel):
    """Observe actual Pydantic AI model boundaries without owning continuation."""

    def __init__(self, model: Model | str, bridge: RunBridge, scope: TaskScope) -> None:
        super().__init__(cast(Any, model))
        self.bridge = bridge
        self.scope = scope

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        request_id = uuid4().hex
        self.bridge.environment.state.request_attempts += 1
        self.bridge.record_model_input(
            self.scope, ModelMessagesTypeAdapter.dump_python(messages, mode="json"), request_id
        )
        try:
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
        except BaseException as exc:
            self.bridge.environment.state.usage_complete = False
            self.bridge.event(
                "model.failed",
                self.scope,
                request_id=request_id,
                error_category=type(exc).__name__,
            )
            raise
        self.bridge.environment.state.request_responses += 1
        if response.usage.input_tokens == response.usage.output_tokens == 0:
            self.bridge.environment.state.usage_complete = False
        self.bridge.record_model_response(self.scope, response.parts, request_id)
        return response


async def _run_agent(bridge: RunBridge, scope: TaskScope, model: Model | str) -> str:
    bridge.event("task.started", scope, role=scope.role)
    capabilities = [BackgroundTools()] if scope.role != "leaf" else []
    agent: Agent[None, str] = Agent(
        TracedModel(model, bridge, scope),
        name=f"{scope.role}_{scope.id[:8]}",
        instructions=(
            f"You are a {scope.role}. Use native tools to solve your task. "
            "You may launch more than one independent delegation in a model response. "
            "The background tool's initial acknowledgement is not its result. "
            "Wait for actual child results. Do not repeat or poll pending delegations. "
            "Call finish_task with JSON output once the task is complete."
        ),
        capabilities=capabilities,
    )

    @agent.tool_plain
    async def query_sql(sql: str) -> dict[str, Any]:
        """Execute read-only SQL against the public task tables."""
        return bridge.query(scope, sql)

    if scope.role == "leaf" and bridge.approvals is not None:

        @agent.tool_plain
        async def write_ledger(entry: str) -> dict[str, Any]:
            """Write one synthetic ledger entry after exact-call approval."""
            return await bridge.write_ledger(scope, entry)

    if scope.role != "leaf":

        @agent.tool_plain(metadata={"background": True})
        async def delegate(role_key: str, task: str) -> str:
            """Launch one allowed child agent; the result arrives later automatically."""
            child = bridge.start_task(role_key, task, scope)
            try:
                result = await _run_agent(bridge, child, model)
            except Exception as exc:
                bridge.completed(child, "", error=type(exc).__name__)
                return json.dumps(
                    {"task_id": child.id, "status": "failed", "error_category": type(exc).__name__}
                )
            bridge.completed(child, result)
            return json.dumps({"task_id": child.id, "status": "succeeded", "output": result})

    @agent.tool_plain
    async def finish_task(output: str, used_result_ids: list[str]) -> dict[str, Any]:
        """Propose a final JSON result after all delegated children finish."""
        if scope.active_children:
            return {
                "accepted": False,
                "reason": "children still running",
                "pending": sorted(scope.active_children),
            }
        if not set(used_result_ids) <= scope.completed_children:
            return {"accepted": False, "reason": "unknown result ID"}
        try:
            result_answer(output)
        except Exception:
            return {"accepted": False, "reason": "output must be a JSON object of answer fields"}
        scope.finished = output
        bridge.event("task.finish_accepted", scope, used_result_ids=used_result_ids)
        return {"accepted": True}

    async with agent.iter(
        initial_prompt(bridge, scope),
        model_settings=bridge.settings,
        usage=bridge.usage,
        usage_limits=UsageLimits(
            request_limit=bridge.budget.requests,
            total_tokens_limit=bridge.budget.total_tokens,
        ),
    ) as agent_run:
        async for _ in agent_run:
            # The tool node has executed when the next node is yielded. Stop before
            # that next model request can run after an accepted submission.
            if scope.finished is not None:
                break
    if scope.active_children:
        raise RuntimeError("native run ended with unfinished child tasks")
    if scope.finished is None:
        raise RuntimeError("native run ended without accepted finish_task")
    bridge.event(
        "model.run_stopped", scope, native_run_id=agent_run.run_id, reason="accepted_finish_task"
    )
    return scope.finished


async def run(bridge: RunBridge, model: Model | str) -> Answer:
    root = bridge.start_task("root", bridge.task.prompt)
    try:
        output = await _run_agent(bridge, root, model)
        bridge.completed(root, output)
        return result_answer(output)
    except BaseException as exc:
        bridge.completed(root, "", error=type(exc).__name__)
        raise
    finally:
        bridge.close()
