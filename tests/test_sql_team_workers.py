from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import ArchitectureOptions, Budget, ModelSpec, TaskInput
from poc.execution.sql_ports import PhaseTimeout
from poc.execution.sql_strategy import ContextBoundModel, single_json
from poc.execution.sql_team_worker import TeamWorker

TASK = TaskInput(prompt="Return x from data.", tables={"data": [{"x": 7}]})


@pytest.mark.parametrize("native", [True, False])
async def test_critic_schema_retries_before_artifact_commit(native: bool) -> None:
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        values = (
            {"x": 7}
            if calls == 1
            else dict.fromkeys(
                ("validity", "evidence", "usefulness", "novelty", "constraint_satisfaction"), 4
            )
        )
        payload = {"values": values}
        if native:
            assert "CritiqueOutput" in str(
                info.output_tools[0].parameters_json_schema
            ) or "EvaluationVector" in str(info.output_tools[0].parameters_json_schema)
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    try:
        worker = TeamWorker(
            "hybrid_v1",
            TASK,
            env,
            FunctionModel(respond),
            {},
            Budget(),
            RunUsage(),
            json_protocol=not native,
        )
        result = await worker("Critique the candidate", "critique")
        assert result.values["validity"] == 4
        assert calls == 2
        assert not env.state.candidates  # Critique scores are not candidate task answers.
    finally:
        env.close()


@pytest.mark.parametrize("team", [True, False])
async def test_malformed_json_has_finite_retry_budget(team: bool) -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{"sql":"SELECT\n x FROM data"}')])

    env = TaskEnvironment(TASK, 20)
    usage = RunUsage()
    try:
        with pytest.raises(UnexpectedModelBehavior):
            if team:
                await TeamWorker(
                    "managed_pool",
                    TASK,
                    env,
                    FunctionModel(respond),
                    {},
                    Budget(),
                    usage,
                    json_protocol=True,
                )("Plan", "plan")
            else:
                await single_json(TASK, env, FunctionModel(respond), {}, Budget(), usage)
        assert usage.requests == 2
        assert env.tool_calls == 0
    finally:
        env.close()


async def test_stage_deadline_leaves_future_time_and_cancellation_propagates() -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    try:
        worker = TeamWorker(
            "board_claim",
            TASK,
            env,
            FunctionModel(respond),
            {},
            Budget(seconds=0.1),
            RunUsage(),
            json_protocol=True,
        )
        with pytest.raises(PhaseTimeout):
            await worker("Plan", "plan")
        assert math.isclose(env.state.phases[0]["deadline_seconds"], 0.02)
        assert env.state.phases[0]["status"] == "timeout"
    finally:
        env.close()


async def test_role_model_is_called_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def resolve(spec: ModelSpec) -> FunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append(spec.model)
            return ModelResponse(parts=[TextPart('{"values":{"plan":"Inspect data"}}')])

        return FunctionModel(respond, model_name=spec.model)

    monkeypatch.setattr("poc.execution.sql_team_worker.resolve_model", resolve)
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(
        phase_models={
            "plan": ModelSpec(name="planner", model="different-planner", model_class="7B")
        }
    )
    try:
        worker = TeamWorker(
            "managed_pool", TASK, env, "unused-model", {}, Budget(), RunUsage(), json_protocol=True
        )
        await worker("Plan", "plan")
        assert calls == ["different-planner"]
        assert env.state.phases[0]["model"] == "different-planner"
        assert env.state.phases[0]["token_limit"] == 20000
    finally:
        env.close()


@pytest.mark.parametrize("fallback", [True, False])
async def test_speculative_retention_uses_first_committed_not_best(fallback: bool) -> None:
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[TextPart('{"values":{"x":-999}}')])
        raise UsageLimitExceeded("second candidate cannot finish")

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(
        speculative_failure_policy="return_first_submitted" if fallback else "fail"
    )
    try:
        if fallback:
            answer = await ADAPTERS["speculative-json"](
                TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
            )
            assert answer.values == {"x": -999}  # Known wrong: no grader-driven selection.
            assert env.state.answer_source == "first_submitted_fallback"
        else:
            with pytest.raises(UsageLimitExceeded):
                await ADAPTERS["speculative-json"](
                    TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
                )
    finally:
        env.close()


async def test_context_guard_caps_completion_without_changing_messages() -> None:
    settings_seen: list[Any] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings_seen.append(info.model_settings)
        return ModelResponse(parts=[TextPart('{"values":{"x":7}}')])

    env = TaskEnvironment(TASK, 20)
    try:
        model = ContextBoundModel(FunctionModel(respond), 5000)
        result = await single_json(TASK, env, model, {"max_tokens": 20000}, Budget(), RunUsage())
        assert result.values == {"x": 7}
        assert 0 < settings_seen[0]["max_tokens"] < 5000
    finally:
        env.close()


@pytest.mark.parametrize("method", ["hierarchical_dag", "board_claim", "managed_pool"])
async def test_native_sql_queries_stay_on_environment_thread(method: str) -> None:
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(
            isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts)
            for m in messages
        ):
            return ModelResponse(
                parts=[ToolCallPart(info.function_tools[0].name, {"sql": "SELECT x FROM data"})]
            )
        values = (
            {"plan": "Read x from data"} if "Your role is planner" in str(messages) else {"x": 7}
        )
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"values": values})])

    env = TaskEnvironment(TASK, 20)
    try:
        result = await ADAPTERS[method](TASK, env, FunctionModel(respond), {}, Budget(), RunUsage())
        assert result.values == {"x": 7}
        assert env.tool_calls == 2
    finally:
        env.close()
