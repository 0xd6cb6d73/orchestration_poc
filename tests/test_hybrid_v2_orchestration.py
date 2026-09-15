from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import ArchitectureOptions, Budget, TaskInput
from poc.hybrid.collaboration_controller import CollaborationError

TASK = TaskInput(prompt="Return the value of x.", tables={"data": [{"x": 7}]})

EVALUATION: dict[str, int] = dict.fromkeys(
    ("validity", "evidence", "usefulness", "novelty", "constraint_satisfaction"), 4
)

PLAN: dict[str, Any] = {
    "parent_objective": "Return the value of x.",
    "tasks": [
        {
            "task_id": "t1",
            "objective": "Look up x",
            "local_scope": "the data table only",
            "definition_of_done": ["return the single row of the data table"],
        },
        {
            "task_id": "t2",
            "objective": "Package the answer",
            "local_scope": "the t1 output only",
            "definition_of_done": ["answer values equal the data table row"],
            "dependencies": ["t1"],
        },
    ],
    "integration_definition_of_done": ["values cover the data table row"],
}


def revised_task() -> dict[str, Any]:
    revision = dict(PLAN["tasks"][0])
    revision["local_scope"] = "the data table only; exclude derived columns"
    return revision


TASK_ID_PATTERN = re.compile(r'"task_id":\s*"(t\d+)"')


def hybrid_v2_model(
    *,
    invalid_first_plan: bool = False,
    fail_first_task_critique: bool = False,
    fail_all_task_critiques: bool = False,
    decision: str = "revise",
) -> FunctionModel:
    state = {
        "plan_repaired": False,
        "task_critiques": 0,
        "decisions": 0,
    }

    def payload(values: dict[str, Any], info: AgentInfo) -> ModelResponse:
        answer = {"values": values}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if "decompose it into between 2 and 8" in prompt:
            if invalid_first_plan and not state["plan_repaired"]:
                state["plan_repaired"] = True
                return payload({"parent_objective": "only one task"}, info)
            return payload(PLAN, info)
        if "Judge this proposed" in prompt:
            return payload(dict(EVALUATION), info)
        if "Execute ONLY this scoped task" in prompt:
            return payload({"x": 7}, info)
        if "Critique this scoped task output" in prompt:
            state["task_critiques"] += 1
            if fail_all_task_critiques or (
                fail_first_task_critique and state["task_critiques"] == 1
            ):
                return payload(dict.fromkeys(EVALUATION, 0), info)
            return payload(dict(EVALUATION), info)
        if "One scoped task failed its critique" in prompt:
            state["decisions"] += 1
            if decision == "revise":
                match = TASK_ID_PATTERN.search(prompt)
                target = match.group(1) if match else "t1"
                original = next(task for task in PLAN["tasks"] if task["task_id"] == target)
                revision = dict(original)
                revision["local_scope"] = f"{original['local_scope']}; exclude derived columns"
                return payload(
                    {"decision": "revise", "rationale": "tighten scope", "revision": revision},
                    info,
                )
            return payload({"decision": "escalate", "rationale": "cannot be fixed"}, info)
        if "Combine these scoped task outputs" in prompt:
            return payload({"x": 7}, info)
        if "Independently verify this integrated answer" in prompt:
            return payload({"answer_supported": True}, info)
        return payload({"x": 7}, info)

    return FunctionModel(respond)


async def run_v2(
    stub: Callable[[], FunctionModel],
    *,
    options: ArchitectureOptions | None = None,
    suffix: str = "",
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = options or ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        answer = await ADAPTERS["hybrid_v2" + suffix](TASK, env, stub(), {}, Budget(), RunUsage())
        return answer, events, env.state.phases
    finally:
        env.close()


def orchestration_events(events: list[dict[str, Any]]) -> set[str]:
    return {e["orchestration"]["event_type"] for e in events if "orchestration" in e}


async def test_hybrid_v2_completes_through_scoped_tasks() -> None:
    answer, events, phases = await run_v2(lambda: hybrid_v2_model())
    assert answer.values == {"x": 7}
    kinds = orchestration_events(events)
    assert {
        "candidate.released",
        "candidate.selected",
        "claim.acquired",
        "task.completed",
        "worker.attached",
        "hybrid_v2.plan_selected",
        "verification.completed",
        "delivery.gated",
    } <= kinds
    assert len(phases) == 10


async def test_hybrid_v2_repairs_rejected_plan_once() -> None:
    answer, events, _ = await run_v2(lambda: hybrid_v2_model(invalid_first_plan=True))
    assert answer.values == {"x": 7}
    assert "hybrid_v2.plan_selected" in orchestration_events(events)


async def test_hybrid_v2_revision_loop_recovers_refuted_task() -> None:
    answer, events, _ = await run_v2(lambda: hybrid_v2_model(fail_first_task_critique=True))
    assert answer.values == {"x": 7}
    kinds = orchestration_events(events)
    assert "candidate.revised" in kinds
    assert "hybrid_v2.plan_selected" in kinds


async def test_hybrid_v2_escalation_blocks_delivery() -> None:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        with pytest.raises(CollaborationError, match="escalated"):
            await ADAPTERS["hybrid_v2"](
                TASK,
                env,
                hybrid_v2_model(fail_first_task_critique=True, decision="escalate"),
                {},
                Budget(),
                RunUsage(),
            )
        assert "hybrid_v2.escalated" in orchestration_events(events)
    finally:
        env.close()


async def test_hybrid_v2_decision_rounds_are_bounded() -> None:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1", hybrid_max_decision_rounds=1)
    env.state.emit = events.append
    try:
        with pytest.raises(CollaborationError, match="decision rounds exhausted"):
            await ADAPTERS["hybrid_v2"](
                TASK,
                env,
                hybrid_v2_model(fail_all_task_critiques=True),
                {},
                Budget(),
                RunUsage(),
            )
    finally:
        env.close()


def test_hybrid_v2_unknown_method_is_rejected() -> None:
    from poc.execution.sql_orchestration import adapter

    with pytest.raises(ValueError, match="unknown SQL orchestration method"):
        adapter("hybrid_v3")
