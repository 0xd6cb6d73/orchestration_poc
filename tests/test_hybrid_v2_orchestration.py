from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import ArchitectureOptions, Budget, TaskInput
from poc.execution.sql_orchestration import DecisionRoundsExhausted
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

WIDE_PLAN: dict[str, Any] = {
    "parent_objective": "Return the value of x.",
    "tasks": [
        {
            "task_id": f"t{index}",
            "objective": f"Scoped step {index}",
            "local_scope": f"step {index} data only",
            "definition_of_done": [f"step {index} result recorded"],
            "dependencies": [f"t{index - 1}"] if index > 1 else [],
        }
        for index in range(1, 6)
    ],
    "integration_definition_of_done": ["values cover the data table row"],
}


def hybrid_v2_model(
    *,
    invalid_first_plan: bool = False,
    fail_first_task_critique: bool = False,
    fail_all_task_critiques: bool = False,
    plan_judge_failures: int = 0,
    wide_plan: bool = False,
    fail_task_t1_twice: bool = False,
    decision: str = "revise",
) -> FunctionModel:
    state = {
        "plan_repaired": False,
        "task_critiques": 0,
        "decisions": 0,
        "plan_judges": 0,
        "t1_task_calls": 0,
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
            return payload(WIDE_PLAN if wide_plan else PLAN, info)
        if "Judge this proposed" in prompt:
            state["plan_judges"] += 1
            if state["plan_judges"] <= plan_judge_failures:
                return payload(dict.fromkeys(EVALUATION, 0), info)
            return payload(dict(EVALUATION), info)
        if "Execute ONLY this scoped task" in prompt:
            if fail_task_t1_twice and "Task id: t1" in prompt:
                state["t1_task_calls"] += 1
                if state["t1_task_calls"] == 1:
                    raise UnexpectedModelBehavior("injected task failure")
            return payload({"x": 7}, info)
        if "Audit this scoped task output" in prompt:
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


async def test_hybrid_v2_planning_repair_recovers_rejected_plans() -> None:
    answer, events, _ = await run_v2(lambda: hybrid_v2_model(plan_judge_failures=2))
    assert answer.values == {"x": 7}
    kinds = orchestration_events(events)
    assert "candidate.revised" in kinds
    assert "hybrid_v2.plan_selected" in kinds


async def test_hybrid_v2_planning_repair_is_bounded() -> None:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        with pytest.raises(CollaborationError, match="no candidate survived"):
            await ADAPTERS["hybrid_v2"](
                TASK,
                env,
                hybrid_v2_model(plan_judge_failures=99),
                {},
                Budget(),
                RunUsage(),
            )
        assert "candidate.revised" in orchestration_events(events)
    finally:
        env.close()


async def test_hybrid_v2_wide_plan_scales_round_membership() -> None:
    answer, events, _ = await run_v2(lambda: hybrid_v2_model(wide_plan=True))
    assert answer.values == {"x": 7}
    kinds = orchestration_events(events)
    assert "hybrid_v2.plan_selected" in kinds
    assert "delivery.gated" in kinds


async def test_hybrid_v2_blocked_dependency_recovers_through_decisions() -> None:
    """A task whose dependency failed is re-planned by the decision loop, not fatal."""
    state = {"t1_calls": 0, "task_critiques": 0, "decisions": 0}

    def payload(values: dict[str, Any], info: AgentInfo) -> ModelResponse:
        answer = {"values": values}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if "decompose it into between 2 and 8" in prompt:
            return payload(WIDE_PLAN, info)
        if "Your role is critic" in prompt or "Your role is verifier" in prompt:
            refs = re.findall(r"query:(?:critique|verify):\d+:\d+", prompt)
            if not refs:
                if info.function_tools:
                    return ModelResponse(
                        parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                    )
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            if "Your role is critic" in prompt:
                state["task_critiques"] += 1
                if state["task_critiques"] == 1:
                    return payload(dict.fromkeys(EVALUATION, 0), info)
                return payload(
                    {
                        "structural_validity": True,
                        "feasibility": "supported",
                        "evidence_refs": [refs[-1]],
                        "reason": "Audited the task contract",
                        "evaluation": dict(EVALUATION),
                    },
                    info,
                )
            own = [ref for ref in refs if ref.startswith("query:verify:")]
            if not own:
                if info.function_tools:
                    return ModelResponse(
                        parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                    )
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            return payload(
                {"answer_supported": True, "evidence_refs": [own[-1]], "reason": "Checked"},
                info,
            )
        if "Execute ONLY this scoped task" in prompt:
            if "Task id: t1" in prompt:
                state["t1_calls"] += 1
                if state["t1_calls"] <= 2:
                    raise UnexpectedModelBehavior("injected task failure")
            return payload({"x": 7}, info)
        if "One scoped task failed its critique" in prompt:
            state["decisions"] += 1
            match = TASK_ID_PATTERN.search(prompt)
            target = match.group(1) if match else "t1"
            original = next(task for task in WIDE_PLAN["tasks"] if task["task_id"] == target)
            revision = dict(original)
            revision["local_scope"] = f"{original['local_scope']}; sample queries only"
            return payload(
                {"decision": "revise", "rationale": "tighten scope", "revision": revision},
                info,
            )
        if "Combine these scoped task outputs" in prompt:
            return payload({"x": 7}, info)
        return payload({"x": 7}, info)

    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="reliable-v2")
    env.state.emit = events.append
    try:
        answer = await ADAPTERS["hybrid_v2"](
            TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
        )
        assert answer.values == {"x": 7}
        kinds = orchestration_events(events)
        assert "sql.branch_failed" in kinds
        assert "sql.task_blocked" in kinds
        assert "candidate.revised" in kinds
        assert "delivery.gated" in kinds
    finally:
        env.close()


async def test_hybrid_v2_stage_retry_survives_transient_stall() -> None:
    """One recoverable stage failure is retried under reliable policies."""
    state = {"orchestrate_calls": 0}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        payload_values: dict[str, Any]
        if "decompose it into between 2 and 8" in prompt:
            state["orchestrate_calls"] += 1
            if state["orchestrate_calls"] == 1:
                raise UnexpectedModelBehavior("injected provider stall")
            payload_values = PLAN
        elif "Your role is critic" in prompt or "Your role is verifier" in prompt:
            refs = re.findall(r"query:(?:critique|verify):\d+:\d+", prompt)
            if not refs:
                if info.function_tools:
                    return ModelResponse(
                        parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                    )
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            if "Your role is critic" in prompt:
                payload_values = {
                    "structural_validity": True,
                    "feasibility": "supported",
                    "evidence_refs": [refs[-1]],
                    "reason": "Checked the referenced tables",
                    "evaluation": dict(EVALUATION),
                }
            else:
                own = [ref for ref in refs if ref.startswith("query:verify:")]
                if not own:
                    if info.function_tools:
                        return ModelResponse(
                            parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                        )
                    return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
                payload_values = {
                    "answer_supported": True,
                    "evidence_refs": [own[-1]],
                    "reason": "Checked the integrated answer",
                }
        elif "Combine these scoped task outputs" in prompt:
            payload_values = {"x": 7}
        else:
            payload_values = {"x": 7}
        answer = {"values": payload_values}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="reliable-v2")
    env.state.emit = events.append
    try:
        answer = await ADAPTERS["hybrid_v2"](
            TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
        )
        assert answer.values == {"x": 7}
        assert "sql.stage_retried" in orchestration_events(events)
    finally:
        env.close()


async def test_hybrid_v2_revision_loop_recovers_refuted_task() -> None:
    answer, events, _ = await run_v2(lambda: hybrid_v2_model(fail_first_task_critique=True))
    assert answer.values == {"x": 7}
    kinds = orchestration_events(events)
    assert "candidate.revised" in kinds
    assert "hybrid_v2.plan_selected" in kinds


async def test_hybrid_v2_escalation_defers_and_integrates_viable_tasks() -> None:
    """An escalation blocks only its own task; viable siblings still integrate."""
    prompts: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        prompts.append(prompt)
        values: dict[str, Any]
        if "decompose it into between 2 and 8" in prompt:
            values = PLAN
        elif "Judge this proposed" in prompt:
            values = dict(EVALUATION)
        elif "Execute ONLY this scoped task" in prompt:
            values = {"x": 1} if "Task id: t1" in prompt else {"x": 7}
        elif "Audit this scoped task output" in prompt:
            state["task_critiques"] += 1
            values = (
                dict.fromkeys(EVALUATION, 0) if state["task_critiques"] == 1 else dict(EVALUATION)
            )
        elif "One scoped task failed its critique" in prompt:
            values = {"decision": "escalate", "rationale": "cannot be fixed"}
        elif "Combine these scoped task outputs" in prompt:
            values = {"x": 7}
        elif "Independently verify this integrated answer" in prompt:
            values = {"answer_supported": True}
        else:
            values = {"x": 7}
        answer = {"values": values}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    state = {"task_critiques": 0}
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        answer = await ADAPTERS["hybrid_v2"](
            TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
        )
        assert answer.values == {"x": 7}
        assert "hybrid_v2.escalated" in orchestration_events(events)
        verify_prompt = next(p for p in prompts if "Independently verify this integrated" in p)
        assert '"t1": "blocked"' in verify_prompt
        assert '"t2": "viable"' in verify_prompt
    finally:
        env.close()


async def test_hybrid_v2_all_blocked_tasks_end_in_decision_rounds_exhausted() -> None:
    """When every task ends blocked the run fails typed, not via the quorum catch-all."""
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        with pytest.raises(CollaborationError) as excinfo:
            await ADAPTERS["hybrid_v2"](
                TASK,
                env,
                hybrid_v2_model(fail_all_task_critiques=True, decision="escalate"),
                {},
                Budget(),
                RunUsage(),
            )
        assert isinstance(excinfo.value, DecisionRoundsExhausted)
        escalations = [
            e["orchestration"]["data"]
            for e in events
            if e.get("orchestration", {}).get("event_type") == "hybrid_v2.escalated"
        ]
        assert {e["task"] for e in escalations} == {"t1", "t2"}
        blocked = next(e["delivery_blocked"] for e in events if "delivery_blocked" in e)
        assert blocked["code"] == "decision_rounds_exhausted"
        assert blocked["blocked"] == {"t1": "escalated", "t2": "escalated"}
    finally:
        env.close()


async def test_hybrid_v2_context_infeasible_prompt_defers_without_a_model_call() -> None:
    """A scoped prompt that cannot fit the bound window is blocked before any request."""
    from poc.execution.sql_contracts import ModelSpec
    from poc.execution.sql_orchestration import TaskContextInfeasible

    calls = {"task": 0}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if "Execute ONLY this scoped task" in prompt:
            calls["task"] += 1
        values: dict[str, Any]
        if "decompose it into between 2 and 8" in prompt:
            values = PLAN
        elif "Judge this proposed" in prompt:
            values = dict(EVALUATION)
        else:
            values = {"x": 7}
        answer = {"values": values}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(
        team_policy="bounded-v1",
        phase_models={
            "task": ModelSpec(
                name="small", model_class="7B", model="fixture-small", context_window=64
            )
        },
    )
    env.state.emit = events.append
    try:
        with pytest.raises(TaskContextInfeasible):
            await ADAPTERS["hybrid_v2"](TASK, env, FunctionModel(respond), {}, Budget(), RunUsage())
        assert calls["task"] == 0
        assert "sql.task_context_infeasible" in orchestration_events(events)
        blocked = next(e["delivery_blocked"] for e in events if "delivery_blocked" in e)
        assert blocked["code"] == "task_context_infeasible"
        assert blocked["consumption"]["estimated_tokens"] > 0
        assert blocked["threshold"]["context_window"] == 64
        assert blocked["consumption"]["model"] == "fixture-small"
    finally:
        env.close()


async def test_hybrid_v2_context_preflight_falls_back_to_the_resolved_model_window() -> None:
    """Without phase-model bindings, the resolved model's own window bounds admission."""
    import tempfile
    from pathlib import Path

    from poc.execution.sql_contracts import Budget, TaskInput
    from poc.execution.sql_orchestration import SQLTeam, TaskContextInfeasible

    task = TaskInput(prompt="Infrastructure canary", tables={"data": [{"x": 7}]})
    with (
        tempfile.TemporaryDirectory() as directory,
        tempfile.TemporaryDirectory() as unbound_directory,
    ):
        team = SQLTeam(
            Path(directory),
            "hybrid_v2",
            task,
            Budget(max_output_tokens=32),
            backend="sql-json",
            model_name="fixture",
            context_window=64,
        )
        with pytest.raises(TaskContextInfeasible):
            team.require_context_feasible("x" * 8000, "task")
        # A prompt that fits the declared window is admitted without a binding.
        team.require_context_feasible("x" * 64, "task")
        # An unknown window leaves admission to request time.
        unbound = SQLTeam(
            Path(unbound_directory),
            "hybrid_v2",
            task,
            Budget(max_output_tokens=32),
            backend="sql-json",
            model_name="fixture",
        )
        try:
            unbound.require_context_feasible("x" * 8000, "task")
        finally:
            unbound.db.close()
        team.db.close()


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
