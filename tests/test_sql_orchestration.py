from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import ArchitectureOptions, Budget, TaskInput
from poc.execution.sql_orchestration import METHODS
from poc.models import ExecutionMode

TASK = TaskInput(prompt="Return the value of x.", tables={"data": [{"x": 7}]})


def model() -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if "decompose it into between 2 and 8" in prompt:
            values = {
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
        elif "Judge this proposed" in prompt or "Audit this scoped task output" in prompt:
            values = dict.fromkeys(
                ("validity", "evidence", "usefulness", "novelty", "constraint_satisfaction"), 4
            )
        elif "One scoped task failed its critique" in prompt:
            values = {
                "decision": "revise",
                "rationale": "tighten scope",
                "revision": {
                    "task_id": "t1",
                    "objective": "Look up x",
                    "local_scope": "the data table only",
                    "definition_of_done": ["return the single row of the data table"],
                },
            }
        elif "Critique this proposed" in prompt:
            values = dict.fromkeys(
                ("validity", "evidence", "usefulness", "novelty", "constraint_satisfaction"), 4
            )
        elif (
            "Independently verify this integrated answer" in prompt
            or "Independently verify this selected" in prompt
        ):
            values = {"answer_supported": True}
        elif "Develop a solution plan" in prompt:
            values = {"plan": "SELECT x FROM data"}
        else:
            values = {"x": 7}
        answer = {"values": values}
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
        return ModelResponse(parts=[TextPart(json.dumps(answer))])

    return FunctionModel(respond)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("suffix", ["", "-json"])
async def test_all_methods_execute_real_controllers(method: str, suffix: str) -> None:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    usage = RunUsage()
    try:
        answer = await ADAPTERS[method + suffix](TASK, env, model(), {}, Budget(), usage)
        assert answer.values == {"x": 7}
        assert (
            usage.requests
            == {
                "hierarchical_dag": 2,
                "board_claim": 2,
                "managed_pool": 2,
                "speculative": 3,
                "hybrid_v1": 5,
                "hybrid_v2": 10,
            }[method]
        )
        kinds = {e["orchestration"]["event_type"] for e in events if "orchestration" in e}
        required = {
            "hierarchical_dag": {"plan.execution_mode_selected"},
            "board_claim": {"claim.acquired", "task.completed", "worker.attached"},
            "managed_pool": {"offer.published", "assignment.created", "assignment.completed"},
            "speculative": {"candidate.started", "candidate.completed", "reconciliation.completed"},
            "hybrid_v1": {
                "candidate.released",
                "candidate.selected",
                "claim.acquired",
                "verification.completed",
                "delivery.gated",
            },
            "hybrid_v2": {
                "candidate.released",
                "candidate.selected",
                "claim.acquired",
                "task.completed",
                "worker.attached",
                "hybrid_v2.plan_selected",
                "verification.completed",
                "delivery.gated",
            },
        }
        assert required[method] <= kinds, kinds
        assert len(env.state.phases) == usage.requests
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
async def test_workers_share_request_ceiling(method: str) -> None:
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    usage = RunUsage()
    try:
        with pytest.raises(UsageLimitExceeded):
            await ADAPTERS[method](TASK, env, model(), {}, Budget(requests=1), usage)
        assert usage.requests == 1
    finally:
        env.close()


def test_every_runtime_mode_has_both_protocols() -> None:
    for mode in ExecutionMode:
        assert mode.value in ADAPTERS
        assert mode.value + "-json" in ADAPTERS
    assert "hybrid_v1" in ADAPTERS
    assert "hybrid_v2" in ADAPTERS
    assert "hybrid_v2-json" in ADAPTERS


async def test_cancelled_trial_cleans_scheduler_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    started = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    task = asyncio.ensure_future(
        ADAPTERS["board_claim"](TASK, env, FunctionModel(respond), {}, Budget(), RunUsage())
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert list(tmp_path.iterdir()) == []
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
async def test_workers_share_tool_ceiling(method: str) -> None:
    from poc.execution.sql_ports import ToolBudgetExceeded

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if "Your role is orchestrator" in (info.instructions or ""):
            # The tool-less orchestrator cannot query; it plans directly.
            return ModelResponse(
                parts=[
                    TextPart(
                        json.dumps(
                            {
                                "values": {
                                    "parent_objective": "Return the value of x.",
                                    "tasks": [
                                        {
                                            "task_id": "t1",
                                            "objective": "Look up x",
                                            "local_scope": "data table",
                                            "definition_of_done": ["return the row"],
                                        },
                                        {
                                            "task_id": "t2",
                                            "objective": "Package the answer",
                                            "local_scope": "the t1 output only",
                                            "definition_of_done": ["values equal the row"],
                                            "dependencies": ["t1"],
                                        },
                                    ],
                                    "integration_definition_of_done": ["values cover the row"],
                                }
                            }
                        )
                    )
                ]
            )
        if "SQL result:" not in prompt:
            return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
        if "Develop a solution plan" in prompt:
            return ModelResponse(parts=[TextPart('{"values":{"plan":"SELECT x FROM data"}}')])
        return ModelResponse(
            parts=[
                TextPart(
                    '{"values":{"validity":4,"evidence":4,"usefulness":4,'
                    '"novelty":4,"constraint_satisfaction":4}}'
                )
            ]
        )

    env = TaskEnvironment(TASK, 1)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    usage = RunUsage()
    try:
        with pytest.raises(ToolBudgetExceeded):
            await ADAPTERS[method + "-json"](
                TASK, env, FunctionModel(respond), {}, Budget(tool_calls=1), usage
            )
        assert env.tool_calls == 1
        expected_requests = 5 if method == "hybrid_v2" else 3
        assert usage.requests == expected_requests
    finally:
        env.close()


async def test_full_matrix_records_and_grades_all_methods(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from poc.evaluation.suite import runner
    from poc.evaluation.suite.models import Matrix, ModelSpec, TaskCase

    case = TaskCase(
        id="sql-public",
        family="ledger",
        seed=0,
        split="dev",
        difficulty="standard",
        input=TASK,
        expected={"x": 7},
    )

    def cases(matrix: Matrix) -> list[TaskCase]:
        return [case]

    def resolve(spec: ModelSpec) -> FunctionModel:
        return model()

    monkeypatch.setattr(runner, "make_cases", cases)
    monkeypatch.setattr(runner, "resolve_model", resolve)
    strategies = [method + suffix for method in METHODS for suffix in ("", "-json")]
    matrix = Matrix(
        models=[ModelSpec(name="fake", model_class="baseline", model="test")],
        strategies=strategies,
        architecture_options={s: ArchitectureOptions(team_policy="bounded-v1") for s in strategies},
        families=["ledger"],
        seeds=[0],
        repetitions=1,
    )
    path = tmp_path / "report.jsonl"
    report = await runner.run_matrix(matrix, path)
    assert {t["strategy"] for t in report["trials"]} == set(strategies)
    assert all(t["status"] == "completed" and t["scores"]["exact"] == 1 for t in report["trials"])
    loaded = runner.read_report(path)
    assert loaded["trials"] == report["trials"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {
        r["event"]["orchestration"]["method"] for r in rows if "orchestration" in r.get("event", {})
    } == set(METHODS)


async def test_hybrid_rejected_verification_does_not_submit_selected_candidate() -> None:
    from poc.hybrid.completion_gate import DeliveryBlocked

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        values: dict[str, Any]
        if "Independently verify this selected" in prompt:
            values = {"answer_supported": False}
        elif "Critique this proposed" in prompt:
            values = dict.fromkeys(
                ("validity", "evidence", "usefulness", "novelty", "constraint_satisfaction"), 4
            )
        else:
            values = {"x": 7}
        return ModelResponse(parts=[TextPart(json.dumps({"values": values}))])

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    try:
        with pytest.raises(DeliveryBlocked):
            await ADAPTERS["hybrid_v1-json"](
                TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
            )
    finally:
        env.close()
