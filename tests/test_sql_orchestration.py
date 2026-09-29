from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import (
    Answer,
    ArchitectureOptions,
    Budget,
    PhaseName,
    TaskInput,
)
from poc.execution.sql_orchestration import (
    METHODS,
    DecisionRoundsExhausted,
    OrchestratorEscalated,
    PlanningRejected,
    StageBudgetExhausted,
    TaskContextInfeasible,
)
from poc.hybrid.collaboration_controller import CollaborationError
from poc.models import ExecutionMode

TASK = TaskInput(prompt="Return the value of x.", tables={"data": [{"x": 7}]})


def model() -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if "decompose it into between 2 and 8" in prompt:
            values: dict[str, Any] = {
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
            if "TaskEvidence" in prompt:
                for task in cast(list[dict[str, Any]], values["tasks"]):
                    task.update(
                        allowed_tables=["data"],
                        output_schema="TaskEvidence",
                        budgets={
                            "requests": 12,
                            "tool_calls": 6,
                            "total_tokens": 30000,
                            "seconds": 120,
                        },
                    )
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
        if (
            "Return proposed_values only for this task" in (info.instructions or "")
            or "Return final values plus status" in (info.instructions or "")
            or "Independently check each claim" in (info.instructions or "")
        ):
            role = (
                "task"
                if "Return proposed_values" in (info.instructions or "")
                else "integrate"
                if "Return final values" in (info.instructions or "")
                else "verify"
            )
            refs = re.findall(rf"query:{role}:[A-Za-z0-9_-]+:[0-9]+", prompt)
            query_seen = "SQL result:" in prompt or "ToolReturnPart" in prompt
            if not refs or not query_seen:
                if info.function_tools:
                    return ModelResponse(
                        parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                    )
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            if role == "task":
                values = {
                    "proposed_values": {"x": 7},
                    "findings": [{"claim": "x is 7", "evidence_refs": [refs[-1]]}],
                    "assumptions": [],
                    "unresolved_questions": [],
                }
            elif role == "integrate":
                answer = {
                    "values": {"x": 7},
                    "status": "supported",
                    "claims": [{"claim": "x is 7", "evidence_refs": [refs[-1]]}],
                    "assumptions": [],
                    "unresolved_questions": [],
                }
                if info.output_tools:
                    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, answer)])
                return ModelResponse(parts=[TextPart(json.dumps(answer))])
            else:
                values = {
                    "status": "supported",
                    "claims_supported": True,
                    "gaps_disclosed": True,
                    "evidence_refs": [refs[-1]],
                    "reason": "Checked x",
                }
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
                "hybrid_v2_1": 14,
                "hybrid_v2_2": 11,
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
            "hybrid_v2_2": {
                "candidate.released",
                "claim.acquired",
                "task.completed",
                "worker.attached",
                "hybrid_v2_2.plan_selected",
                "verification.completed",
                "delivery.gated",
            },
            "hybrid_v2_1": {
                "candidate.released",
                "candidate.selected",
                "claim.acquired",
                "task.completed",
                "worker.attached",
                "hybrid_v2_1.plan_selected",
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
        assert len(env.state.phases) == (
            7 if method == "hybrid_v2_2" else 10 if method == "hybrid_v2_1" else usage.requests
        )
    finally:
        env.close()


@pytest.mark.parametrize("method", METHODS)
async def test_workers_share_request_ceiling(method: str) -> None:
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    usage = RunUsage()
    try:
        # hybrid_v2 terminates budget-aware before re-claiming a spent budget; the
        # other strategies still hit the provider-side usage limit.
        expected = (
            StageBudgetExhausted
            if method in {"hybrid_v2", "hybrid_v2_1", "hybrid_v2_2"}
            else UsageLimitExceeded
        )
        with pytest.raises(expected):
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
    assert "hybrid_v2_1-json" in ADAPTERS
    assert "hybrid_v2_2-json" in ADAPTERS


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
                                            **(
                                                {
                                                    "allowed_tables": ["data"],
                                                    "output_schema": "TaskEvidence",
                                                    "budgets": {
                                                        "requests": 12,
                                                        "tool_calls": 6,
                                                        "total_tokens": 30000,
                                                        "seconds": 120,
                                                    },
                                                }
                                                if method == "hybrid_v2_1"
                                                else {}
                                            ),
                                            "definition_of_done": ["return the row"],
                                        },
                                        {
                                            "task_id": "t2",
                                            "objective": "Package the answer",
                                            "local_scope": "the t1 output only",
                                            **(
                                                {
                                                    "allowed_tables": ["data"],
                                                    "output_schema": "TaskEvidence",
                                                    "budgets": {
                                                        "requests": 12,
                                                        "tool_calls": 6,
                                                        "total_tokens": 30000,
                                                        "seconds": 120,
                                                    },
                                                }
                                                if method == "hybrid_v2_1"
                                                else {}
                                            ),
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
        if method == "hybrid_v2_2" and "Execute ONLY this scoped task" in prompt:
            refs = re.findall(r"query:task:[A-Za-z0-9_-]+:[0-9]+", prompt)
            return ModelResponse(
                parts=[
                    TextPart(
                        json.dumps(
                            {
                                "values": {
                                    "proposed_values": {"x": 7},
                                    "findings": [{"claim": "x is 7", "evidence_refs": [refs[-1]]}],
                                    "assumptions": [],
                                    "unresolved_questions": [],
                                }
                            }
                        )
                    )
                ]
            )
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
        expected_requests = (
            4 if method == "hybrid_v2_2" else 5 if method in {"hybrid_v2", "hybrid_v2_1"} else 3
        )
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

    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        with pytest.raises(DeliveryBlocked):
            await ADAPTERS["hybrid_v1-json"](
                TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
            )
        blocked = next(e["delivery_blocked"] for e in events if "delivery_blocked" in e)
        assert blocked["code"] == "verification_rejected"
    finally:
        env.close()


@pytest.mark.parametrize(
    "error,code",
    [
        (
            OrchestratorEscalated("t1", "cannot proceed"),
            "orchestrator_escalated",
        ),
        (
            PlanningRejected(["every plan candidate was refuted by the plan judges"]),
            "planning_rejected",
        ),
        (
            TaskContextInfeasible(
                prompt_chars=9000,
                estimated_tokens=2250,
                context_window=2048,
                model="fixture-small",
                reserve_tokens=1024,
            ),
            "task_context_infeasible",
        ),
        (
            StageBudgetExhausted(
                stage="task-t1",
                tokens=100000,
                requests=40,
                token_limit=100000,
                request_limit=40,
            ),
            "stage_budget_exhausted",
        ),
        (
            DecisionRoundsExhausted(
                "hybrid_v2 decision rounds exhausted before integration",
                {"t1": {"reason": "escalated"}},
            ),
            "decision_rounds_exhausted",
        ),
        (
            CollaborationError("team is smaller than the pinned proposal population"),
            "proposal_quorum_not_met",
        ),
    ],
)
async def test_typed_collaboration_errors_map_to_failure_codes(
    error: CollaborationError, code: str
) -> None:
    """The adapter attributes failures by exception type, not a quorum catch-all."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise error

    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="bounded-v1")
    env.state.emit = events.append
    try:
        with pytest.raises(type(error)):
            await ADAPTERS["hybrid_v2"](TASK, env, FunctionModel(respond), {}, Budget(), RunUsage())
        blocked = next(e["delivery_blocked"] for e in events if "delivery_blocked" in e)
        assert blocked["code"] == code
        assert blocked["type"] == type(error).__name__
    finally:
        env.close()


async def test_bounded_board_work_refuses_claim_when_run_budget_spent() -> None:
    from tempfile import TemporaryDirectory

    from poc.execution.sql_orchestration import SQLTeam

    usage = RunUsage()
    usage.requests = Budget().requests
    calls = 0

    async def solve(prompt: str, role: PhaseName) -> Answer:
        nonlocal calls
        calls += 1
        return Answer(values={})

    with TemporaryDirectory(prefix="sql-budget-") as directory:
        team = SQLTeam(
            Path(directory),
            "hybrid_v2",
            TASK,
            Budget(),
            backend="sql-native",
            model_name="fixture",
            options=ArchitectureOptions(),
            usage=usage,
        )
        try:
            await team.start()
            with pytest.raises(StageBudgetExhausted):
                await team.bounded_board_work(0, "task-t1", solve, "prompt")
            assert calls == 0
            kinds = {e["event_type"] for e in team.db.events("sql-run")}
            assert "claim.acquired" not in kinds
        finally:
            team.db.close()


async def test_bounded_board_work_abandons_retry_when_run_budget_spent() -> None:
    from tempfile import TemporaryDirectory

    from pydantic_ai.exceptions import UnexpectedModelBehavior

    from poc.execution.sql_orchestration import SQLTeam

    usage = RunUsage()
    calls = 0

    async def solve(prompt: str, role: PhaseName) -> Answer:
        nonlocal calls
        calls += 1
        usage.requests += 1
        raise UnexpectedModelBehavior("injected stage failure")

    with TemporaryDirectory(prefix="sql-retry-") as directory:
        # One request budget: the failed solve consumes it, so the retry re-claim
        # must be refused instead of spinning through zero-token attempts.
        team = SQLTeam(
            Path(directory),
            "hybrid_v2",
            TASK,
            Budget(requests=1),
            backend="sql-native",
            model_name="fixture",
            options=ArchitectureOptions(),
            usage=usage,
        )
        try:
            await team.start()
            with pytest.raises(StageBudgetExhausted):
                await team.bounded_board_work(0, "task-t1", solve, "prompt", role="task")
            assert calls == 1
            assert usage.requests == 1
            kinds = {e["event_type"] for e in team.db.events("sql-run")}
            assert "sql.stage_retried" not in kinds
        finally:
            team.db.close()
