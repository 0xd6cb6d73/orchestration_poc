"""Behavior checks for the evidence-first hybrid_v2_1 SQL strategy."""

from __future__ import annotations

import json
import re
from typing import Any, cast

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import ArchitectureOptions, Budget, TaskInput
from poc.execution.sql_orchestration import DecisionRoundsExhausted
from poc.execution.sql_team_worker import V21Answer
from poc.hybrid.completion_gate import DeliveryBlocked

TASK = TaskInput(
    prompt="Find x and report how well the answer is supported.",
    tables={"data": [{"x": 7}], "secret": [{"value": 99}]},
)
TASKS = [
    {
        "task_id": "t1",
        "objective": "Inspect x",
        "local_scope": "data.x",
        "out_of_scope": ["secret"],
        "allowed_tables": ["data"],
        "definition_of_done": ["Record observed x and cite its query"],
        "output_schema": "TaskEvidence",
        "budgets": {"requests": 12, "tool_calls": 6, "total_tokens": 30000, "seconds": 120},
    },
    {
        "task_id": "t2",
        "objective": "Check the first finding",
        "local_scope": "the t1 finding and data.x",
        "out_of_scope": ["secret"],
        "allowed_tables": ["data"],
        "definition_of_done": ["Check x against public data"],
        "dependencies": ["t1"],
        "output_schema": "TaskEvidence",
        "budgets": {"requests": 12, "tool_calls": 6, "total_tokens": 30000, "seconds": 120},
    },
]
PLAN = {
    "parent_objective": TASK.prompt,
    "tasks": TASKS,
    "integration_definition_of_done": ["State x with its support and disclose any gap"],
}
SCORES = {
    "validity": 4,
    "evidence": 4,
    "usefulness": 4,
    "novelty": 4,
    "constraint_satisfaction": 4,
}


def model_for_v21(
    stages: list[str],
    *,
    reject_first_t1: bool = False,
    wrong_target: bool = False,
    status: str = "supported",
    task_request_limit: int | None = None,
    wide_plan: bool = False,
    fail_t2: bool = False,
) -> FunctionModel:
    state = {"t1_audits": 0}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        refs = re.findall(r"query:(?:task|critique|integrate|verify):[^\"' ]+", prompt)

        def own(role: str) -> list[str]:
            return [ref for ref in refs if ref.startswith(f"query:{role}:")]

        if "decompose it into between 2 and 8" in prompt:
            stages.append("plan")
            tasks = [dict(task) for task in TASKS]
            if wide_plan:
                tasks.extend(
                    {
                        **TASKS[0],
                        "task_id": f"t{index}",
                        "objective": f"Inspect x independently {index}",
                    }
                    for index in range(3, 6)
                )
            plan: dict[str, Any] = {**PLAN, "tasks": tasks}
            if task_request_limit is not None:
                cast(list[dict[str, Any]], plan["tasks"])[0]["budgets"] = {
                    **cast(dict[str, int], TASKS[0]["budgets"]),
                    "requests": task_request_limit,
                }
            values: dict[str, Any] = plan
        elif "Judge this proposed" in prompt:
            stages.append("plan_judge")
            values = {
                "structural_validity": True,
                "feasibility": "abstain",
                "evidence_refs": [],
                "reason": "Feasibility needs the planned investigation",
                "evaluation": {**SCORES, "evidence": 0},
            }
        elif "Execute ONLY this scoped task" in prompt:
            task_id = "t2" if "Task id: t2" in prompt else "t1"
            if task_id == "t2":
                assert "Accepted prerequisite outputs:" in prompt
            if "SQL result:" not in prompt:
                stages.append(f"{task_id}_query")
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            stages.append(f"{task_id}_output")
            values = {
                "proposed_values": {"x": 7},
                "findings": [
                    {
                        "claim": "x is 7",
                        "evidence_refs": ["foreign"]
                        if fail_t2 and task_id == "t2"
                        else [own("task")[-1]],
                    }
                ],
                "assumptions": [],
                "unresolved_questions": [],
            }
        elif "Audit this scoped task output" in prompt:
            task_id = "t2" if '"task_id": "t2"' in prompt else "t1"
            if not own("critique"):
                stages.append(f"{task_id}_audit_query")
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            stages.append(f"{task_id}_audit")
            reject = reject_first_t1 and task_id == "t1" and state["t1_audits"] == 0
            if task_id == "t1":
                state["t1_audits"] += 1
            values = {
                "structural_validity": True,
                "feasibility": "unsupported" if reject else "supported",
                "evidence_refs": [own("critique")[-1]],
                "reason": "Needs revision" if reject else "Checked against SQL",
                "evaluation": {**SCORES, "validity": 0} if reject else SCORES,
            }
        elif "One scoped task failed its critique" in prompt:
            stages.append("decision")
            revision = dict(TASKS[1] if wrong_target else TASKS[0])
            revision["local_scope"] = str(revision["local_scope"]) + "; revised after critique"
            values = {
                "decision": "revise",
                "rationale": "tighten the finding",
                "revision": revision,
            }
        elif "Combine these scoped task outputs" in prompt:
            if not own("integrate"):
                stages.append("integrate_query")
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            stages.append("integrate")
            result = {
                "values": {} if status == "inconclusive" else {"x": 7},
                "status": status,
                "claims": []
                if status == "inconclusive"
                else [{"claim": "x is 7", "evidence_refs": [own("integrate")[-1]]}],
                "assumptions": [],
                "unresolved_questions": []
                if status == "supported"
                else ["Other requested values are unresolved"],
            }
            return ModelResponse(parts=[TextPart(json.dumps(result))])
        elif "Independently verify this integrated answer" in prompt:
            if not own("verify"):
                stages.append("verify_query")
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            stages.append("verify")
            values = {
                "status": status,
                "claims_supported": True,
                "gaps_disclosed": True,
                "evidence_refs": [own("verify")[-1]],
                "reason": "Claims match the data and gaps are disclosed",
            }
        else:
            raise AssertionError(f"unexpected model call: {prompt[-400:]}")
        return ModelResponse(parts=[TextPart(json.dumps({"values": values}))])

    return FunctionModel(respond)


async def run_v21(model: FunctionModel) -> tuple[V21Answer, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(team_policy="reliable-v2")
    env.state.emit = events.append
    try:
        answer = await ADAPTERS["hybrid_v2_1-json"](
            TASK,
            env,
            model,
            {},
            Budget(requests=200, tool_calls=100, total_tokens=500000, seconds=300),
            RunUsage(),
        )
        return V21Answer.model_validate(answer), events
    finally:
        env.close()


async def test_v21_accepts_exploratory_plan_and_audits_before_dependents() -> None:
    stages: list[str] = []
    answer, events = await run_v21(model_for_v21(stages))
    assert answer.status == "supported"
    assert answer.values == {"x": 7}
    assert stages.index("t1_audit") < stages.index("t2_query")
    assert any(e.get("orchestration", {}).get("event_type") == "delivery.gated" for e in events)
    assessed = next(
        e["orchestration"]["data"]
        for e in events
        if e.get("orchestration", {}).get("event_type") == "hybrid_v2_1.answer_assessed"
    )
    assert {ref.split(":")[1] for ref in assessed["evidence_refs"]} == {
        "task",
        "integrate",
        "verify",
    }


async def test_v21_revises_prerequisite_before_running_dependent() -> None:
    stages: list[str] = []
    answer, _ = await run_v21(model_for_v21(stages, reject_first_t1=True))
    assert answer.values == {"x": 7}
    assert stages.count("t1_audit") == 2
    assert stages.index("t2_query") > stages.index("decision")


async def test_v21_assigns_distinct_workers_to_wide_plan() -> None:
    answer, events = await run_v21(model_for_v21([], wide_plan=True))
    assert answer.status == "supported"
    produced = [
        e["orchestration"]["data"]["task_id"]
        for e in events
        if e.get("orchestration", {}).get("event_type") == "task.completed"
        and e["orchestration"]["data"]["task_id"].startswith("task-t")
    ]
    assert len(produced) == 5
    assert len(set(produced)) == 5


async def test_v21_reports_partial_answer_with_qualified_claims() -> None:
    answer, events = await run_v21(model_for_v21([], status="partial"))
    assert answer.status == "partial"
    assert answer.unresolved_questions
    assert any(
        e.get("orchestration", {}).get("event_type") == "hybrid_v2_1.answer_assessed"
        for e in events
    )


async def test_v21_requires_qualified_status_when_a_task_is_unaccepted() -> None:
    partial, _ = await run_v21(model_for_v21([], fail_t2=True, status="partial"))
    assert partial.status == "partial"
    with pytest.raises(DeliveryBlocked, match="accepted coverage"):
        await run_v21(model_for_v21([], fail_t2=True, status="supported"))


async def test_v21_reports_inconclusive_answer_without_claims() -> None:
    answer, _ = await run_v21(model_for_v21([], status="inconclusive"))
    assert answer.status == "inconclusive"
    assert answer.values == {}
    assert answer.claims == []
    assert answer.unresolved_questions


def test_sql_table_allowlist_denies_out_of_scope_reads() -> None:
    env = TaskEnvironment(TASK, 3)
    try:
        assert env.query("SELECT x FROM data", allowed_tables=frozenset({"data"}))["rows"] == [[7]]
        assert "error" in env.query("SELECT value FROM secret", allowed_tables=frozenset({"data"}))
    finally:
        env.close()


async def test_v21_rejects_revision_of_another_pending_task() -> None:
    stages: list[str] = []
    with pytest.raises(DecisionRoundsExhausted):
        await run_v21(model_for_v21(stages, reject_first_t1=True, wrong_target=True))
    assert "decision" in stages
    assert "t2_query" not in stages


async def test_v21_task_request_limit_covers_the_retry_pair() -> None:
    stages: list[str] = []
    with pytest.raises(DecisionRoundsExhausted):
        await run_v21(model_for_v21(stages, wrong_target=True, task_request_limit=1))
    assert stages.count("t1_query") == 1
    assert "t1_output" not in stages
