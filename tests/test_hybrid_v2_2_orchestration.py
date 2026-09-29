"""The v2.2 planner sees the question and delegates evidence work to specialists."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import ArchitectureOptions, Budget, TaskInput
from poc.execution.sql_orchestration import PlanningRejected
from poc.execution.sql_team_worker import V21Answer

TASK = TaskInput(
    prompt="Determine x and explain whether the answer is supported.",
    tables={"data": [{"x": 7}], "secret": [{"other": 99}]},
)
CONCEPTUAL_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "t1",
        "objective": "Establish the candidate value for x",
        "local_scope": "the requested value",
        "out_of_scope": ["final synthesis"],
        "definition_of_done": ["Report a value with evidence or an unresolved gap"],
    },
    {
        "task_id": "t2",
        "objective": "Assess support for the candidate value",
        "local_scope": "support for the t1 finding",
        "out_of_scope": ["final synthesis"],
        "definition_of_done": ["Report whether the candidate has support"],
        "dependencies": ["t1"],
    },
]


def scripted_model(
    observed: list[tuple[str, str]], *, reject_first: bool = False, forbidden_metadata: bool = False
) -> FunctionModel:
    state = {"t1_audits": 0}

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        instructions = info.instructions or ""
        role = next(
            (
                key
                for marker, key in (
                    ("Your role is orchestrator", "orchestrate"),
                    ("Your role is scoped task executor", "task"),
                    ("Your role is critic", "critique"),
                    ("Your role is integrator", "integrate"),
                    ("Your role is verifier", "verify"),
                )
                if marker in instructions
            ),
            "unknown",
        )
        observed.append((role, prompt))
        refs = re.findall(r"query:(?:task|critique|integrate|verify):[^\"' ]+", prompt)
        own = [ref for ref in refs if ref.startswith(f"query:{role}:")]
        if role == "orchestrate" and "One scoped task failed its critique" not in prompt:
            tasks = [dict(task) for task in CONCEPTUAL_TASKS]
            if forbidden_metadata:
                tasks[0]["allowed_tables"] = ["data"]
            values: dict[str, Any] = {
                "parent_objective": TASK.prompt,
                "tasks": tasks,
                "integration_definition_of_done": ["Answer x and state support or gaps"],
            }
        elif role == "orchestrate":
            revision = dict(CONCEPTUAL_TASKS[0])
            revision["local_scope"] = "recheck the requested value"
            values = {"decision": "revise", "rationale": "retry the finding", "revision": revision}
        elif (role == "task" and "SQL result:" not in prompt) or (
            role in {"critique", "integrate", "verify"} and not own
        ):
            return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
        elif role == "task":
            values = {
                "proposed_values": {"x": 7},
                "findings": [{"claim": "x is 7", "evidence_refs": [own[-1]]}],
                "assumptions": [],
                "unresolved_questions": [],
            }
        elif role == "critique":
            t1 = '"task_id": "t1"' in prompt
            reject = reject_first and t1 and state["t1_audits"] == 0
            if t1:
                state["t1_audits"] += 1
            values = {
                "structural_validity": True,
                "feasibility": "unsupported" if reject else "supported",
                "evidence_refs": [own[-1]],
                "reason": "retry needed" if reject else "checked",
                "evaluation": {
                    "validity": 0 if reject else 4,
                    "evidence": 4,
                    "usefulness": 4,
                    "novelty": 4,
                    "constraint_satisfaction": 4,
                },
            }
        elif role == "integrate":
            return ModelResponse(
                parts=[
                    TextPart(
                        json.dumps(
                            {
                                "values": {"x": 7},
                                "status": "supported",
                                "claims": [{"claim": "x is 7", "evidence_refs": [own[-1]]}],
                                "assumptions": [],
                                "unresolved_questions": [],
                            }
                        )
                    )
                ]
            )
        elif role == "verify":
            values = {
                "status": "supported",
                "claims_supported": True,
                "gaps_disclosed": True,
                "evidence_refs": [own[-1]],
                "reason": "checked x",
            }
        else:
            raise AssertionError(f"unexpected role: {role}")
        return ModelResponse(parts=[TextPart(json.dumps({"values": values}))])

    return FunctionModel(respond)


async def run_v22(
    model: FunctionModel,
) -> tuple[V21Answer, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(team_policy="reliable-v2")
    env.state.emit = events.append
    try:
        answer = await ADAPTERS["hybrid_v2_2-json"](
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


async def test_v22_dispatches_conceptual_plan_without_schema_or_plan_judge() -> None:
    observed: list[tuple[str, str]] = []
    answer, events = await run_v22(scripted_model(observed))
    assert answer.status == "supported"
    assert answer.values == {"x": 7}
    planner_inputs = [prompt for role, prompt in observed if role == "orchestrate"]
    assert len(planner_inputs) == 1
    assert "Specialist capabilities:" in planner_inputs[0]
    assert "SQL schema:" not in planner_inputs[0]
    assert "secret" not in planner_inputs[0]
    assert not any("Judge this proposed" in prompt for _, prompt in observed)
    assert any(
        "SQL schema:" in prompt and "secret" in prompt
        for role, prompt in observed
        if role == "task"
    )
    selected = next(
        event["orchestration"]["data"]
        for event in events
        if event.get("orchestration", {}).get("event_type") == "hybrid_v2_2.plan_selected"
    )
    assert "artifact_id" in selected
    for task in selected["plan"]["tasks"]:
        assert set(task["allowed_tables"]) == {"data", "secret"}
        assert task["output_schema"] == "TaskEvidence"
        assert set(task["budgets"]) == {"requests", "tool_calls", "total_tokens", "seconds"}


async def test_v22_rejects_planner_assigned_execution_details() -> None:
    observed: list[tuple[str, str]] = []
    with pytest.raises(PlanningRejected, match="planner cannot assign"):
        await run_v22(scripted_model(observed, forbidden_metadata=True))
    assert [role for role, _ in observed] == ["orchestrate", "orchestrate"]


async def test_v22_replanning_sees_only_abstract_task_failure() -> None:
    observed: list[tuple[str, str]] = []
    answer, _ = await run_v22(scripted_model(observed, reject_first=True))
    assert answer.status == "supported"
    decisions = [
        prompt
        for role, prompt in observed
        if role == "orchestrate" and "One scoped task failed its critique" in prompt
    ]
    assert len(decisions) == 1
    for forbidden in ("SQL schema:", "SQL result:", "query:", "allowed_tables", "evidence_refs"):
        assert forbidden not in decisions[0]
    assert "task_audit_refuted" in decisions[0]
