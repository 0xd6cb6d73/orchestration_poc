from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from poc.blackboard.service import BlackboardService
from poc.hybrid.collaboration_controller import CollaborationController, CollaborationError
from poc.hybrid.communication import CommunicationService
from poc.hybrid.contracts import CollaborationPhase, VisibilityPolicy
from poc.hybrid.planning import (
    MAX_PLAN_TASKS,
    PlanValidationError,
    apply_decision,
    parse_decision,
    parse_plan,
    validate_decision,
)
from poc.hybrid.policies import SwarmPolicyResolver
from poc.models import (
    AgentInstance,
    ExecutionMode,
    ExecutionPolicy,
    HybridConfig,
    MissionPlan,
    RunCreate,
    SwarmStrategy,
    Tier,
)
from poc.persistence.database import Database
from poc.services.artifact_store import ArtifactStore


def _valid_values() -> dict[str, Any]:
    return {
        "parent_objective": "Produce an evidence-backed incident report",
        "tasks": [
            {
                "task_id": "t1",
                "objective": "Summarize latency metrics",
                "local_scope": "metrics table only",
                "definition_of_done": ["p95 and p99 quoted with timestamps"],
                "out_of_scope": ["remediation"],
            },
            {
                "task_id": "t2",
                "objective": "Correlate metrics with deploys",
                "local_scope": "deployments table plus t1 artifact",
                "definition_of_done": ["each spike matched to a deploy or marked unexplained"],
                "dependencies": ["t1"],
            },
        ],
        "integration_definition_of_done": ["report cites every task artifact"],
    }


def test_parse_plan_accepts_valid_plan() -> None:
    plan = parse_plan(_valid_values())
    assert [task.task_id for task in plan.tasks] == ["t1", "t2"]
    assert plan.tasks[1].dependencies == ("t1",)
    assert plan.integration_definition_of_done == ("report cites every task artifact",)


def test_parse_plan_accepts_string_criteria_and_fills_task_ids() -> None:
    values = _valid_values()
    values["tasks"][0]["definition_of_done"] = "one criterion string"
    del values["tasks"][1]["task_id"]
    plan = parse_plan(values)
    assert plan.tasks[0].definition_of_done == ("one criterion string",)
    assert plan.tasks[1].task_id == "t2"


def test_parse_plan_rejects_task_count_outside_bounds() -> None:
    values = _valid_values()
    values["tasks"] = values["tasks"][:1]
    with pytest.raises(PlanValidationError) as err:
        parse_plan(values)
    assert any("between" in problem for problem in err.value.problems)

    filler = {"task_id": "t9", "objective": "x", "local_scope": "y", "definition_of_done": ["d"]}
    values["tasks"] = [
        {**filler, "task_id": f"t{i}", "local_scope": f"scope {i}"}
        for i in range(MAX_PLAN_TASKS + 1)
    ]
    with pytest.raises(PlanValidationError) as err:
        parse_plan(values)
    assert any(str(MAX_PLAN_TASKS) in problem for problem in err.value.problems)


def _empty_dod(values: dict[str, Any]) -> None:
    values["tasks"][0]["definition_of_done"] = []


def _blank_scope(values: dict[str, Any]) -> None:
    values["tasks"][0]["local_scope"] = "  "


def _blank_objective(values: dict[str, Any]) -> None:
    values["tasks"][0]["objective"] = ""


def _duplicate_id(values: dict[str, Any]) -> None:
    values["tasks"][1]["task_id"] = "t1"


def _unknown_dependency(values: dict[str, Any]) -> None:
    values["tasks"][1]["dependencies"] = ["t3"]


def _self_dependency(values: dict[str, Any]) -> None:
    values["tasks"][1]["dependencies"] = ["t2"]


def _dependency_cycle(values: dict[str, Any]) -> None:
    values["tasks"][0]["dependencies"] = ["t2"]
    values["tasks"][1]["dependencies"] = ["t1"]


def _empty_integration(values: dict[str, Any]) -> None:
    values["integration_definition_of_done"] = []


def _blank_output_schema(values: dict[str, Any]) -> None:
    values["tasks"][0]["output_schema"] = " "


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (_empty_dod, "definition_of_done"),
        (_blank_scope, "local_scope"),
        (_blank_objective, "objective"),
        (_duplicate_id, "duplicate"),
        (_unknown_dependency, "unknown dependency"),
        (_self_dependency, "cannot depend on itself"),
        (_dependency_cycle, "acyclic"),
        (_empty_integration, "integration"),
        (_blank_output_schema, "output_schema"),
    ],
)
def test_parse_plan_hard_validates(
    mutate: Callable[[dict[str, Any]], None], fragment: str
) -> None:
    values = _valid_values()
    mutate(values)
    with pytest.raises(PlanValidationError) as err:
        parse_plan(values)
    assert any(fragment in problem for problem in err.value.problems)


def test_decision_parsing_and_application() -> None:
    plan = parse_plan(_valid_values())
    accept = parse_decision({"decision": "accept", "rationale": "critique passed"})
    validate_decision(accept, plan)

    revise = parse_decision(
        {
            "decision": "revise",
            "rationale": "scope creep beyond metrics table",
            "revision": {
                "task_id": "t1",
                "objective": "Summarize latency metrics",
                "local_scope": "metrics table only, no deployment columns",
                "definition_of_done": ["p95 and p99 quoted"],
            },
        }
    )
    validate_decision(revise, plan, pending_task_id="t1")
    assert apply_decision(plan, revise).version == 2

    added = parse_decision(
        {
            "decision": "add_task",
            "rationale": "missing coverage for log signals",
            "added_task": {
                "task_id": "t3",
                "objective": "Extract error logs",
                "local_scope": "logs table",
                "definition_of_done": ["top three error classes named"],
                "dependencies": ["t1"],
            },
        }
    )
    extended = apply_decision(plan, added)
    assert [task.task_id for task in extended.tasks] == ["t1", "t2", "t3"]


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ({"decision": "accept", "rationale": ""}, "rationale"),
        (
            {
                "decision": "accept",
                "rationale": "r",
                "revision": {
                    "task_id": "t1",
                    "objective": "o",
                    "local_scope": "s",
                    "definition_of_done": ["d"],
                },
            },
            "only revise decisions",
        ),
        ({"decision": "revise", "rationale": "r"}, "requires a revised task"),
        (
            {
                "decision": "revise",
                "rationale": "r",
                "revision": {
                    "task_id": "t9",
                    "objective": "o",
                    "local_scope": "s",
                    "definition_of_done": ["d"],
                },
            },
            "unknown revision target",
        ),
        ({"decision": "add_task", "rationale": "r"}, "requires an added task"),
        (
            {
                "decision": "add_task",
                "rationale": "r",
                "added_task": {
                    "task_id": "t1",
                    "objective": "o",
                    "local_scope": "s",
                    "definition_of_done": ["d"],
                },
            },
            "already exists",
        ),
        (
            {
                "decision": "add_task",
                "rationale": "r",
                "added_task": {
                    "task_id": "t3",
                    "objective": "o",
                    "local_scope": "s",
                    "definition_of_done": ["d"],
                    "dependencies": ["t9"],
                },
            },
            "unknown dependency",
        ),
    ],
)
def test_decision_hard_validation(payload: dict[str, Any], fragment: str) -> None:
    plan = parse_plan(_valid_values())
    decision = parse_decision(payload)
    with pytest.raises(PlanValidationError) as err:
        validate_decision(decision, plan)
    assert any(fragment in problem for problem in err.value.problems)


def test_swarm_policy_resolver_registers_hybrid_v2() -> None:
    resolved = SwarmPolicyResolver().resolve(SwarmStrategy.HYBRID_V2)
    assert resolved.strategy == SwarmStrategy.HYBRID_V2
    assert resolved.collaboration_policy == "orchestrator_planned_v2"
    assert resolved.allocation_policy == "board_scoped_wave_v1"
    assert resolved.context_policy == "artifact_manifest_v1"
    assert resolved.completion_policy == "attested_delivery_v1"


def test_hybrid_v2_requires_board_claim_mode() -> None:
    policy = ExecutionPolicy(
        mode=ExecutionMode.BOARD_CLAIM,
        swarm_strategy=SwarmStrategy.HYBRID_V2,
        policy_set=SwarmPolicyResolver().resolve(SwarmStrategy.HYBRID_V2),
        agent_backend="custom_python",
        agent_model="stub",
        allowed_roles=frozenset({"worker"}),
    )
    assert policy.mode == ExecutionMode.BOARD_CLAIM
    with pytest.raises(ValueError, match="board_claim mode"):
        ExecutionPolicy.model_validate(
            policy.model_copy(update={"mode": ExecutionMode.HIERARCHICAL_DAG}).model_dump()
        )
    with pytest.raises(ValueError, match="board_claim"):
        RunCreate(
            swarm_strategy=SwarmStrategy.HYBRID_V2,
            execution_mode=ExecutionMode.HIERARCHICAL_DAG,
        )


def _mission_plan() -> MissionPlan:
    return MissionPlan(
        plan_id="plan",
        run_id="run",
        objective="objective",
        execution_mode=ExecutionMode.BOARD_CLAIM,
        swarm_strategy=SwarmStrategy.HYBRID_V2,
        policy_set=SwarmPolicyResolver().resolve(SwarmStrategy.HYBRID_V2),
        constraints=[],
        permitted_sources=[],
        permitted_tools=[],
        completion_criteria=["submit an answer"],
        areas=[],
    )


def _controller(tmp_path: Path) -> tuple[CollaborationController, Database, list[AgentInstance]]:
    db = Database(tmp_path / "scheduler.sqlite")
    artifacts = ArtifactStore(tmp_path / "artifacts", db)
    controller = CollaborationController(
        db, artifacts, BlackboardService(db), CommunicationService(db)
    )
    plan = _mission_plan()
    db.create_run("run", "objective", "steward", plan)
    db.approve_plan(plan, plan.model_copy(update={"status": "approved"}))
    agents = [
        AgentInstance(
            agent_instance_id="steward",
            run_id="run",
            parent_agent_id="main",
            tier=Tier.SUB,
            role_id="steward",
            role_version=1,
            plan_version=1,
            agent_backend="custom_python",
            agent_model="stub",
        ),
        *[
            AgentInstance(
                agent_instance_id=f"worker-{index}",
                run_id="run",
                parent_agent_id="steward",
                tier=Tier.WORKER,
                role_id="worker",
                role_version=1,
                plan_version=1,
                agent_backend="custom_python",
                agent_model="stub",
            )
            for index in range(1, 3)
        ],
    ]
    for agent in agents:
        db.put_agent(agent)
    return controller, db, agents


def _drive_to_tested(
    controller: CollaborationController, agents: list[AgentInstance]
) -> tuple[str, list[str]]:
    round_ = controller.frame(
        run_id="run",
        domain_id="domain",
        steward=agents[0],
        member_agent_ids=tuple(agent.agent_instance_id for agent in agents[1:]),
        config=HybridConfig(proposals_per_round=2),
    )
    artifact_ids: list[str] = []
    for index, agent in enumerate(agents[1:]):
        written = controller.artifacts.write(
            "run", {"candidate": index}, producer_task_id=f"task-{index}"
        )
        artifact_ids.append(written.artifact_id)
        controller.submit_candidate(
            round_id=round_.round_id,
            author=agent,
            hypothesis_key=f"task-{index}",
            artifact_id=written.artifact_id,
            evidence_refs=(),
        )
    controller.release(round_.round_id)
    controller.cluster_candidates(round_.round_id)
    controller.advance(round_.round_id, CollaborationPhase.CRITIQUED)
    controller.advance(round_.round_id, CollaborationPhase.TESTED)
    return round_.round_id, artifact_ids


def test_revise_candidate_publishes_derived_candidate(tmp_path: Path) -> None:
    controller, db, agents = _controller(tmp_path)
    round_id, artifact_ids = _drive_to_tested(controller, agents)
    candidates = [item for item in db.list_candidates("run") if item.round_id == round_id]
    artifact = controller.artifacts.write("run", {"revised": True}, producer_task_id="task-1-rev")
    revised = controller.revise_candidate(
        round_id=round_id,
        author=agents[1],
        hypothesis_key=candidates[1].hypothesis_key,
        artifact_id=artifact.artifact_id,
        evidence_refs=(artifact_ids[1],),
        parent_candidate_ids=(candidates[1].candidate_id,),
    )
    assert revised.parent_candidate_ids == (candidates[1].candidate_id,)
    assert revised.visibility == VisibilityPolicy.RELEASED_TO_TEAM
    assert revised.candidate_id.endswith(":1")

    added = controller.revise_candidate(
        round_id=round_id,
        author=agents[1],
        hypothesis_key="task-3",
        artifact_id=artifact.artifact_id,
    )
    assert added.parent_candidate_ids == ()
    assert added.candidate_id.endswith(":1")
    follow_up = controller.revise_candidate(
        round_id=round_id,
        author=agents[1],
        hypothesis_key="task-3",
        artifact_id=artifact.artifact_id,
    )
    assert follow_up.candidate_id.endswith(":2")


def test_revise_candidate_rejects_unknown_parent_and_wrong_phase(tmp_path: Path) -> None:
    controller, db, agents = _controller(tmp_path)
    round_id, _ = _drive_to_tested(controller, agents)
    artifact = controller.artifacts.write("run", {}, producer_task_id="x")
    with pytest.raises(CollaborationError, match="unknown candidate"):
        controller.revise_candidate(
            round_id=round_id,
            author=agents[1],
            hypothesis_key="task-1",
            artifact_id=artifact.artifact_id,
            parent_candidate_ids=("candidate:missing",),
        )
    round_ = db.get_collaboration_round(round_id)
    assert round_ is not None
    db.put_collaboration_round(round_.model_copy(update={"phase": CollaborationPhase.RELEASED}))
    with pytest.raises(CollaborationError, match="phase"):
        controller.revise_candidate(
            round_id=round_id,
            author=agents[1],
            hypothesis_key="task-1",
            artifact_id=artifact.artifact_id,
        )
