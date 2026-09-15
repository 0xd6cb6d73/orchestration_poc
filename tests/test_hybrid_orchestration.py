from __future__ import annotations

import pytest

from poc.blackboard.models import RecordType
from poc.control.runtime import Runtime
from poc.hybrid.capability_router import NoCompatibleCapability
from poc.hybrid.communication import CommunicationDenied
from poc.hybrid.context_assembler import DependencyNotReady
from poc.hybrid.contracts import TeamSpec, VisibilityPolicy
from poc.hybrid.verification_service import VerificationDenied
from poc.models import (
    ApprovalRequest,
    DependencyRequirement,
    ExecutionMode,
    RunCreate,
    SwarmStrategy,
    TaskSpec,
)


@pytest.mark.asyncio
async def test_hybrid_run_preserves_minority_and_gates_delivery(runtime: Runtime) -> None:
    run, proposed = runtime.create_run(
        RunCreate(
            execution_mode=ExecutionMode.BOARD_CLAIM,
            swarm_strategy=SwarmStrategy.HYBRID_V1,
        )
    )
    assert proposed.policy_set.strategy == SwarmStrategy.HYBRID_V1
    assert proposed.policy_set.collaboration_policy == "diverge_test_select_v1"

    await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1))
    await runtime.wait(run["run_id"])
    status = runtime.status(run["run_id"])

    assert status["run"]["status"] == "completed", status["run"]["error"]
    assert any(agent["role_id"] == "assurance_supervisor" for agent in status["agents"])
    hybrid = status["hybrid"]
    assert hybrid["collaboration_rounds"][0]["phase"] == "complete"
    candidates = hybrid["candidates"]
    assert len(candidates) == 3
    assert len(hybrid["candidate_clusters"]) == 3
    selected = next(candidate for candidate in candidates if candidate["state"] == "selected")
    assert selected["hypothesis_key"] == "cache_queueing"
    assert sum(candidate["state"] == "refuted" for candidate in candidates) == 2
    assert hybrid["verifications"][0]["verdict"] == "pass"
    assert hybrid["acceptance_decisions"][0]["accepted"] is True
    assert hybrid["delivery_decisions"][0]["permitted"] is True
    event_types = {event["event_type"] for event in status["events"]}
    assert {
        "context.assembled",
        "allocation.decided",
        "candidate.released",
        "candidate.retained",
        "verification.completed",
        "delivery.gated",
    } <= event_types


@pytest.mark.asyncio
async def test_hybrid_v2_orchestrated_plan_revises_scoped_tasks(runtime: Runtime) -> None:
    run, proposed = runtime.create_run(
        RunCreate(
            execution_mode=ExecutionMode.BOARD_CLAIM,
            swarm_strategy=SwarmStrategy.HYBRID_V2,
        )
    )
    assert proposed.policy_set.strategy == SwarmStrategy.HYBRID_V2
    assert proposed.policy_set.collaboration_policy == "orchestrator_planned_v2"

    await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1))
    await runtime.wait(run["run_id"])
    status = runtime.status(run["run_id"])

    assert status["run"]["status"] == "completed", status["run"]["error"]
    rounds = status["hybrid"]["collaboration_rounds"]
    assert {round_["phase"] for round_ in rounds} == {"complete"}
    assert len(rounds) == 2
    candidates = status["hybrid"]["candidates"]
    plan_candidates = [c for c in candidates if c["hypothesis_key"].startswith("plan_")]
    assert len(plan_candidates) == 3
    assert sum(c["state"] == "selected" for c in plan_candidates) == 1
    assert (
        sum(c["hypothesis_key"] == "integration" and c["state"] == "viable" for c in candidates)
        == 1
    )
    assert any(c["hypothesis_key"] == "t1" and c["parent_candidate_ids"] for c in candidates)
    assert any(
        c["hypothesis_key"] == "integration" and not c["parent_candidate_ids"] for c in candidates
    )
    event_types = {event["event_type"] for event in status["events"]}
    assert {
        "hybrid_v2.plan_selected",
        "candidate.revised",
        "verification.completed",
        "delivery.gated",
    } <= event_types
    assert status["hybrid"]["verifications"][0]["verdict"] == "pass"
    assert status["hybrid"]["delivery_decisions"][0]["permitted"] is True


@pytest.mark.asyncio
async def test_hybrid_visibility_communication_and_verification_boundaries(
    runtime: Runtime,
) -> None:
    run, _ = runtime.create_run(RunCreate())
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    assert main is not None
    supervisor = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=main,
        child_role="metrics_supervisor",
        plan_version=plan.version,
        stable_key="hybrid-boundary-supervisor",
    )
    author = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=supervisor,
        child_role="manifest_reader",
        plan_version=plan.version,
        stable_key="hybrid-boundary-author",
    )
    peer = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=supervisor,
        child_role="percentile_calculator",
        plan_version=plan.version,
        stable_key="hybrid-boundary-peer",
    )
    sealed = runtime.artifacts.write(
        run["run_id"],
        {"hypothesis": "private proposal"},
        producer_task_id="sealed-proposal",
        visibility=VisibilityPolicy.SEALED_TO_ROUND,
        visibility_ref="round:test",
        access_labels=frozenset({f"agent:{author.agent_instance_id}"}),
    )

    assert runtime.artifacts.read_for(author, sealed.artifact_id)[0] == sealed
    with pytest.raises(PermissionError, match="visible context"):
        runtime.artifacts.read_for(peer, sealed.artifact_id, round_id="round:test")

    first = runtime.blackboard.publish(
        actor=author,
        domain_id="metrics",
        record_type=RecordType.CLAIM,
        statement="The deployment caused the incident.",
        supporting_artifacts=(sealed.artifact_id,),
    )
    second = runtime.blackboard.publish(
        actor=peer,
        domain_id="metrics",
        record_type=RecordType.CONTRADICTION,
        statement="The deployment is only temporally correlated.",
        challenging_artifacts=(sealed.artifact_id,),
    )
    visible = runtime.blackboard.visible_records(actor=author, domain_id="metrics")
    assert {record.record_id for record in visible} >= {first.record_id, second.record_id}

    team = TeamSpec(
        team_id="team:boundary",
        run_id=run["run_id"],
        domain_id="metrics",
        steward_agent_id=supervisor.agent_instance_id,
        member_agent_ids=(author.agent_instance_id, peer.agent_instance_id),
        permitted_edges=frozenset({f"{author.agent_instance_id}->{peer.agent_instance_id}"}),
    )
    runtime.communications.register_team(team)
    with pytest.raises(CommunicationDenied, match="sealed"):
        runtime.communications.send(
            team_id=team.team_id,
            sender=author,
            recipient=peer,
            message_type="PUBLISH",
            payload={"kind": "candidate"},
            payload_ref=sealed.artifact_id,
        )
    runtime.communications.send(
        team_id=team.team_id,
        sender=author,
        recipient=peer,
        message_type="CHALLENGE",
        payload={"record_id": first.record_id},
    )
    with pytest.raises(CommunicationDenied, match="edge"):
        runtime.communications.send(
            team_id=team.team_id,
            sender=peer,
            recipient=author,
            message_type="CHALLENGE",
            payload={"record_id": second.record_id},
        )

    source = runtime.artifacts.write(
        run["run_id"],
        {
            "summary": "The source omits the operational qualifier.",
            "consequential_detail": "timezone-less timestamps are UTC",
        },
        producer_task_id="context-source",
    )
    context_task = TaskSpec(
        id="context-fidelity",
        role="manifest_reader",
        goal="Recover the exact source detail.",
        output_schema="ManifestFact",
        static_artifact_refs=[source.artifact_id],
        domain_id="metrics",
    )
    assembled = runtime.context.assemble(
        workflow_id="context-fidelity",
        workflow_revision=1,
        task=context_task,
        plan=plan,
        agent=author,
        attempt_id="attempt-context-fidelity",
        inputs={"summary": "The source omits the operational qualifier."},
        input_artifacts=[source.artifact_id],
    )
    assert (
        assembled.inputs["_delegation"]["artifact_excerpts"][source.artifact_id][
            "consequential_detail"
        ]
        == "timezone-less timestamps are UTC"
    )
    with pytest.raises(DependencyNotReady, match="artifact_verified"):
        runtime.context.assemble(
            workflow_id="verified-dependency",
            workflow_revision=1,
            task=context_task.model_copy(
                update={
                    "id": "verified-dependency",
                    "artifact_requirements": {
                        source.artifact_id: DependencyRequirement.ARTIFACT_VERIFIED
                    },
                }
            ),
            plan=plan,
            agent=author,
            attempt_id="attempt-verified-dependency",
            inputs={},
            input_artifacts=[source.artifact_id],
        )

    verification = runtime.verification.request(
        run_id=run["run_id"],
        subject_artifact_id=sealed.artifact_id,
        producer_agent_id=author.agent_instance_id,
        schema="Hypothesis",
        required_checks=("evidence",),
        evidence_refs=(sealed.artifact_id,),
    )
    with pytest.raises(VerificationDenied, match="cannot verify"):
        runtime.verification.complete(
            verification_id=verification.verification_id,
            verifier=author,
            checks={"evidence": True},
            findings=("self-check",),
            evidence_refs=(sealed.artifact_id,),
        )


def test_capability_router_rejects_incompatible_profiles(runtime: Runtime) -> None:
    role = runtime.roles.get("manifest_reader")
    profile = runtime.capabilities.profile_for_role(
        role,
        RunCreate().agent_runtime,
        task_classes=frozenset({"ManifestFact"}),
    )

    decision = runtime.capabilities.route(
        run_id="run-routing",
        task_id="manifest",
        task_class="ManifestFact",
        required_tools=frozenset({"read_manifest"}),
        profiles=[profile],
    )
    assert decision.selected_profile_id == profile.profile_id
    assert decision.reasons

    with pytest.raises(NoCompatibleCapability):
        runtime.capabilities.route(
            run_id="run-routing",
            task_id="write",
            task_class="ManifestFact",
            required_tools=frozenset({"write_artifact"}),
            profiles=[profile],
        )
