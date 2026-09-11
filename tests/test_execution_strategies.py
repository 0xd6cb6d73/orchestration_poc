from datetime import datetime, timedelta, timezone

import pytest

from poc.execution import (
    BoardClaimStrategy,
    ManagedPoolStrategy,
    SpeculativeStrategy,
    StaleClaim,
    StrategyError,
)
from poc.models import ApprovalRequest, EffectPolicy, ExecutionMode, ExecutionPolicy, RunCreate
from poc.services.tool_gateway import ToolDenied


async def _authorized_team(runtime, supervisor_role: str, worker_roles: list[str]):
    run, _ = runtime.create_run(RunCreate())
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    supervisor = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=main,
        child_role=supervisor_role,
        plan_version=plan.version,
        stable_key=f"{supervisor_role}-strategy-test",
    )
    workers = [
        runtime.spawns.spawn(
            run_id=run["run_id"],
            parent=supervisor,
            child_role=role,
            plan_version=plan.version,
            stable_key=f"{role}-{index}-strategy-test",
        )
        for index, role in enumerate(worker_roles)
    ]
    return run, supervisor, workers


@pytest.mark.asyncio
async def test_board_claim_is_atomic_and_fenced_at_tool_gateway(runtime):
    run, supervisor, workers = await _authorized_team(
        runtime, "metrics_supervisor", []
    )
    policy = ExecutionPolicy(
        mode=ExecutionMode.BOARD_CLAIM,
        max_workers=2,
        allowed_roles={"manifest_reader"},
        lease_seconds=30,
    )
    handle = await runtime.submit_execution(
        run_id=run["run_id"],
        owner_suborchestrator_id=supervisor.agent_instance_id,
        goal_ref="manifest-board",
        policy=policy,
    )
    workers = runtime.capacity.reconcile(handle, {"manifest_reader": 2})
    assert len(workers) == 2
    with pytest.raises(StrategyError, match="population envelope"):
        runtime.capacity.reconcile(handle, {"manifest_reader": 3})
    board = runtime.strategies.get(ExecutionMode.BOARD_CLAIM)
    assert isinstance(board, BoardClaimStrategy)
    board.post_task(
        execution_id=handle.execution_id,
        task_id="read-timezone",
        required_role="manifest_reader",
        task_spec_ref="fixture:manifest-timezone",
    )
    claim = board.claim_next(execution_id=handle.execution_id, worker_id=workers[0].agent_instance_id)
    assert claim is not None
    assert board.claim_next(execution_id=handle.execution_id, worker_id=workers[1].agent_instance_id) is None
    assert "token" not in repr(claim)
    with pytest.raises(ToolDenied, match="ownership grant"):
        runtime.tools.execute(
            operation_id="claim:manifest:missing-grant",
            actor=workers[0],
            task_id="read-timezone",
            tool_name="read_manifest",
            arguments={},
        )
    result = runtime.tools.execute(
        operation_id="claim:manifest:read",
        actor=workers[0],
        task_id="read-timezone",
        tool_name="read_manifest",
        arguments={},
        grant=claim,
    )
    board.complete(claim, result_ref=f"inline:{result['timestamp_timezone']}")
    with pytest.raises(ToolDenied, match="stale"):
        runtime.tools.execute(
            operation_id="claim:manifest:stale",
            actor=workers[0],
            task_id="read-timezone",
            tool_name="read_manifest",
            arguments={},
            grant=claim,
        )
    with pytest.raises(StaleClaim):
        board.complete(claim, result_ref="late")

    board.post_task(
        execution_id=handle.execution_id,
        task_id="retry-timezone",
        required_role="manifest_reader",
        task_spec_ref="fixture:retry",
    )
    expired_claim = board.claim_next(execution_id=handle.execution_id, worker_id=workers[0].agent_instance_id)
    runtime.db.conn.execute(
        "UPDATE swarm_tasks SET lease_expires_at=? WHERE execution_id=? AND task_id=?",
        ("2000-01-01T00:00:00+00:00", handle.execution_id, "retry-timezone"),
    )
    runtime.db.conn.commit()
    assert board.reap_expired(handle.execution_id) == ["retry-timezone"]
    replacement = board.claim_next(execution_id=handle.execution_id, worker_id=workers[1].agent_instance_id)
    assert replacement is not None
    assert replacement.generation == expired_claim.generation + 1


@pytest.mark.asyncio
async def test_managed_pool_uses_deterministic_tie_break(runtime):
    run, supervisor, workers = await _authorized_team(
        runtime, "metrics_supervisor", ["manifest_reader", "manifest_reader"]
    )
    policy = ExecutionPolicy(
        mode=ExecutionMode.MANAGED_POOL,
        max_workers=2,
        allowed_roles={"manifest_reader"},
        options={"role_quotas": {"manifest_reader": 2}},
    )
    handle = await runtime.submit_execution(
        run_id=run["run_id"],
        owner_suborchestrator_id=supervisor.agent_instance_id,
        goal_ref="manifest-pool",
        policy=policy,
    )
    pool = runtime.strategies.get(ExecutionMode.MANAGED_POOL)
    assert isinstance(pool, ManagedPoolStrategy)
    slots = [pool.start_slot(execution_id=handle.execution_id, worker_id=worker.agent_instance_id) for worker in workers]
    offer = pool.publish_offer(
        execution_id=handle.execution_id,
        task_id="pool-task",
        eligible_roles=["manifest_reader"],
        bid_deadline=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    for slot in slots:
        pool.submit_bid(execution_id=handle.execution_id, offer_id=offer, slot_id=slot, fit_features={"score": 1})
    assignment = pool.arbitrate(execution_id=handle.execution_id, offer_id=offer)
    assert assignment.slot_id == min(slots)
    assert pool.validate_assignment(assignment, task_id="pool-task", worker_id=assignment.worker_id)
    runtime.db.conn.execute(
        "UPDATE swarm_assignments SET lease_expires_at=? WHERE assignment_id=?",
        ("2000-01-01T00:00:00+00:00", assignment.assignment_id),
    )
    runtime.db.conn.commit()
    assert pool.reap_expired(handle.execution_id) == [assignment.assignment_id]
    assert not pool.validate_assignment(assignment, task_id="pool-task", worker_id=assignment.worker_id)
    replacement = pool.arbitrate(execution_id=handle.execution_id, offer_id=offer)
    assert replacement.generation == assignment.generation + 1
    pool.complete(replacement, result_ref="inline:UTC")


@pytest.mark.asyncio
async def test_speculative_candidates_are_read_only_and_reconciled_explicitly(runtime):
    run, supervisor, workers = await _authorized_team(
        runtime, "reporting_supervisor", ["claim_drafter", "claim_drafter"]
    )
    policy = ExecutionPolicy(
        mode=ExecutionMode.SPECULATIVE,
        max_workers=2,
        speculative_fanout=2,
        allowed_roles={"claim_drafter"},
        effect_policy=EffectPolicy.READ_ONLY,
    )
    handle = await runtime.submit_execution(
        run_id=run["run_id"],
        owner_suborchestrator_id=supervisor.agent_instance_id,
        goal_ref="hypothesis-group",
        policy=policy,
    )
    speculative = runtime.strategies.get(ExecutionMode.SPECULATIVE)
    assert isinstance(speculative, SpeculativeStrategy)
    group = speculative.start_group(execution_id=handle.execution_id, logical_task_id="candidate-cause")
    grants = [
        speculative.authorize_candidate(
            execution_id=handle.execution_id, group_id=group, worker_id=worker.agent_instance_id
        )
        for worker in workers
    ]
    with pytest.raises(ToolDenied, match="external effects"):
        runtime.tools.execute(
            operation_id="candidate:forbidden-effect",
            actor=workers[0],
            task_id="candidate-cause",
            tool_name="write_artifact",
            arguments={"content": "must not be committed"},
            grant=grants[0],
            effect_class="external_effect",
        )
    speculative.complete_candidate(grants[0], result_ref="candidate:valid", valid=True)
    speculative.complete_candidate(grants[1], result_ref="candidate:invalid", valid=False)
    decision = speculative.reconcile(execution_id=handle.execution_id, group_id=group)
    assert decision.accepted_candidate_ids == (grants[0].candidate_id,)
    assert decision.rejected_candidate_ids == (grants[1].candidate_id,)
    with pytest.raises(StrategyError, match="not open"):
        speculative.reconcile(execution_id=handle.execution_id, group_id=group)
