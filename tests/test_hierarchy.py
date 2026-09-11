import pytest

from poc.control.spawn_policy import SpawnDenied
from poc.models import AgentInstance, ApprovalRequest, RunCreate, Tier


@pytest.mark.asyncio
async def test_three_tier_spawn_authority(runtime):
    run, _ = runtime.create_run(RunCreate())
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    assert main is not None
    sub = runtime.spawns.spawn(run_id=run["run_id"], parent=main, child_role="metrics_supervisor",
                               plan_version=plan.version, stable_key="test-sub")
    worker = runtime.spawns.spawn(run_id=run["run_id"], parent=sub, child_role="window_selector",
                                  plan_version=plan.version, stable_key="test-worker")

    with pytest.raises(SpawnDenied):
        runtime.spawns.spawn(run_id=run["run_id"], parent=main, child_role="window_selector",
                              plan_version=plan.version)
    with pytest.raises(SpawnDenied):
        runtime.spawns.spawn(run_id=run["run_id"], parent=sub, child_role="evidence_supervisor",
                              plan_version=plan.version)
    with pytest.raises(SpawnDenied):
        runtime.spawns.spawn(run_id=run["run_id"], parent=worker, child_role="window_selector",
                              plan_version=plan.version)

