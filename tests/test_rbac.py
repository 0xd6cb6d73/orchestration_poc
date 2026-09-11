import pytest

from poc.models import ApprovalRequest, RunCreate
from poc.services.tool_gateway import ToolDenied


@pytest.mark.asyncio
async def test_supervisor_cannot_invoke_domain_tool(runtime):
    run, _ = runtime.create_run(RunCreate())
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    sub = runtime.spawns.spawn(run_id=run["run_id"], parent=main, child_role="metrics_supervisor",
                               plan_version=plan.version, stable_key="rbac-sub")
    with pytest.raises(ToolDenied, match="restricted to workers"):
        runtime.tools.execute(operation_id="denied:1", actor=sub, task_id="bad",
                              tool_name="read_metric_slice",
                              arguments={"start": "2026-04-17T12:00:00Z", "end": "2026-04-17T12:01:00Z"})


@pytest.mark.asyncio
async def test_cancel_revokes_worker_authority(runtime):
    run, _ = runtime.create_run(RunCreate())
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    sub = runtime.spawns.spawn(run_id=run["run_id"], parent=main, child_role="metrics_supervisor",
                               plan_version=plan.version, stable_key="cancel-sub")
    worker = runtime.spawns.spawn(run_id=run["run_id"], parent=sub, child_role="window_selector",
                                  plan_version=plan.version, stable_key="cancel-worker")
    assert runtime.db.request_cancellation(run["run_id"])
    with pytest.raises(ToolDenied, match="inactive"):
        runtime.tools.execute(operation_id="denied:cancel", actor=worker, task_id="stale",
                              tool_name="read_metric_slice",
                              arguments={"start": "2026-04-17T12:00:00Z", "end": "2026-04-17T12:01:00Z"})

