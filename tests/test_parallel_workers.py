import pytest

from poc.control.runtime import Runtime
from poc.models import ApprovalRequest, RunCreate


@pytest.mark.asyncio
async def test_same_role_workers_are_isolated(runtime: Runtime) -> None:
    run, _ = runtime.create_run(RunCreate())
    await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1))
    await runtime.wait(run["run_id"])
    status = runtime.status(run["run_id"])
    workers = [a for a in status["agents"] if a["role_id"] == "percentile_calculator"]
    assert len(workers) == 2
    assert workers[0]["agent_instance_id"] != workers[1]["agent_instance_id"]
    metrics = next(w for w in status["workflows"] if w["workflow_id"].endswith("-metrics"))
    task_ids = {task["task_id"]: task for task in metrics["tasks"]}
    assert (
        task_ids["baseline_p95"]["agent_instance_id"]
        != task_ids["incident_p95"]["agent_instance_id"]
    )
    assert "baseline_p95" in task_ids["baseline_p95"]["attempt_id"]
    assert "incident_p95" in task_ids["incident_p95"]["attempt_id"]
