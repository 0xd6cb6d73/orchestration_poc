import pytest

from poc.control.runtime import Runtime
from poc.models import ApprovalRequest, PlanEdit, RunCreate


@pytest.mark.asyncio
async def test_validation_delegates_lookup_and_resumes_correct_checkpoint(runtime: Runtime) -> None:
    run, proposed = runtime.create_run(RunCreate())
    constraints = [*proposed.constraints, "Never emit customer identifiers"]
    approved = await runtime.approve(
        run["run_id"], ApprovalRequest(plan_version=1, edits=PlanEdit(constraints=constraints))
    )
    await runtime.wait(run["run_id"])
    status = runtime.status(run["run_id"])

    assert approved.version == 2
    assert status["run"]["status"] == "completed"
    types = [event["event_type"] for event in status["events"]]
    assert "validation.requested" in types
    assert "validation.resolved" in types
    assert "workflow.paused" in types
    assert "workflow.resumed" in types
    assert any(agent["role_id"] == "manifest_reader" for agent in status["agents"])
    requested = next(
        event for event in status["events"] if event["event_type"] == "validation.requested"
    )
    resolved = next(
        event for event in status["events"] if event["event_type"] == "validation.resolved"
    )
    assert requested["data"]["request_id"] == resolved["data"]["request_id"]


@pytest.mark.asyncio
async def test_stale_plan_approval_is_rejected(runtime: Runtime) -> None:
    run, _ = runtime.create_run(RunCreate())
    await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    with pytest.raises(ValueError):
        await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
