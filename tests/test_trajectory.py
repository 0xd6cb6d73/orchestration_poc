import pytest

from poc.models import ApprovalRequest, RunCreate


@pytest.mark.asyncio
async def test_report_has_exact_lineage_and_ooda_identity(runtime):
    run, _ = runtime.create_run(RunCreate())
    await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1))
    await runtime.wait(run["run_id"])
    status = runtime.status(run["run_id"])
    delivery = next(e for e in status["events"] if e["event_type"] == "deliverable.accepted")
    _, content = runtime.artifacts.read(delivery["data"]["artifact_id"])
    report = content.decode()
    assert "3.56x" in report
    assert "Temporal association is not proof of causation" in report
    for artifact_id in delivery["data"]["evidence_artifacts"]:
        assert artifact_id in report
    phases = {
        e["data"].get("ooda_phase") for e in status["events"] if e["event_type"].startswith("ooda.")
    }
    assert phases == {"observe", "orient", "decide", "act"}
    assert all(e["actor_id"] for e in status["events"] if e["event_type"].startswith("ooda."))
