import pytest

from poc.control.runtime import Runtime
from poc.models import ApprovalRequest, RunCreate


@pytest.mark.asyncio
async def test_tool_and_artifact_publication_are_idempotent(runtime):
    run, _ = runtime.create_run(RunCreate())
    plan = await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=False)
    main = runtime.db.get_agent(run["main_agent_id"])
    sub = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=main,
        child_role="metrics_supervisor",
        plan_version=plan.version,
        stable_key="recover-sub",
    )
    worker = runtime.spawns.spawn(
        run_id=run["run_id"],
        parent=sub,
        child_role="manifest_reader",
        plan_version=plan.version,
        stable_key="recover-worker",
    )
    first = runtime.tools.execute(
        operation_id="stable-operation",
        actor=worker,
        task_id="lookup",
        tool_name="read_manifest",
        arguments={},
    )
    second = runtime.tools.execute(
        operation_id="stable-operation",
        actor=worker,
        task_id="lookup",
        tool_name="read_manifest",
        arguments={},
    )
    assert first == second
    assert (
        runtime.db.conn.execute(
            "SELECT count(*) FROM operations WHERE operation_id='stable-operation'"
        ).fetchone()[0]
        == 1
    )
    a = runtime.artifacts.write(run["run_id"], {"same": "content"}, producer_task_id="lookup")
    b = runtime.artifacts.write(run["run_id"], {"same": "content"}, producer_task_id="lookup")
    assert a.artifact_id == b.artifact_id
    assert (
        runtime.db.conn.execute(
            "SELECT count(*) FROM artifacts WHERE artifact_id=?", (a.artifact_id,)
        ).fetchone()[0]
        == 1
    )


@pytest.mark.asyncio
async def test_restart_reuses_completed_workflows_without_republication(tmp_path):
    data = tmp_path / "restart-data"
    first = Runtime(data)
    run, _ = first.create_run(RunCreate())
    await first.approve(run["run_id"], ApprovalRequest(plan_version=1))
    await first.wait(run["run_id"])
    agent_count = len(first.db.list_agents(run["run_id"]))
    artifact_count = len(first.db.list_artifacts(run["run_id"]))
    # Simulate a crash after work committed but before the terminal run transition persisted.
    first.db.conn.execute("UPDATE runs SET status='running' WHERE run_id=?", (run["run_id"],))
    first.db.conn.commit()
    first.runner.close()
    first.db.close()

    restarted = Runtime(data)
    assert await restarted.recover() == [run["run_id"]]
    await restarted.wait(run["run_id"])
    restarted_run = restarted.db.get_run(run["run_id"])
    assert restarted_run is not None
    assert restarted_run["status"] == "completed"
    assert len(restarted.db.list_agents(run["run_id"])) == agent_count
    assert len(restarted.db.list_artifacts(run["run_id"])) == artifact_count
    restarted.runner.close()
    restarted.db.close()
