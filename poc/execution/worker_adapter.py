from __future__ import annotations

from typing import Any

from poc.control.spawn_policy import SpawnPolicy
from poc.execution.ooda_graph import OODAHarness
from poc.models import AgentInstance, Outcome, TaskSpec, new_id
from poc.persistence.database import Database


class WorkerAdapter:
    def __init__(self, db: Database, spawns: SpawnPolicy, ooda: OODAHarness):
        self.db = db
        self.spawns = spawns
        self.ooda = ooda

    def execute(self, *, task: TaskSpec, workflow_id: str, workflow_revision: int,
                run_id: str, plan_version: int, owner: AgentInstance,
                inputs: dict[str, Any], input_artifacts: list[str]) -> dict[str, Any]:
        stable_key = f"{workflow_id}-r{workflow_revision}-{task.id}"
        agent = self.spawns.spawn(run_id=run_id, parent=owner, child_role=task.role,
                                  plan_version=plan_version, stable_key=stable_key)
        attempt_id = f"attempt-{workflow_id}-r{workflow_revision}-{task.id}-1"
        self.db.update_task(workflow_id, workflow_revision, task.id, "running",
                            agent.agent_instance_id, attempt_id)
        state = {
            "run_id": run_id, "workflow_id": workflow_id, "workflow_revision": workflow_revision,
            "task_id": task.id, "goal": task.goal, "output_schema": task.output_schema,
            "acceptance_criteria": task.acceptance_criteria, "inputs": inputs,
            "input_artifacts": input_artifacts, "agent": agent.model_dump(mode="json"),
            "attempt_id": attempt_id, "cycle": 0, "tool_calls": 0, "validations": 0,
            "working": {}, "evidence_artifacts": [],
        }
        output = self.ooda.graph.invoke(state)
        result = output.get("worker_result")
        if result:
            status = "succeeded" if result["outcome"] == Outcome.SUCCEEDED else "failed"
            self.db.update_task(workflow_id, workflow_revision, task.id, status, result=result)
        else:
            self.db.update_task(workflow_id, workflow_revision, task.id, "paused")
        return output
