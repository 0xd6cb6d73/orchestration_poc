from __future__ import annotations

from typing import Any

from poc.control.spawn_policy import SpawnPolicy
from poc.execution.agent_executor import AgentExecutionRequest, AgentExecutorRegistry
from poc.models import AgentInstance, Outcome, TaskSpec
from poc.persistence.database import Database


class WorkerAdapter:
    def __init__(self, db: Database, spawns: SpawnPolicy, executors: AgentExecutorRegistry):
        self.db = db
        self.spawns = spawns
        self.executors = executors

    def execute(
        self,
        *,
        task: TaskSpec,
        workflow_id: str,
        workflow_revision: int,
        run_id: str,
        plan_version: int,
        owner: AgentInstance,
        inputs: dict[str, Any],
        input_artifacts: list[str],
    ) -> dict[str, Any]:
        stable_key = f"{workflow_id}-r{workflow_revision}-{task.id}"
        selected_backend = task.agent_backend or self.spawns.roles.get(task.role).agent_backend
        executor = self.executors.get(selected_backend)
        agent = self.spawns.spawn(
            run_id=run_id,
            parent=owner,
            child_role=task.role,
            plan_version=plan_version,
            stable_key=stable_key,
            agent_backend=selected_backend,
        )
        attempt_id = f"attempt-{workflow_id}-r{workflow_revision}-{task.id}-1"
        self.db.update_task(
            workflow_id, workflow_revision, task.id, "running", agent.agent_instance_id, attempt_id
        )
        request = AgentExecutionRequest(
            task=task,
            workflow_id=workflow_id,
            workflow_revision=workflow_revision,
            run_id=run_id,
            agent=agent,
            attempt_id=attempt_id,
            inputs=inputs,
            input_artifacts=input_artifacts,
        )
        output = executor.execute(request)
        result = output.get("worker_result")
        if result:
            status = "succeeded" if result["outcome"] == Outcome.SUCCEEDED else "failed"
            self.db.update_task(workflow_id, workflow_revision, task.id, status, result=result)
        else:
            self.db.update_task(workflow_id, workflow_revision, task.id, "paused")
        return output
