from __future__ import annotations

from contextlib import suppress
from typing import Any

from langgraph.errors import GraphInterrupt

from poc.control.spawn_policy import SpawnPolicy
from poc.execution.agent_executor import AgentExecutionRequest, AgentExecutorRegistry
from poc.execution.board_claim import BoardClaimStrategy, Claim
from poc.execution.capacity import CapacityScheduler
from poc.hybrid.context_assembler import ContextAssembler
from poc.hybrid.contracts import ContextManifest, VisibilityPolicy
from poc.models import (
    HYBRID_STRATEGIES,
    AgentInstance,
    ExecutionHandle,
    ExecutionMode,
    ExecutionPolicy,
    Outcome,
    TaskSpec,
    utc_now,
)
from poc.persistence.database import Database


class WorkerAdapter:
    def __init__(
        self,
        db: Database,
        spawns: SpawnPolicy,
        executors: AgentExecutorRegistry,
        context: ContextAssembler,
        capacity: CapacityScheduler,
        board: BoardClaimStrategy,
    ):
        self.db = db
        self.spawns = spawns
        self.executors = executors
        self.context = context
        self.capacity = capacity
        self.board = board

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
        role = self.spawns.roles.get(task.role)
        selected_provider = task.agent_provider or role.provider
        selected_model = task.agent_model or role.model
        executor = self.executors.get(selected_backend)
        agent = self.spawns.spawn(
            run_id=run_id,
            parent=owner,
            child_role=task.role,
            plan_version=plan_version,
            stable_key=stable_key,
            agent_backend=selected_backend,
            agent_provider=selected_provider,
            agent_model=selected_model,
        )
        plan = self.db.get_plan(run_id, plan_version)
        if plan is None:
            raise RuntimeError(f"approved plan version {plan_version} was not found")
        authority_task_id = task.id
        claim: Claim | None = None
        resumed_claim = False
        handle: ExecutionHandle | None = None
        attempt_id = f"attempt-{authority_task_id}-1"
        if plan.swarm_strategy in HYBRID_STRATEGIES:
            authority_task_id = f"{workflow_id}:r{workflow_revision}:{task.id}"
            handle = self._hybrid_execution(run_id, owner)
            self.capacity.activate_existing(handle, agent)
            existing_task = self.db.conn.execute(
                "SELECT state FROM swarm_tasks WHERE execution_id=? AND task_id=?",
                (handle.execution_id, authority_task_id),
            ).fetchone()
            if existing_task is None:
                self.board.post_task(
                    execution_id=handle.execution_id,
                    task_id=authority_task_id,
                    required_role=task.role,
                    task_spec_ref=f"workflow:{workflow_id}:r{workflow_revision}:{task.id}",
                    acceptance_criteria_ref=f"workflow:{workflow_id}:{task.id}:acceptance",
                )
            elif existing_task["state"] == "claimed":
                active_attempt = self.db.conn.execute(
                    "SELECT a.attempt_id FROM swarm_tasks t JOIN swarm_task_attempts a "
                    "ON a.execution_id=t.execution_id AND a.task_id=t.task_id "
                    "AND a.ownership_generation=t.claim_generation "
                    "WHERE t.execution_id=? AND t.task_id=? AND t.claimant_id=? "
                    "AND t.lease_expires_at>? AND a.status='running'",
                    (
                        handle.execution_id,
                        authority_task_id,
                        agent.agent_instance_id,
                        utc_now(),
                    ),
                ).fetchone()
                if active_attempt is None:
                    raise RuntimeError(
                        f"hybrid task {authority_task_id!r} has an unavailable active claim"
                    )
                attempt_id = str(active_attempt["attempt_id"])
                resumed_claim = True
            if not resumed_claim:
                claim = self.board.claim_task(
                    execution_id=handle.execution_id,
                    task_id=authority_task_id,
                    worker_id=agent.agent_instance_id,
                )
                if claim is None:
                    self.capacity.deactivate(handle, agent.agent_instance_id)
                    raise RuntimeError(
                        f"hybrid task {authority_task_id!r} could not acquire its claim"
                    )
                attempt_id = claim.attempt_id
        self.db.update_task(
            workflow_id, workflow_revision, task.id, "running", agent.agent_instance_id, attempt_id
        )
        try:
            assembled = self.context.assemble(
                workflow_id=workflow_id,
                workflow_revision=workflow_revision,
                task=task,
                plan=plan,
                agent=agent,
                attempt_id=attempt_id,
                inputs=inputs,
                input_artifacts=list(dict.fromkeys([*task.static_artifact_refs, *input_artifacts])),
            )
            request = AgentExecutionRequest(
                task=task,
                workflow_id=workflow_id,
                workflow_revision=workflow_revision,
                run_id=run_id,
                agent=agent,
                attempt_id=attempt_id,
                authority_task_id=authority_task_id,
                goal_contract_id=assembled.goal_contract.goal_contract_id,
                context_manifest_id=assembled.manifest.context_manifest_id,
                ownership_grant=claim,
                inputs=assembled.inputs,
                input_artifacts=assembled.input_artifacts,
            )
            output = executor.execute(request)
            effective_claim = claim
            resumed_grant = output.get("ownership_grant")
            if effective_claim is None and isinstance(resumed_grant, Claim):
                effective_claim = resumed_grant
            result = output.get("worker_result")
            if result:
                self._record_artifact_metadata(
                    task=task,
                    agent=agent,
                    attempt_id=attempt_id,
                    input_artifacts=assembled.input_artifacts,
                    manifest=assembled.manifest,
                    result=result,
                )
                status = "succeeded" if result["outcome"] == Outcome.SUCCEEDED else "failed"
                self.db.update_task(workflow_id, workflow_revision, task.id, status, result=result)
                if effective_claim is not None:
                    if status == "succeeded":
                        result_ref = result.get("output_artifact") or f"worker-result:{attempt_id}"
                        self.board.complete(effective_claim, result_ref=str(result_ref))
                    else:
                        self.board.release(
                            effective_claim,
                            reason=str(result.get("completion_summary", "failed")),
                        )
            else:
                self.db.update_task(workflow_id, workflow_revision, task.id, "paused")
                if effective_claim is not None:
                    self.board.release(effective_claim, reason="worker paused")
            return output
        except GraphInterrupt:
            # The outer workflow checkpointer owns the pause/resume lifecycle. Keep
            # the fenced claim live because the nested OODA state resumes with that
            # exact opaque token; replacing it would correctly make the token stale.
            self.db.update_task(workflow_id, workflow_revision, task.id, "paused")
            raise
        except BaseException as exc:
            if claim is not None:
                with suppress(Exception):
                    self.board.release(claim, reason=f"executor error: {exc}")
            raise
        finally:
            if handle is not None:
                self.capacity.deactivate(handle, agent.agent_instance_id)

    def _hybrid_execution(self, run_id: str, owner: AgentInstance) -> ExecutionHandle:
        row = self.db.conn.execute(
            "SELECT * FROM executions WHERE run_id=? AND owner_suborchestrator_id=? "
            "AND mode=? AND status='active' ORDER BY created_at DESC LIMIT 1",
            (run_id, owner.agent_instance_id, ExecutionMode.BOARD_CLAIM),
        ).fetchone()
        if row is None:
            raise RuntimeError("hybrid worker has no active board execution")
        policy = ExecutionPolicy.model_validate_json(row["policy"])
        if policy.swarm_strategy not in HYBRID_STRATEGIES:
            raise RuntimeError("board execution is not pinned to a hybrid strategy")
        return ExecutionHandle(
            execution_id=row["execution_id"],
            run_id=run_id,
            owner_suborchestrator_id=owner.agent_instance_id,
            mode=ExecutionMode.BOARD_CLAIM,
        )

    def _record_artifact_metadata(
        self,
        *,
        task: TaskSpec,
        agent: AgentInstance,
        attempt_id: str,
        input_artifacts: list[str],
        manifest: ContextManifest,
        result: dict[str, Any],
    ) -> None:
        artifact_ids = [result.get("output_artifact")]
        published = result.get("result", {}).get("published", {}).get("artifact_id")
        artifact_ids.append(published)
        visibility = VisibilityPolicy(task.artifact_visibility)
        visibility_ref = (
            task.collaboration_round_id
            if visibility == VisibilityPolicy.SEALED_TO_ROUND
            else task.team_id
            if visibility == VisibilityPolicy.RELEASED_TO_TEAM
            else task.domain_id
            if visibility == VisibilityPolicy.DOMAIN
            else attempt_id
            if visibility == VisibilityPolicy.PRIVATE_TO_ATTEMPT
            else None
        )
        labels = frozenset(
            value
            for value in {
                f"agent:{agent.agent_instance_id}",
                f"attempt:{attempt_id}",
                f"round:{task.collaboration_round_id}" if task.collaboration_round_id else None,
                f"team:{task.team_id}" if task.team_id else None,
                f"domain:{task.domain_id}" if task.domain_id else None,
            }
            if value is not None
        )
        for artifact_id in dict.fromkeys(
            artifact_id for artifact_id in artifact_ids if isinstance(artifact_id, str)
        ):
            if visibility != VisibilityPolicy.RUN_WIDE:
                self.context.artifacts.set_visibility(
                    artifact_id,
                    visibility,
                    visibility_ref=visibility_ref,
                    access_labels=labels,
                )
            self.context.record_provenance(
                artifact_id=artifact_id,
                task=task,
                agent=agent,
                attempt_id=attempt_id,
                input_artifacts=input_artifacts,
                manifest=manifest,
            )
