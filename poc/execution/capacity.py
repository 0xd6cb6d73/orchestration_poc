from __future__ import annotations

from typing import Protocol

from poc.control.spawn_policy import SpawnPolicy
from poc.execution.strategy import StrategyError
from poc.models import (
    AgentInstance,
    EventRecord,
    ExecutionHandle,
    ExecutionMode,
    ExecutionPolicy,
    utc_now,
)
from poc.persistence.database import Database


class WorkerMaterializer(Protocol):
    """Trusted runtime boundary that creates an already-authorized worker."""

    def materialize(
        self,
        *,
        run_id: str,
        parent: AgentInstance,
        role_id: str,
        plan_version: int,
        stable_key: str,
        agent_backend: str,
        agent_provider: str | None,
        agent_model: str | None,
    ) -> AgentInstance: ...


def _invalid_capacity_count(count: object) -> bool:
    return not isinstance(count, int) or isinstance(count, bool) or count < 0


class InProcessWorkerMaterializer:
    def __init__(self, spawns: SpawnPolicy):
        self.spawns = spawns

    def materialize(
        self,
        *,
        run_id: str,
        parent: AgentInstance,
        role_id: str,
        plan_version: int,
        stable_key: str,
        agent_backend: str,
        agent_provider: str | None,
        agent_model: str | None,
    ) -> AgentInstance:
        return self.spawns.spawn(
            run_id=run_id,
            parent=parent,
            child_role=role_id,
            plan_version=plan_version,
            stable_key=stable_key,
            agent_backend=agent_backend,
            agent_provider=agent_provider,
            agent_model=agent_model,
        )


class CapacityScheduler:
    """Reconciles board worker population without granting workers spawn authority."""

    def __init__(self, db: Database, materializer: WorkerMaterializer):
        self.db = db
        self.materializer = materializer

    def reconcile(self, handle: ExecutionHandle, desired: dict[str, int]) -> list[AgentInstance]:
        row = self.db.conn.execute(
            "SELECT * FROM executions WHERE execution_id=? AND run_id=? AND owner_suborchestrator_id=?",
            (handle.execution_id, handle.run_id, handle.owner_suborchestrator_id),
        ).fetchone()
        if row is None or row["status"] != "active" or row["mode"] != ExecutionMode.BOARD_CLAIM:
            raise StrategyError("capacity scheduler requires an active board-and-claim execution")
        policy = ExecutionPolicy.model_validate_json(row["policy"])
        if (
            set(desired) - policy.allowed_roles
            or any(_invalid_capacity_count(count) for count in desired.values())
            or sum(desired.values()) > policy.max_workers
        ):
            raise StrategyError("desired capacity exceeds the authorized role/population envelope")
        owner = self.db.get_agent(handle.owner_suborchestrator_id)
        if owner is None or not self.db.authority_active(handle.run_id, owner.plan_version):
            raise StrategyError("sub-orchestrator authority is inactive")

        active: list[AgentInstance] = []
        now = utc_now()
        for role_id in sorted(policy.allowed_roles):
            target = desired.get(role_id, 0)
            records = self.db.conn.execute(
                "SELECT * FROM swarm_execution_workers WHERE execution_id=? AND role_id=? ORDER BY created_at,worker_instance_id",
                (handle.execution_id, role_id),
            ).fetchall()
            for index in range(target):
                if index < len(records):
                    worker = self.db.get_agent(records[index]["worker_instance_id"])
                    with self.db.transaction() as tx:
                        tx.execute(
                            "UPDATE swarm_execution_workers SET status='active',updated_at=? "
                            "WHERE execution_id=? AND worker_instance_id=?",
                            (now, handle.execution_id, records[index]["worker_instance_id"]),
                        )
                else:
                    worker = self.materializer.materialize(
                        run_id=handle.run_id,
                        parent=owner,
                        role_id=role_id,
                        plan_version=owner.plan_version,
                        stable_key=f"{handle.execution_id}-{role_id}-{index}",
                        agent_backend=policy.agent_backend,
                        agent_provider=policy.agent_provider,
                        agent_model=policy.agent_model,
                    )
                    with self.db.transaction() as tx:
                        tx.execute(
                            "INSERT INTO swarm_execution_workers VALUES (?,?,?,'active',?,?)",
                            (handle.execution_id, worker.agent_instance_id, role_id, now, now),
                        )
                        self.db.append_event(
                            tx,
                            EventRecord(
                                run_id=handle.run_id,
                                event_type="worker.started",
                                actor_id=worker.agent_instance_id,
                                data={
                                    "execution_id": handle.execution_id,
                                    "role": role_id,
                                    "agent_backend": worker.agent_backend,
                                    "agent_provider": worker.agent_provider,
                                    "agent_model": worker.agent_model,
                                    "parent_authority": owner.agent_instance_id,
                                },
                            ),
                        )
                if worker is not None:
                    active.append(worker)
            for record in records[target:]:
                with self.db.transaction() as tx:
                    tx.execute(
                        "UPDATE swarm_execution_workers SET status='stopped',updated_at=? "
                        "WHERE execution_id=? AND worker_instance_id=?",
                        (now, handle.execution_id, record["worker_instance_id"]),
                    )
                    self.db.append_event(
                        tx,
                        EventRecord(
                            run_id=handle.run_id,
                            event_type="worker.terminated",
                            actor_id=record["worker_instance_id"],
                            data={
                                "execution_id": handle.execution_id,
                                "reason": "capacity reconciled",
                            },
                        ),
                    )
        return active

    def activate_existing(self, handle: ExecutionHandle, worker: AgentInstance) -> None:
        """Attach a pre-materialized graph worker to a bounded board population."""
        now = utc_now()
        with self.db.transaction() as tx:
            row = tx.execute(
                "SELECT * FROM executions WHERE execution_id=? AND run_id=?",
                (handle.execution_id, handle.run_id),
            ).fetchone()
            if row is None or row["status"] != "active" or row["mode"] != ExecutionMode.BOARD_CLAIM:
                raise StrategyError("worker activation requires an active board execution")
            policy = ExecutionPolicy.model_validate_json(row["policy"])
            if (
                worker.run_id != handle.run_id
                or worker.parent_agent_id != handle.owner_suborchestrator_id
                or worker.role_id not in policy.allowed_roles
                or worker.agent_backend != policy.agent_backend
            ):
                raise StrategyError("worker is incompatible with the board execution policy")
            existing = tx.execute(
                "SELECT status FROM swarm_execution_workers WHERE execution_id=? "
                "AND worker_instance_id=?",
                (handle.execution_id, worker.agent_instance_id),
            ).fetchone()
            active_count = int(
                tx.execute(
                    "SELECT count(*) FROM swarm_execution_workers WHERE execution_id=? "
                    "AND status='active'",
                    (handle.execution_id,),
                ).fetchone()[0]
            )
            if (
                existing is None or existing["status"] != "active"
            ) and active_count >= policy.max_workers:
                raise StrategyError("board worker population limit is exhausted")
            tx.execute(
                "INSERT INTO swarm_execution_workers VALUES (?,?,?,'active',?,?) "
                "ON CONFLICT(execution_id,worker_instance_id) DO UPDATE SET "
                "role_id=excluded.role_id,status='active',updated_at=excluded.updated_at",
                (handle.execution_id, worker.agent_instance_id, worker.role_id, now, now),
            )
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=handle.run_id,
                    event_type="worker.attached",
                    actor_id=worker.agent_instance_id,
                    data={
                        "execution_id": handle.execution_id,
                        "role": worker.role_id,
                        "parent_authority": handle.owner_suborchestrator_id,
                    },
                ),
            )

    def deactivate(self, handle: ExecutionHandle, worker_id: str) -> None:
        now = utc_now()
        with self.db.transaction() as tx:
            changed = tx.execute(
                "UPDATE swarm_execution_workers SET status='stopped',updated_at=? "
                "WHERE execution_id=? AND worker_instance_id=? AND status='active'",
                (now, handle.execution_id, worker_id),
            )
            if changed.rowcount:
                self.db.append_event(
                    tx,
                    EventRecord(
                        run_id=handle.run_id,
                        event_type="worker.detached",
                        actor_id=worker_id,
                        data={"execution_id": handle.execution_id},
                    ),
                )
