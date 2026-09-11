from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from poc.execution.strategy import PersistentExecutionStrategy, StrategyError
from poc.models import EventRecord, ExecutionMode, new_id, utc_now


class StaleClaim(StrategyError):
    pass


@dataclass(frozen=True)
class Claim:
    execution_id: str
    task_id: str
    attempt_id: str
    worker_id: str
    generation: int
    expires_at: str
    token: str = field(repr=False)

    @property
    def token_fingerprint(self) -> str:
        return _fingerprint(self.token)


class BoardClaimStrategy(PersistentExecutionStrategy):
    """SQLite PoC task board with atomic fenced, leased claims."""

    mode = ExecutionMode.BOARD_CLAIM

    def post_task(
        self,
        *,
        execution_id: str,
        task_id: str,
        required_role: str,
        task_spec_ref: str,
        acceptance_criteria_ref: str | None = None,
        dependency_ids: Sequence[str] = (),
        priority: int = 0,
        available_at: datetime | str | None = None,
        revision: int = 1,
        max_attempts: int | None = None,
    ) -> None:
        execution, policy = self._active_execution(execution_id)
        if required_role not in policy.allowed_roles:
            raise StrategyError(f"role {required_role!r} is outside the execution policy")
        attempts = policy.max_task_attempts if max_attempts is None else max_attempts
        if attempts < 1 or attempts > policy.max_task_attempts:
            raise StrategyError("task attempts exceed the execution policy")
        dependencies = list(dict.fromkeys(dependency_ids))
        if task_id in dependencies:
            raise StrategyError("a task cannot depend on itself")
        now = utc_now()
        ready_at = _timestamp(available_at) if available_at else now
        with self.db.transaction() as tx:
            missing = [
                dep
                for dep in dependencies
                if tx.execute(
                    "SELECT 1 FROM swarm_tasks WHERE execution_id=? AND task_id=?",
                    (execution_id, dep),
                ).fetchone()
                is None
            ]
            if missing:
                raise StrategyError(f"unknown dependencies: {missing}")
            state = "blocked" if dependencies else "ready"
            try:
                tx.execute(
                    "INSERT INTO swarm_tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        execution_id,
                        task_id,
                        revision,
                        required_role,
                        task_spec_ref,
                        acceptance_criteria_ref,
                        json.dumps(dependencies),
                        priority,
                        ready_at,
                        state,
                        None,
                        None,
                        0,
                        None,
                        0,
                        attempts,
                        None,
                        now,
                        now,
                    ),
                )
            except Exception as exc:
                raise StrategyError(f"task {task_id!r} already exists") from exc
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="task.posted",
                    actor_id=execution["owner_suborchestrator_id"],
                    data={
                        "execution_id": execution_id,
                        "task_id": task_id,
                        "task_revision": revision,
                        "required_role": required_role,
                        "dependencies": dependencies,
                        "priority": priority,
                    },
                ),
            )

    def claim_next(
        self,
        *,
        execution_id: str,
        worker_id: str,
        eligible_roles: Sequence[str] | None = None,
    ) -> Claim | None:
        execution, policy = self._active_execution(execution_id)
        worker = self.db.get_agent(worker_id)
        if (
            worker is None
            or worker.run_id != execution["run_id"]
            or worker.parent_agent_id != execution["owner_suborchestrator_id"]
            or worker.role_id not in policy.allowed_roles
        ):
            raise StrategyError("worker identity is not authorized for this execution")
        membership = self.db.conn.execute(
            "SELECT 1 FROM swarm_execution_workers WHERE execution_id=? AND worker_instance_id=? "
            "AND role_id=? AND status='active'",
            (execution_id, worker_id, worker.role_id),
        ).fetchone()
        if membership is None:
            raise StrategyError("worker is not in the scheduler-authorized active population")
        roles = frozenset(eligible_roles or (worker.role_id,))
        if worker.role_id not in roles or not roles <= policy.allowed_roles:
            raise StrategyError("eligible roles cannot enlarge the worker's authority")

        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=policy.lease_seconds)).isoformat()
        token = secrets.token_urlsafe(32)
        token_hash = _token_hash(token)
        attempt_id = new_id("attempt")
        with self.db.transaction() as tx:
            placeholders = ",".join("?" for _ in roles)
            row = tx.execute(
                f"SELECT * FROM swarm_tasks WHERE execution_id=? AND state='ready' "
                f"AND required_role IN ({placeholders}) AND available_at<=? "
                "AND attempt_count<max_attempts "
                "ORDER BY priority DESC,created_at ASC,task_id ASC LIMIT 1",
                (execution_id, *sorted(roles), now),
            ).fetchone()
            if row is None:
                return None
            generation = int(row["claim_generation"]) + 1
            changed = tx.execute(
                "UPDATE swarm_tasks SET state='claimed',claimant_id=?,claim_token_hash=?,"
                "claim_generation=?,lease_expires_at=?,attempt_count=attempt_count+1,updated_at=? "
                "WHERE execution_id=? AND task_id=? AND state='ready'",
                (worker_id, token_hash, generation, expires_at, now, execution_id, row["task_id"]),
            )
            if changed.rowcount != 1:
                return None
            tx.execute(
                "INSERT INTO swarm_task_attempts VALUES (?,?,?,?,?,?,?,NULL,'running',NULL,NULL)",
                (attempt_id, execution_id, row["task_id"], generation, worker_id, now, now),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="claim.acquired",
                    actor_id=worker_id,
                    data={
                        "execution_id": execution_id,
                        "task_id": row["task_id"],
                        "attempt_id": attempt_id,
                        "claim_generation": generation,
                        "token_fingerprint": _fingerprint(token),
                        "lease_expires_at": expires_at,
                        "role": worker.role_id,
                    },
                ),
            )
        return Claim(
            execution_id, row["task_id"], attempt_id, worker_id, generation, expires_at, token
        )

    def heartbeat(self, claim: Claim) -> str:
        execution, policy = self._active_execution(claim.execution_id)
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=policy.lease_seconds)).isoformat()
        with self.db.transaction() as tx:
            changed = tx.execute(
                "UPDATE swarm_tasks SET lease_expires_at=?,updated_at=? WHERE execution_id=? AND task_id=? "
                "AND state='claimed' AND claimant_id=? AND claim_token_hash=? AND claim_generation=? "
                "AND lease_expires_at>?",
                (
                    expires_at,
                    now,
                    claim.execution_id,
                    claim.task_id,
                    claim.worker_id,
                    _token_hash(claim.token),
                    claim.generation,
                    now,
                ),
            )
            if changed.rowcount != 1:
                raise StaleClaim("claim no longer owns the task")
            tx.execute(
                "UPDATE swarm_task_attempts SET last_heartbeat_at=? WHERE attempt_id=? AND status='running'",
                (now, claim.attempt_id),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="claim.heartbeat",
                    actor_id=claim.worker_id,
                    data={
                        "task_id": claim.task_id,
                        "attempt_id": claim.attempt_id,
                        "claim_generation": claim.generation,
                        "lease_expires_at": expires_at,
                    },
                ),
            )
        return expires_at

    def release(
        self, claim: Claim, *, retry_at: datetime | str | None = None, reason: str = "released"
    ) -> None:
        execution, _ = self._active_execution(claim.execution_id)
        now = utc_now()
        available_at = _timestamp(retry_at) if retry_at else now
        with self.db.transaction() as tx:
            row = self._owned_row(tx, claim, require_unexpired=False)
            state = "ready" if row["attempt_count"] < row["max_attempts"] else "failed"
            tx.execute(
                "UPDATE swarm_tasks SET state=?,claimant_id=NULL,claim_token_hash=NULL,"
                "lease_expires_at=NULL,available_at=?,updated_at=? WHERE execution_id=? AND task_id=?",
                (state, available_at, now, claim.execution_id, claim.task_id),
            )
            tx.execute(
                "UPDATE swarm_task_attempts SET status='abandoned',failure_ref=?,ended_at=? "
                "WHERE attempt_id=? AND status='running'",
                (reason, now, claim.attempt_id),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="claim.released",
                    actor_id=claim.worker_id,
                    data={
                        "task_id": claim.task_id,
                        "attempt_id": claim.attempt_id,
                        "claim_generation": claim.generation,
                        "reason": reason,
                        "next_eligible_at": available_at if state == "ready" else None,
                    },
                ),
            )

    def complete(self, claim: Claim, *, result_ref: str) -> None:
        execution, _ = self._active_execution(claim.execution_id)
        now = utc_now()
        with self.db.transaction() as tx:
            self._owned_row(tx, claim, require_unexpired=True)
            tx.execute(
                "UPDATE swarm_tasks SET state='completed',accepted_result_ref=?,claimant_id=NULL,"
                "claim_token_hash=NULL,lease_expires_at=NULL,updated_at=? "
                "WHERE execution_id=? AND task_id=?",
                (result_ref, now, claim.execution_id, claim.task_id),
            )
            tx.execute(
                "UPDATE swarm_task_attempts SET status='succeeded',result_ref=?,ended_at=? "
                "WHERE attempt_id=? AND status='running'",
                (result_ref, now, claim.attempt_id),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="task.completed",
                    actor_id=claim.worker_id,
                    data={
                        "execution_id": claim.execution_id,
                        "task_id": claim.task_id,
                        "attempt_id": claim.attempt_id,
                        "claim_generation": claim.generation,
                        "result_ref": result_ref,
                    },
                ),
            )
            self._promote_dependents(
                tx, execution["run_id"], claim.execution_id, claim.task_id, now
            )

    def reap_expired(self, execution_id: str) -> list[str]:
        execution, _ = self._active_execution(execution_id)
        now = utc_now()
        expired: list[str] = []
        with self.db.transaction() as tx:
            rows = tx.execute(
                "SELECT * FROM swarm_tasks WHERE execution_id=? AND state='claimed' AND lease_expires_at<=?",
                (execution_id, now),
            ).fetchall()
            for row in rows:
                next_state = "ready" if row["attempt_count"] < row["max_attempts"] else "failed"
                tx.execute(
                    "UPDATE swarm_tasks SET state=?,claimant_id=NULL,claim_token_hash=NULL,"
                    "lease_expires_at=NULL,available_at=?,updated_at=? WHERE execution_id=? AND task_id=?",
                    (next_state, now, now, execution_id, row["task_id"]),
                )
                attempt = tx.execute(
                    "SELECT attempt_id,last_heartbeat_at FROM swarm_task_attempts WHERE execution_id=? "
                    "AND task_id=? AND ownership_generation=? AND status='running'",
                    (execution_id, row["task_id"], row["claim_generation"]),
                ).fetchone()
                if attempt:
                    tx.execute(
                        "UPDATE swarm_task_attempts SET status='abandoned',failure_ref='lease expired',ended_at=? "
                        "WHERE attempt_id=?",
                        (now, attempt["attempt_id"]),
                    )
                self.db._append_event(
                    tx,
                    EventRecord(
                        run_id=execution["run_id"],
                        event_type="claim.expired",
                        data={
                            "execution_id": execution_id,
                            "task_id": row["task_id"],
                            "attempt_id": attempt["attempt_id"] if attempt else None,
                            "claim_generation": row["claim_generation"],
                            "lease_expires_at": row["lease_expires_at"],
                            "last_heartbeat_at": attempt["last_heartbeat_at"] if attempt else None,
                            "next_state": next_state,
                        },
                    ),
                )
                expired.append(row["task_id"])
        return expired

    def validate_claim(self, claim: Claim, *, task_id: str, worker_id: str) -> bool:
        now = utc_now()
        row = self.db.conn.execute(
            "SELECT 1 FROM swarm_tasks WHERE execution_id=? AND task_id=? AND state='claimed' "
            "AND claimant_id=? AND claim_token_hash=? AND claim_generation=? AND lease_expires_at>?",
            (
                claim.execution_id,
                task_id,
                worker_id,
                _token_hash(claim.token),
                claim.generation,
                now,
            ),
        ).fetchone()
        return row is not None

    def _owned_row(self, tx, claim: Claim, *, require_unexpired: bool):
        suffix = " AND lease_expires_at>?" if require_unexpired else ""
        params: tuple[Any, ...] = (
            claim.execution_id,
            claim.task_id,
            claim.worker_id,
            _token_hash(claim.token),
            claim.generation,
        )
        if require_unexpired:
            params += (utc_now(),)
        row = tx.execute(
            "SELECT * FROM swarm_tasks WHERE execution_id=? AND task_id=? AND state='claimed' "
            "AND claimant_id=? AND claim_token_hash=? AND claim_generation=?" + suffix,
            params,
        ).fetchone()
        if row is None:
            raise StaleClaim("claim no longer owns the task")
        return row

    def _promote_dependents(
        self, tx, run_id: str, execution_id: str, completed_task_id: str, now: str
    ) -> None:
        rows = tx.execute(
            "SELECT task_id,dependency_ids FROM swarm_tasks WHERE execution_id=? AND state='blocked'",
            (execution_id,),
        ).fetchall()
        completed = {
            row["task_id"]
            for row in tx.execute(
                "SELECT task_id FROM swarm_tasks WHERE execution_id=? AND state='completed'",
                (execution_id,),
            )
        }
        for row in rows:
            if set(json.loads(row["dependency_ids"])) <= completed:
                tx.execute(
                    "UPDATE swarm_tasks SET state='ready',updated_at=? WHERE execution_id=? AND task_id=?",
                    (now, execution_id, row["task_id"]),
                )
                self.db._append_event(
                    tx,
                    EventRecord(
                        run_id=run_id,
                        event_type="task.ready",
                        data={
                            "execution_id": execution_id,
                            "task_id": row["task_id"],
                            "unblocked_by": completed_task_id,
                        },
                    ),
                )

    def _mode_snapshot(self, execution_id: str) -> dict[str, Any]:
        tasks = [
            {**dict(row), "dependency_ids": json.loads(row["dependency_ids"])}
            for row in self.db.conn.execute(
                "SELECT * FROM swarm_tasks WHERE execution_id=? ORDER BY priority DESC,created_at,task_id",
                (execution_id,),
            )
        ]
        attempts = [
            dict(row)
            for row in self.db.conn.execute(
                "SELECT * FROM swarm_task_attempts WHERE execution_id=? ORDER BY started_at,attempt_id",
                (execution_id,),
            )
        ]
        return {"tasks": tasks, "attempts": attempts}


def _timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _fingerprint(token: str) -> str:
    return _token_hash(token)[:16]
