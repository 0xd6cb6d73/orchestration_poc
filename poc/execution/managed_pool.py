from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from poc.execution.strategy import PersistentExecutionStrategy, StrategyError
from poc.models import EventRecord, ExecutionHandle, ExecutionMode, ExecutionPolicy, new_id, utc_now


class StaleAssignment(StrategyError):
    pass


@dataclass(frozen=True)
class Assignment:
    execution_id: str
    pool_id: str
    assignment_id: str
    offer_id: str
    task_id: str
    slot_id: str
    worker_id: str
    generation: int
    expires_at: str
    token: str = field(repr=False)

    @property
    def token_fingerprint(self) -> str:
        return _token_hash(self.token)[:16]


class ManagedPoolStrategy(PersistentExecutionStrategy):
    """Bounded persistent worker pool with deterministic structured arbitration."""

    mode = ExecutionMode.MANAGED_POOL

    def _validate_policy(self, policy: ExecutionPolicy) -> None:
        role_quotas = dict(policy.options.get("role_quotas", {}))
        if not role_quotas:
            return
        unknown = set(role_quotas) - policy.allowed_roles
        if unknown or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in role_quotas.values()
        ):
            raise StrategyError("role quotas must be non-negative integers within allowed_roles")
        if sum(role_quotas.values()) < policy.max_workers:
            raise StrategyError("role quotas cannot be smaller than max_workers in aggregate")

    async def _after_submit(self, handle: ExecutionHandle, policy: ExecutionPolicy) -> None:
        role_quotas = dict(policy.options.get("role_quotas", {}))
        if not role_quotas:
            role_quotas = {role: policy.max_workers for role in policy.allowed_roles}
        now = utc_now()
        pool_id = new_id("pool")
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO swarm_pools VALUES (?,?,?,?,?,'active',?)",
                (
                    pool_id,
                    handle.execution_id,
                    json.dumps(role_quotas, sort_keys=True),
                    policy.max_workers,
                    str(policy.options.get("arbitration_policy", "deterministic-v1")),
                    now,
                ),
            )
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=handle.run_id,
                    event_type="pool.created",
                    actor_id=handle.owner_suborchestrator_id,
                    data={
                        "execution_id": handle.execution_id,
                        "pool_id": pool_id,
                        "role_quotas": role_quotas,
                        "max_slots": policy.max_workers,
                        "arbitration_policy": policy.options.get(
                            "arbitration_policy", "deterministic-v1"
                        ),
                    },
                ),
            )

    def pool_id(self, execution_id: str) -> str:
        self._active_execution(execution_id)
        row = self.db.conn.execute(
            "SELECT pool_id FROM swarm_pools WHERE execution_id=?", (execution_id,)
        ).fetchone()
        if row is None:
            raise StrategyError("managed pool was not initialized")
        return str(row[0])

    def start_slot(self, *, execution_id: str, worker_id: str) -> str:
        execution, policy = self._active_execution(execution_id)
        pool_id = self.pool_id(execution_id)
        worker = self.db.get_agent(worker_id)
        if (
            worker is None
            or worker.run_id != execution["run_id"]
            or worker.parent_agent_id != execution["owner_suborchestrator_id"]
            or worker.role_id not in policy.allowed_roles
        ):
            raise StrategyError("worker identity is not authorized for this pool")
        with self.db.transaction() as tx:
            pool = tx.execute(
                "SELECT * FROM swarm_pools WHERE pool_id=? AND state='active'", (pool_id,)
            ).fetchone()
            count = tx.execute(
                "SELECT count(*) FROM swarm_pool_slots WHERE pool_id=? AND status!='dead'",
                (pool_id,),
            ).fetchone()[0]
            role_count = tx.execute(
                "SELECT count(*) FROM swarm_pool_slots WHERE pool_id=? AND role_id=? AND status!='dead'",
                (pool_id, worker.role_id),
            ).fetchone()[0]
            quotas = json.loads(pool["role_quotas"])
            if count >= pool["max_slots"] or role_count >= quotas.get(worker.role_id, 0):
                raise StrategyError("pool capacity or role quota is exhausted")
            prior = tx.execute(
                "SELECT max(generation) FROM swarm_pool_slots WHERE pool_id=? AND worker_instance_id=?",
                (pool_id, worker_id),
            ).fetchone()[0]
            generation = int(prior or 0) + 1
            slot_id = new_id("slot")
            now = utc_now()
            tx.execute(
                "INSERT INTO swarm_pool_slots VALUES (?,?,?,?,?,'idle',?,NULL)",
                (slot_id, pool_id, worker_id, worker.role_id, generation, now),
            )
            self._register_worker(execution_id, worker_id, worker.role_id)
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="pool.slot_started",
                    actor_id=execution["owner_suborchestrator_id"],
                    data={
                        "execution_id": execution_id,
                        "pool_id": pool_id,
                        "slot_id": slot_id,
                        "worker_id": worker_id,
                        "role": worker.role_id,
                        "slot_generation": generation,
                    },
                ),
            )
        return slot_id

    def publish_offer(
        self,
        *,
        execution_id: str,
        task_id: str,
        eligible_roles: Sequence[str],
        bid_deadline: datetime | str,
        priority: int = 0,
        acceptance_criteria_ref: str | None = None,
    ) -> str:
        execution, policy = self._active_execution(execution_id)
        roles = frozenset(eligible_roles)
        if not roles or not roles <= policy.allowed_roles:
            raise StrategyError("offer roles must be a non-empty subset of allowed_roles")
        offer_id = new_id("offer")
        deadline = _timestamp(bid_deadline)
        now = utc_now()
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO swarm_offers VALUES (?,?,?,?,?,?,?,'open',?)",
                (
                    offer_id,
                    self.pool_id(execution_id),
                    task_id,
                    json.dumps(sorted(roles)),
                    priority,
                    deadline,
                    acceptance_criteria_ref,
                    now,
                ),
            )
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="offer.published",
                    actor_id=execution["owner_suborchestrator_id"],
                    data={
                        "execution_id": execution_id,
                        "offer_id": offer_id,
                        "task_id": task_id,
                        "eligible_roles": sorted(roles),
                        "bid_deadline": deadline,
                        "priority": priority,
                    },
                ),
            )
        return offer_id

    def submit_bid(
        self,
        *,
        execution_id: str,
        offer_id: str,
        slot_id: str,
        fit_features: dict[str, int | float | str | bool] | None = None,
    ) -> str:
        execution, _ = self._active_execution(execution_id)
        with self.db.transaction() as tx:
            offer = tx.execute(
                "SELECT * FROM swarm_offers WHERE offer_id=? AND pool_id=? AND state='open'",
                (offer_id, self.pool_id(execution_id)),
            ).fetchone()
            slot = tx.execute(
                "SELECT * FROM swarm_pool_slots WHERE slot_id=? AND pool_id=?",
                (slot_id, self.pool_id(execution_id)),
            ).fetchone()
            if offer is None or slot is None:
                raise StrategyError("offer or slot is not active in this pool")
            if offer["bid_deadline"] < utc_now():
                raise StrategyError("offer bid deadline has passed")
            eligible = slot["role_id"] in json.loads(offer["eligible_roles"])
            available = slot["status"] == "idle"
            if not eligible or not available:
                raise StrategyError("slot is not eligible and idle for this offer")
            bid_id = new_id("bid")
            now = utc_now()
            try:
                tx.execute(
                    "INSERT INTO swarm_bids VALUES (?,?,?,?,?,?,?)",
                    (
                        bid_id,
                        offer_id,
                        slot_id,
                        1,
                        1,
                        json.dumps(fit_features or {}, sort_keys=True),
                        now,
                    ),
                )
            except Exception as exc:
                raise StrategyError("the slot already bid on this offer") from exc
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="bid.submitted",
                    actor_id=slot["worker_instance_id"],
                    data={
                        "offer_id": offer_id,
                        "slot_id": slot_id,
                        "fit_features": fit_features or {},
                    },
                ),
            )
        return bid_id

    def arbitrate(self, *, execution_id: str, offer_id: str) -> Assignment:
        execution, policy = self._active_execution(execution_id)
        pool_id = self.pool_id(execution_id)
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        with self.db.transaction() as tx:
            offer = tx.execute(
                "SELECT * FROM swarm_offers WHERE offer_id=? AND pool_id=? AND state='open'",
                (offer_id, pool_id),
            ).fetchone()
            if offer is None:
                raise StrategyError("offer is not open")
            rows = tx.execute(
                "SELECT b.*,s.worker_instance_id,s.status,s.last_assigned_at FROM swarm_bids b "
                "JOIN swarm_pool_slots s ON s.slot_id=b.slot_id "
                "WHERE b.offer_id=? AND b.eligibility=1 AND b.availability=1 AND s.status='idle'",
                (offer_id,),
            ).fetchall()
            if not rows:
                raise StrategyError("offer has no eligible available bids")

            def rank(row: sqlite3.Row) -> tuple[float, str, str]:
                features = json.loads(row["fit_features"])
                score = float(features.get("score", 0))
                return (-score, row["last_assigned_at"] or "", row["slot_id"])

            winner = min(rows, key=rank)
            previous = tx.execute(
                "SELECT max(a.generation) FROM swarm_assignments a "
                "JOIN swarm_offers o ON o.offer_id=a.offer_id WHERE a.task_id=? AND o.pool_id=?",
                (offer["task_id"], pool_id),
            ).fetchone()[0]
            generation = int(previous or 0) + 1
            assignment_id = new_id("assignment")
            token = secrets.token_urlsafe(32)
            expires_at = (now_dt + timedelta(seconds=policy.lease_seconds)).isoformat()
            tx.execute(
                "INSERT INTO swarm_assignments VALUES (?,?,?,?,?,?,?,'active',NULL,?,?)",
                (
                    assignment_id,
                    offer_id,
                    offer["task_id"],
                    winner["slot_id"],
                    generation,
                    _token_hash(token),
                    expires_at,
                    now,
                    now,
                ),
            )
            tx.execute("UPDATE swarm_offers SET state='assigned' WHERE offer_id=?", (offer_id,))
            tx.execute(
                "UPDATE swarm_pool_slots SET status='assigned',last_assigned_at=?,last_seen_at=? WHERE slot_id=?",
                (now, now, winner["slot_id"]),
            )
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="assignment.created",
                    actor_id=execution["owner_suborchestrator_id"],
                    data={
                        "execution_id": execution_id,
                        "pool_id": pool_id,
                        "offer_id": offer_id,
                        "task_id": offer["task_id"],
                        "slot_id": winner["slot_id"],
                        "worker_id": winner["worker_instance_id"],
                        "assignment_id": assignment_id,
                        "assignment_generation": generation,
                        "token_fingerprint": _token_hash(token)[:16],
                        "lease_expires_at": expires_at,
                    },
                ),
            )
        return Assignment(
            execution_id,
            pool_id,
            assignment_id,
            offer_id,
            offer["task_id"],
            winner["slot_id"],
            winner["worker_instance_id"],
            generation,
            expires_at,
            token,
        )

    def heartbeat(self, assignment: Assignment) -> str:
        _, policy = self._active_execution(assignment.execution_id)
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(seconds=policy.lease_seconds)).isoformat()
        with self.db.transaction() as tx:
            changed = tx.execute(
                "UPDATE swarm_assignments SET lease_expires_at=?,updated_at=? WHERE assignment_id=? "
                "AND slot_id=? AND generation=? AND token_hash=? AND state='active' AND lease_expires_at>?",
                (
                    expires_at,
                    now,
                    assignment.assignment_id,
                    assignment.slot_id,
                    assignment.generation,
                    _token_hash(assignment.token),
                    now,
                ),
            )
            if changed.rowcount != 1:
                raise StaleAssignment("assignment no longer owns the task")
            tx.execute(
                "UPDATE swarm_pool_slots SET last_seen_at=? WHERE slot_id=?",
                (now, assignment.slot_id),
            )
        return expires_at

    def reap_expired(self, execution_id: str) -> list[str]:
        execution, _ = self._active_execution(execution_id)
        pool_id = self.pool_id(execution_id)
        now = utc_now()
        expired: list[str] = []
        with self.db.transaction() as tx:
            rows = tx.execute(
                "SELECT a.* FROM swarm_assignments a JOIN swarm_offers o ON o.offer_id=a.offer_id "
                "WHERE o.pool_id=? AND a.state='active' AND a.lease_expires_at<=?",
                (pool_id, now),
            ).fetchall()
            for row in rows:
                tx.execute(
                    "UPDATE swarm_assignments SET state='expired',updated_at=? "
                    "WHERE assignment_id=? AND state='active'",
                    (now, row["assignment_id"]),
                )
                tx.execute(
                    "UPDATE swarm_offers SET state='open' WHERE offer_id=?", (row["offer_id"],)
                )
                tx.execute(
                    "UPDATE swarm_pool_slots SET status='idle',last_seen_at=? WHERE slot_id=?",
                    (now, row["slot_id"]),
                )
                self.db.append_event(
                    tx,
                    EventRecord(
                        run_id=execution["run_id"],
                        event_type="assignment.expired",
                        data={
                            "execution_id": execution_id,
                            "pool_id": pool_id,
                            "assignment_id": row["assignment_id"],
                            "task_id": row["task_id"],
                            "assignment_generation": row["generation"],
                            "lease_expires_at": row["lease_expires_at"],
                        },
                    ),
                )
                expired.append(row["assignment_id"])
        return expired

    def complete(self, assignment: Assignment, *, result_ref: str) -> None:
        execution, _ = self._active_execution(assignment.execution_id)
        now = utc_now()
        with self.db.transaction() as tx:
            changed = tx.execute(
                "UPDATE swarm_assignments SET state='completed',result_ref=?,updated_at=? "
                "WHERE assignment_id=? AND slot_id=? AND generation=? AND token_hash=? "
                "AND state='active' AND lease_expires_at>?",
                (
                    result_ref,
                    now,
                    assignment.assignment_id,
                    assignment.slot_id,
                    assignment.generation,
                    _token_hash(assignment.token),
                    now,
                ),
            )
            if changed.rowcount != 1:
                raise StaleAssignment("assignment no longer owns the task")
            tx.execute(
                "UPDATE swarm_pool_slots SET status='idle',last_seen_at=? WHERE slot_id=?",
                (now, assignment.slot_id),
            )
            self.db.append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="assignment.completed",
                    actor_id=assignment.worker_id,
                    data={
                        "assignment_id": assignment.assignment_id,
                        "task_id": assignment.task_id,
                        "assignment_generation": assignment.generation,
                        "result_ref": result_ref,
                    },
                ),
            )

    def validate_assignment(self, assignment: Assignment, *, task_id: str, worker_id: str) -> bool:
        row = self.db.conn.execute(
            "SELECT 1 FROM swarm_assignments a JOIN swarm_pool_slots s ON s.slot_id=a.slot_id "
            "WHERE a.assignment_id=? AND a.task_id=? AND s.worker_instance_id=? AND a.generation=? "
            "AND a.token_hash=? AND a.state='active' AND a.lease_expires_at>?",
            (
                assignment.assignment_id,
                task_id,
                worker_id,
                assignment.generation,
                _token_hash(assignment.token),
                utc_now(),
            ),
        ).fetchone()
        return row is not None

    def _mode_snapshot(self, execution_id: str) -> dict[str, Any]:
        pool_id = self.db.conn.execute(
            "SELECT pool_id FROM swarm_pools WHERE execution_id=?", (execution_id,)
        ).fetchone()
        if pool_id is None:
            return {"pool": None, "slots": [], "offers": [], "assignments": []}
        key = pool_id[0]
        return {
            "pool": dict(
                self.db.conn.execute("SELECT * FROM swarm_pools WHERE pool_id=?", (key,)).fetchone()
            ),
            "slots": [
                dict(row)
                for row in self.db.conn.execute(
                    "SELECT * FROM swarm_pool_slots WHERE pool_id=?", (key,)
                )
            ],
            "offers": [
                dict(row)
                for row in self.db.conn.execute(
                    "SELECT * FROM swarm_offers WHERE pool_id=?", (key,)
                )
            ],
            "assignments": [
                dict(row)
                for row in self.db.conn.execute(
                    "SELECT a.* FROM swarm_assignments a JOIN swarm_offers o ON o.offer_id=a.offer_id WHERE o.pool_id=?",
                    (key,),
                )
            ],
        }


def _timestamp(value: datetime | str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
