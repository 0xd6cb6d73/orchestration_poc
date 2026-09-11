from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from poc.execution.strategy import PersistentExecutionStrategy, StrategyError
from poc.models import EffectPolicy, EventRecord, ExecutionMode, new_id, utc_now


@dataclass(frozen=True)
class CandidateGrant:
    execution_id: str
    group_id: str
    logical_task_id: str
    candidate_id: str
    worker_id: str
    role_id: str
    generation: int
    effect_policy: EffectPolicy
    token: str = field(repr=False)

    @property
    def token_fingerprint(self) -> str:
        return _token_hash(self.token)[:16]


@dataclass(frozen=True)
class ReconciliationDecision:
    decision_id: str
    group_id: str
    revision: int
    considered_candidate_ids: tuple[str, ...]
    accepted_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    final_result_ref: str
    reason: str


class SpeculativeStrategy(PersistentExecutionStrategy):
    """Read-mostly candidate fan-out with an explicit serialized reconciliation."""

    mode = ExecutionMode.SPECULATIVE

    def start_group(
        self,
        *,
        execution_id: str,
        logical_task_id: str,
        authorized_roles: Sequence[str] | None = None,
        fanout_k: int | None = None,
        stop_policy: str = "require_all",
        reconciliation_policy: str = "deterministic-v1",
    ) -> str:
        execution, policy = self._active_execution(execution_id)
        roles = frozenset(authorized_roles or policy.allowed_roles)
        fanout = policy.speculative_fanout if fanout_k is None else fanout_k
        if not roles or not roles <= policy.allowed_roles:
            raise StrategyError("candidate roles must be a non-empty subset of allowed_roles")
        if fanout < 1 or fanout > policy.speculative_fanout or fanout > policy.max_workers:
            raise StrategyError("fan-out exceeds the approved execution policy")
        if stop_policy not in {"require_all", "quorum"}:
            raise StrategyError("stop_policy must be 'require_all' or 'quorum'")
        group_id = new_id("speculation")
        now = utc_now()
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO swarm_speculation_groups VALUES (?,?,?,?,?,?,?,?,0,'open',NULL,?,?)",
                (
                    group_id,
                    execution_id,
                    logical_task_id,
                    fanout,
                    json.dumps(sorted(roles)),
                    stop_policy,
                    reconciliation_policy,
                    policy.effect_policy,
                    now,
                    now,
                ),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="speculation.started",
                    actor_id=execution["owner_suborchestrator_id"],
                    data={
                        "execution_id": execution_id,
                        "group_id": group_id,
                        "logical_task_id": logical_task_id,
                        "fanout_k": fanout,
                        "authorized_roles": sorted(roles),
                        "stop_policy": stop_policy,
                        "reconciliation_policy": reconciliation_policy,
                        "effect_policy": policy.effect_policy,
                    },
                ),
            )
        return group_id

    def authorize_candidate(
        self, *, execution_id: str, group_id: str, worker_id: str
    ) -> CandidateGrant:
        execution, policy = self._active_execution(execution_id)
        worker = self.db.get_agent(worker_id)
        if (
            worker is None
            or worker.run_id != execution["run_id"]
            or worker.parent_agent_id != execution["owner_suborchestrator_id"]
            or worker.role_id not in policy.allowed_roles
        ):
            raise StrategyError("worker identity is not authorized for this speculation")
        token = secrets.token_urlsafe(32)
        candidate_id = new_id("candidate")
        now = utc_now()
        with self.db.transaction() as tx:
            group = tx.execute(
                "SELECT * FROM swarm_speculation_groups WHERE group_id=? AND execution_id=? AND status='open'",
                (group_id, execution_id),
            ).fetchone()
            if group is None or worker.role_id not in json.loads(group["authorized_roles"]):
                raise StrategyError("speculation group is not open for this worker role")
            count = tx.execute(
                "SELECT count(*) FROM swarm_candidates WHERE group_id=?", (group_id,)
            ).fetchone()[0]
            if count >= group["fanout_k"]:
                raise StrategyError("speculation fan-out is exhausted")
            tx.execute(
                "INSERT INTO swarm_candidates VALUES (?,?,?,?,?,'running',NULL,'[]',NULL,?,NULL)",
                (candidate_id, group_id, worker_id, worker.role_id, _token_hash(token), now),
            )
            self._register_worker(execution_id, worker_id, worker.role_id)
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="candidate.started",
                    actor_id=worker_id,
                    data={
                        "execution_id": execution_id,
                        "group_id": group_id,
                        "logical_task_id": group["logical_task_id"],
                        "candidate_id": candidate_id,
                        "role": worker.role_id,
                        "token_fingerprint": _token_hash(token)[:16],
                    },
                ),
            )
        return CandidateGrant(
            execution_id,
            group_id,
            group["logical_task_id"],
            candidate_id,
            worker_id,
            worker.role_id,
            1,
            EffectPolicy(group["effect_policy"]),
            token,
        )

    def complete_candidate(
        self,
        grant: CandidateGrant,
        *,
        result_ref: str,
        evidence_refs: Sequence[str] = (),
        valid: bool = True,
    ) -> None:
        execution, _ = self._active_execution(grant.execution_id)
        now = utc_now()
        validation_status = "valid" if valid else "invalid"
        with self.db.transaction() as tx:
            changed = tx.execute(
                "UPDATE swarm_candidates SET state='completed',result_ref=?,evidence_refs=?,"
                "validation_status=?,completed_at=? WHERE candidate_id=? AND group_id=? "
                "AND worker_instance_id=? AND token_hash=? AND state='running'",
                (
                    result_ref,
                    json.dumps(list(dict.fromkeys(evidence_refs))),
                    validation_status,
                    now,
                    grant.candidate_id,
                    grant.group_id,
                    grant.worker_id,
                    _token_hash(grant.token),
                ),
            )
            if changed.rowcount != 1:
                raise StrategyError("candidate grant is stale or already completed")
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="candidate.completed",
                    actor_id=grant.worker_id,
                    data={
                        "group_id": grant.group_id,
                        "candidate_id": grant.candidate_id,
                        "result_ref": result_ref,
                        "evidence_refs": list(evidence_refs),
                        "validation_status": validation_status,
                    },
                ),
            )

    def reconcile(
        self,
        *,
        execution_id: str,
        group_id: str,
        accepted_candidate_ids: Sequence[str] | None = None,
        final_result_ref: str | None = None,
        reason: str = "deterministic valid-candidate selection",
    ) -> ReconciliationDecision:
        execution, _ = self._active_execution(execution_id)
        with self.db.transaction() as tx:
            group = tx.execute(
                "SELECT * FROM swarm_speculation_groups WHERE group_id=? AND execution_id=? AND status='open'",
                (group_id, execution_id),
            ).fetchone()
            if group is None:
                raise StrategyError("speculation group is not open")
            candidates = tx.execute(
                "SELECT * FROM swarm_candidates WHERE group_id=? AND state='completed' ORDER BY candidate_id",
                (group_id,),
            ).fetchall()
            required = (
                group["fanout_k"]
                if group["stop_policy"] == "require_all"
                else (group["fanout_k"] // 2 + 1)
            )
            if len(candidates) < required:
                raise StrategyError(f"reconciliation requires {required} completed candidates")
            valid_ids = [
                row["candidate_id"] for row in candidates if row["validation_status"] == "valid"
            ]
            if not valid_ids:
                raise StrategyError("no candidate passed deterministic validation")
            accepted = list(dict.fromkeys(accepted_candidate_ids or [valid_ids[0]]))
            if not accepted or not set(accepted) <= set(valid_ids):
                raise StrategyError("only valid completed candidates may be accepted")
            considered = [row["candidate_id"] for row in candidates]
            rejected = [candidate_id for candidate_id in considered if candidate_id not in accepted]
            result_ref = final_result_ref
            if result_ref is None:
                selected = next(row for row in candidates if row["candidate_id"] == accepted[0])
                result_ref = selected["result_ref"]
            revision = int(group["revision"]) + 1
            decision_id = new_id("reconciliation")
            now = utc_now()
            changed = tx.execute(
                "UPDATE swarm_speculation_groups SET status='finalized',revision=?,final_result_ref=?,updated_at=? "
                "WHERE group_id=? AND status='open' AND revision=?",
                (revision, result_ref, now, group_id, group["revision"]),
            )
            if changed.rowcount != 1:
                raise StrategyError("another reconciler finalized the group")
            tx.execute(
                "INSERT INTO swarm_reconciliations VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    decision_id,
                    group_id,
                    revision,
                    json.dumps(considered),
                    json.dumps(accepted),
                    json.dumps(rejected),
                    reason,
                    result_ref,
                    group["reconciliation_policy"],
                    now,
                ),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=execution["run_id"],
                    event_type="reconciliation.completed",
                    actor_id=execution["owner_suborchestrator_id"],
                    data={
                        "execution_id": execution_id,
                        "group_id": group_id,
                        "decision_id": decision_id,
                        "revision": revision,
                        "candidate_ids_considered": considered,
                        "accepted_candidate_ids": accepted,
                        "rejected_candidate_ids": rejected,
                        "reason": reason,
                        "final_result_ref": result_ref,
                    },
                ),
            )
        return ReconciliationDecision(
            decision_id,
            group_id,
            revision,
            tuple(considered),
            tuple(accepted),
            tuple(rejected),
            result_ref,
            reason,
        )

    def validate_candidate(self, grant: CandidateGrant, *, task_id: str, worker_id: str) -> bool:
        row = self.db.conn.execute(
            "SELECT 1 FROM swarm_candidates c JOIN swarm_speculation_groups g ON g.group_id=c.group_id "
            "WHERE c.candidate_id=? AND c.group_id=? AND g.logical_task_id=? AND c.worker_instance_id=? "
            "AND c.token_hash=? AND c.state='running' AND g.status='open'",
            (
                grant.candidate_id,
                grant.group_id,
                task_id,
                worker_id,
                _token_hash(grant.token),
            ),
        ).fetchone()
        return row is not None

    def _mode_snapshot(self, execution_id: str) -> dict[str, Any]:
        groups = [
            dict(row)
            for row in self.db.conn.execute(
                "SELECT * FROM swarm_speculation_groups WHERE execution_id=? ORDER BY created_at",
                (execution_id,),
            )
        ]
        group_ids = [row["group_id"] for row in groups]
        if not group_ids:
            return {"groups": [], "candidates": [], "reconciliations": []}
        placeholders = ",".join("?" for _ in group_ids)
        return {
            "groups": groups,
            "candidates": [
                dict(row)
                for row in self.db.conn.execute(
                    f"SELECT * FROM swarm_candidates WHERE group_id IN ({placeholders}) ORDER BY created_at",
                    group_ids,
                )
            ],
            "reconciliations": [
                dict(row)
                for row in self.db.conn.execute(
                    f"SELECT * FROM swarm_reconciliations WHERE group_id IN ({placeholders}) ORDER BY created_at",
                    group_ids,
                )
            ],
        }


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
