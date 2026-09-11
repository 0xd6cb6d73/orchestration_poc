from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from poc.models import (
    AgentInstance,
    ArtifactRecord,
    EventRecord,
    MessageEnvelope,
    MissionPlan,
    RunStatus,
    utc_now,
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    status TEXT NOT NULL,
    active_plan_version INTEGER NOT NULL,
    main_agent_id TEXT NOT NULL,
    cancellation_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS plans (
    run_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (run_id, version)
);
CREATE TABLE IF NOT EXISTS agents (
    agent_instance_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    parent_agent_id TEXT,
    tier TEXT NOT NULL,
    role_id TEXT NOT NULL,
    role_version INTEGER NOT NULL,
    plan_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS supervisor_state (
    agent_instance_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    state_version INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS agents_identity
ON agents(run_id, parent_agent_id, role_id, plan_version, agent_instance_id);
CREATE TABLE IF NOT EXISTS workflows (
    run_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    spec TEXT NOT NULL,
    result TEXT,
    thread_id TEXT NOT NULL,
    PRIMARY KEY(workflow_id, revision)
);
CREATE TABLE IF NOT EXISTS tasks (
    workflow_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    agent_instance_id TEXT,
    status TEXT NOT NULL,
    attempt_id TEXT,
    result TEXT,
    PRIMARY KEY(workflow_id, revision, task_id)
);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    delivered_at TEXT,
    consumed_at TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    run_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor_id TEXT,
    correlation_id TEXT,
    causation_id TEXT,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    event_id TEXT PRIMARY KEY,
    exported_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS command_outbox (
    command_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    supervisor_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    executed_at TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    media_type TEXT NOT NULL,
    path TEXT NOT NULL,
    producer_task_id TEXT,
    payload TEXT NOT NULL,
    UNIQUE(run_id, sha256)
);
CREATE TABLE IF NOT EXISTS operations (
    operation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS validations (
    request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    interrupt_id TEXT,
    status TEXT NOT NULL,
    request TEXT NOT NULL,
    resolution TEXT
);
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    owner_suborchestrator_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    goal_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    policy TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS executions_run ON executions(run_id, created_at);
CREATE TABLE IF NOT EXISTS swarm_execution_workers (
    execution_id TEXT NOT NULL,
    worker_instance_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(execution_id, worker_instance_id)
);
CREATE INDEX IF NOT EXISTS swarm_execution_workers_active
ON swarm_execution_workers(execution_id, role_id, status);
CREATE TABLE IF NOT EXISTS swarm_tasks (
    execution_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    required_role TEXT NOT NULL,
    task_spec_ref TEXT NOT NULL,
    acceptance_criteria_ref TEXT,
    dependency_ids TEXT NOT NULL,
    priority INTEGER NOT NULL,
    available_at TEXT NOT NULL,
    state TEXT NOT NULL,
    claimant_id TEXT,
    claim_token_hash TEXT,
    claim_generation INTEGER NOT NULL DEFAULT 0,
    lease_expires_at TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL,
    accepted_result_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(execution_id, task_id)
);
CREATE INDEX IF NOT EXISTS swarm_tasks_claimable
ON swarm_tasks(execution_id, state, required_role, priority, available_at);
CREATE TABLE IF NOT EXISTS swarm_task_attempts (
    attempt_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    ownership_generation INTEGER NOT NULL,
    worker_instance_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    result_ref TEXT,
    failure_ref TEXT
);
CREATE TABLE IF NOT EXISTS swarm_pools (
    pool_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL UNIQUE,
    role_quotas TEXT NOT NULL,
    max_slots INTEGER NOT NULL,
    arbitration_policy TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS swarm_pool_slots (
    slot_id TEXT PRIMARY KEY,
    pool_id TEXT NOT NULL,
    worker_instance_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_assigned_at TEXT
);
CREATE TABLE IF NOT EXISTS swarm_offers (
    offer_id TEXT PRIMARY KEY,
    pool_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    eligible_roles TEXT NOT NULL,
    priority INTEGER NOT NULL,
    bid_deadline TEXT NOT NULL,
    acceptance_criteria_ref TEXT,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS swarm_bids (
    bid_id TEXT PRIMARY KEY,
    offer_id TEXT NOT NULL,
    slot_id TEXT NOT NULL,
    eligibility INTEGER NOT NULL,
    availability INTEGER NOT NULL,
    fit_features TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(offer_id, slot_id)
);
CREATE TABLE IF NOT EXISTS swarm_assignments (
    assignment_id TEXT PRIMARY KEY,
    offer_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    slot_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    token_hash TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL,
    state TEXT NOT NULL,
    result_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
DROP INDEX IF EXISTS swarm_one_active_assignment;
CREATE UNIQUE INDEX IF NOT EXISTS swarm_one_active_offer_assignment
ON swarm_assignments(offer_id) WHERE state = 'active';
CREATE TABLE IF NOT EXISTS swarm_speculation_groups (
    group_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    logical_task_id TEXT NOT NULL,
    fanout_k INTEGER NOT NULL,
    authorized_roles TEXT NOT NULL,
    stop_policy TEXT NOT NULL,
    reconciliation_policy TEXT NOT NULL,
    effect_policy TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    final_result_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS swarm_candidates (
    candidate_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    worker_instance_id TEXT NOT NULL,
    role_id TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    result_ref TEXT,
    evidence_refs TEXT NOT NULL,
    validation_status TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS swarm_reconciliations (
    decision_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    group_revision INTEGER NOT NULL,
    considered_ids TEXT NOT NULL,
    accepted_ids TEXT NOT NULL,
    rejected_ids TEXT NOT NULL,
    reason TEXT NOT NULL,
    final_result_ref TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(group_id, group_revision)
);
"""


class Database:
    """Small synchronous SQLite unit-of-work.

    Each transaction is protected because the PoC deliberately runs one process and
    uses asyncio tasks around a shared connection.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                yield self.conn
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def create_run(
        self, run_id: str, objective: str, main_agent_id: str, plan: MissionPlan
    ) -> None:
        now = utc_now()
        with self.transaction() as tx:
            tx.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, 0, NULL, ?, ?)",
                (
                    run_id,
                    objective,
                    RunStatus.AWAITING_APPROVAL,
                    plan.version,
                    main_agent_id,
                    now,
                    now,
                ),
            )
            tx.execute(
                "INSERT INTO plans VALUES (?, ?, ?, ?)",
                (run_id, plan.version, plan.status, plan.model_dump_json()),
            )
            self._append_event(
                tx,
                EventRecord(
                    run_id=run_id,
                    event_type="plan.proposed",
                    actor_id=main_agent_id,
                    data={"plan_id": plan.plan_id, "plan_version": plan.version},
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            dict(row) for row in self.conn.execute("SELECT * FROM runs ORDER BY created_at DESC")
        ]

    def get_plan(self, run_id: str, version: int | None = None) -> MissionPlan | None:
        if version is None:
            row = self.conn.execute(
                "SELECT p.payload FROM plans p JOIN runs r ON r.run_id=p.run_id AND r.active_plan_version=p.version WHERE p.run_id=?",
                (run_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT payload FROM plans WHERE run_id=? AND version=?", (run_id, version)
            ).fetchone()
        return MissionPlan.model_validate_json(row[0]) if row else None

    def approve_plan(self, old_plan: MissionPlan, approved: MissionPlan) -> None:
        with self.transaction() as tx:
            tx.execute(
                "UPDATE plans SET status='superseded' WHERE run_id=? AND version=?",
                (old_plan.run_id, old_plan.version),
            )
            if approved.version == old_plan.version:
                tx.execute(
                    "UPDATE plans SET status='approved', payload=? WHERE run_id=? AND version=?",
                    (approved.model_dump_json(), approved.run_id, approved.version),
                )
            else:
                tx.execute(
                    "INSERT INTO plans VALUES (?, ?, 'approved', ?)",
                    (approved.run_id, approved.version, approved.model_dump_json()),
                )
            tx.execute(
                "UPDATE runs SET status=?, active_plan_version=?, updated_at=? WHERE run_id=?",
                (RunStatus.RUNNING, approved.version, utc_now(), approved.run_id),
            )
            self._append_event(
                tx,
                EventRecord(
                    run_id=approved.run_id,
                    event_type="plan.approved",
                    data={"plan_id": approved.plan_id, "plan_version": approved.version},
                ),
            )

    def update_run_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        with self.transaction() as tx:
            tx.execute(
                "UPDATE runs SET status=?, error=?, updated_at=? WHERE run_id=?",
                (status, error, utc_now(), run_id),
            )
            self._append_event(
                tx, EventRecord(run_id=run_id, event_type=f"run.{status}", data={"error": error})
            )

    def request_cancellation(self, run_id: str) -> bool:
        with self.transaction() as tx:
            row = tx.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                return False
            tx.execute(
                "UPDATE runs SET cancellation_requested=1, status=?, updated_at=? WHERE run_id=?",
                (RunStatus.CANCELLED, utc_now(), run_id),
            )
            self._append_event(
                tx,
                EventRecord(
                    run_id=run_id, event_type="workflow.cancelled", data={"authority_revoked": True}
                ),
            )
        return True

    def authority_active(self, run_id: str, plan_version: int) -> bool:
        row = self.conn.execute(
            "SELECT status, active_plan_version, cancellation_requested FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return bool(row and row[0] == RunStatus.RUNNING and row[1] == plan_version and not row[2])

    def put_agent(self, agent: AgentInstance) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent.agent_instance_id,
                    agent.run_id,
                    agent.parent_agent_id,
                    agent.tier,
                    agent.role_id,
                    agent.role_version,
                    agent.plan_version,
                    agent.status,
                    agent.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                if agent.tier != "worker":
                    tx.execute(
                        "INSERT OR IGNORE INTO supervisor_state VALUES (?, ?, 0, '{}', ?)",
                        (agent.agent_instance_id, agent.run_id, utc_now()),
                    )
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=agent.run_id,
                        event_type="agent.started",
                        actor_id=agent.agent_instance_id,
                        data={
                            "tier": agent.tier,
                            "role_id": agent.role_id,
                            "parent_agent_id": agent.parent_agent_id,
                        },
                    ),
                )

    def list_agents(self, run_id: str) -> list[dict[str, Any]]:
        return [
            AgentInstance.model_validate_json(row[0]).model_dump(mode="json")
            for row in self.conn.execute(
                "SELECT payload FROM agents WHERE run_id=? ORDER BY rowid", (run_id,)
            )
        ]

    def get_agent(self, agent_id: str) -> AgentInstance | None:
        row = self.conn.execute(
            "SELECT payload FROM agents WHERE agent_instance_id=?", (agent_id,)
        ).fetchone()
        return AgentInstance.model_validate_json(row[0]) if row else None

    def put_workflow(self, spec: Any, thread_id: str) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO workflows VALUES (?, ?, ?, ?, 'submitted', ?, NULL, ?)",
                (
                    spec.run_id,
                    spec.workflow_id,
                    spec.revision,
                    spec.owner,
                    spec.model_dump_json(),
                    thread_id,
                ),
            )
            for task in spec.tasks:
                tx.execute(
                    "INSERT OR IGNORE INTO tasks VALUES (?, ?, ?, NULL, 'pending', NULL, NULL)",
                    (spec.workflow_id, spec.revision, task.id),
                )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=spec.run_id,
                        event_type="workflow.submitted",
                        actor_id=spec.owner,
                        data={"workflow_id": spec.workflow_id, "revision": spec.revision},
                    ),
                )

    def get_workflow(self, workflow_id: str, revision: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM workflows WHERE workflow_id=? AND revision=?", (workflow_id, revision)
        ).fetchone()
        return dict(row) if row else None

    def update_workflow(
        self, workflow_id: str, revision: int, status: str, result: Any = None
    ) -> None:
        with self.transaction() as tx:
            row = tx.execute(
                "SELECT run_id,owner FROM workflows WHERE workflow_id=? AND revision=?",
                (workflow_id, revision),
            ).fetchone()
            tx.execute(
                "UPDATE workflows SET status=?, result=? WHERE workflow_id=? AND revision=?",
                (status, json.dumps(result) if result is not None else None, workflow_id, revision),
            )
            if row:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=row[0],
                        event_type=f"workflow.{status}",
                        actor_id=row[1],
                        data={"workflow_id": workflow_id, "revision": revision},
                    ),
                )

    def update_task(
        self,
        workflow_id: str,
        revision: int,
        task_id: str,
        status: str,
        agent_id: str | None = None,
        attempt_id: str | None = None,
        result: Any = None,
    ) -> None:
        with self.transaction() as tx:
            tx.execute(
                "UPDATE tasks SET status=?, agent_instance_id=COALESCE(?,agent_instance_id), "
                "attempt_id=COALESCE(?,attempt_id), result=COALESCE(?,result) WHERE workflow_id=? AND revision=? AND task_id=?",
                (
                    status,
                    agent_id,
                    attempt_id,
                    json.dumps(result) if result is not None else None,
                    workflow_id,
                    revision,
                    task_id,
                ),
            )

    def list_workflows(self, run_id: str) -> list[dict[str, Any]]:
        workflows: list[dict[str, Any]] = []
        for row in self.conn.execute(
            "SELECT * FROM workflows WHERE run_id=? ORDER BY rowid", (run_id,)
        ):
            item = dict(row)
            item["tasks"] = [
                dict(task)
                for task in self.conn.execute(
                    "SELECT task_id,agent_instance_id,status,attempt_id,result FROM tasks WHERE workflow_id=? AND revision=? ORDER BY rowid",
                    (row["workflow_id"], row["revision"]),
                )
            ]
            workflows.append(item)
        return workflows

    def put_message(self, message: MessageEnvelope) -> None:
        with self.transaction() as tx:
            tx.execute(
                "INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?, ?, ?)",
                (
                    message.message_id,
                    message.run_id,
                    message.recipient_id,
                    message.delivered_at,
                    message.consumed_at,
                    message.model_dump_json(),
                ),
            )
            self._append_event(
                tx,
                EventRecord(
                    run_id=message.run_id,
                    event_type="message.sent",
                    actor_id=message.sender_id,
                    correlation_id=message.correlation_id,
                    causation_id=message.causation_id,
                    data={
                        "message_id": message.message_id,
                        "recipient_id": message.recipient_id,
                        "message_type": message.message_type,
                        "task_id": message.task_id,
                        "attempt_id": message.attempt_id,
                        "assignment_id": message.assignment_id,
                        "payload_ref": message.payload_ref,
                    },
                ),
            )
            if message.delivered_at:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=message.run_id,
                        event_type="message.delivered",
                        actor_id=message.recipient_id,
                        data={"message_id": message.message_id},
                    ),
                )
            if message.consumed_at:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=message.run_id,
                        event_type="message.consumed",
                        actor_id=message.recipient_id,
                        data={"message_id": message.message_id},
                    ),
                )

    def supervisor_version(self, agent_id: str) -> int:
        row = self.conn.execute(
            "SELECT state_version FROM supervisor_state WHERE agent_instance_id=?", (agent_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def commit_supervisor_turn(
        self,
        *,
        agent_id: str,
        run_id: str,
        input_version: int,
        snapshot: dict[str, Any],
        accepted_commands: list[dict[str, Any]],
        event: EventRecord,
    ) -> int:
        """Atomically commit the snapshot, decision record, and idempotent command outbox."""
        with self.transaction() as tx:
            row = tx.execute(
                "SELECT state_version FROM supervisor_state WHERE agent_instance_id=?", (agent_id,)
            ).fetchone()
            current = int(row[0]) if row else 0
            if current != input_version:
                raise RuntimeError(
                    f"stale supervisor snapshot: expected {input_version}, found {current}"
                )
            resulting = current + 1
            tx.execute(
                "INSERT INTO supervisor_state VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(agent_instance_id) DO UPDATE SET state_version=excluded.state_version,"
                "snapshot=excluded.snapshot,updated_at=excluded.updated_at",
                (agent_id, run_id, resulting, json.dumps(snapshot), utc_now()),
            )
            for command in accepted_commands:
                tx.execute(
                    "INSERT OR IGNORE INTO command_outbox(command_id,run_id,supervisor_id,payload,created_at) "
                    "VALUES(?,?,?,?,?)",
                    (command["command_id"], run_id, agent_id, json.dumps(command), utc_now()),
                )
            event.data["resulting_state_version"] = resulting
            self._append_event(tx, event)
        return resulting

    def record_event(self, event: EventRecord) -> None:
        with self.transaction() as tx:
            self._append_event(tx, event)

    def append_event(self, tx: sqlite3.Connection, event: EventRecord) -> None:
        """Append an event inside a caller-owned transaction."""
        self._append_event(tx, event)

    def _append_event(self, tx: sqlite3.Connection, event: EventRecord) -> None:
        inserted = tx.execute(
            "INSERT OR IGNORE INTO events(event_id,run_id,event_type,actor_id,correlation_id,causation_id,data,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                event.event_id,
                event.run_id,
                event.event_type,
                event.actor_id,
                event.correlation_id,
                event.causation_id,
                json.dumps(event.data),
                event.created_at,
            ),
        )
        tx.execute("INSERT OR IGNORE INTO outbox(event_id) VALUES (?)", (event.event_id,))
        # Best-effort telemetry is intentionally downstream of the durable journal write.
        if inserted.rowcount and inserted.lastrowid is not None:
            from poc.telemetry.domain_spans import emit_event_span

            with suppress(Exception):
                emit_event_span(event, event_seq=inserted.lastrowid)

    def events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE run_id=? ORDER BY seq", (run_id,)
        ).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def put_artifact(self, artifact: ArtifactRecord) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.sha256,
                    artifact.media_type,
                    artifact.path,
                    artifact.producer_task_id,
                    artifact.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=artifact.run_id,
                        event_type="artifact.created",
                        data={
                            "artifact_id": artifact.artifact_id,
                            "sha256": artifact.sha256,
                            "producer_task_id": artifact.producer_task_id,
                        },
                    ),
                )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        row = self.conn.execute(
            "SELECT payload FROM artifacts WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        return ArtifactRecord.model_validate_json(row[0]) if row else None

    def list_artifacts(self, run_id: str) -> list[ArtifactRecord]:
        return [
            ArtifactRecord.model_validate_json(row[0])
            for row in self.conn.execute(
                "SELECT payload FROM artifacts WHERE run_id=? ORDER BY rowid", (run_id,)
            )
        ]

    def begin_operation(
        self, operation_id: str, run_id: str, task_id: str, kind: str, request: Any
    ) -> str:
        with self.transaction() as tx:
            existing = tx.execute(
                "SELECT status,response FROM operations WHERE operation_id=?", (operation_id,)
            ).fetchone()
            if existing:
                return existing[0]
            tx.execute(
                "INSERT INTO operations VALUES (?, ?, ?, ?, 'started', ?, NULL, ?, NULL)",
                (operation_id, run_id, task_id, kind, json.dumps(request), utc_now()),
            )
        return "started"

    def finish_operation(self, operation_id: str, status: str, response: Any) -> None:
        with self.transaction() as tx:
            tx.execute(
                "UPDATE operations SET status=?,response=?,finished_at=? WHERE operation_id=?",
                (status, json.dumps(response), utc_now(), operation_id),
            )

    def get_operation(self, operation_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM operations WHERE operation_id=?", (operation_id,)
        ).fetchone()
        return dict(row) if row else None
