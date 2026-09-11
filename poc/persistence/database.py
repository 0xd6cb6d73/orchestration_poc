from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from poc.blackboard.models import BlackboardRecord
from poc.hybrid.contracts import (
    AcceptanceDecision,
    Candidate,
    CandidateCluster,
    CapabilityProfile,
    Challenge,
    CollaborationRound,
    ContextManifest,
    DeliveryDecision,
    GoalContract,
    ProvenanceRecord,
    RouteDecision,
    TeamSpec,
    TestResult,
    VerificationRequest,
    VerificationVerdict,
)
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
CREATE TABLE IF NOT EXISTS goal_contracts (
    goal_contract_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(goal_contract_id, version)
);
CREATE TABLE IF NOT EXISTS context_manifests (
    context_manifest_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    agent_instance_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(context_manifest_id, version)
);
CREATE TABLE IF NOT EXISTS provenance_records (
    provenance_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE(run_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS blackboard_records (
    record_id TEXT NOT NULL,
    record_version INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    epistemic_status TEXT NOT NULL,
    visibility_policy TEXT NOT NULL,
    visibility_ref TEXT,
    payload TEXT NOT NULL,
    PRIMARY KEY(record_id, record_version)
);
CREATE TABLE IF NOT EXISTS team_specs (
    team_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(team_id, version)
);
CREATE TABLE IF NOT EXISTS capability_profiles (
    profile_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(profile_id, version)
);
CREATE TABLE IF NOT EXISTS route_decisions (
    route_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    offer_generation INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collaboration_rounds (
    round_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hybrid_candidates (
    candidate_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    round_id TEXT NOT NULL,
    visibility TEXT NOT NULL,
    state TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(candidate_id, version)
);
CREATE TABLE IF NOT EXISTS hybrid_candidate_clusters (
    cluster_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    round_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(cluster_id, version)
);
CREATE TABLE IF NOT EXISTS hybrid_challenges (
    challenge_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hybrid_test_results (
    test_result_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verification_requests (
    verification_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    subject_artifact_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verification_verdicts (
    verdict_id TEXT PRIMARY KEY,
    verification_id TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS acceptance_decisions (
    acceptance_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    subject_artifact_id TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delivery_decisions (
    delivery_decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    permitted INTEGER NOT NULL,
    payload TEXT NOT NULL
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
                            "agent_backend": agent.agent_backend,
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

    def update_artifact_access(
        self,
        artifact_id: str,
        *,
        visibility: str,
        visibility_ref: str | None,
        access_labels: frozenset[str] = frozenset(),
    ) -> ArtifactRecord:
        current = self.get_artifact(artifact_id)
        if current is None:
            raise KeyError(artifact_id)
        updated = current.model_copy(
            update={
                "visibility": visibility,
                "visibility_ref": visibility_ref,
                "access_labels": access_labels,
            }
        )
        with self.transaction() as tx:
            tx.execute(
                "UPDATE artifacts SET payload=? WHERE artifact_id=?",
                (updated.model_dump_json(), artifact_id),
            )
            self._append_event(
                tx,
                EventRecord(
                    run_id=updated.run_id,
                    event_type="artifact.visibility_changed",
                    data={
                        "artifact_id": artifact_id,
                        "visibility": visibility,
                        "visibility_ref": visibility_ref,
                    },
                ),
            )
        return updated

    def put_goal_contract(self, contract: GoalContract) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO goal_contracts VALUES (?,?,?,?)",
                (
                    contract.goal_contract_id,
                    contract.version,
                    contract.run_id,
                    contract.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=contract.run_id,
                        event_type="goal_contract.created",
                        data={
                            "goal_contract_id": contract.goal_contract_id,
                            "goal_contract_version": contract.version,
                            "plan_version": contract.plan_version,
                        },
                    ),
                )

    def put_context_manifest(self, manifest: ContextManifest) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO context_manifests VALUES (?,?,?,?,?,?)",
                (
                    manifest.context_manifest_id,
                    manifest.version,
                    manifest.run_id,
                    manifest.agent_instance_id,
                    manifest.task_id,
                    manifest.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=manifest.run_id,
                        event_type="context.assembled",
                        actor_id=manifest.agent_instance_id,
                        data={
                            "context_manifest_id": manifest.context_manifest_id,
                            "goal_contract_id": manifest.goal_contract_id,
                            "task_id": manifest.task_id,
                            "attempt_id": manifest.attempt_id,
                            "available_artifacts": len(manifest.available_artifact_refs),
                            "retrieved_artifacts": len(manifest.retrieved_artifact_refs),
                            "included_artifacts": len(manifest.included_artifact_refs),
                            "context_policy_version": manifest.context_policy_version,
                        },
                    ),
                )

    def put_provenance(self, record: ProvenanceRecord) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO provenance_records VALUES (?,?,?,?)",
                (
                    record.provenance_id,
                    record.run_id,
                    record.artifact_id,
                    record.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=record.run_id,
                        event_type="provenance.recorded",
                        actor_id=record.producer_agent_id,
                        data={
                            "provenance_id": record.provenance_id,
                            "artifact_id": record.artifact_id,
                            "context_manifest_id": record.context_manifest_id,
                            "task_id": record.task_id,
                            "attempt_id": record.attempt_id,
                        },
                    ),
                )

    def get_provenance(self, artifact_id: str) -> ProvenanceRecord | None:
        row = self.conn.execute(
            "SELECT payload FROM provenance_records WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
        return ProvenanceRecord.model_validate_json(row[0]) if row else None

    def put_blackboard_record(self, record: BlackboardRecord, *, event_type: str) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO blackboard_records VALUES (?,?,?,?,?,?,?,?)",
                (
                    record.record_id,
                    record.record_version,
                    record.run_id,
                    record.domain_id,
                    record.epistemic_status,
                    record.visibility_policy,
                    record.visibility_ref,
                    record.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=record.run_id,
                        event_type=event_type,
                        actor_id=record.author_agent_id,
                        data={
                            "blackboard_record_id": record.record_id,
                            "record_version": record.record_version,
                            "domain_id": record.domain_id,
                            "epistemic_status": record.epistemic_status,
                            "visibility": record.visibility_policy,
                        },
                    ),
                )

    def latest_blackboard_record(self, record_id: str) -> BlackboardRecord | None:
        row = self.conn.execute(
            "SELECT payload FROM blackboard_records WHERE record_id=? "
            "ORDER BY record_version DESC LIMIT 1",
            (record_id,),
        ).fetchone()
        return BlackboardRecord.model_validate_json(row[0]) if row else None

    def list_blackboard_records(self, run_id: str) -> list[BlackboardRecord]:
        rows = self.conn.execute(
            "SELECT b.payload FROM blackboard_records b "
            "WHERE b.run_id=? AND b.record_version=(SELECT max(b2.record_version) "
            "FROM blackboard_records b2 WHERE b2.record_id=b.record_id) ORDER BY b.rowid",
            (run_id,),
        ).fetchall()
        return [BlackboardRecord.model_validate_json(row[0]) for row in rows]

    def put_team(self, team: TeamSpec) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO team_specs VALUES (?,?,?,?)",
                (team.team_id, team.version, team.run_id, team.model_dump_json()),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=team.run_id,
                        event_type="team.registered",
                        actor_id=team.steward_agent_id,
                        data={
                            "team_id": team.team_id,
                            "domain_id": team.domain_id,
                            "member_count": len(team.member_agent_ids),
                            "message_budget": team.message_budget,
                            "communication_policy_version": (team.communication_policy_version),
                        },
                    ),
                )

    def get_team(self, team_id: str) -> TeamSpec | None:
        row = self.conn.execute(
            "SELECT payload FROM team_specs WHERE team_id=? ORDER BY version DESC LIMIT 1",
            (team_id,),
        ).fetchone()
        return TeamSpec.model_validate_json(row[0]) if row else None

    def put_capability_profile(self, profile: CapabilityProfile) -> None:
        with self.transaction() as tx:
            tx.execute(
                "INSERT OR REPLACE INTO capability_profiles VALUES (?,?,?)",
                (profile.profile_id, profile.version, profile.model_dump_json()),
            )

    def list_capability_profiles(self) -> list[CapabilityProfile]:
        return [
            CapabilityProfile.model_validate_json(row[0])
            for row in self.conn.execute("SELECT payload FROM capability_profiles ORDER BY rowid")
        ]

    def put_route_decision(self, decision: RouteDecision) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO route_decisions VALUES (?,?,?,?,?)",
                (
                    decision.route_decision_id,
                    decision.run_id,
                    decision.task_id,
                    decision.offer_generation,
                    decision.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=decision.run_id,
                        event_type="allocation.decided",
                        data={
                            "route_decision_id": decision.route_decision_id,
                            "task_id": decision.task_id,
                            "selected_profile_id": decision.selected_profile_id,
                            "offer_generation": decision.offer_generation,
                            "policy_version": decision.policy_version,
                            "reasons": list(decision.reasons),
                        },
                    ),
                )

    def put_collaboration_round(self, round_: CollaborationRound) -> None:
        with self.transaction() as tx:
            previous = tx.execute(
                "SELECT phase FROM collaboration_rounds WHERE round_id=?", (round_.round_id,)
            ).fetchone()
            tx.execute(
                "INSERT INTO collaboration_rounds VALUES (?,?,?,?,?) "
                "ON CONFLICT(round_id) DO UPDATE SET phase=excluded.phase,payload=excluded.payload",
                (
                    round_.round_id,
                    round_.run_id,
                    round_.domain_id,
                    round_.phase,
                    round_.model_dump_json(),
                ),
            )
            if previous is None or previous[0] != round_.phase:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=round_.run_id,
                        event_type="collaboration.transitioned",
                        actor_id=round_.steward_agent_id,
                        data={
                            "round_id": round_.round_id,
                            "team_id": round_.team_id,
                            "from_phase": previous[0] if previous else None,
                            "phase": round_.phase,
                            "policy_version": round_.policy_version,
                        },
                    ),
                )

    def get_collaboration_round(self, round_id: str) -> CollaborationRound | None:
        row = self.conn.execute(
            "SELECT payload FROM collaboration_rounds WHERE round_id=?", (round_id,)
        ).fetchone()
        return CollaborationRound.model_validate_json(row[0]) if row else None

    def list_collaboration_rounds(self, run_id: str) -> list[CollaborationRound]:
        return [
            CollaborationRound.model_validate_json(row[0])
            for row in self.conn.execute(
                "SELECT payload FROM collaboration_rounds WHERE run_id=? ORDER BY rowid", (run_id,)
            )
        ]

    def put_candidate(self, candidate: Candidate, *, event_type: str) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO hybrid_candidates VALUES (?,?,?,?,?,?)",
                (
                    candidate.candidate_id,
                    candidate.version,
                    candidate.round_id,
                    candidate.visibility,
                    candidate.state,
                    candidate.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                round_row = tx.execute(
                    "SELECT run_id FROM collaboration_rounds WHERE round_id=?",
                    (candidate.round_id,),
                ).fetchone()
                if round_row:
                    self._append_event(
                        tx,
                        EventRecord(
                            run_id=round_row[0],
                            event_type=event_type,
                            actor_id=candidate.author_agent_id,
                            data={
                                "round_id": candidate.round_id,
                                "candidate_id": candidate.candidate_id,
                                "candidate_version": candidate.version,
                                "artifact_id": candidate.artifact_id,
                                "visibility": candidate.visibility,
                                "state": candidate.state,
                                "hypothesis_key": candidate.hypothesis_key,
                            },
                        ),
                    )

    def latest_candidate(self, candidate_id: str) -> Candidate | None:
        row = self.conn.execute(
            "SELECT payload FROM hybrid_candidates WHERE candidate_id=? "
            "ORDER BY version DESC LIMIT 1",
            (candidate_id,),
        ).fetchone()
        return Candidate.model_validate_json(row[0]) if row else None

    def put_candidate_cluster(self, cluster: CandidateCluster) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO hybrid_candidate_clusters VALUES (?,?,?,?)",
                (
                    cluster.cluster_id,
                    cluster.version,
                    cluster.round_id,
                    cluster.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                round_row = tx.execute(
                    "SELECT run_id FROM collaboration_rounds WHERE round_id=?",
                    (cluster.round_id,),
                ).fetchone()
                if round_row:
                    self._append_event(
                        tx,
                        EventRecord(
                            run_id=round_row[0],
                            event_type="candidate.clustered",
                            data={
                                "round_id": cluster.round_id,
                                "cluster_id": cluster.cluster_id,
                                "cluster_version": cluster.version,
                                "candidate_ids": list(cluster.candidate_ids),
                                "policy_version": cluster.assignment_policy_version,
                                "rationale": cluster.rationale,
                            },
                        ),
                    )

    def list_candidate_clusters(self, run_id: str) -> list[CandidateCluster]:
        rows = self.conn.execute(
            "SELECT c.payload FROM hybrid_candidate_clusters c JOIN collaboration_rounds r "
            "ON r.round_id=c.round_id WHERE r.run_id=? AND c.version=(SELECT max(c2.version) "
            "FROM hybrid_candidate_clusters c2 WHERE c2.cluster_id=c.cluster_id) ORDER BY c.rowid",
            (run_id,),
        ).fetchall()
        return [CandidateCluster.model_validate_json(row[0]) for row in rows]

    def list_candidates(self, run_id: str) -> list[Candidate]:
        rows = self.conn.execute(
            "SELECT c.payload FROM hybrid_candidates c JOIN collaboration_rounds r "
            "ON r.round_id=c.round_id WHERE r.run_id=? AND c.version=(SELECT max(c2.version) "
            "FROM hybrid_candidates c2 WHERE c2.candidate_id=c.candidate_id) ORDER BY c.rowid",
            (run_id,),
        ).fetchall()
        return [Candidate.model_validate_json(row[0]) for row in rows]

    def put_challenge(self, challenge: Challenge) -> None:
        with self.transaction() as tx:
            tx.execute(
                "INSERT OR IGNORE INTO hybrid_challenges VALUES (?,?,?,?)",
                (
                    challenge.challenge_id,
                    challenge.round_id,
                    challenge.candidate_id,
                    challenge.model_dump_json(),
                ),
            )

    def put_test_result(self, result: TestResult) -> None:
        with self.transaction() as tx:
            tx.execute(
                "INSERT OR IGNORE INTO hybrid_test_results VALUES (?,?,?,?)",
                (
                    result.test_result_id,
                    result.round_id,
                    result.candidate_id,
                    result.model_dump_json(),
                ),
            )

    def put_verification_request(self, request: VerificationRequest) -> None:
        with self.transaction() as tx:
            tx.execute(
                "INSERT OR IGNORE INTO verification_requests VALUES (?,?,?,?)",
                (
                    request.verification_id,
                    request.run_id,
                    request.subject_artifact_id,
                    request.model_dump_json(),
                ),
            )

    def get_verification_request(self, verification_id: str) -> VerificationRequest | None:
        row = self.conn.execute(
            "SELECT payload FROM verification_requests WHERE verification_id=?",
            (verification_id,),
        ).fetchone()
        return VerificationRequest.model_validate_json(row[0]) if row else None

    def put_verification_verdict(self, run_id: str, verdict: VerificationVerdict) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO verification_verdicts VALUES (?,?,?)",
                (verdict.verdict_id, verdict.verification_id, verdict.model_dump_json()),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=run_id,
                        event_type="verification.completed",
                        actor_id=verdict.verifier_agent_id,
                        data={
                            "verification_id": verdict.verification_id,
                            "verdict_id": verdict.verdict_id,
                            "verdict": verdict.verdict,
                            "checks_performed": list(verdict.checks_performed),
                        },
                    ),
                )

    def get_verification_verdict(self, verdict_id: str) -> VerificationVerdict | None:
        row = self.conn.execute(
            "SELECT payload FROM verification_verdicts WHERE verdict_id=?", (verdict_id,)
        ).fetchone()
        return VerificationVerdict.model_validate_json(row[0]) if row else None

    def put_acceptance_decision(self, decision: AcceptanceDecision) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO acceptance_decisions VALUES (?,?,?,?,?)",
                (
                    decision.acceptance_decision_id,
                    decision.run_id,
                    decision.subject_artifact_id,
                    decision.accepted,
                    decision.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=decision.run_id,
                        event_type="acceptance.decided",
                        data={
                            "acceptance_decision_id": decision.acceptance_decision_id,
                            "artifact_id": decision.subject_artifact_id,
                            "accepted": decision.accepted,
                            "verdict_refs": list(decision.verdict_refs),
                            "acceptance_policy_version": decision.acceptance_policy_version,
                        },
                    ),
                )

    def accepted_artifact(self, run_id: str, artifact_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM acceptance_decisions WHERE run_id=? AND subject_artifact_id=? "
            "AND accepted=1 LIMIT 1",
            (run_id, artifact_id),
        ).fetchone()
        return row is not None

    def get_acceptance_decision(self, decision_id: str) -> AcceptanceDecision | None:
        row = self.conn.execute(
            "SELECT payload FROM acceptance_decisions WHERE acceptance_decision_id=?",
            (decision_id,),
        ).fetchone()
        return AcceptanceDecision.model_validate_json(row[0]) if row else None

    def delivery_permitted(self, run_id: str, artifact_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM delivery_decisions WHERE run_id=? AND artifact_id=? "
            "AND permitted=1 LIMIT 1",
            (run_id, artifact_id),
        ).fetchone()
        return row is not None

    def put_delivery_decision(self, decision: DeliveryDecision) -> None:
        with self.transaction() as tx:
            inserted = tx.execute(
                "INSERT OR IGNORE INTO delivery_decisions VALUES (?,?,?,?,?)",
                (
                    decision.delivery_decision_id,
                    decision.run_id,
                    decision.artifact_id,
                    decision.permitted,
                    decision.model_dump_json(),
                ),
            )
            if inserted.rowcount:
                self._append_event(
                    tx,
                    EventRecord(
                        run_id=decision.run_id,
                        event_type="delivery.gated",
                        data={
                            "delivery_decision_id": decision.delivery_decision_id,
                            "artifact_id": decision.artifact_id,
                            "plan_version": decision.plan_version,
                            "permitted": decision.permitted,
                            "checks": decision.checks,
                            "reasons": list(decision.reasons),
                        },
                    ),
                )

    def hybrid_status(self, run_id: str) -> dict[str, Any]:
        def payloads(query: str, parameters: tuple[str, ...]) -> list[dict[str, Any]]:
            return [json.loads(row[0]) for row in self.conn.execute(query, parameters)]

        return {
            "blackboard": [
                item.model_dump(mode="json") for item in self.list_blackboard_records(run_id)
            ],
            "collaboration_rounds": [
                item.model_dump(mode="json") for item in self.list_collaboration_rounds(run_id)
            ],
            "candidates": [item.model_dump(mode="json") for item in self.list_candidates(run_id)],
            "candidate_clusters": [
                item.model_dump(mode="json") for item in self.list_candidate_clusters(run_id)
            ],
            "route_decisions": payloads(
                "SELECT payload FROM route_decisions WHERE run_id=? ORDER BY rowid", (run_id,)
            ),
            "verifications": payloads(
                "SELECT v.payload FROM verification_verdicts v JOIN verification_requests r "
                "ON r.verification_id=v.verification_id WHERE r.run_id=? ORDER BY v.rowid",
                (run_id,),
            ),
            "acceptance_decisions": payloads(
                "SELECT payload FROM acceptance_decisions WHERE run_id=? ORDER BY rowid", (run_id,)
            ),
            "delivery_decisions": payloads(
                "SELECT payload FROM delivery_decisions WHERE run_id=? ORDER BY rowid", (run_id,)
            ),
        }

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
