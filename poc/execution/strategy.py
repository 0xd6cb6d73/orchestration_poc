from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from poc.models import (
    EventRecord,
    ExecutionHandle,
    ExecutionMode,
    ExecutionPolicy,
    Tier,
    new_id,
    utc_now,
)
from poc.persistence.database import Database


class StrategyError(RuntimeError):
    pass


class StrategyNotRegistered(StrategyError):
    pass


@runtime_checkable
class ExecutionStrategy(Protocol):
    mode: ExecutionMode

    async def submit(
        self,
        *,
        run_id: str,
        owner_suborchestrator_id: str,
        goal_ref: str,
        policy: ExecutionPolicy,
    ) -> ExecutionHandle: ...

    async def cancel(self, handle: ExecutionHandle) -> None: ...

    async def snapshot(self, handle: ExecutionHandle) -> dict[str, Any]: ...


StrategyFactory = Callable[[Database], ExecutionStrategy]


class StrategyRegistry:
    """Maps stable mode identifiers to factories.

    Factories, rather than a conditional in the runtime, make third-party or future
    schedulers installable without modifying the coordinator.
    """

    def __init__(self, db: Database):
        self.db = db
        self._factories: dict[ExecutionMode, StrategyFactory] = {}
        self._instances: dict[ExecutionMode, ExecutionStrategy] = {}

    def register(self, mode: ExecutionMode | str, factory: StrategyFactory, *, replace: bool = False) -> None:
        normalized = ExecutionMode(mode)
        if normalized in self._factories and not replace:
            raise ValueError(f"strategy already registered for {normalized.value!r}")
        self._factories[normalized] = factory
        self._instances.pop(normalized, None)

    def get(self, mode: ExecutionMode | str) -> ExecutionStrategy:
        normalized = ExecutionMode(mode)
        if normalized not in self._factories:
            raise StrategyNotRegistered(f"no execution strategy registered for {normalized.value!r}")
        if normalized not in self._instances:
            strategy = self._factories[normalized](self.db)
            if strategy.mode != normalized:
                raise ValueError(
                    f"strategy reports mode {strategy.mode.value!r}, expected {normalized.value!r}"
                )
            self._instances[normalized] = strategy
        return self._instances[normalized]

    @property
    def modes(self) -> frozenset[ExecutionMode]:
        return frozenset(self._factories)


class ExecutionCoordinator:
    def __init__(self, registry: StrategyRegistry):
        self.registry = registry

    async def submit(
        self,
        *,
        run_id: str,
        owner_suborchestrator_id: str,
        goal_ref: str,
        policy: ExecutionPolicy,
    ) -> ExecutionHandle:
        return await self.registry.get(policy.mode).submit(
            run_id=run_id,
            owner_suborchestrator_id=owner_suborchestrator_id,
            goal_ref=goal_ref,
            policy=policy,
        )

    async def cancel(self, handle: ExecutionHandle) -> None:
        await self.registry.get(handle.mode).cancel(handle)

    async def snapshot(self, handle: ExecutionHandle) -> dict[str, Any]:
        return await self.registry.get(handle.mode).snapshot(handle)


class PersistentExecutionStrategy:
    """Shared durable lifecycle; subclasses add only scheduling semantics."""

    mode: ExecutionMode

    def __init__(self, db: Database):
        self.db = db

    async def submit(
        self,
        *,
        run_id: str,
        owner_suborchestrator_id: str,
        goal_ref: str,
        policy: ExecutionPolicy,
    ) -> ExecutionHandle:
        if policy.mode != self.mode:
            raise StrategyError(
                f"policy mode {policy.mode.value!r} cannot be submitted to {self.mode.value!r}"
            )
        self._validate_policy(policy)
        owner = self.db.get_agent(owner_suborchestrator_id)
        if owner is None or owner.run_id != run_id or owner.tier != Tier.SUB:
            raise StrategyError("execution owner must be a sub-orchestrator in the same run")
        if not self.db.authority_active(run_id, owner.plan_version):
            raise StrategyError("plan authority is inactive or superseded")
        plan = self.db.get_plan(run_id)
        if plan is None:
            raise StrategyError("approved plan was not found")
        area = next((area for area in plan.areas if area.owner_role == owner.role_id), None)
        if area is not None and policy.mode not in area.allowed_execution_modes:
            raise StrategyError(f"mode {policy.mode.value!r} is not authorized for this plan area")
        if policy.max_workers > plan.budgets.get("max_workers", policy.max_workers):
            raise StrategyError("execution exceeds the plan worker budget")

        existing = self.db.conn.execute(
            "SELECT * FROM executions WHERE run_id=? AND owner_suborchestrator_id=? "
            "AND mode=? AND goal_ref=? AND status='active' ORDER BY created_at DESC LIMIT 1",
            (run_id, owner_suborchestrator_id, self.mode, goal_ref),
        ).fetchone()
        if existing is not None:
            stored_policy = ExecutionPolicy.model_validate_json(existing["policy"])
            if stored_policy != policy:
                raise StrategyError("goal_ref is already registered with a different execution policy")
            return ExecutionHandle(
                execution_id=existing["execution_id"],
                run_id=run_id,
                owner_suborchestrator_id=owner_suborchestrator_id,
                mode=self.mode,
            )

        execution_id = new_id("exec")
        now = utc_now()
        policy_payload = policy.model_dump(mode="json")
        policy_payload["allowed_roles"] = sorted(policy.allowed_roles)
        policy_hash = hashlib.sha256(
            json.dumps(policy_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self.db.transaction() as tx:
            tx.execute(
                "INSERT INTO executions VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                (
                    execution_id,
                    run_id,
                    owner_suborchestrator_id,
                    self.mode,
                    goal_ref,
                    policy.model_dump_json(),
                    now,
                    now,
                ),
            )
            self.db._append_event(
                tx,
                EventRecord(
                    run_id=run_id,
                    event_type="plan.execution_mode_selected",
                    actor_id=owner_suborchestrator_id,
                    data={
                        "execution_id": execution_id,
                        "plan_id": plan.plan_id,
                        "plan_version": owner.plan_version,
                        "mode": self.mode,
                        "policy_hash": policy_hash,
                        "tool_policy_id": policy.tool_policy_id,
                        "budget_id": policy.budget_id,
                        "max_workers": policy.max_workers,
                        "allowed_roles": sorted(policy.allowed_roles),
                        "speculative_fanout": policy.speculative_fanout,
                    },
                ),
            )
        handle = ExecutionHandle(
            execution_id=execution_id,
            run_id=run_id,
            owner_suborchestrator_id=owner_suborchestrator_id,
            mode=self.mode,
        )
        await self._after_submit(handle, policy)
        return handle

    async def _after_submit(self, handle: ExecutionHandle, policy: ExecutionPolicy) -> None:
        return None

    def _validate_policy(self, policy: ExecutionPolicy) -> None:
        """Validate mode-specific options before any durable state is written."""
        return None

    async def cancel(self, handle: ExecutionHandle) -> None:
        self._require_handle(handle)
        with self.db.transaction() as tx:
            changed = tx.execute(
                "UPDATE executions SET status='cancelled',updated_at=? "
                "WHERE execution_id=? AND status='active'",
                (utc_now(), handle.execution_id),
            )
            if changed.rowcount:
                self.db._append_event(
                    tx,
                    EventRecord(
                        run_id=handle.run_id,
                        event_type="execution.cancelled",
                        actor_id=handle.owner_suborchestrator_id,
                        data={"execution_id": handle.execution_id, "mode": handle.mode},
                    ),
                )

    async def snapshot(self, handle: ExecutionHandle) -> dict[str, Any]:
        row = self._require_handle(handle)
        result = dict(row)
        result["policy"] = json.loads(result["policy"])
        result.update(self._mode_snapshot(handle.execution_id))
        return result

    def _mode_snapshot(self, execution_id: str) -> dict[str, Any]:
        return {}

    def _require_handle(self, handle: ExecutionHandle):
        if handle.mode != self.mode:
            raise StrategyError("execution handle belongs to another strategy")
        row = self.db.conn.execute(
            "SELECT * FROM executions WHERE execution_id=?", (handle.execution_id,)
        ).fetchone()
        if (
            row is None
            or row["run_id"] != handle.run_id
            or row["owner_suborchestrator_id"] != handle.owner_suborchestrator_id
        ):
            raise StrategyError("unknown or mismatched execution handle")
        return row

    def _active_execution(self, execution_id: str):
        row = self.db.conn.execute(
            "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
        ).fetchone()
        if row is None or row["mode"] != self.mode or row["status"] != "active":
            raise StrategyError("execution is not active for this strategy")
        policy = ExecutionPolicy.model_validate_json(row["policy"])
        if not self.db.authority_active(row["run_id"], self.db.get_agent(row["owner_suborchestrator_id"]).plan_version):
            raise StrategyError("plan authority is inactive or superseded")
        return row, policy

    def _register_worker(self, execution_id: str, worker_id: str, role_id: str) -> None:
        now = utc_now()
        self.db.conn.execute(
            "INSERT INTO swarm_execution_workers VALUES (?,?,?,'active',?,?) "
            "ON CONFLICT(execution_id,worker_instance_id) DO UPDATE SET "
            "role_id=excluded.role_id,status='active',updated_at=excluded.updated_at",
            (execution_id, worker_id, role_id, now, now),
        )
