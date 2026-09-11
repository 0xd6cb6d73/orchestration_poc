from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poc.execution.board_claim import BoardClaimStrategy, Claim
from poc.execution.managed_pool import Assignment, ManagedPoolStrategy
from poc.execution.speculative import CandidateGrant, SpeculativeStrategy
from poc.models import AgentInstance, EventRecord
from poc.persistence.database import Database
from poc.services.artifact_store import ArtifactStore
from poc.services.rbac_adapter import RBACAdapter


class ToolDenied(PermissionError):
    pass


class ToolGateway:
    """Only path from a worker decision to fixture I/O or artifact publication."""

    def __init__(
        self, db: Database, rbac: RBACAdapter, artifacts: ArtifactStore, fixture_root: str | Path
    ):
        self.db = db
        self.rbac = rbac
        self.artifacts = artifacts
        self.fixture_root = Path(fixture_root)

    def execute(
        self,
        *,
        operation_id: str,
        actor: AgentInstance,
        task_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        grant: Claim | Assignment | CandidateGrant | None = None,
        effect_class: str = "read",
    ) -> dict[str, Any]:
        decision = self.rbac.authorize(actor, tool_name)
        event_data: dict[str, Any] = {
            "operation_id": operation_id,
            "task_id": task_id,
            "tool": tool_name,
            "role_id": actor.role_id,
            "reason": decision.reason,
        }
        ownership_error = self._validate_ownership(actor, task_id, grant, effect_class)
        if grant is not None:
            event_data.update(
                {
                    "execution_id": grant.execution_id,
                    "ownership_type": (
                        "claim"
                        if isinstance(grant, Claim)
                        else "assignment"
                        if isinstance(grant, Assignment)
                        else "speculation"
                    ),
                    "ownership_generation": grant.generation,
                    "ownership_token_fingerprint": grant.token_fingerprint,
                }
            )
        if decision.allowed and ownership_error:
            decision = type(decision)(False, ownership_error)
            event_data["reason"] = ownership_error
        self.db.record_event(
            EventRecord(
                run_id=actor.run_id,
                event_type=f"tool.authorization_{'allowed' if decision.allowed else 'denied'}",
                actor_id=actor.agent_instance_id,
                data=event_data,
            )
        )
        if not decision.allowed:
            raise ToolDenied(decision.reason)

        existing = self.db.get_operation(operation_id)
        if existing and existing["status"] == "succeeded":
            return json.loads(existing["response"])
        self.db.begin_operation(operation_id, actor.run_id, task_id, tool_name, arguments)
        try:
            result = getattr(self, f"_tool_{tool_name}")(actor.run_id, task_id, arguments)
        except BaseException as exc:
            self.db.finish_operation(operation_id, "failed", {"error": str(exc)})
            raise
        self.db.finish_operation(operation_id, "succeeded", result)
        self.db.record_event(
            EventRecord(
                run_id=actor.run_id,
                event_type="tool.executed",
                actor_id=actor.agent_instance_id,
                data={**event_data, "status": "succeeded"},
            )
        )
        return result

    def _validate_ownership(
        self,
        actor: AgentInstance,
        task_id: str,
        grant: Claim | Assignment | CandidateGrant | None,
        effect_class: str,
    ) -> str | None:
        # The legacy DAG strategy has explicit graph-node assignment and therefore
        # no lease credential. Swarm strategies always supply one of these opaque
        # runtime-held grants.
        if grant is None:
            membership = self.db.conn.execute(
                "SELECT 1 FROM swarm_execution_workers w JOIN executions e ON e.execution_id=w.execution_id "
                "WHERE w.worker_instance_id=? AND w.status='active' AND e.status='active' "
                "AND e.mode!='hierarchical_dag'",
                (actor.agent_instance_id,),
            ).fetchone()
            if membership:
                return "an active swarm execution requires a current ownership grant"
            return None
        if grant.execution_id not in {
            row[0]
            for row in self.db.conn.execute(
                "SELECT execution_id FROM executions WHERE run_id=? AND status='active'",
                (actor.run_id,),
            )
        }:
            return "execution is inactive or belongs to another run"
        if isinstance(grant, Claim):
            if not BoardClaimStrategy(self.db).validate_claim(
                grant, task_id=task_id, worker_id=actor.agent_instance_id
            ):
                return "claim is stale, forged, or scoped to another task"
            return None
        if isinstance(grant, Assignment):
            if not ManagedPoolStrategy(self.db).validate_assignment(
                grant, task_id=task_id, worker_id=actor.agent_instance_id
            ):
                return "assignment is stale, forged, or scoped to another task"
            return None
        allowed_effects = {
            "pure_only": {"pure"},
            "read_only": {"pure", "read", "artifact"},
            "staged_effects": {"pure", "read", "artifact", "stage"},
        }[grant.effect_policy.value]
        if effect_class not in allowed_effects:
            return "speculative candidates cannot perform direct external effects"
        if not SpeculativeStrategy(self.db).validate_candidate(
            grant, task_id=task_id, worker_id=actor.agent_instance_id
        ):
            return "candidate grant is stale, forged, or scoped to another task"
        return None

    def _tool_read_metric_slice(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        start = _parse_ts(args["start"], args.get("assume_timezone"))
        end = _parse_ts(args["end"], args.get("assume_timezone"))
        rows: list[dict[str, Any]] = []
        with (self.fixture_root / "metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                at = _parse_ts(row["timestamp"], None)
                if row["service"] == args.get("service", "checkout") and start <= at <= end:
                    rows.append({**row, "latency_ms": float(row["latency_ms"])})
        return {
            "rows": rows,
            "count": len(rows),
            "start": args["start"],
            "end": args["end"],
            "units": "ms",
        }

    def _tool_calculate_percentile(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        values = sorted(float(v) for v in args["values"])
        if not values:
            raise ValueError("cannot calculate a percentile from an empty sample")
        percentile = float(args.get("percentile", 95))
        rank = (percentile / 100) * (len(values) - 1)
        low, high = math.floor(rank), math.ceil(rank)
        value = (
            values[low]
            if low == high
            else values[low] + (values[high] - values[low]) * (rank - low)
        )
        return {
            "percentile": percentile,
            "value": round(value, 2),
            "units": args.get("units", "ms"),
            "sample_count": len(values),
            "method": "linear interpolation",
        }

    def _tool_read_manifest(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        return json.loads((self.fixture_root / "manifest.json").read_text())

    def _tool_read_log_slice(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        start, end = _parse_ts(args["start"], "UTC"), _parse_ts(args["end"], "UTC")
        rows: list[dict[str, Any]] = []
        for line in (self.fixture_root / "logs.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if start <= _parse_ts(row["timestamp"], None) <= end:
                rows.append(row)
        return {"rows": rows, "count": len(rows), "start": args["start"], "end": args["end"]}

    def _tool_count_log_pattern(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        rows = args["rows"]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["pattern"]] = counts.get(row["pattern"], 0) + 1
        return {
            "counts": counts,
            "total": len(rows),
            "interpretation": "cache_miss is incident-correlated; payment_timeout also appears in baseline and is not sufficient alone",
        }

    def _tool_read_deployment_record(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        deployments = json.loads((self.fixture_root / "deployments.json").read_text())
        incident_start = _parse_ts(args["incident_start"], "UTC")
        matched: list[dict[str, Any]] = []
        for deployment in deployments:
            delta = (incident_start - _parse_ts(deployment["timestamp"], None)).total_seconds() / 60
            if deployment["service"] == "checkout" and 0 <= delta <= args.get(
                "lookback_minutes", 30
            ):
                matched.append({**deployment, "minutes_before_incident": int(delta)})
        return {"matches": matched, "count": len(matched)}

    def _tool_write_artifact(
        self, run_id: str, task_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        record = self.artifacts.write(
            run_id,
            args["content"],
            producer_task_id=task_id,
            media_type=args.get("media_type", "application/json"),
        )
        return {
            "artifact_id": record.artifact_id,
            "sha256": record.sha256,
            "media_type": record.media_type,
        }


def _parse_ts(value: str, assume_timezone: str | None) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        if assume_timezone != "UTC":
            raise ValueError(f"timezone is ambiguous for {value!r}")
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
