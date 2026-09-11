from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poc.models import AgentInstance, EventRecord
from poc.persistence.database import Database
from poc.services.artifact_store import ArtifactStore
from poc.services.rbac_adapter import RBACAdapter


class ToolDenied(PermissionError):
    pass


class ToolGateway:
    """Only path from a worker decision to fixture I/O or artifact publication."""

    def __init__(self, db: Database, rbac: RBACAdapter, artifacts: ArtifactStore, fixture_root: str | Path):
        self.db = db
        self.rbac = rbac
        self.artifacts = artifacts
        self.fixture_root = Path(fixture_root)

    def execute(self, *, operation_id: str, actor: AgentInstance, task_id: str,
                tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        decision = self.rbac.authorize(actor, tool_name)
        event_data = {"operation_id": operation_id, "task_id": task_id, "tool": tool_name,
                      "role_id": actor.role_id, "reason": decision.reason}
        self.db.record_event(EventRecord(run_id=actor.run_id,
                                         event_type=f"tool.authorization_{'allowed' if decision.allowed else 'denied'}",
                                         actor_id=actor.agent_instance_id, data=event_data))
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
        self.db.record_event(EventRecord(run_id=actor.run_id, event_type="tool.executed",
                                         actor_id=actor.agent_instance_id,
                                         data={**event_data, "status": "succeeded"}))
        return result

    def _tool_read_metric_slice(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        start = _parse_ts(args["start"], args.get("assume_timezone"))
        end = _parse_ts(args["end"], args.get("assume_timezone"))
        rows = []
        with (self.fixture_root / "metrics.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                at = _parse_ts(row["timestamp"], None)
                if row["service"] == args.get("service", "checkout") and start <= at <= end:
                    rows.append({**row, "latency_ms": float(row["latency_ms"])})
        return {"rows": rows, "count": len(rows), "start": args["start"], "end": args["end"], "units": "ms"}

    def _tool_calculate_percentile(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        values = sorted(float(v) for v in args["values"])
        if not values:
            raise ValueError("cannot calculate a percentile from an empty sample")
        percentile = float(args.get("percentile", 95))
        rank = (percentile / 100) * (len(values) - 1)
        low, high = math.floor(rank), math.ceil(rank)
        value = values[low] if low == high else values[low] + (values[high] - values[low]) * (rank - low)
        return {"percentile": percentile, "value": round(value, 2), "units": args.get("units", "ms"),
                "sample_count": len(values), "method": "linear interpolation"}

    def _tool_read_manifest(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        return json.loads((self.fixture_root / "manifest.json").read_text())

    def _tool_read_log_slice(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        start, end = _parse_ts(args["start"], "UTC"), _parse_ts(args["end"], "UTC")
        rows = []
        for line in (self.fixture_root / "logs.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if start <= _parse_ts(row["timestamp"], None) <= end:
                rows.append(row)
        return {"rows": rows, "count": len(rows), "start": args["start"], "end": args["end"]}

    def _tool_count_log_pattern(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        rows = args["rows"]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["pattern"]] = counts.get(row["pattern"], 0) + 1
        return {"counts": counts, "total": len(rows),
                "interpretation": "cache_miss is incident-correlated; payment_timeout also appears in baseline and is not sufficient alone"}

    def _tool_read_deployment_record(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        deployments = json.loads((self.fixture_root / "deployments.json").read_text())
        incident_start = _parse_ts(args["incident_start"], "UTC")
        matched = []
        for deployment in deployments:
            delta = (incident_start - _parse_ts(deployment["timestamp"], None)).total_seconds() / 60
            if deployment["service"] == "checkout" and 0 <= delta <= args.get("lookback_minutes", 30):
                matched.append({**deployment, "minutes_before_incident": int(delta)})
        return {"matches": matched, "count": len(matched)}

    def _tool_write_artifact(self, run_id: str, task_id: str, args: dict[str, Any]) -> dict[str, Any]:
        record = self.artifacts.write(run_id, args["content"], producer_task_id=task_id,
                                      media_type=args.get("media_type", "application/json"))
        return {"artifact_id": record.artifact_id, "sha256": record.sha256, "media_type": record.media_type}


def _parse_ts(value: str, assume_timezone: str | None) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        if assume_timezone != "UTC":
            raise ValueError(f"timezone is ambiguous for {value!r}")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
