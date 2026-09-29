"""Versioned public shape contracts. This module never checks task feasibility."""

from __future__ import annotations

from typing import Any, cast

from pydantic_ai.exceptions import UnexpectedModelBehavior

from poc.execution.sql_contracts import Answer, TaskInput


class ArtifactContractError(UnexpectedModelBehavior, ValueError):
    pass


def validate_artifact(
    task: TaskInput, answer: Answer, policy: str = "public-v1", *, allow_partial: bool = False
) -> Answer:
    if policy == "legacy-v1":
        return answer
    values, tables = answer.values, task.tables
    if not values:
        raise ArtifactContractError("public-v1: empty task artifact")
    table, key, kind = "", "id", "integer"
    if "schedule_jobs" in tables:
        table = "schedule_jobs"
    elif "stops" in tables and "vehicles" in tables:
        table, kind = "stops", "route"
    elif "invoices" in tables:
        table, key = "invoices", "invoice"
    elif "jobs" in tables and "dependencies" in tables:
        table = "jobs"
    elif "queries" in tables and "policies" in tables:
        table, kind = "queries", "decision"
    # Unknown public schemas have only a nonempty mapping contract. No ID guessing.
    if not table:
        return answer
    required = {row[key] for row in tables[table]}
    if (not set(values) <= required) or (not allow_partial and set(values) != required):
        raise ArtifactContractError(
            f"public-v1: identifier coverage mismatch; missing={sorted(required - set(values))}, "
            f"unexpected={sorted(set(values) - required)}"
        )
    for name, value in values.items():
        if kind == "integer":
            valid = type(value) is int
        elif kind == "decision":
            valid = type(value) is str and value in ("allow", "deny")
        else:
            valid = (
                isinstance(value, dict)
                and set(cast(dict[str, Any], value)) == {"vehicle", "position"}
                and type(cast(dict[str, Any], value)["vehicle"]) is int
                and type(cast(dict[str, Any], value)["position"]) is int
                and value["vehicle"] in {row["id"] for row in tables["vehicles"]}
            )
        if not valid:
            raise ArtifactContractError(f"public-v1: invalid value type or identifier for {name}")
    return answer


class CommittedCandidates:
    """Serialized snapshots cannot be changed by downstream workers or callers."""

    def __init__(self) -> None:
        self._snapshots: list[str] = []

    def append(self, answer: Answer) -> None:
        self._snapshots.append(answer.model_dump_json())

    def __bool__(self) -> bool:
        return bool(self._snapshots)

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, index: int) -> Answer:
        return Answer.model_validate_json(self._snapshots[index])
