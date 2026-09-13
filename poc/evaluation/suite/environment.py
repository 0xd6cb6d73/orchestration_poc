from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from opentelemetry import trace

from poc.evaluation.suite.models import Answer, TaskInput
from poc.evaluation.suite.runtime import ToolBudgetExceeded, TrialState


class TaskEnvironment:
    """Read-only SQL over public data only; no host files, network, or gold answers."""

    def __init__(self, task: TaskInput, max_calls: int):
        self.db = sqlite3.connect(":memory:")
        self.calls: list[dict[str, Any]] = []
        self.max_calls = max_calls
        self.state = TrialState(task)
        self.validation_calls: list[dict[str, Any]] = []
        self.schema: dict[str, list[str]] = {}
        for table, rows in task.tables.items():
            if not rows:
                raise ValueError(f"empty table without a schema: {table}")
            columns = list(rows[0])
            if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n) for n in [table, *columns]):
                raise ValueError("invalid SQL identifier")
            self.schema[table] = columns
            names = ",".join(f'"{c}"' for c in columns)
            self.db.execute(f'CREATE TABLE "{table}" ({names})')
            placeholders = ",".join("?" for _ in columns)
            self.db.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [tuple(row[c] for c in columns) for row in rows],
            )
        self.db.commit()
        self.db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 64000)
        self.db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16000)
        self.db.set_authorizer(self._authorize)

    @staticmethod
    def _authorize(
        action: int, arg1: str | None, arg2: str | None, database: str | None, source: str | None
    ) -> int:
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }
        if action == sqlite3.SQLITE_FUNCTION and arg2 == "load_extension":
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    def query(self, sql: str) -> dict[str, Any]:
        """Run read-only SQLite SQL. Up to 200 rows; use LIMIT/OFFSET to paginate.

        Joins, aggregates, window functions and recursive CTEs are available.
        """
        if self.tool_calls >= self.max_calls:
            raise ToolBudgetExceeded("shared tool-call budget exhausted")
        call: dict[str, Any] = {"sql": sql}
        self.calls.append(call)
        ticks = 0

        def progress() -> int:
            nonlocal ticks
            ticks += 1
            return int(ticks > 20000)

        self.db.set_progress_handler(progress, 1000)
        with trace.get_tracer(__name__).start_as_current_span("evaluation.query") as span:
            span.set_attribute("openinference.span.kind", "TOOL")
            span.set_attribute("evaluation.phase", self.state.phase)
            span.set_attribute("input.value", sql)
            try:
                cursor = self.db.execute(sql)
                rows = cursor.fetchmany(201)
                result: dict[str, Any] = {
                    "columns": [d[0] for d in cursor.description or []],
                    "rows": [list(row) for row in rows[:200]],
                    "truncated": len(rows) > 200,
                }
                # Bound result size as well as row count (e.g. group_concat/hex).
                if len(json.dumps(result)) > 64000:
                    result = {"error": "Result exceeds 64KB; select fewer rows or columns."}
            except sqlite3.Error as exc:
                result = {"error": str(exc)}
            call["result"] = result
            span.set_attribute("output.value", json.dumps(result))
            return result

    def close(self) -> None:
        self.db.close()

    def fork(self, max_calls: int) -> TaskEnvironment:
        """Independent SQLite connection with a reserved share of the trial tool budget."""
        child = TaskEnvironment(self.state.task, max_calls)
        child.state = TrialState(self.state.task, self.state.options, self.state.emit)
        return child

    def absorb(self, child: object) -> None:
        if not isinstance(child, TaskEnvironment):
            raise TypeError("cannot absorb a foreign environment")
        self.calls.extend(child.calls)
        self.validation_calls.extend(child.validation_calls)
        self.state.candidates.extend(child.state.candidates)
        self.state.diagnostic_candidates.extend(child.state.diagnostic_candidates)

    @property
    def tool_calls(self) -> int:
        return len(self.calls) + len(self.validation_calls)

    def submit_candidate(self, values: dict[str, Any]) -> dict[str, Any]:
        if self.tool_calls >= self.max_calls:
            raise ToolBudgetExceeded("shared tool-call budget exhausted")
        answer = Answer(values=values)
        self.validation_calls.append({"action": "submit_candidate", "values": answer.model_dump()})
        self.state.candidate(answer, "submitted")
        return {"recorded": True}
