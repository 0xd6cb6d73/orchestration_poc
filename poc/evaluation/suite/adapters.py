"""Benchmark bindings for execution-layer SQL strategies and the harness oracle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Budget, TaskInput
from poc.execution.sql_orchestration import METHODS, adapter
from poc.execution.sql_strategy import (
    resolve_model as resolve_model,
)
from poc.execution.sql_strategy import (
    review,
    review_json,
    single,
    single_json,
)

Adapter = Callable[
    [TaskInput, TaskEnvironment, Model | str, ModelSettings, Budget, RunUsage], Awaitable[Answer]
]
ADAPTERS: dict[str, Adapter] = {}


def register_adapter(name: str, adapter: Adapter) -> None:
    if name in ADAPTERS:
        raise ValueError(f"adapter already registered: {name}")
    ADAPTERS[name] = adapter


async def sql_baseline(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    """Non-LLM oracle checks the environment and grader using independent SQL algorithms."""
    if "invoices" in env.schema:
        sql = """WITH dedup AS (SELECT DISTINCT * FROM payments),
        paid AS (SELECT invoice, SUM(CASE WHEN kind='refund' THEN -amount ELSE amount END) n
                 FROM dedup WHERE status='settled' GROUP BY invoice)
        SELECT i.invoice, i.amount-COALESCE(p.n,0) FROM invoices i
        LEFT JOIN paid p ON p.invoice=i.invoice
        WHERE i.revision=(SELECT MAX(revision) FROM invoices x WHERE x.invoice=i.invoice)"""
    elif "jobs" in env.schema:
        # A topological dynamic program, independent from the generating loop.
        jobs = {r[0]: r[1] for r in _rows(env, "SELECT id,duration FROM jobs")}
        edges = _rows(env, "SELECT job,requires FROM dependencies")
        finished: dict[str, Any] = {}
        while jobs:
            ready = [j for j in jobs if all(p in finished for child, p in edges if child == j)]
            if not ready:
                raise ValueError("dependency cycle")
            for job in ready:
                finished[job] = jobs.pop(job) + max(
                    (finished[p] for child, p in edges if child == job), default=0
                )
        return Answer(values=finished)
    else:
        sql = """WITH RECURSIVE effective(user,group_id) AS (
        SELECT user,group_id FROM memberships UNION
        SELECT e.user,n.parent FROM effective e JOIN nesting n ON n.child=e.group_id)
        SELECT q.id, CASE WHEN
        SUM(CASE WHEN p.effect='allow' THEN 1 ELSE 0 END)>0 AND
        SUM(CASE WHEN p.effect='deny' THEN 1 ELSE 0 END)=0 THEN 'allow' ELSE 'deny' END
        FROM queries q LEFT JOIN effective e ON e.user=q.user
        LEFT JOIN policies p ON p.group_id=e.group_id AND p.resource=q.resource GROUP BY q.id"""
    return Answer(values={str(row[0]): row[1] for row in _rows(env, sql)})


def _rows(env: TaskEnvironment, sql: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    while True:
        result = env.query(f"SELECT * FROM ({sql}) LIMIT 200 OFFSET {len(rows)}")
        if "error" in result:
            raise RuntimeError(result["error"])
        batch = result["rows"]
        rows.extend(batch)
        if len(batch) < 200:
            return rows


register_adapter("single", single)
register_adapter("review", review)
register_adapter("sql-baseline", sql_baseline)

register_adapter("single-json", single_json)
register_adapter("review-json", review_json)

# Bind the real execution controllers, with the same public task/tool boundary.

for method in METHODS:
    register_adapter(method, adapter(method))
    register_adapter(f"{method}-json", adapter(method, json_protocol=True))
