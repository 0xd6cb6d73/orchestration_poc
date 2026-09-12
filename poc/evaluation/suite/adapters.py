from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any, cast

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Budget, ModelSpec, TaskInput

# Trusted in-process plugins receive only public inputs, a shared environment and usage.
Adapter = Callable[
    [TaskInput, TaskEnvironment, Model | str, ModelSettings, Budget, RunUsage], Awaitable[Answer]
]
ADAPTERS: dict[str, Adapter] = {}


def register_adapter(name: str, adapter: Adapter) -> None:
    if name in ADAPTERS:
        raise ValueError(f"adapter already registered: {name}")
    ADAPTERS[name] = adapter


def resolve_model(spec: ModelSpec) -> Model | str:
    if spec.base_url_env:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        endpoint = os.environ[spec.base_url_env]
        return OpenAIChatModel(
            spec.model,
            provider=OpenAIProvider(
                base_url=endpoint,
                api_key=os.environ[spec.api_key_env],
            ),
        )
    return spec.model


async def _solve_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
    draft: Answer | None = None,
) -> Answer:
    # The same text JSON protocol works even when a provider lacks native function calling.
    agent = Agent(
        model,
        output_type=str,
        model_settings=settings,
        capabilities=[Instrumentation()],
        instructions=(
            "Solve the task with the read-only SQLite tool. On EACH turn return exactly one "
            'JSON object: {"sql": "SELECT ..."} to execute a query, or '
            '{"values": {"id": value}} for your COMPLETE final answer. '
            "No markdown. SQLite joins, windows and recursive CTEs are supported. "
            "Queries return at most 200 rows / 64KB; use LIMIT/OFFSET pagination. "
            "Inspect and compute over the tables, check edge cases, never invent results."
        ),
    )
    prompt = task.prompt + "\nSQL schema: " + json.dumps(env.schema)
    if draft is not None:
        prompt += "\nIndependently verify and correct this draft: " + draft.model_dump_json()
    history = None
    while True:
        result = await agent.run(
            prompt,
            message_history=history,
            usage=usage,
            usage_limits=UsageLimits(
                request_limit=budget.requests, total_tokens_limit=budget.total_tokens
            ),
        )
        history = result.all_messages()
        try:
            action = json.loads(result.output)
            if (
                isinstance(action, dict)
                and set(cast(dict[str, Any], action)) == {"sql"}
                and isinstance(action["sql"], str)
            ):
                prompt = "SQL result: " + json.dumps(env.query(action["sql"]))
            else:
                return Answer.model_validate(action)
        except (ValueError, TypeError):
            prompt = 'Invalid action. Return ONLY {"sql":"..."} or {"values":{...}}.'


async def _solve_native(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
    draft: Answer | None = None,
) -> Answer:
    async def query(sql: str) -> dict[str, Any]:
        """Read-only SQLite query; 200 rows and 64KB maximum. Paginate with LIMIT/OFFSET."""
        return env.query(sql)

    agent = Agent(
        model,
        output_type=Answer,
        tools=[query],
        model_settings=settings,
        capabilities=[Instrumentation()],
        instructions=(
            "Solve the task using SQL over the provided schema. SQLite joins, "
            "windows and recursive CTEs are supported. Check all constraints "
            "and return the complete values mapping."
        ),
    )
    prompt = task.prompt + "\nSQL schema: " + json.dumps(env.schema)
    if draft is not None:
        prompt += "\nIndependently verify and correct this draft: " + draft.model_dump_json()
    result = await agent.run(
        prompt,
        usage=usage,
        usage_limits=UsageLimits(
            request_limit=budget.requests,
            tool_calls_limit=budget.tool_calls,
            total_tokens_limit=budget.total_tokens,
        ),
    )
    return result.output


async def single_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    return await _solve_json(task, env, model, settings, budget, usage)


async def review_json(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    draft = await _solve_json(task, env, model, settings, budget, usage)
    return await _solve_json(task, env, model, settings, budget, usage, draft)


async def single(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    return await _solve_native(task, env, model, settings, budget, usage)


async def review(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    draft = await _solve_native(task, env, model, settings, budget, usage)
    return await _solve_native(task, env, model, settings, budget, usage, draft)


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
