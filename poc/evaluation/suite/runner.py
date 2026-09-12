from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from opentelemetry import trace
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS, resolve_model
from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Matrix, ModelSpec, TaskCase
from poc.evaluation.suite.tasks import FACTORIES, GRADERS, generate


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_cases(matrix: Matrix) -> list[TaskCase]:
    unknown = set(matrix.families) - FACTORIES.keys()
    if unknown:
        raise ValueError(f"unknown task families: {sorted(unknown)}")
    return [
        generate(f, s, matrix.split, matrix.difficulty)
        for f in matrix.families
        for s in matrix.seeds
    ]


async def run_trial(
    case: TaskCase, spec: ModelSpec, strategy: str, repetition: int, matrix: Matrix
) -> dict[str, Any]:
    start_time = datetime.now(UTC)
    started = perf_counter()
    usage = RunUsage()
    env = TaskEnvironment(case.input, matrix.budget.tool_calls)
    answer = Answer(values={})
    error: str | None = None
    error_status_code: int | None = None
    status = "completed"
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("evaluation.trial") as span:
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("evaluation.case_id", case.id)
        span.set_attribute("evaluation.case_sha256", case.digest)
        span.set_attribute("evaluation.strategy", strategy)
        span.set_attribute("evaluation.model", spec.model)
        span.set_attribute("evaluation.repetition", repetition)
        context = span.get_span_context()
        trace_id = format(context.trace_id, "032x") if context.is_valid else None
        try:
            model = "test" if strategy == "sql-baseline" else resolve_model(spec)
            async with asyncio.timeout(matrix.budget.seconds):
                answer = await ADAPTERS[strategy](
                    case.input.model_copy(deep=True),
                    env,
                    model,
                    cast(ModelSettings, spec.settings),
                    matrix.budget,
                    usage,
                )
            if (
                usage.requests > matrix.budget.requests
                or usage.total_tokens > matrix.budget.total_tokens
                or len(env.calls) > matrix.budget.tool_calls
            ):
                raise RuntimeError("shared trial budget exceeded")
        except UsageLimitExceeded:
            status, error = "budget_exhausted", "model usage budget exhausted"
        except TimeoutError:
            status, error = "timeout", "trial wall-clock budget exhausted"
        except Exception as exc:
            # Do not persist provider exception bodies, which may contain request credentials.
            status, error = "error", type(exc).__name__
            error_status_code = getattr(exc, "status_code", None)
        finally:
            env.close()
        scores = (
            GRADERS[case.family](case, answer)
            if status == "completed"
            else {
                "exact": 0.0,
                "fraction_correct": 0.0,
            }
        )
        for name, score in scores.items():
            span.set_attribute(f"evaluation.{name}", score)
        span.set_attribute("evaluation.status", status)
        if error is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
    cost = None
    if spec.input_usd_per_million is not None and spec.output_usd_per_million is not None:
        cost = (
            usage.input_tokens * spec.input_usd_per_million
            + usage.output_tokens * spec.output_usd_per_million
        ) / 1_000_000
    return {
        "case_id": case.id,
        "case_sha256": case.digest,
        "family": case.family,
        "model_name": spec.name,
        "model_class": spec.model_class,
        "strategy": strategy,
        "repetition": repetition,
        "status": status,
        "error": error,
        "error_status_code": error_status_code,
        "scores": scores,
        "answer": answer.model_dump(),
        "tool_calls": env.calls,
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": cost,
        "elapsed_seconds": perf_counter() - started,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
    }


def summarize(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        groups[trial["model_name"], trial["strategy"], trial["family"]].append(trial)
    summaries: list[dict[str, Any]] = []
    for (model, strategy, family), rows in sorted(groups.items()):
        summaries.append(
            {
                "model_name": model,
                "strategy": strategy,
                "family": family,
                "trials": len(rows),
                "cases": len({r["case_id"] for r in rows}),
                "pass_rate": statistics.mean(r["scores"]["exact"] for r in rows),
                "mean_fraction_correct": statistics.mean(
                    r["scores"]["fraction_correct"] for r in rows
                ),
                "errors": sum(r["status"] != "completed" for r in rows),
                "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in rows),
                "p95_seconds": sorted(r["elapsed_seconds"] for r in rows)[
                    max(0, (95 * len(rows) + 99) // 100 - 1)
                ],
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "mean_tool_calls": statistics.mean(len(r["tool_calls"]) for r in rows),
            }
        )
    return summaries


async def run_matrix(matrix: Matrix, output: Path) -> dict[str, Any]:
    unknown = set(matrix.strategies) - ADAPTERS.keys()
    if unknown:
        raise ValueError(f"unknown adapters: {sorted(unknown)}")
    if set(matrix.strategies) & {"single", "review", "single-json", "review-json"}:
        for spec in matrix.models:
            if spec.base_url_env:
                for key in (spec.base_url_env, spec.api_key_env):
                    if not os.environ.get(key):
                        raise ValueError(f"missing environment variable {key} for {spec.name}")
    cases = make_cases(matrix)
    schedule = [
        (case, model, strategy, rep)
        for case in cases
        for model in matrix.models
        for strategy in matrix.strategies
        for rep in range(1, matrix.repetitions + 1)
    ]
    random.Random(matrix.order_seed).shuffle(schedule)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of expensive benchmark results.
    trials: list[dict[str, Any]] = []
    with output.open("x") as stream:
        header = {
            "schema_version": 1,
            "implementation_sha256": implementation_digest(),
            "config": matrix.model_dump(),
            "cases": [{"id": c.id, "sha256": c.digest} for c in cases],
            "suite_sha256": hashlib.sha256("".join(c.digest for c in cases).encode()).hexdigest(),
        }
        stream.write(json.dumps({"manifest": header}) + "\n")
        stream.flush()
        pending = iter(schedule)

        async def worker() -> None:
            # Only these fixed workers are scheduled. Each claims its next trial before
            # awaiting, so queue time does not consume that trial's wall-clock budget.
            for case, model, strategy, rep in pending:
                trial = await run_trial(case, model, strategy, rep, matrix)
                # All workers share the event loop; there is no await during a record write.
                trials.append(trial)
                stream.write(json.dumps({"trial": trial}) + "\n")
                stream.flush()
                print(
                    f"{case.id} {model.name}/{strategy} r{rep}: "
                    f"{trial['status']} exact={trial['scores']['exact']}",
                    flush=True,
                )

        # Cancellation or an unexpected runner failure joins/cancels all workers before
        # closing the report. Ordinary trial errors remain records and do not abort peers.
        async with asyncio.TaskGroup() as group:
            for _ in range(min(matrix.max_concurrency, len(schedule))):
                group.create_task(worker())
        stream.write(json.dumps({"summaries": summarize(trials)}) + "\n")
    return {**header, "trials": trials, "summaries": summarize(trials)}


def read_report(path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    manifest = records[0]["manifest"]
    trials = [r["trial"] for r in records if "trial" in r]
    return {**manifest, "trials": trials, "summaries": summarize(trials)}
