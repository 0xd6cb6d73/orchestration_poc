from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from opentelemetry import trace
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS, resolve_model
from poc.evaluation.suite.diagnostics import diagnose
from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Matrix, ModelSpec, TaskCase
from poc.evaluation.suite.reports import append, comparisons, coverage, locked, records, trial_key
from poc.evaluation.suite.runtime import RequestTimeout, ToolBudgetExceeded, TrialState
from poc.evaluation.suite.tasks import FACTORIES, GRADERS, generate


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    lockfile = Path(__file__).resolve().parents[3] / "uv.lock"
    if lockfile.exists():
        digest.update(lockfile.read_bytes())
    # Include registered plugin code, not just built-in suite files.
    sources = {
        inspect.getsourcefile(f)
        for f in [*ADAPTERS.values(), *FACTORIES.values(), *GRADERS.values()]
    }
    for source in sorted(s for s in sources if s):
        path = Path(source)
        if path.parent != Path(__file__).parent:
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
    case: TaskCase,
    spec: ModelSpec,
    strategy: str,
    repetition: int,
    matrix: Matrix,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    start_time = datetime.now(UTC)
    started = perf_counter()
    usage = RunUsage()
    env = TaskEnvironment(case.input, matrix.budget.tool_calls)
    env.state = TrialState(case.input, matrix.architecture_options.get(strategy), emit)
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
                or env.tool_calls > matrix.budget.tool_calls
            ):
                raise UsageLimitExceeded("shared trial budget exceeded")
        except (UsageLimitExceeded, ToolBudgetExceeded):
            status, error = "budget_exhausted", "shared trial budget exhausted"
        except RequestTimeout:
            status, error = "request_timeout", "model request timeout"
        except TimeoutError:
            status, error = "timeout", "trial wall-clock budget exhausted"
        except Exception as exc:
            # Do not persist provider exception bodies, which may contain request credentials.
            status, error = "error", type(exc).__name__
            error_status_code = getattr(exc, "status_code", None)
        finally:
            env.close()
        if status == "completed" and not env.state.candidates:
            env.state.candidate(answer, "adapter_return")
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
        span.set_attribute("evaluation.request_attempts", env.state.request_attempts)
        span.set_attribute("evaluation.request_responses", env.state.request_responses)
        if env.state.request_attempts:
            span.set_attribute("evaluation.usage_complete", env.state.usage_complete)
        span.set_attribute("evaluation.candidate_count", len(env.state.candidates))
        span.set_attribute(
            "evaluation.review_started", any(p["name"] == "review" for p in env.state.phases)
        )
        if error is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
    cost = None
    if (
        spec.input_usd_per_million is not None
        and spec.output_usd_per_million is not None
        and env.state.usage_complete
    ):
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
        "validation_calls": env.validation_calls,
        "total_tool_calls": env.tool_calls,
        "diagnostics": diagnose(case.input, answer),
        "candidates": env.state.candidates,
        "best_candidate": env.state.best_candidate(),
        "phases": env.state.phases,
        "review_started": any(p["name"] == "review" for p in env.state.phases),
        "exhausted_phase": env.state.exhausted_phase
        or (env.state.phase if status in {"timeout", "budget_exhausted"} else None),
        "architecture_options": env.state.options.model_dump(),
        "request_attempts": env.state.request_attempts,
        "request_responses": env.state.request_responses,
        "usage_complete": env.state.usage_complete
        if env.state.request_attempts
        else (True if strategy == "sql-baseline" else None),
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "estimated_cost_usd": cost,
        "elapsed_seconds": perf_counter() - started,
        "start_time": start_time.isoformat(),
        "end_time": datetime.now(UTC).isoformat(),
        "trace_id": trace_id,
    }


def summarize(trials: list[dict[str, Any]], matrix: Matrix | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        groups[trial["model_name"], trial["strategy"], trial["family"]].append(trial)
    if matrix:
        for m in matrix.models:
            for strategy in matrix.strategies:
                for family in matrix.families:
                    groups[m.name, strategy, family]
    summaries: list[dict[str, Any]] = []
    for (model, strategy, family), rows in sorted(groups.items()):
        expected = len(matrix.seeds) * matrix.repetitions if matrix else None
        if not rows:
            summaries.append(
                {
                    "model_name": model,
                    "strategy": strategy,
                    "family": family,
                    "trials": 0,
                    "cases": 0,
                    "expected_trials": expected,
                    "coverage_complete": False,
                    "pass_rate": None,
                    "mean_fraction_correct": None,
                }
            )
            continue
        dimension_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        for row in rows:
            for name, counts in row.get("diagnostics", {}).get("dimensions", {}).items():
                for key in ("passed", "total"):
                    dimension_totals[name][key] += counts[key]
        summaries.append(
            {
                "model_name": model,
                "strategy": strategy,
                "family": family,
                "trials": len(rows),
                "expected_trials": expected,
                "coverage_complete": len(rows) == expected if expected is not None else None,
                "constraint_checks": dict(dimension_totals),
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
                "mean_tool_calls": statistics.mean(
                    r.get("total_tool_calls", len(r["tool_calls"])) for r in rows
                ),
                "invalid_answers": sum(
                    r.get("diagnostics", {}).get("answer_valid") is False for r in rows
                ),
                "feasible_candidates": sum(
                    bool(
                        r.get("best_candidate")
                        and r["best_candidate"]["diagnostics"].get("feasible")
                    )
                    for r in rows
                ),
                "review_started": sum(r.get("review_started", False) for r in rows),
                "usage_complete": all(r.get("usage_complete") is True for r in rows),
            }
        )
    return summaries


async def run_matrix(matrix: Matrix, output: Path, *, resume: bool = False) -> dict[str, Any]:
    with locked(output):
        return await _run_matrix(matrix, output, resume=resume)


async def _run_matrix(matrix: Matrix, output: Path, *, resume: bool) -> dict[str, Any]:
    unknown = set(matrix.strategies) - ADAPTERS.keys()
    if unknown:
        raise ValueError(f"unknown adapters: {sorted(unknown)}")
    cases = make_cases(matrix)
    header = {
        "schema_version": 2,
        "report_id": uuid4().hex,
        "implementation_sha256": implementation_digest(),
        "config": matrix.model_dump(),
        "cases": [{"id": c.id, "sha256": c.digest} for c in cases],
        "suite_sha256": hashlib.sha256("".join(c.digest for c in cases).encode()).hexdigest(),
    }
    trials: list[dict[str, Any]] = []
    if resume:
        saved = read_report(output)
        for field in ("implementation_sha256", "config", "cases", "suite_sha256"):
            if saved[field] != header[field]:
                raise ValueError(f"cannot resume: {field} changed; use a new report")
        header = {
            k: v
            for k, v in saved.items()
            if k not in {"trials", "summaries", "coverage", "comparisons", "events"}
        }
        trials = saved["trials"]
        records(output, repair=True)
    completed = {trial_key(t) for t in trials}
    schedule = [
        (case, model, strategy, rep)
        for case in cases
        for model in matrix.models
        for strategy in matrix.strategies
        for rep in range(1, matrix.repetitions + 1)
    ]
    random.Random(matrix.order_seed).shuffle(schedule)
    total = len(schedule)
    schedule = [
        (c, m, s, r)
        for c, m, s, r in schedule
        if trial_key({"case_id": c.id, "model_name": m.name, "strategy": s, "repetition": r})
        not in completed
    ]
    if schedule and set(matrix.strategies) & {"single", "review", "single-json", "review-json"}:
        for spec in matrix.models:
            if spec.base_url_env:
                for key in (spec.base_url_env, spec.api_key_env):
                    if not os.environ.get(key):
                        raise ValueError(f"missing environment variable {key} for {spec.name}")
    with output.open("a" if resume else "x") as stream:
        if not resume:
            append(stream, {"manifest": header})
        pending = iter(schedule)
        finished = len(completed)
        running = 0

        def progress() -> None:
            print(
                f"Progress: {finished}/{total} finished, {running} running",
                file=sys.stderr,
                flush=True,
            )

        progress()

        async def worker() -> None:
            nonlocal finished, running
            for case, model, strategy, rep in pending:
                identity = {
                    "case_id": case.id,
                    "model_name": model.name,
                    "strategy": strategy,
                    "repetition": rep,
                    "attempt_id": uuid4().hex,
                }

                def emit(event: dict[str, Any], identity: dict[str, Any] = identity) -> None:
                    append(stream, {"event": {**identity, **event}})

                emit({"started": datetime.now(UTC).isoformat()})
                running += 1
                progress()
                try:
                    trial = await run_trial(case, model, strategy, rep, matrix, emit)
                    append(stream, {"trial": trial})
                    trials.append(trial)
                    finished += 1
                finally:
                    running -= 1
                    progress()
                print(
                    f"{case.id} {model.name}/{strategy} r{rep}: "
                    f"{trial['status']} exact={trial['scores']['exact']}",
                    flush=True,
                )

        async with asyncio.TaskGroup() as group:
            for _ in range(min(matrix.max_concurrency, len(schedule))):
                group.create_task(worker())
        report = {**header, "trials": trials, "summaries": summarize(trials, matrix)}
        report["coverage"] = coverage(report)
        report["comparisons"] = comparisons(trials)
        if not report["coverage"]["complete"]:
            raise RuntimeError(
                "workers interrupted before all trials were recorded; resume this report"
            )
        append(stream, {k: report[k] for k in ("summaries", "coverage", "comparisons")})
    return report


def read_report(path: Path) -> dict[str, Any]:
    entries = records(path)
    manifest = entries[0]["manifest"]
    trials = [r["trial"] for r in entries if "trial" in r]
    report = {
        **manifest,
        "trials": trials,
        "summaries": summarize(trials, Matrix.model_validate(manifest["config"])),
        "events": [r["event"] for r in entries if "event" in r],
    }
    report["coverage"] = coverage(report)
    report["comparisons"] = comparisons(trials)
    # Refuse foreign records before deciding they can be skipped during resume/publication.
    case_hashes = {c["id"]: c["sha256"] for c in manifest["cases"]}
    models = {m["name"] for m in manifest["config"]["models"]}
    for t in trials:
        if (
            t["case_id"] not in case_hashes
            or t["case_sha256"] != case_hashes[t["case_id"]]
            or t["model_name"] not in models
            or t["strategy"] not in manifest["config"]["strategies"]
            or not 1 <= t["repetition"] <= manifest["config"]["repetitions"]
        ):
            raise ValueError("trial does not belong to report manifest")
    return report
