from __future__ import annotations

import asyncio
import hashlib
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from pydantic import TypeAdapter

from poc.control.runtime import Runtime
from poc.evaluation.benchmark_models import (
    OrchestrationBenchmarkReport,
    OrchestrationBenchmarkSummary,
    OrchestrationBenchmarkTrial,
    OrchestrationBenchmarkVariant,
    OrchestrationWorkload,
    TokenUsage,
)
from poc.execution.pydantic_ai_executor import PydanticModelFactory
from poc.models import AgentRuntimeConfig, ApprovalRequest, RunCreate, RunStatus, Tier


def incident_investigation_workload() -> OrchestrationWorkload:
    """The production demo as a non-trivial, multi-agent benchmark workload."""
    return OrchestrationWorkload(
        name="incident-investigation-v1",
        objective=(
            "Investigate the supplied checkout latency incident across metrics, logs, "
            "deployment records, and fixture metadata. Quantify the regression, separate "
            "correlation from causation, preserve exact evidence lineage, and propose safe "
            "falsification tests. Do not modify any live system."
        ),
        required_report_concepts={
            "quantified_regression": ("3.56x",),
            "nearby_deployment": ("2026.04.17.2",),
            "incident_specific_signal": ("cache_miss", "cache miss"),
            "misleading_signal_qualified": ("payment_timeout", "payment timeout"),
            "causal_caveat": (
                "not proof of causation",
                "does not prove causation",
                "correlation is not causation",
            ),
            "safe_follow_up": ("isolated environment", "fixture workload"),
            "evidence_lineage": ("evidence lineage",),
        },
        minimum_workflows=4,
        minimum_tasks=12,
        minimum_workers=12,
        required_tools=frozenset(
            {
                "read_metric_slice",
                "calculate_percentile",
                "read_log_slice",
                "count_log_pattern",
                "read_deployment_record",
                "read_manifest",
                "write_artifact",
            }
        ),
    )


class OrchestrationBenchmarkRunner:
    """Runs complete approved missions and compares orchestration-level outcomes."""

    def __init__(
        self,
        workload: OrchestrationWorkload | None = None,
        *,
        fixture_root: str | Path | None = None,
        work_dir: str | Path | None = None,
        pydantic_model_factory: PydanticModelFactory | None = None,
    ):
        self.workload = workload or incident_investigation_workload()
        self.fixture_root = Path(fixture_root) if fixture_root is not None else None
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.pydantic_model_factory = pydantic_model_factory

    async def evaluate(
        self,
        variants: Sequence[OrchestrationBenchmarkVariant],
        *,
        repeat: int = 1,
        max_concurrency: int = 1,
        include_output: bool = False,
    ) -> OrchestrationBenchmarkReport:
        if not variants:
            raise ValueError("at least one benchmark variant is required")
        if repeat < 1:
            raise ValueError("repeat must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        names = [variant.name for variant in variants]
        if len(names) != len(set(names)):
            raise ValueError("benchmark variant names must be unique")
        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def run(
            variant: OrchestrationBenchmarkVariant, repetition: int
        ) -> OrchestrationBenchmarkTrial:
            async with semaphore:
                return await self._run_trial(variant, repetition, include_output=include_output)

        trials = await asyncio.gather(
            *(
                run(variant, repetition)
                for variant in variants
                for repetition in range(1, repeat + 1)
            )
        )
        return OrchestrationBenchmarkReport(
            workload=self.workload,
            variants=list(variants),
            trials=list(trials),
            summaries=[_summarize(variant, trials) for variant in variants],
        )

    async def _run_trial(
        self,
        variant: OrchestrationBenchmarkVariant,
        repetition: int,
        *,
        include_output: bool,
    ) -> OrchestrationBenchmarkTrial:
        parent = str(self.work_dir) if self.work_dir is not None else None
        with TemporaryDirectory(prefix="orchestration-eval-", dir=parent) as data_dir:
            runtime = Runtime(
                data_dir,
                fixture_root=self.fixture_root,
                pydantic_model_factory=self.pydantic_model_factory,
            )
            started = perf_counter()
            try:
                request = RunCreate(
                    objective=self.workload.objective,
                    agent_runtime=AgentRuntimeConfig(
                        backend=variant.agent_backend,
                        provider=variant.provider,
                        model=variant.model,
                        options=variant.model_settings,
                    ),
                    execution_mode=variant.execution_mode,
                    execution_options=variant.execution_options,
                    swarm_strategy=variant.swarm_strategy,
                    hybrid=variant.hybrid,
                )
                run, _ = runtime.create_run(request)
                await runtime.approve(run["run_id"], ApprovalRequest(plan_version=1), start=True)
                await runtime.wait(run["run_id"])
                elapsed = perf_counter() - started
                return _trial_from_runtime(
                    runtime,
                    run["run_id"],
                    self.workload,
                    variant,
                    repetition,
                    elapsed,
                    include_output=include_output,
                )
            except Exception as exc:
                return OrchestrationBenchmarkTrial(
                    workload=self.workload.name,
                    variant=variant.name,
                    repetition=repetition,
                    execution_mode=variant.execution_mode,
                    swarm_strategy=variant.swarm_strategy,
                    agent_backend=variant.agent_backend,
                    provider=variant.provider,
                    model=variant.model,
                    status=RunStatus.FAILED,
                    passed=False,
                    elapsed_seconds=perf_counter() - started,
                    quality_checks={"runner_completed": False},
                    workflow_count=0,
                    task_count=0,
                    successful_task_count=0,
                    worker_count=0,
                    tool_call_count=0,
                    failed_tool_call_count=0,
                    event_count=0,
                    error=str(exc),
                )
            finally:
                await runtime.close()


def _trial_from_runtime(
    runtime: Runtime,
    run_id: str,
    workload: OrchestrationWorkload,
    variant: OrchestrationBenchmarkVariant,
    repetition: int,
    elapsed: float,
    *,
    include_output: bool,
) -> OrchestrationBenchmarkTrial:
    status = runtime.status(run_id)
    run = status["run"]
    workflows = status["workflows"]
    tasks = [task for workflow in workflows for task in workflow["tasks"]]
    workers = [agent for agent in status["agents"] if agent["tier"] == Tier.WORKER]
    operations = [
        dict(row)
        for row in runtime.db.conn.execute(
            "SELECT kind,status FROM operations WHERE run_id=? ORDER BY started_at", (run_id,)
        )
    ]
    events = status["events"]
    event_counts = Counter(event["event_type"] for event in events)
    final_report, report_digest = _final_report(runtime, events)
    normalized_report = final_report.casefold()
    concept_checks = {
        f"report:{concept}": any(term.casefold() in normalized_report for term in alternatives)
        for concept, alternatives in workload.required_report_concepts.items()
    }
    used_tools = {operation["kind"] for operation in operations}
    execution_modes = {execution["mode"] for execution in status["executions"]}
    accepted_events = [event for event in events if event["event_type"] == "deliverable.accepted"]
    expected_lineage = TypeAdapter(list[str]).validate_python(
        accepted_events[-1]["data"].get("evidence_artifacts", []) if accepted_events else []
    )
    checks = {
        "run_completed": run["status"] == RunStatus.COMPLETED,
        "multi_workflow_plan_executed": len(workflows) >= workload.minimum_workflows,
        "nontrivial_task_graph_executed": len(tasks) >= workload.minimum_tasks,
        "worker_population_materialized": len(workers) >= workload.minimum_workers,
        "all_tasks_succeeded": bool(tasks) and all(task["status"] == "succeeded" for task in tasks),
        "all_tool_calls_succeeded": bool(operations)
        and all(operation["status"] == "succeeded" for operation in operations),
        "required_tools_used": workload.required_tools <= used_tools,
        "selected_mode_applied": bool(status["executions"])
        and execution_modes == {variant.execution_mode},
        "timestamp_ambiguity_resolved": "read_manifest" in used_tools
        and (event_counts["workflow.resumed"] >= 1 or event_counts["validation.completed"] >= 1),
        "deliverable_accepted": event_counts["deliverable.accepted"] == 1,
        "exact_evidence_lineage_rendered": bool(expected_lineage)
        and all(artifact_id in final_report for artifact_id in expected_lineage),
        **concept_checks,
    }
    usage = _aggregate_usage(events)
    return OrchestrationBenchmarkTrial(
        workload=workload.name,
        variant=variant.name,
        repetition=repetition,
        execution_mode=variant.execution_mode,
        swarm_strategy=variant.swarm_strategy,
        agent_backend=variant.agent_backend,
        provider=variant.provider,
        model=variant.model,
        status=run["status"],
        passed=all(checks.values()),
        elapsed_seconds=round(elapsed, 6),
        quality_checks=checks,
        workflow_count=len(workflows),
        task_count=len(tasks),
        successful_task_count=sum(task["status"] == "succeeded" for task in tasks),
        worker_count=len(workers),
        tool_call_count=len(operations),
        failed_tool_call_count=sum(operation["status"] != "succeeded" for operation in operations),
        event_count=len(events),
        usage=usage,
        final_report_sha256=report_digest,
        final_report=final_report if include_output else None,
        error=run.get("error"),
    )


def _final_report(runtime: Runtime, events: list[dict[str, Any]]) -> tuple[str, str | None]:
    accepted = [event for event in events if event["event_type"] == "deliverable.accepted"]
    if not accepted:
        return "", None
    artifact_id = accepted[-1]["data"].get("artifact_id")
    if not isinstance(artifact_id, str):
        return "", None
    _, content = runtime.artifacts.read(artifact_id)
    return content.decode(errors="replace"), hashlib.sha256(content).hexdigest()


def _aggregate_usage(events: list[dict[str, Any]]) -> TokenUsage:
    totals: Counter[str] = Counter()
    for event in events:
        if event["event_type"] != "agent.execution_usage":
            continue
        for key, value in event["data"].get("usage", {}).items():
            if isinstance(value, int):
                totals[key] += value
    return TokenUsage(
        requests=totals["requests"],
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        cache_read_tokens=totals["cache_read_tokens"],
        cache_write_tokens=totals["cache_write_tokens"],
    )


def _summarize(
    variant: OrchestrationBenchmarkVariant,
    all_trials: Sequence[OrchestrationBenchmarkTrial],
) -> OrchestrationBenchmarkSummary:
    trials = [trial for trial in all_trials if trial.variant == variant.name]
    elapsed = [trial.elapsed_seconds for trial in trials]
    return OrchestrationBenchmarkSummary(
        variant=variant.name,
        execution_mode=variant.execution_mode,
        swarm_strategy=variant.swarm_strategy,
        trials=len(trials),
        completed=sum(trial.status == RunStatus.COMPLETED for trial in trials),
        passed=sum(trial.passed for trial in trials),
        completion_rate=sum(trial.status == RunStatus.COMPLETED for trial in trials) / len(trials),
        pass_rate=sum(trial.passed for trial in trials) / len(trials),
        mean_elapsed_seconds=round(statistics.fmean(elapsed), 6),
        p50_elapsed_seconds=round(statistics.median(elapsed), 6),
        p95_elapsed_seconds=round(_percentile(elapsed, 0.95), 6),
        mean_tool_calls=round(statistics.fmean(trial.tool_call_count for trial in trials), 3),
        mean_workers=round(statistics.fmean(trial.worker_count for trial in trials), 3),
        total_input_tokens=sum(trial.usage.input_tokens for trial in trials),
        total_output_tokens=sum(trial.usage.output_tokens for trial in trials),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = quantile * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def save_benchmark_report(report: OrchestrationBenchmarkReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.model_dump_json(indent=2) + "\n")
