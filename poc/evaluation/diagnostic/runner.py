"""Infrastructure acceptance probes: stage coverage and invariants, never rankings."""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.diagnostic.fixtures import TARGET, TASK, ScriptedProvider
from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import Answer, ArchitectureOptions, Budget, ModelSpec, TaskInput
from poc.execution.sql_orchestration import METHODS
from poc.execution.sql_ports import PhaseTimeout, RequestTimeout
from poc.execution.sql_strategy import ContextAdmissionExceeded, ContextBoundModel, resolve_model
from poc.hybrid.collaboration_controller import CollaborationError
from poc.hybrid.completion_gate import DeliveryBlocked

STRATEGIES = ("single", "review", *METHODS, "sql-baseline")
POLICIES = ("reliable-v2", "concurrent-v1")
PROTOCOLS = ("native", "json")
REQUIRED_PHASES: dict[str, set[str]] = {
    "single": {"solve"},
    "review": {"draft", "review"},
    "hierarchical_dag": {"plan", "solve"},
    "board_claim": {"plan", "solve"},
    "managed_pool": {"plan", "solve"},
    "speculative": {"proposal", "reconcile"},
    "hybrid_v1": {"proposal", "critique", "verify"},
    "hybrid_v2": {"orchestrate", "task", "critique", "integrate", "verify"},
    "hybrid_v2_1": {"orchestrate", "task", "critique", "integrate", "verify"},
    "hybrid_v2_2": {"orchestrate", "task", "critique", "integrate", "verify"},
    "sql-baseline": set(),
}
REQUIRED_EVENTS = {
    "hierarchical_dag": {"plan.execution_mode_selected"},
    "board_claim": {"claim.acquired", "task.completed"},
    "managed_pool": {"assignment.created", "assignment.completed"},
    "speculative": {"candidate.completed", "reconciliation.completed"},
    "hybrid_v1": {"candidate.selected", "verification.completed", "delivery.gated"},
    "hybrid_v2_2": {
        "candidate.released",
        "hybrid_v2_2.plan_selected",
        "verification.completed",
        "delivery.gated",
    },
    "hybrid_v2_1": {
        "candidate.released",
        "candidate.selected",
        "hybrid_v2_1.plan_selected",
        "verification.completed",
        "delivery.gated",
    },
    "hybrid_v2": {
        "candidate.released",
        "candidate.selected",
        "hybrid_v2.plan_selected",
        "verification.completed",
        "delivery.gated",
    },
}


@dataclass(frozen=True)
class Probe:
    strategy: str
    protocol: str
    policy: str
    scenario: str = "healthy"

    @property
    def id(self) -> str:
        return f"{self.strategy}/{self.protocol}/{self.policy}/{self.scenario}"

    @property
    def adapter(self) -> str:
        return self.strategy + ("-json" if self.protocol == "json" else "")


def plan_probes(
    *,
    strategies: Sequence[str] = STRATEGIES,
    protocols: Sequence[str] = PROTOCOLS,
    policies: Sequence[str] = POLICIES,
    live: bool = False,
) -> list[Probe]:
    for supplied, allowed, name in (
        (strategies, STRATEGIES, "strategies"),
        (protocols, PROTOCOLS, "protocols"),
        (policies, POLICIES, "policies"),
    ):
        if not supplied or len(set(supplied)) != len(supplied) or set(supplied) - set(allowed):
            raise ValueError(f"invalid or duplicate {name}")
    probes: list[Probe] = []
    for strategy in strategies:
        if strategy == "sql-baseline":
            probes.append(Probe(strategy, "native", "reliable-v2"))
            continue
        for protocol in protocols:
            for policy in policies if strategy in METHODS else ("reliable-v2",):
                scenarios = ["healthy"]
                if not live:
                    scenarios += ["malformed_artifact", "request_budget"]
                    if strategy == "single":
                        scenarios += [
                            "admission",
                            "rate_limit",
                            "reasoning_only",
                            "duplicate_keys",
                            "request_timeout",
                            "phase_timeout",
                        ]
                        if protocol == "json":
                            scenarios += ["multiple_actions", "escaped_json"]
                    elif strategy == "review":
                        scenarios += ["empty_revision"]
                    elif strategy == "speculative":
                        scenarios += ["reconcile_timeout"]
                        if policy == "concurrent-v1":
                            scenarios += ["cancellation"]
                    elif strategy == "hybrid_v1":
                        scenarios += [
                            "critic_contract",
                            "false_approval",
                            "foreign_evidence",
                            "verifier_reject",
                            "proposal_loss",
                        ]
                    elif strategy == "managed_pool" and policy == "concurrent-v1":
                        scenarios += ["pool_reassign"]
                    elif strategy == "board_claim":
                        scenarios += ["sql_no_progress"]
                probes.extend(Probe(strategy, protocol, policy, scenario) for scenario in scenarios)
    return probes


def _options(probe: Probe) -> ArchitectureOptions:
    return ArchitectureOptions.model_validate(
        {
            "team_policy": probe.policy,
            "artifact_contract": "public-v1",
            "output_retries": 1,
            "review_protocol": "decision-v1" if probe.strategy == "review" else "replace",
            "review_failure_policy": "return_submitted_draft"
            if probe.scenario == "empty_revision"
            else "fail",
            "speculative_failure_policy": "return_first_submitted"
            if probe.scenario == "reconcile_timeout"
            else "fail",
            "hybrid_proposal_quorum": 1 if probe.scenario == "proposal_loss" else 2,
            "transport_policy": "json-normalize-v1"
            if probe.scenario == "escaped_json"
            else "strict-v1",
            "provider_retries": 1 if probe.scenario == "rate_limit" else 0,
            "return_reserve_seconds": 0.05,
        }
    )


def _task(probe: Probe) -> TaskInput:
    if probe.strategy != "sql-baseline":
        return TASK.model_copy(deep=True)
    return TaskInput(
        prompt="Read the one public invoice balance.",
        tables={
            "invoices": [{"invoice": "invoice-A", "revision": 1, "amount": 7}],
            "payments": [
                {
                    "invoice": "invoice-A",
                    "event": "event-A",
                    "amount": 0,
                    "status": "settled",
                    "kind": "payment",
                }
            ],
        },
    )


def _shape(answer: dict[str, Any], strategy: str) -> bool:
    values = answer.get("values", {})
    ids = {"invoice-A"} if strategy == "sql-baseline" else set(TARGET)
    return set(values) == ids and all(type(v) is int for v in values.values())


def _checks(
    probe: Probe,
    env: TaskEnvironment,
    answer: Answer | None,
    error: BaseException | None,
    events: list[dict[str, Any]],
    script: ScriptedProvider | None,
    usage: RunUsage,
    budget: Budget,
) -> dict[str, bool]:
    checks = {
        "request_ceiling": env.state.request_attempts <= budget.requests
        and usage.requests <= budget.requests,
        "tool_ceiling": env.tool_calls <= budget.tool_calls,
        "token_ceiling": usage.total_tokens + env.state.unreported_token_reserve
        <= budget.total_tokens,
        "committed_artifact_contracts": all(
            _shape(c["answer"], probe.strategy) for c in env.state.candidates
        ),
        "workers_drained": script is None or script.active == 0,
    }
    if script is not None:
        checks["request_accounting"] = script.calls == env.state.request_attempts
    if script is not None and probe.scenario not in {"healthy", "admission", "request_budget"}:
        checks["fault_exercised"] = script.injected > 0
    scenario = probe.scenario
    rejected = {
        "malformed_artifact",
        "critic_contract",
        "false_approval",
        "foreign_evidence",
        "verifier_reject",
        "reasoning_only",
        "duplicate_keys",
        "multiple_actions",
        "sql_no_progress",
        "request_budget",
        "admission",
        "request_timeout",
        "phase_timeout",
        "cancellation",
    }
    if scenario in rejected:
        expected: tuple[type[BaseException], ...] = (UnexpectedModelBehavior,)
        if scenario in {"critic_contract", "false_approval", "foreign_evidence"} or (
            scenario == "malformed_artifact"
            and probe.strategy in {"hybrid_v1", "hybrid_v2_1", "hybrid_v2_2"}
        ):
            expected = (CollaborationError,)
        elif scenario == "verifier_reject":
            expected = (DeliveryBlocked,)
        elif scenario == "admission":
            expected = (ContextAdmissionExceeded,)
        elif scenario == "request_budget":
            expected = (UsageLimitExceeded, CollaborationError)
        elif scenario == "request_timeout":
            expected = (RequestTimeout,)
        elif scenario in {"phase_timeout", "cancellation"}:
            expected = (PhaseTimeout,) if scenario == "phase_timeout" else (asyncio.CancelledError,)
        checks["expected_rejection"] = answer is None and isinstance(error, expected)
        if (
            scenario == "request_budget"
            and probe.strategy == "managed_pool"
            and probe.policy == "concurrent-v1"
        ):
            # Zero-sized reservations fail before provider calls. Match this exact controller path.
            checks["expected_rejection"] = (
                answer is None
                and isinstance(error, UnexpectedModelBehavior)
                and str(error) == "pool retry allocation exhausted: two failed branches"
                and env.state.request_attempts == 0
            )
        if scenario in {"admission", "reasoning_only", "duplicate_keys", "multiple_actions"}:
            checks["no_tool_execution"] = env.tool_calls == 0
        if scenario == "admission":
            checks["no_provider_request"] = (
                env.state.request_attempts == 0 and script is not None and script.calls == 0
            )
        if scenario in {"request_timeout", "phase_timeout"}:
            checks["failure_scope"] = getattr(error, "failure_details", {}).get("scope") == (
                "request" if scenario == "request_timeout" else "phase"
            )
        if scenario in {"critic_contract", "false_approval", "foreign_evidence"}:
            checks["verification_not_bypassed"] = not any(
                p["name"] == "verify" for p in env.state.phases
            )
            critiques = [p for p in env.state.phases if p["name"] == "critique"]
            checks["both_critics_exercised"] = len(critiques) == 2 and all(
                p["status"] == "completed"
                if scenario == "false_approval"
                else p.get("failure", {}).get("type") == "UnexpectedModelBehavior"
                for p in critiques
            )
        if scenario == "verifier_reject":
            checks["verdict_reached_gate"] = any(
                p["name"] == "verify" and p["status"] == "completed" for p in env.state.phases
            )
    else:
        checks["answer_delivered"] = error is None and answer is not None
        checks["final_artifact_contract"] = answer is not None and _shape(
            answer.model_dump(), probe.strategy
        )
        checks["sql_exercised"] = env.tool_calls > 0
        if script is not None:
            checks["public_data_transferred"] = answer is not None and answer.values == (
                TARGET if probe.strategy != "sql-baseline" else {"invoice-A": 7}
            )
        completed = {p["name"] for p in env.state.phases if p["status"] == "completed"}
        needed = REQUIRED_PHASES[probe.strategy]
        if scenario == "empty_revision":
            needed = {"draft"}
            checks["draft_retained"] = env.state.answer_source == "draft_fallback"
        elif scenario == "reconcile_timeout":
            needed = {"proposal"}
            checks["committed_candidate_retained"] = (
                env.state.answer_source == "first_submitted_fallback"
            )
            commits = [
                e["orchestration"]["data"]["answer"]
                for e in events
                if e.get("orchestration", {}).get("event_type") == "sql.candidate_committed"
            ]
            checks["distinct_candidates_committed"] = len(commits) == 2 and commits[0] != commits[1]
            checks.pop("public_data_transferred", None)
            checks["fallback_artifact_identity"] = (
                bool(commits) and answer is not None and answer.model_dump() == commits[0]
            )
        checks["required_stages_completed"] = needed <= completed
        kinds = {e["orchestration"]["event_type"] for e in events if "orchestration" in e}
        if scenario == "healthy":
            checks["controller_events"] = REQUIRED_EVENTS.get(probe.strategy, set()) <= kinds
            if probe.policy == "concurrent-v1":
                first = sorted(env.state.phases, key=lambda p: p.get("branch_index", 0))[:2]
                if probe.strategy != "hybrid_v2_2":
                    checks["overlapping_branches"] = len(first) == 2 and max(
                        p["start_seconds"] for p in first
                    ) < min(p["start_seconds"] + p["elapsed_seconds"] for p in first)
                reservations = [
                    e["budget_reservation"] for e in events if "budget_reservation" in e
                ]
                checks["shared_reservations"] = bool(reservations) and all(
                    sum(r[key] for r in reservations) <= getattr(budget, key)
                    for key in ("requests", "tool_calls", "total_tokens")
                )
                if probe.strategy == "board_claim":
                    checks["claim_contention"] = "sql.claim_contended" in kinds
        if scenario == "pool_reassign":
            assignments = [
                e["orchestration"]["data"]
                for e in events
                if e.get("orchestration", {}).get("event_type") == "assignment.created"
            ]
            attempts = [a for a in assignments if a["task_id"] == "plan-0"]
            checks["reassigned_worker_generation"] = (
                len(attempts) == 2
                and attempts[0]["worker_id"] != attempts[1]["worker_id"]
                and attempts[1]["assignment_generation"] > attempts[0]["assignment_generation"]
            )
        if probe.strategy in {"hybrid_v1", "hybrid_v2", "hybrid_v2_1", "hybrid_v2_2"}:
            refs = [e["evidence"]["ref"] for e in events if "evidence" in e]
            checks["independent_evidence"] = len(refs) >= (
                2 if scenario == "proposal_loss" else 3
            ) and len(set(refs)) == len(refs)
    return checks


async def run_probe(
    probe: Probe, *, case_seconds: float = 3, live_model: ModelSpec | None = None
) -> dict[str, Any]:
    task = _task(probe)
    seconds = (
        min(case_seconds, 0.3)
        if probe.scenario in {"phase_timeout", "reconcile_timeout", "cancellation"}
        else case_seconds
    )
    budget = Budget(
        seconds=seconds,
        requests=1 if probe.scenario == "request_budget" else 60,
        tool_calls=40,
        total_tokens=200000,
        max_output_tokens=2048,
        request_timeout_seconds=0.03 if probe.scenario == "request_timeout" else None,
        draft_fraction=0.5,
    )
    env = TaskEnvironment(task, budget.tool_calls)
    env.state.options = _options(probe)
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        # Keep causal evidence, excluding benchmark scores and private correctness diagnostics.
        copied = dict(event)
        if "candidate" in copied:
            copied["candidate"] = {
                k: v for k, v in copied["candidate"].items() if k != "diagnostics"
            }
        events.append(copied)

    env.state.emit = emit
    usage = RunUsage()
    script = (
        None
        if live_model is not None
        else ScriptedProvider(probe.strategy, probe.scenario, probe.policy == "concurrent-v1")
    )
    answer: Answer | None = None
    error: BaseException | None = None
    settings: ModelSettings = {"max_tokens": 131072} if probe.scenario == "admission" else {}
    began = perf_counter()
    cancelled_by_suite = False
    try:
        if live_model is not None:
            model: Model | str = resolve_model(live_model)
            settings = cast(ModelSettings, live_model.settings)
        else:
            assert script is not None
            model = ContextBoundModel(
                script.model, 1 if probe.scenario == "admission" else 128000, 2048
            )
        async with asyncio.timeout(case_seconds):
            execution = asyncio.ensure_future(
                ADAPTERS[probe.adapter](task, env, model, settings, budget, usage)
            )
            try:
                if probe.scenario == "cancellation":
                    assert script is not None
                    while script.active < 2 and not execution.done():
                        await asyncio.sleep(0)
                    execution.cancel()
                returned = cast(object, await execution)
                if not isinstance(returned, Answer):
                    raise TypeError("adapter returned a non-Answer artifact")
                answer = returned
            finally:
                if not execution.done():
                    execution.cancel()
                await asyncio.gather(execution, return_exceptions=True)
    except asyncio.CancelledError as exc:
        current = asyncio.current_task()
        if probe.scenario != "cancellation" or (current is not None and current.cancelling()):
            cancelled_by_suite = True
        error = exc
    except Exception as exc:
        error = exc
    finally:
        env.close()
    checks = _checks(probe, env, answer, error, events, script, usage, budget)
    checks["case_watchdog"] = type(error) is not TimeoutError
    if cancelled_by_suite:
        checks["suite_deadline"] = False
    phases = env.state.phases
    checks["stage_records_finalized"] = all(p["status"] != "running" for p in phases)
    admitted = [e["request_settings"] for e in events if "request_settings" in e]
    checks["effective_completion_caps"] = all(
        isinstance(request.get("max_tokens"), int)
        and 0 < request["max_tokens"] <= 2048
        and (
            request.get("completion_limit") is None
            or request["max_tokens"] <= request["completion_limit"]
        )
        and (
            request.get("context_window") is None
            or request["max_tokens"] + request["input_token_bound"] <= request["context_window"]
        )
        for request in admitted
    )
    return {
        "id": probe.id,
        "strategy": probe.strategy,
        "protocol": probe.protocol,
        "policy": probe.policy,
        "scenario": probe.scenario,
        "status": "not_run" if cancelled_by_suite else "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "findings": [name for name, passed in checks.items() if not passed],
        "error_type": type(error).__name__ if error else None,
        "provider_status_code": getattr(error, "status_code", None),
        "failure": getattr(error, "failure_details", None),
        "elapsed_seconds": perf_counter() - began,
        "phases": phases,
        "events": events,
        "budget": budget.model_dump(),
        "options": env.state.options.model_dump(),
        "public_task": task.model_dump(),
        "requests": usage.requests,
        "request_attempts": env.state.request_attempts,
        "tool_calls": env.tool_calls,
        "tokens": usage.total_tokens,
        "usage_complete": env.state.usage_complete,
        "unreported_token_reserve": env.state.unreported_token_reserve,
        "injections": script.injected if script else 0,
        "answer": answer.model_dump() if answer else None,
        "answer_source": env.state.answer_source,
    }


def validate_limits(case_seconds: float, suite_seconds: float, concurrency: int) -> None:
    if (
        any(not math.isfinite(value) or value <= 0 for value in (case_seconds, suite_seconds))
        or concurrency < 1
    ):
        raise ValueError("time limits and concurrency must be positive and finite")


async def run_diagnostics(
    *,
    strategies: Sequence[str] = STRATEGIES,
    protocols: Sequence[str] = PROTOCOLS,
    policies: Sequence[str] = POLICIES,
    case_seconds: float = 3,
    suite_seconds: float = 30,
    concurrency: int = 4,
    live_model: ModelSpec | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    validate_limits(case_seconds, suite_seconds, concurrency)
    planned = plan_probes(
        strategies=strategies, protocols=protocols, policies=policies, live=live_model is not None
    )
    rows: dict[str, dict[str, Any]] = {}
    pending = iter(planned)
    began = perf_counter()

    async def worker() -> None:
        for probe in pending:
            row = await run_probe(probe, case_seconds=case_seconds, live_model=live_model)
            rows[probe.id] = row
            if progress:
                progress(row)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                return

    expired = False
    try:
        async with asyncio.timeout(suite_seconds):
            async with asyncio.TaskGroup() as group:
                for _ in range(min(concurrency, len(planned))):
                    group.create_task(worker())
    except TimeoutError:
        expired = True
    results = [
        rows.get(
            p.id,
            {
                "id": p.id,
                "strategy": p.strategy,
                "protocol": p.protocol,
                "policy": p.policy,
                "scenario": p.scenario,
                "status": "not_run",
                "findings": ["suite_deadline"],
                "checks": {},
            },
        )
        for p in planned
    ]
    coverage: dict[str, Any] = {}
    for strategy in strategies:
        canaries = [r for r in results if r["strategy"] == strategy and r["scenario"] == "healthy"]
        coverage[strategy] = {
            "expected_variants": len(canaries),
            "passed_variants": sum(r["status"] == "pass" for r in canaries),
            "complete": all(r["status"] == "pass" for r in canaries),
            "reached_stages": sorted({p["name"] for r in canaries for p in r.get("phases", [])}),
        }
    counts = dict(Counter(row["status"] for row in results))
    return {
        "schema_version": "infrastructure-diagnostic-v1",
        "purpose": "infrastructure_acceptance",
        "mode": "live_canary" if live_model else "offline_fault_injection",
        "model_binding": live_model.model_dump() if live_model else None,
        "limits": {
            "case_seconds": case_seconds,
            "suite_seconds": suite_seconds,
            "concurrency": concurrency,
        },
        "status": "pass" if not expired and all(r["status"] == "pass" for r in results) else "fail",
        "suite_deadline_reached": expired,
        "elapsed_seconds": perf_counter() - began,
        "counts": counts,
        "coverage": coverage,
        "probes": results,
    }
