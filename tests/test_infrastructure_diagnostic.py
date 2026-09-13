from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.models.function import FunctionModel

from poc.evaluation.diagnostic import runner
from poc.evaluation.diagnostic.cli import select_model
from poc.evaluation.diagnostic.fixtures import TARGET, ScriptedProvider
from poc.evaluation.diagnostic.runner import (
    STRATEGIES,
    Probe,
    plan_probes,
    run_diagnostics,
    run_probe,
)
from poc.evaluation.suite.__main__ import main
from poc.evaluation.suite.adapters import ADAPTERS
from poc.execution.sql_contracts import Answer, ModelSpec


async def test_default_scan_exercises_all_strategies_and_injected_failures() -> None:
    report = await run_diagnostics()
    assert report["status"] == "pass", [
        (r["id"], r["findings"]) for r in report["probes"] if r["status"] != "pass"
    ]
    assert report["counts"] == {"pass": len(plan_probes())}
    assert set(report["coverage"]) == set(STRATEGIES)
    assert all(row["complete"] for row in report["coverage"].values())
    assert report["coverage"]["hybrid_v1"]["expected_variants"] == 4
    assert set(report["coverage"]["hybrid_v1"]["reached_stages"]) == {
        "proposal",
        "critique",
        "verify",
    }
    assert {r["scenario"] for r in report["probes"]} >= {
        "malformed_artifact",
        "false_approval",
        "pool_reassign",
        "request_timeout",
        "reconcile_timeout",
        "admission",
        "request_budget",
        "reasoning_only",
    }
    # This is an acceptance report, not a new route to benchmark grading or rankings.
    assert all("scores" not in row for row in report["probes"])
    assert not any(
        "diagnostics" in e.get("candidate", {}) for row in report["probes"] for e in row["events"]
    )


async def test_noop_adapter_cannot_pass_stage_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def noop(*args: Any) -> Answer:
        return Answer(values=TARGET)

    monkeypatch.setitem(ADAPTERS, "hierarchical_dag-json", noop)
    row = await run_probe(Probe("hierarchical_dag", "json", "reliable-v2"))
    assert row["status"] == "fail"
    assert row["checks"]["final_artifact_contract"]
    assert {"required_stages_completed", "controller_events", "sql_exercised"} <= set(
        row["findings"]
    )


async def test_disabled_artifact_gate_is_detected_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bypass(task: Any, answer: Answer, policy: str) -> Answer:
        return answer

    monkeypatch.setattr("poc.execution.sql_strategy.validate_artifact", bypass)
    row = await run_probe(Probe("single", "json", "reliable-v2", "malformed_artifact"))
    assert row["status"] == "fail"
    assert row["injections"] > 0
    assert {"expected_rejection", "committed_artifact_contracts"} <= set(row["findings"])


async def test_unrelated_crash_does_not_pass_an_expected_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: Any) -> Answer:
        raise RuntimeError("unrelated adapter regression")

    monkeypatch.setitem(ADAPTERS, "single-json", broken)
    row = await run_probe(Probe("single", "json", "reliable-v2", "duplicate_keys"))
    assert row["status"] == "fail"
    assert row["error_type"] == "RuntimeError"
    assert {"fault_exercised", "expected_rejection"} <= set(row["findings"])


async def test_suite_deadline_records_unrun_cases_and_drains_active_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = False

    async def blocked(*args: Any) -> Answer:
        nonlocal stopped
        try:
            await asyncio.Event().wait()
        finally:
            stopped = True
        raise AssertionError("unreachable")

    monkeypatch.setitem(ADAPTERS, "single-json", blocked)
    report = await run_diagnostics(
        strategies=["single"], protocols=["json"], concurrency=1, case_seconds=2, suite_seconds=0.03
    )
    assert stopped and report["suite_deadline_reached"]
    assert report["status"] == "fail" and not report["coverage"]["single"]["complete"]
    assert all(row["status"] == "not_run" for row in report["probes"])
    assert len(report["probes"]) == len(plan_probes(strategies=["single"], protocols=["json"]))


async def test_live_mode_uses_one_binding_and_only_canaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def resolve(spec: ModelSpec) -> FunctionModel:
        calls.append(spec.name)
        return ScriptedProvider("single", "healthy", False).model

    monkeypatch.setattr(runner, "resolve_model", resolve)
    spec = ModelSpec(name="one-endpoint", model="test", model_class="baseline")
    report = await run_diagnostics(strategies=["single"], protocols=["json"], live_model=spec)
    assert report["status"] == "pass"
    assert calls == ["one-endpoint"]
    assert report["mode"] == "live_canary" and report["counts"] == {"pass": 1}
    assert report["probes"][0]["scenario"] == "healthy"
    assert "public_data_transferred" not in report["probes"][0]["checks"]


def test_live_binding_selection_does_not_expand_the_benchmark_matrix(tmp_path: Path) -> None:
    path = tmp_path / "many-models.json"
    path.write_text(
        json.dumps(
            {
                "models": [{"name": n, "model": n, "model_class": "baseline"} for n in ("a", "b")],
                "seeds": list(range(1000)),
                "repetitions": 999,
            }
        )
    )
    with pytest.raises(ValueError, match="select exactly one"):
        select_model(path, None)
    selected = select_model(path, "b")
    assert selected is not None and selected.model == "b"
    with pytest.raises(ValueError, match="--live-config"):
        select_model(None, "b")


def test_cli_persists_evidence_and_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "diagnostic.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "suite",
            "--no-env-file",
            "diagnose",
            "--quiet",
            "--strategies",
            "single",
            "--protocols",
            "json",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exit_:
        main()
    assert exit_.value.code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "pass"
    original = output.read_bytes()
    entries = [json.loads(line) for line in original.splitlines()]
    assert entries[0]["manifest"]["purpose"] == "infrastructure_acceptance"
    assert entries[-1]["summary"]["status"] == "pass"
    assert len([e for e in entries if "probe" in e]) == len(
        entries[0]["manifest"]["planned_probes"]
    )
    with pytest.raises(SystemExit) as exit_:
        main()
    assert exit_.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
async def test_invalid_deadlines_are_rejected(limit: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        await run_diagnostics(suite_seconds=limit)


def test_new_execution_strategies_require_explicit_diagnostic_coverage() -> None:
    assert set(ADAPTERS) == {p.adapter for p in plan_probes()}


async def test_wrong_candidate_fallback_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ADAPTERS["speculative-json"]

    async def replace_fallback(*args: Any) -> Answer:
        await original(*args)
        # Pretend a controller chose the better second candidate, leaving its provenance
        # label untouched. The diagnostic must check the exact committed artifact.
        return Answer(values=TARGET)

    monkeypatch.setitem(ADAPTERS, "speculative-json", replace_fallback)
    row = await run_probe(Probe("speculative", "json", "reliable-v2", "reconcile_timeout"))
    assert row["checks"]["distinct_candidates_committed"]
    assert row["checks"]["committed_candidate_retained"]
    assert row["status"] == "fail" and "fallback_artifact_identity" in row["findings"]
