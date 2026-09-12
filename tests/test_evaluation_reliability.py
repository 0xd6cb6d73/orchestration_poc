from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.evaluation.suite import runner
from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.diagnostics import diagnose
from poc.evaluation.suite.models import Answer, ArchitectureOptions, Budget, Matrix, ModelSpec
from poc.evaluation.suite.phoenix import publish, reconcile
from poc.evaluation.suite.reports import comparisons, locked, records
from poc.evaluation.suite.runner import read_report, run_matrix, run_trial
from poc.evaluation.suite.tasks import generate


def _model_for_spec(model: FunctionModel, spec: ModelSpec) -> FunctionModel:
    return model


def matrix(**kwargs: Any) -> Matrix:
    return Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="test")],
        strategies=["single"],
        families=["scheduling"],
        seeds=[0],
        repetitions=1,
        **kwargs,
    )


async def test_baseline_has_no_extra_tools_or_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert [t.name for t in info.function_tools] == ["query"]
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": {"error": "bad SQL"}})]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix()
    t = await run_trial(
        generate("scheduling", 0, "dev", "standard"), cfg.models[0], "single", 1, cfg
    )
    assert t["status"] == "completed" and t["requests"] == 1
    assert t["scores"]["exact"] == 0 and t["diagnostics"]["answer_valid"] is False
    assert t["candidates"][0]["answer"]["values"] == {"error": "bad SQL"}


async def test_explicit_output_validation_repairs_shape_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        # A full but infeasible answer must still be scored normally, without repair hints.
        values = {} if calls == 1 else dict.fromkeys(case.expected, 0)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"values": values})])

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix(architecture_options={"single": ArchitectureOptions(output_validation=True)})
    t = await run_trial(case, cfg.models[0], "single", 1, cfg)
    assert calls == 2 and t["status"] == "completed"
    assert t["diagnostics"]["answer_valid"] is True and t["scores"]["exact"] == 0


async def test_checkpoint_survives_timeout_without_becoming_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = generate("scheduling", 0, "dev", "standard")

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("submit_candidate", {"values": case.expected})]
            )
        await asyncio.sleep(5)
        raise AssertionError("cancelled request should never finish")

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix(
        budget=Budget(seconds=0.2),
        architecture_options={"single": ArchitectureOptions(candidate_submission=True)},
    )
    path = tmp_path / "timeout.jsonl"
    report = await run_matrix(cfg, path)
    t = report["trials"][0]
    assert t["status"] == "timeout" and t["scores"]["exact"] == 0
    assert t["best_candidate"]["diagnostics"]["feasible"] is True
    assert t["answer"] == {"values": {}} and t["usage_complete"] is False
    assert t["request_attempts"] == 2 and t["request_responses"] == 1
    assert any("candidate" in event for event in read_report(path)["events"])


async def test_finalization_requires_model_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("submit_candidate", {"values": case.expected})]
            )
        if calls == 2:
            await asyncio.sleep(5)
        assert not info.function_tools
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": case.expected})]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix(
        budget=Budget(seconds=0.5, finalization_seconds=0.25),
        architecture_options={"single": ArchitectureOptions(candidate_submission=True)},
    )
    t = await run_trial(case, cfg.models[0], "single", 1, cfg)
    assert calls == 3 and t["status"] == "completed" and t["scores"]["exact"] == 1
    assert [p["name"] for p in t["phases"]] == ["solve", "finalize"]
    assert t["exhausted_phase"] == "solve"


async def test_reviewer_failure_preserves_draft(monkeypatch: pytest.MonkeyPatch) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("reviewer unavailable")
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": case.expected})]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix().model_copy(update={"strategies": ["review"]})
    t = await run_trial(case, cfg.models[0], "review", 1, cfg)
    assert t["review_started"] and t["status"] == "error"
    assert t["best_candidate"]["diagnostics"]["feasible"] is True
    assert t["scores"]["exact"] == 0 and not t["usage_complete"]


async def test_request_timeout_is_distinguished(monkeypatch: pytest.MonkeyPatch) -> None:
    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(5)
        raise AssertionError

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix(budget=Budget(seconds=1, request_timeout_seconds=0.02))
    t = await run_trial(
        generate("scheduling", 0, "dev", "standard"), cfg.models[0], "single", 1, cfg
    )
    assert t["status"] == "request_timeout" and t["exhausted_phase"] == "solve"
    assert t["request_attempts"] == 1 and t["request_responses"] == 0


def test_public_diagnostics_and_shared_validation_budget() -> None:
    from poc.evaluation.suite.environment import TaskEnvironment

    case = generate("routing", 4, "dev", "standard")
    answer = {f"stop-{n:03}": {"vehicle": n % 4, "position": n // 4 + 1} for n in range(24)}
    d = diagnose(case.input, Answer(values=answer))
    assert d["dimensions"]["service_window"] == {"passed": 5, "total": 24}
    assert d["dimensions"]["capacity"] == {"passed": 2, "total": 4}
    assert d["dimensions"]["return_deadline"] == {"passed": 1, "total": 4}
    assert d["scores"]["fraction_correct"] == 1 / 3
    env = TaskEnvironment(case.input, 2)
    try:
        assert env.submit_candidate(case.expected) == {"recorded": True}
        assert env.validate_candidate(answer)["feasible"] is False
        with pytest.raises(RuntimeError, match="budget"):
            env.query("SELECT 1")
    finally:
        env.close()


async def test_execution_resume_preserves_failed_trials_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = matrix().model_copy(
        update={"strategies": ["resume-test"], "repetitions": 4, "max_concurrency": 1}
    )
    count = 0

    async def adapter(*args: Any) -> Answer:
        nonlocal count
        count += 1
        if count == 1:
            raise RuntimeError("record this failure")
        if count == 3:
            raise asyncio.CancelledError
        return Answer(values={})

    monkeypatch.setitem(ADAPTERS, "resume-test", adapter)
    path = tmp_path / "resume.jsonl"
    # A child cancellation leaves no terminal record for the interrupted trial.
    with pytest.raises(RuntimeError, match="workers interrupted"):
        await run_matrix(cfg, path)
    assert len(read_report(path)["trials"]) == 2
    with path.open("ab") as f:
        f.write(b'{"trial":')
    capsys.readouterr()
    await run_matrix(cfg, path, resume=True)
    progress = capsys.readouterr().err.splitlines()
    assert progress[0] == "Progress: 2/4 finished, 0 running"
    assert progress[-1] == "Progress: 4/4 finished, 0 running"
    assert count == 5
    report = read_report(path)
    assert report["coverage"] == {"expected": 4, "recorded": 4, "missing": 0, "complete": True}
    assert sum(t["status"] == "error" for t in report["trials"]) == 1
    assert path.with_suffix(".jsonl.interrupted").exists()
    await run_matrix(cfg, path, resume=True)
    assert capsys.readouterr().err.strip() == "Progress: 4/4 finished, 0 running"
    assert count == 5
    with pytest.raises(ValueError, match="config changed"):
        await run_matrix(cfg.model_copy(update={"max_concurrency": 2}), path, resume=True)
    monkeypatch.setattr(runner, "implementation_digest", lambda: "different")
    with pytest.raises(ValueError, match="implementation_sha256 changed"):
        await run_matrix(cfg, path, resume=True)


def test_corruption_and_concurrent_writer_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.jsonl"
    path.write_text('{"manifest": {}}\ninvalid\n')
    with pytest.raises(ValueError, match="corrupt"):
        records(path, repair=True)
    with locked(path), pytest.raises(RuntimeError, match="another process"), locked(path):
        pass


def test_comparisons_only_use_matched_trials() -> None:
    trials = [
        {
            "model_name": "m",
            "family": "f",
            "strategy": strategy,
            "case_id": case,
            "repetition": 1,
            "scores": {"exact": score},
            "elapsed_seconds": 1,
        }
        for strategy, case, score in [("single", "a", 0), ("review", "a", 1), ("single", "b", 1)]
    ]
    row = comparisons(trials)[0]
    assert row["matched_trials"] == 1 and row["unmatched_trials"] == 1
    assert row["pass_rate_delta"] == 1


class FakePhoenix:
    def __init__(self, fail: str):
        self.datasets = self.experiments = self
        self._client = self
        self.dataset: Any = None
        self.exps: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.evaluations: dict[tuple[str, str], Any] = {}
        self.fail = fail

    def lost_response(self, stage: str) -> None:
        if self.fail == stage:
            self.fail = ""
            raise ConnectionError("server accepted the write; response lost")

    def list(self, **kwargs: Any) -> list[dict[str, Any]]:
        if "dataset_id" in kwargs:
            return self.exps
        return [{"id": "dataset", "name": self.dataset.name}] if self.dataset else []

    def create_dataset(self, **kwargs: Any) -> Any:
        assert self.dataset is None
        self.dataset = SimpleNamespace(
            id="dataset",
            version_id="version",
            name=kwargs["name"],
            examples=[
                {"id": f"example-{n}", "metadata": m} for n, m in enumerate(kwargs["metadata"])
            ],
        )
        self.lost_response("dataset")
        return self.dataset

    def get_dataset(self, **kwargs: Any) -> Any:
        return self.dataset

    def create(self, **kwargs: Any) -> dict[str, Any]:
        exp = {
            "id": f"exp-{len(self.exps)}",
            "name": kwargs["experiment_name"],
            "dataset_id": kwargs["dataset_id"],
            "metadata": kwargs["experiment_metadata"],
        }
        self.exps.append(exp)
        self.lost_response("experiment")
        return exp

    def get(self, path: str = "", **kwargs: Any) -> Any:
        if "experiment_id" in kwargs:
            return next(e for e in self.exps if e["id"] == kwargs["experiment_id"])
        exp = path.split("/")[2]
        runs = [r for r in self.runs if r["experiment_id"] == exp]
        offset = int(kwargs["params"].get("cursor", "0"))
        # Deliberately page at one run to exercise pagination.
        page = {
            "data": runs[offset : offset + 1],
            "next_cursor": str(offset + 1) if offset + 1 < len(runs) else None,
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: page)

    def log_run(self, **kwargs: Any) -> dict[str, Any]:
        assert not any(
            (r["experiment_id"], r["dataset_example_id"], r["repetition_number"])
            == (kwargs["experiment_id"], kwargs["dataset_example_id"], kwargs["repetition_number"])
            for r in self.runs
        )
        run = {**kwargs, "id": f"run-{len(self.runs)}", "output": {"task_output": kwargs["output"]}}
        self.runs.append(run)
        self.lost_response("run")
        return run

    def log_evaluation(self, **kwargs: Any) -> None:
        self.evaluations[kwargs["experiment_run_id"], kwargs["name"]] = kwargs
        self.lost_response("evaluation")


@pytest.mark.parametrize("stage", ["dataset", "experiment", "run", "evaluation"])
async def test_publication_recovers_lost_responses_without_duplicates(
    tmp_path: Path, stage: str
) -> None:
    cfg = matrix().model_copy(
        update={"strategies": ["sql-baseline"], "families": ["ledger"], "repetitions": 2}
    )
    report = await run_matrix(cfg, tmp_path / "report.jsonl")
    client = FakePhoenix(stage)
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ConnectionError):
        publish(report, client, receipt_path=receipt_path)
    receipt = publish(report, client, receipt_path=receipt_path)
    publish(report, client, receipt_path=receipt_path)
    assert len(client.exps) == 1 and len(client.runs) == 2
    assert receipt["published_runs"] == 2
    assert len(client.evaluations) == 6
    check = reconcile(report, receipt, client)
    assert check["unpublished"] == check["mismatched"] == []
    report["config"]["max_concurrency"] = 99
    with pytest.raises(ValueError, match="different report"):
        publish(report, client, receipt_path=receipt_path)


@pytest.mark.parametrize("strategy", ["single", "review"])
async def test_cumulative_request_reserves_and_output_cap(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert info.model_settings and info.model_settings.get("max_tokens") == 17
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("submit_candidate", {"values": case.expected})]
            )
        if (strategy == "single" and calls <= 3) or (strategy == "review" and calls == 2):
            return ModelResponse(parts=[ToolCallPart("query", {"sql": "SELECT 1"})])
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": case.expected})]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    budget = Budget(
        requests=4,
        max_output_tokens=17,
        finalization_requests=1 if strategy == "single" else 0,
        draft_fraction=0.5 if strategy == "review" else 1,
    )
    cfg = matrix(budget=budget).model_copy(
        update={
            "strategies": [strategy],
            "architecture_options": {strategy: ArchitectureOptions(candidate_submission=True)},
        }
    )
    t = await run_trial(case, cfg.models[0], strategy, 1, cfg)
    assert t["status"] == "completed" and t["scores"]["exact"] == 1
    assert t["requests"] == (4 if strategy == "single" else 3)
    assert [p["name"] for p in t["phases"]] == (
        ["solve", "finalize"] if strategy == "single" else ["draft", "review"]
    )
    assert t["phases"][0]["status"] == "budget_exhausted"


async def test_json_candidate_and_validation_are_explicit_and_budgeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai import TextPart

    case = generate("routing", 1, "dev", "standard")
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        action = {1: "candidate", 2: "validate"}.get(calls, "values")
        return ModelResponse(parts=[TextPart(json.dumps({action: case.expected}))])

    monkeypatch.setattr(runner, "resolve_model", partial(_model_for_spec, FunctionModel(model)))
    cfg = matrix().model_copy(
        update={
            "strategies": ["single-json"],
            "architecture_options": {
                "single-json": ArchitectureOptions(
                    candidate_submission=True, constraint_feedback=True
                )
            },
        }
    )
    t = await run_trial(case, cfg.models[0], "single-json", 1, cfg)
    assert t["scores"]["exact"] == 1 and t["total_tool_calls"] == 2
    assert [c["source"] for c in t["candidates"]] == ["submitted", "validated", "output"]


def test_suite_telemetry_does_not_instrument_provider_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    import phoenix.otel
    from openinference.instrumentation.langchain import LangChainInstrumentor

    from poc.telemetry import bootstrap

    provider = object()

    def register(**kwargs: Any) -> Any:
        assert kwargs["auto_instrument"] is False
        return provider

    def instrument(*args: Any, **kwargs: Any) -> None:
        pass

    def sdk(**kwargs: Any) -> bool:
        raise AssertionError("suite must not double-instrument provider requests")

    monkeypatch.setenv("PHOENIX_ENABLED", "1")
    monkeypatch.setattr(phoenix.otel, "register", register)
    monkeypatch.setattr(LangChainInstrumentor, "instrument", instrument)
    monkeypatch.setattr(bootstrap, "_instrument_optional_provider", sdk)
    assert bootstrap.configure_telemetry(instrument_providers=False) is provider
