from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS, review_json, single_json, sql_baseline
from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Budget, Matrix, ModelSpec, grade
from poc.evaluation.suite.phoenix import publish
from poc.evaluation.suite.runner import read_report, run_matrix
from poc.evaluation.suite.scheduling import grade_schedule
from poc.evaluation.suite.tasks import generate


@pytest.mark.parametrize("family", ["ledger", "dependencies", "access"])
@pytest.mark.parametrize("difficulty", ["standard", "hard", "stress"])
async def test_independent_solver_agrees_with_generator(family: str, difficulty: str) -> None:
    for seed in range(4):
        case = generate(family, seed, "test", difficulty)
        env = TaskEnvironment(case.input, 200)
        try:
            output = await sql_baseline(case.input, env, "test", {}, Budget(), RunUsage())
            assert grade(case, output)["exact"] == 1
            assert "expected" not in env.schema
        finally:
            env.close()


def test_seed_split_and_difficulty_identity() -> None:
    case = generate("ledger", 42, "dev", "standard")
    assert case.digest == generate("ledger", 42, "dev", "standard").digest
    assert case.digest != generate("ledger", 42, "test", "standard").digest
    assert case.digest != generate("ledger", 42, "dev", "hard").digest
    answer = dict(case.expected)
    answer[next(iter(answer))] = True
    assert grade(case, Answer(values=answer))["exact"] == 0
    answer = {**case.expected, "extra": 0}
    assert grade(case, Answer(values=answer))["fraction_correct"] < 1
    assert grade(case, Answer(values={}))["fraction_correct"] == 0


@pytest.mark.parametrize("difficulty", ["standard", "hard", "stress"])
def test_scheduling_feasibility_and_invalid_answers(difficulty: str) -> None:
    for seed in range(20):
        case = generate("scheduling", seed, "test", difficulty)
        assert grade_schedule(case, Answer(values=case.expected))["exact"] == 1
        assert grade_schedule(case, Answer(values={}))["exact"] == 0
        assert grade_schedule(case, Answer(values=dict.fromkeys(case.expected, 0)))["exact"] == 0
        late = {k: v + 10000 for k, v in case.expected.items()}
        assert grade_schedule(case, Answer(values=late))["exact"] == 0


def test_schedule_accepts_non_reference_feasible_solution() -> None:
    case = generate("scheduling", 3, "dev", "standard")
    for name in case.expected:
        alternate = {**case.expected, name: case.expected[name] + 1}
        if grade_schedule(case, Answer(values=alternate))["exact"]:
            assert alternate != case.expected
            break
    else:
        pytest.fail("test witness has no one-unit feasible perturbation")


def test_environment_blocks_side_effects_bounds_queries_and_counts_errors() -> None:
    env = TaskEnvironment(generate("ledger", 0, "dev", "standard").input, 8)
    try:
        for sql in [
            "DROP TABLE invoices",
            "ATTACH DATABASE '/tmp/leak' AS leak",
            "PRAGMA database_list",
            "SELECT load_extension('/tmp/x')",
            "SELECT hex(zeroblob(1000000000))",
        ]:
            assert "error" in env.query(sql)
        result = env.query("SELECT * FROM payments")
        assert result["truncated"] and len(result["rows"]) == 200
        assert "error" in env.query(
            "WITH RECURSIVE n(x) AS (SELECT 0 UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n"
        )
        assert "rows" in env.query("SELECT COUNT(*) FROM invoices")
        with pytest.raises(RuntimeError, match="budget"):
            env.query("SELECT 1")
    finally:
        env.close()


async def test_model_json_tool_loop_and_shared_review_budget() -> None:
    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert not info.function_tools
        text = '{"sql":"SELECT COUNT(*) FROM invoices"}' if calls == 1 else '{"values":{}}'
        return ModelResponse(parts=[TextPart(text)])

    case = generate("ledger", 0, "dev", "standard")
    env = TaskEnvironment(case.input, 10)
    usage = RunUsage()
    try:
        result = await single_json(case.input, env, FunctionModel(model), {}, Budget(), usage)
        assert result.values == {}
        assert usage.requests == 2 and len(env.calls) == 1
        with pytest.raises(Exception, match="request_limit"):
            await review_json(case.input, env, FunctionModel(model), {}, Budget(requests=3), usage)
        assert usage.requests == 3  # Review cannot reset the trial-wide budget.
    finally:
        env.close()


async def test_runner_keeps_failed_trials_and_survives_partial_report(tmp_path: Path) -> None:
    async def timeout(*args: Any) -> Answer:
        await asyncio.sleep(0.1)
        return Answer(values={})

    ADAPTERS["test-timeout"] = timeout
    config = Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="offline")],
        strategies=["test-timeout", "sql-baseline"],
        families=["ledger"],
        seeds=[1],
        repetitions=1,
        budget=Budget(seconds=0.01),
    )
    path = tmp_path / "trials.jsonl"
    try:
        report = await run_matrix(config, path)
    finally:
        del ADAPTERS["test-timeout"]
    assert {t["status"] for t in report["trials"]} == {"completed", "timeout"}
    assert next(t for t in report["trials"] if t["status"] == "timeout")["scores"]["exact"] == 0
    assert read_report(path)["summaries"] == report["summaries"]
    # A killed run remains readable without a final summary record.
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    assert len(read_report(path)["trials"]) == 2


async def test_phoenix_publication_links_examples_and_scores(tmp_path: Path) -> None:
    config = Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="offline")],
        strategies=["sql-baseline"],
        families=["ledger"],
        seeds=[1],
        repetitions=1,
    )
    report = await run_matrix(config, tmp_path / "report.jsonl")
    recorded: dict[str, Any] = {}

    class Client:
        def __init__(self) -> None:
            self.datasets = self
            self.experiments = self
            self.id = "dataset"
            self.version_id = "version"
            self.examples: list[dict[str, Any]] = []

        def create_dataset(self, **kwargs: Any) -> Client:
            recorded["dataset"] = kwargs
            self.examples = [{"id": "example", "metadata": kwargs["metadata"][0]}]
            return self

        def create(self, **kwargs: Any) -> dict[str, str]:
            recorded["experiment"] = kwargs
            return {"id": "experiment"}

        def log_run(self, **kwargs: Any) -> dict[str, str]:
            recorded["run"] = kwargs
            return {"id": "run"}

        def log_evaluation(self, **kwargs: Any) -> None:
            recorded[kwargs["name"]] = kwargs

    receipt = publish(report, Client())
    assert receipt["dataset_version_id"] == "version"
    assert recorded["run"]["dataset_example_id"] == "example"
    assert recorded["experiment"]["experiment_metadata"]["max_concurrency"] == 4
    assert recorded["exact"]["score"] == 1
    assert recorded["fraction_correct"]["experiment_run_id"] == "run"
    assert "expected" not in json.dumps(recorded["dataset"]["inputs"])
    report["cases"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="generator changed"):
        publish(report, Client())


@pytest.mark.parametrize("difficulty", ["standard", "hard", "stress"])
def test_routing_feasibility_and_invalid_assignments(difficulty: str) -> None:
    from poc.evaluation.suite.routing import grade_routing

    for seed in range(20):
        case = generate("routing", seed, "test", difficulty)
        assert grade_routing(case, Answer(values=case.expected))["exact"] == 1
        assert grade_routing(case, Answer(values={}))["exact"] == 0
        bad = {k: {"vehicle": 0, "position": 1} for k in case.expected}
        assert grade_routing(case, Answer(values=bad))["exact"] == 0
        assert grade_routing(case, Answer(values=dict.fromkeys(case.expected, True)))["exact"] == 0


async def test_native_tool_adapter_runs_query_before_structured_answer() -> None:
    from pydantic_ai import ToolCallPart

    from poc.evaluation.suite.adapters import single

    calls = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                parts=[ToolCallPart("query", {"sql": "SELECT COUNT(*) FROM jobs"})]
            )
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"values": {}})])

    case = generate("dependencies", 0, "dev", "standard")
    env = TaskEnvironment(case.input, 2)
    usage = RunUsage()
    try:
        await single(case.input, env, FunctionModel(model), {}, Budget(), usage)
        assert len(env.calls) == 1
        assert env.calls[0]["result"]["rows"] == [[80]]
        assert usage.requests == 2
    finally:
        env.close()


def test_dotenv_loads_local_credentials_and_preserves_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from poc.evaluation.suite.__main__ import load_environment

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://exported.example/v1")
    (tmp_path / ".env").write_text(
        'export OPENAI_API_KEY="test-key#${LITERAL}"\nOPENAI_BASE_URL=https://file.example/v1\n'
    )
    load_environment()
    assert os.environ["OPENAI_API_KEY"] == "test-key#${LITERAL}"
    assert os.environ["OPENAI_BASE_URL"] == "https://exported.example/v1"


def test_dotenv_explicit_file_and_missing_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from poc.evaluation.suite.__main__ import load_environment

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_environment()  # Missing default is allowed for CI and exported-only usage.
    path = tmp_path / "models.env"
    with pytest.raises(ValueError, match="does not exist"):
        load_environment(path)
    path.write_text("OPENAI_API_KEY='explicit-key'\n")
    load_environment(path)
    assert os.environ["OPENAI_API_KEY"] == "explicit-key"


@pytest.mark.parametrize("concurrency", [1, 2, 3])
async def test_matrix_bounds_parallel_trials_and_isolates_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    concurrency: int,
) -> None:
    active = 0
    peak = 0
    started = 0
    full = asyncio.Event()
    release = asyncio.Event()
    environments: list[TaskEnvironment] = []
    usages: list[RunUsage] = []

    async def adapter(
        task: Any,
        env: TaskEnvironment,
        model: Any,
        settings: Any,
        budget: Budget,
        usage: RunUsage,
    ) -> Answer:
        nonlocal active, peak, started
        started += 1
        index = started
        active += 1
        peak = max(peak, active)
        environments.append(env)
        usages.append(usage)
        assert usage.requests == 0 and not env.calls
        usage.requests = index
        env.query("SELECT COUNT(*) FROM invoices")
        if active == concurrency:
            full.set()
        try:
            await release.wait()
            if index == 1:
                raise RuntimeError("one trial fails without cancelling peers")
            return Answer(values={})
        finally:
            active -= 1

    monkeypatch.setitem(ADAPTERS, "test-parallel", adapter)
    matrix = Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="offline")],
        families=["ledger"],
        seeds=[0],
        strategies=["test-parallel"],
        repetitions=6,
        max_concurrency=concurrency,
    )
    path = tmp_path / "parallel.jsonl"
    async with asyncio.timeout(3):
        task = asyncio.create_task(run_matrix(matrix, path))
        try:
            await full.wait()
            assert active == concurrency
            release.set()
            report = await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert peak == concurrency and active == 0
    assert len({id(env) for env in environments}) == 6
    assert len({id(usage) for usage in usages}) == 6
    assert sorted(t["repetition"] for t in report["trials"]) == list(range(1, 7))
    assert sorted(t["requests"] for t in report["trials"]) == list(range(1, 7))
    assert all(len(t["tool_calls"]) == 1 for t in report["trials"])
    assert sum(t["status"] == "error" for t in report["trials"]) == 1
    assert read_report(path)["trials"] == report["trials"]
    assert report["config"]["max_concurrency"] == concurrency


async def test_parallel_completion_is_flushed_before_slow_trial_and_cancellation_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlite3

    started = 0
    fast_finished = asyncio.Event()
    cancelled = asyncio.Event()
    environments: list[TaskEnvironment] = []

    async def adapter(
        task: Any,
        env: TaskEnvironment,
        model: Any,
        settings: Any,
        budget: Budget,
        usage: RunUsage,
    ) -> Answer:
        nonlocal started
        started += 1
        environments.append(env)
        if started == 1:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        fast_finished.set()
        return Answer(values={})

    monkeypatch.setitem(ADAPTERS, "test-completion", adapter)
    matrix = Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="offline")],
        families=["ledger"],
        seeds=[0],
        strategies=["test-completion"],
        repetitions=4,
        max_concurrency=2,
    )
    path = tmp_path / "partial.jsonl"
    async with asyncio.timeout(3):
        task = asyncio.create_task(run_matrix(matrix, path))
        try:
            await fast_finished.wait()
            assert not task.done()
            assert len(read_report(path)["trials"]) == 3
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert cancelled.is_set()
    assert len(read_report(path)["trials"]) == 3
    for env in environments:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            env.db.execute("SELECT 1")


def test_matrix_rejects_nonpositive_concurrency() -> None:
    from pydantic import ValidationError

    for concurrency in [0, -1]:
        with pytest.raises(ValidationError):
            Matrix(
                models=[ModelSpec(name="offline", model_class="baseline", model="offline")],
                max_concurrency=concurrency,
            )


def test_cli_concurrency_override_is_validated_and_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from poc.evaluation.suite.__main__ import main

    config = tmp_path / "config.json"
    matrix = Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="offline")],
        families=["ledger"],
        seeds=[0],
        strategies=["sql-baseline"],
        repetitions=1,
        max_concurrency=1,
    )
    config.write_text(matrix.model_dump_json())
    output = tmp_path / "trials.jsonl"
    argv = [
        "suite",
        "--no-env-file",
        "run",
        "--config",
        str(config),
        "--output",
        str(output),
        "--max-concurrency",
    ]
    monkeypatch.setenv("PHOENIX_ENABLED", "0")
    monkeypatch.setattr(sys, "argv", [*argv, "0"])
    with pytest.raises(SystemExit) as invalid:
        main()
    assert invalid.value.code == 2
    assert not output.exists()
    monkeypatch.setattr(sys, "argv", [*argv, "3"])
    with pytest.raises(SystemExit) as valid:
        main()
    assert valid.value.code == 0
    assert read_report(output)["config"]["max_concurrency"] == 3
