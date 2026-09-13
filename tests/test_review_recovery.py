from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.evaluation.suite import runner
from poc.evaluation.suite.models import Answer, ArchitectureOptions, Budget, Matrix, ModelSpec
from poc.evaluation.suite.tasks import generate
from poc.execution.review import review_submission


async def test_execution_retains_submission_without_a_grader() -> None:
    draft = {"arbitrary": ["possibly wrong"]}

    async def reviewer(value: dict[str, list[str]]) -> dict[str, list[str]]:
        value["arbitrary"].clear()
        raise TimeoutError()

    result = await review_submission(draft, reviewer, return_draft_on_error=True)
    assert result.output == draft and result.source == "draft_fallback"
    assert result.output is not draft
    assert result.reviewer_error == "TimeoutError"


@pytest.mark.parametrize("error", [RuntimeError("bug"), asyncio.CancelledError()])
async def test_execution_propagates_bugs_and_cancellation(error: BaseException) -> None:
    async def reviewer(value: str) -> str:
        raise error

    with pytest.raises(type(error)):
        await review_submission("draft", reviewer, return_draft_on_error=True)


async def test_execution_preserves_default_failure_and_success_semantics() -> None:
    async def failed(value: str) -> str:
        raise TimeoutError()

    async def successful(value: str) -> str:
        return "review output"

    with pytest.raises(TimeoutError):
        await review_submission("draft", failed)
    result = await review_submission("draft", successful, return_draft_on_error=True)
    assert result.output == "review output" and result.source == "review"


def bind(model: FunctionModel, spec: ModelSpec) -> FunctionModel:
    return model


@pytest.mark.parametrize("strategy", ["review", "review-json"])
@pytest.mark.parametrize("valid", [True, False])
async def test_harness_grades_whatever_execution_returns(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    valid: bool,
) -> None:
    import json

    case = generate("scheduling", 0, "dev", "standard")
    values = case.expected if valid else {}
    calls = 0

    async def model(messages: Any, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise UnexpectedModelBehavior("empty response")
        output = {"values": values}
        return ModelResponse(
            parts=[TextPart(json.dumps(output))]
            if strategy == "review-json"
            else [ToolCallPart(info.output_tools[0].name, output)]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(bind, FunctionModel(model)))
    cfg = Matrix(
        models=[ModelSpec(name="test", model_class="baseline", model="test")],
        strategies=[strategy],
        architecture_options={
            strategy: ArchitectureOptions(
                artifact_contract="legacy-v1", review_failure_policy="return_submitted_draft"
            )
        },
    )
    result = await runner.run_trial(case, cfg.models[0], strategy, 1, cfg)
    assert result["status"] == "completed" and result["recovered"]
    assert result["scores"]["exact"] == float(valid)
    assert result["answer"]["values"] == values
    assert result["answer_source"] == "draft_fallback"


@pytest.mark.parametrize("protocol", ["replace", "decision-v1"])
async def test_hard_harness_deadline_never_selects_a_candidate(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: Any, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls > 1:
            await asyncio.sleep(10)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": case.expected})]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(bind, FunctionModel(model)))
    cfg = Matrix(
        models=[ModelSpec(name="test", model_class="baseline", model="test")],
        strategies=["review"],
        budget=Budget(seconds=0.3),
        architecture_options={
            "review": ArchitectureOptions.model_validate(
                {
                    "review_failure_policy": "return_submitted_draft",
                    "review_protocol": protocol,
                    "return_reserve_seconds": 0,
                }
            )
        },
    )
    result = await runner.run_trial(case, cfg.models[0], "review", 1, cfg)
    assert result["status"] == "timeout" and not result["recovered"]
    assert result["answer"] == Answer(values={}).model_dump()
    assert result["best_candidate"]["diagnostics"]["feasible"]
    assert result["scores"]["exact"] == 0


async def test_execution_review_uses_different_model_callback() -> None:
    seen: list[str] = []

    async def draft_model() -> str:
        seen.append("drafter")
        return "draft"

    async def reviewer_model(value: str) -> str:
        seen.append("reviewer")
        assert value == "draft"
        return "revised"

    result = await review_submission(await draft_model(), reviewer_model)
    assert seen == ["drafter", "reviewer"] and result.output == "revised"


async def test_heterogeneous_models_settings_usage_and_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.usage import RequestUsage

    from poc.execution import sql_strategy

    case = generate("scheduling", 0, "dev", "standard")
    seen: list[str] = []

    async def drafter(messages: Any, info: AgentInfo) -> ModelResponse:
        seen.append("drafter")
        assert info.model_settings and info.model_settings.get("temperature") == 0
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": case.expected})],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    async def reviewer(messages: Any, info: AgentInfo) -> ModelResponse:
        seen.append("reviewer")
        assert info.model_settings and info.model_settings.get("temperature") == 0.2
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"values": case.expected})],
            usage=RequestUsage(input_tokens=20, output_tokens=7),
        )

    primary = ModelSpec(
        name="draft",
        model_class="baseline",
        model="draft-model",
        input_usd_per_million=1,
        output_usd_per_million=2,
    )
    secondary = ModelSpec(
        name="reviewer",
        model_class="baseline",
        model="review-model",
        settings={"temperature": 0.2},
        input_usd_per_million=3,
        output_usd_per_million=4,
    )
    monkeypatch.setattr(runner, "resolve_model", partial(bind, FunctionModel(drafter)))
    monkeypatch.setattr(sql_strategy, "resolve_model", partial(bind, FunctionModel(reviewer)))
    cfg = Matrix(
        models=[primary],
        strategies=["review"],
        architecture_options={
            "review": ArchitectureOptions(
                artifact_contract="legacy-v1", phase_models={"review": secondary}
            )
        },
    )
    result = await runner.run_trial(case, primary, "review", 1, cfg)
    assert seen == ["drafter", "reviewer"]
    assert result["requests"] == 2
    assert result["input_tokens"] == 30 and result["output_tokens"] == 12
    assert abs(result["estimated_cost_usd"] - (10 + 10 + 60 + 28) / 1_000_000) < 1e-12
    assert result["phases"][1]["model"] == "review-model"
    assert result["architecture_options"]["phase_models"]["review"]["model"] == "review-model"


def test_execution_implementation_has_no_evaluation_imports() -> None:
    import ast
    from pathlib import Path

    for name in ("review.py", "sql_strategy.py", "sql_ports.py", "sql_contracts.py"):
        tree = ast.parse((Path(__file__).parents[1] / "poc" / "execution" / name).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(module and module.startswith("poc.evaluation") for module in imports)


def test_checkpoint_port_does_not_expose_grading_diagnostics() -> None:
    from poc.evaluation.suite.runtime import TrialState

    case = generate("scheduling", 0, "dev", "standard")
    state = TrialState(case.input)
    checkpoint = state.candidate(Answer(values={}), "submitted")
    assert "diagnostics" not in checkpoint
    assert "diagnostics" not in state.candidates[0]
    assert state.diagnostic_candidates[0]["diagnostics"]["answer_valid"] is False
