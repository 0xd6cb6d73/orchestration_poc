from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.evaluation.suite import runner
from poc.evaluation.suite.models import Answer, ArchitectureOptions, Matrix, ModelSpec
from poc.evaluation.suite.tasks import generate
from poc.execution.review import ReviewDecision, review_decision_submission


@pytest.mark.parametrize("action", ["accept", "decline"])
async def test_decision_retains_immutable_submission(action: Literal["accept", "decline"]) -> None:
    draft = {"jobs": [1, 2]}

    async def reviewer(submission: dict[str, list[int]]) -> ReviewDecision[dict[str, list[int]]]:
        submission["jobs"].clear()
        return ReviewDecision(action=action, reason="explicit decision")

    result = await review_decision_submission(draft, reviewer)
    assert result.output == {"jobs": [1, 2]}
    assert result.output is not draft
    assert result.source == "draft_" + action
    assert result.reviewer_error is None


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "revise", "reason": "missing replacement"},
        {"action": "accept", "reason": "rewrite", "replacement": {"values": {}}},
        {"action": "decline", "reason": "rewrite", "replacement": {"values": {}}},
        {"action": "accept", "reason": "", "replacement": None},
        {"values": {}},
    ],
)
def test_protocol_rejects_ambiguous_decisions(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReviewDecision[Answer].model_validate(payload)


@pytest.mark.parametrize("cancel", [False, True])
async def test_decision_timeout_recovery_and_external_cancellation(cancel: bool) -> None:
    async def reviewer(submission: str) -> ReviewDecision[str]:
        if cancel:
            raise asyncio.CancelledError()
        raise TimeoutError()

    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await review_decision_submission("draft", reviewer, return_draft_on_error=True)
    else:
        result = await review_decision_submission("draft", reviewer, return_draft_on_error=True)
        assert result.source == "draft_fallback" and result.output == "draft"


def bind(model: FunctionModel, spec: ModelSpec) -> FunctionModel:
    return model


@pytest.mark.parametrize("strategy", ["review", "review-json"])
@pytest.mark.parametrize("action", ["accept", "revise", "decline"])
async def test_decisions_through_adapter_preserve_actual_outputs(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    action: str,
) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: Any, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        # A fully feasible draft followed by an explicitly wrong revision verifies
        # that orchestration never selects the answer using grader scores.
        payload: dict[str, Any] = (
            {"values": case.expected}
            if calls == 1
            else {
                "action": action,
                "reason": "review decision",
                "replacement": {"values": {}} if action == "revise" else None,
            }
        )
        return ModelResponse(
            parts=[TextPart(json.dumps(payload))]
            if strategy == "review-json"
            else [ToolCallPart(info.output_tools[0].name, payload)]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(bind, FunctionModel(model)))
    cfg = Matrix(
        models=[ModelSpec(name="test", model_class="baseline", model="test")],
        strategies=[strategy],
        architecture_options={
            strategy: ArchitectureOptions(
                artifact_contract="legacy-v1",
                review_protocol="decision-v1",
                review_failure_policy="return_submitted_draft",
            )
        },
    )
    result = await runner.run_trial(case, cfg.models[0], strategy, 1, cfg)
    assert calls == 2
    assert result["status"] == "completed" and not result["recovered"]
    assert result["review_decision"]["action"] == action
    assert result["answer"]["values"] == ({} if action == "revise" else case.expected)
    assert result["scores"]["exact"] == (0 if action == "revise" else 1)
    assert result["answer_source"] == ("review" if action == "revise" else "draft_" + action)
    assert not result["validation_calls"]


@pytest.mark.parametrize("strategy", ["review", "review-json"])
async def test_invalid_decisions_have_bounded_retries_and_explicit_recovery(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    case = generate("scheduling", 0, "dev", "standard")
    calls = 0

    async def model(messages: Any, info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        payload: dict[str, Any] = (
            {"values": {}}
            if calls == 1
            else {"action": "revise", "reason": "no replacement", "replacement": None}
        )
        return ModelResponse(
            parts=[TextPart(json.dumps(payload))]
            if strategy == "review-json"
            else [ToolCallPart(info.output_tools[0].name, payload)]
        )

    monkeypatch.setattr(runner, "resolve_model", partial(bind, FunctionModel(model)))
    cfg = Matrix(
        models=[ModelSpec(name="test", model_class="baseline", model="test")],
        strategies=[strategy],
        architecture_options={
            strategy: ArchitectureOptions(
                artifact_contract="legacy-v1",
                review_protocol="decision-v1",
                output_retries=1,
                review_failure_policy="return_submitted_draft",
            )
        },
    )
    result = await runner.run_trial(case, cfg.models[0], strategy, 1, cfg)
    assert calls == 3  # draft, invalid decision, one retry
    assert result["recovered"] and result["answer_source"] == "draft_fallback"
    assert result["review_decision"] is None
    assert result["scores"]["exact"] == 0
