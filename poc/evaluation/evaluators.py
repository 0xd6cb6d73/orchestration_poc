from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from poc.evaluation.models import (
    AgentEvaluationExpected,
    AgentEvaluationInput,
    AgentEvaluationOutput,
)

AgentEvaluator = Evaluator[
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationExpected,
]
AgentEvaluatorContext = EvaluatorContext[
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationExpected,
]


@dataclass
class SuccessfulOutcome(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        expected = _expected(ctx)
        passed = ctx.output.outcome == expected.outcome
        return EvaluationReason(
            value=passed,
            reason=f"expected {expected.outcome}, got {ctx.output.outcome}",
        )


@dataclass
class ResultContainsExpected(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        expected = _expected(ctx).result_contains
        passed = _contains(ctx.output.result, expected)
        return EvaluationReason(
            value=passed,
            reason=(
                "result contains the expected fields"
                if passed
                else f"result does not contain {expected!r}"
            ),
        )


@dataclass
class RequiredToolsUsed(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        required = _expected(ctx).required_tools
        used = {call.tool_name for call in ctx.output.tool_calls}
        missing = required - used
        return EvaluationReason(
            value=not missing,
            reason=(
                "all required tools were used"
                if not missing
                else f"missing tools: {sorted(missing)}"
            ),
        )


@dataclass
class ForbiddenToolsAvoided(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        forbidden = _expected(ctx).forbidden_tools
        used = {call.tool_name for call in ctx.output.tool_calls}
        unexpected = forbidden & used
        return EvaluationReason(
            value=not unexpected,
            reason=(
                "no forbidden tools were used"
                if not unexpected
                else f"forbidden tools used: {sorted(unexpected)}"
            ),
        )


@dataclass
class ToolArgumentsContainExpected(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        expected = _expected(ctx).tool_arguments_contain
        mismatched = [
            tool_name
            for tool_name, arguments in expected.items()
            if not any(
                call.tool_name == tool_name and _contains(call.arguments, arguments)
                for call in ctx.output.tool_calls
            )
        ]
        return EvaluationReason(
            value=not mismatched,
            reason=(
                "tool arguments contain the expected fields"
                if not mismatched
                else f"arguments did not match for: {sorted(mismatched)}"
            ),
        )


@dataclass
class ToolCallBudgetRespected(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        maximum = _expected(ctx).max_tool_calls
        count = len(ctx.output.tool_calls)
        passed = maximum is None or count <= maximum
        return EvaluationReason(
            value=passed,
            reason=(
                f"used {count} tool calls"
                if maximum is None
                else f"used {count} of at most {maximum} tool calls"
            ),
        )


@dataclass
class ToolCallsSucceeded(AgentEvaluator):
    def evaluate(self, ctx: AgentEvaluatorContext) -> EvaluationReason:
        failed = [call.tool_name for call in ctx.output.tool_calls if call.status != "succeeded"]
        return EvaluationReason(
            value=not failed,
            reason=(
                "all invoked tools succeeded" if not failed else f"failed tool operations: {failed}"
            ),
        )


def default_evaluators() -> list[AgentEvaluator]:
    return [
        SuccessfulOutcome(),
        ResultContainsExpected(),
        RequiredToolsUsed(),
        ForbiddenToolsAvoided(),
        ToolArgumentsContainExpected(),
        ToolCallBudgetRespected(),
        ToolCallsSucceeded(),
    ]


def _expected(ctx: AgentEvaluatorContext) -> AgentEvaluationExpected:
    if ctx.metadata is None:
        raise ValueError("agent evaluation cases require expected-behavior metadata")
    return ctx.metadata


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        expected_mapping = cast(dict[str, Any], expected)
        actual_mapping = cast(dict[str, Any], actual)
        return all(
            key in actual_mapping and _contains(actual_mapping[key], value)
            for key, value in expected_mapping.items()
        )
    if isinstance(expected, list):
        expected_items = cast(list[Any], expected)
        if not isinstance(actual, list):
            return False
        actual_items = cast(list[Any], actual)
        if len(actual_items) < len(expected_items):
            return False
        return all(
            _contains(actual_items[index], value) for index, value in enumerate(expected_items)
        )
    return actual == expected
