"""Review orchestration independent of tasks, graders, providers, and benchmark state."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

T = TypeVar("T")


class ReviewDecision(BaseModel, Generic[T]):
    """A decision about the single immutable submission supplied to this reviewer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["accept", "revise", "decline"]
    reason: str = Field(min_length=1)
    replacement: T | None = None

    @model_validator(mode="after")
    def valid_replacement(self) -> ReviewDecision[T]:
        if (self.action == "revise") != (self.replacement is not None):
            raise ValueError("revise requires a replacement; accept and decline forbid one")
        return self


@dataclass(frozen=True)
class ReviewResult(Generic[T]):
    output: T
    source: Literal["review", "draft_fallback", "draft_accept", "draft_decline"]
    reviewer_error: str | None = None


async def review_submission(
    draft: T,
    reviewer: Callable[[T], Awaitable[T]],
    *,
    return_draft_on_error: bool = False,
    recoverable_errors: tuple[type[Exception], ...] = (TimeoutError,),
) -> ReviewResult[T]:
    """Review a submitted output, retaining that submission on explicitly allowed failures.

    The caller owns model bindings, resource limits, and the output protocol. No
    correctness oracle is consulted: a retained draft can still be wrong. External
    cancellation always propagates, including a benchmark's hard deadline.
    """
    committed = deepcopy(draft)
    try:
        result = await reviewer(deepcopy(committed))
    except asyncio.CancelledError:
        raise
    except recoverable_errors as exc:
        if not return_draft_on_error:
            raise
        return ReviewResult(committed, "draft_fallback", type(exc).__name__)
    return ReviewResult(result, "review")


async def review_decision_submission(
    draft: T,
    reviewer: Callable[[T], Awaitable[ReviewDecision[T]]],
    *,
    return_draft_on_error: bool = False,
    recoverable_errors: tuple[type[Exception], ...] = (TimeoutError,),
) -> ReviewResult[T]:
    """Accept/decline retain the artifact; revise explicitly replaces it, even if wrong."""
    decisions: list[ReviewDecision[T]] = []

    async def decide(submission: T) -> T:
        decision = await reviewer(deepcopy(submission))
        decisions.append(decision)
        if decision.action == "revise":
            assert decision.replacement is not None
            return deepcopy(decision.replacement)
        return submission

    result = await review_submission(
        draft,
        decide,
        return_draft_on_error=return_draft_on_error,
        recoverable_errors=recoverable_errors,
    )
    if result.source == "draft_fallback":
        return result
    if decisions[0].action == "accept":
        return ReviewResult(result.output, "draft_accept")
    if decisions[0].action == "decline":
        return ReviewResult(result.output, "draft_decline")
    return result
