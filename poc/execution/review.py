"""Review orchestration independent of tasks, graders, providers, and benchmark state."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ReviewResult(Generic[T]):
    output: T
    source: Literal["review", "draft_fallback"]
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
