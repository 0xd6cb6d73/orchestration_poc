"""Concurrent SQL stages with fixed reservations and isolated connections.

Reservations sum to one trial budget. Unused branch allocations are not borrowed;
ambiguous provider consumption cannot enlarge another branch's reservation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.execution.sql_contracts import Answer, Budget, PhaseName, TaskInput
from poc.execution.sql_ports import TaskEnvironment
from poc.execution.sql_team_worker import POOL_PLANNED, ROLE_POOLS, TeamWorker

T = TypeVar("T")


async def gather_branches(branches: list[Awaitable[T]]) -> list[T]:
    tasks = [asyncio.ensure_future(branch) for branch in branches]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


CONCURRENT_WEIGHTS = {
    "hierarchical_dag": [0.2, 0.2, 0.6],
    "board_claim": [0.2, 0.2, 0.6],
    "managed_pool": [0.15, 0.15, 0.2, 0.5],
    "speculative": [0.35, 0.35, 0.3],
    "hybrid_v1": [0.3, 0.3, 0.12, 0.12, 0.16],
}


class ConcurrentTeamWorker:
    def __init__(
        self,
        method: str,
        task: TaskInput,
        env: TaskEnvironment,
        model: Model | str,
        settings: ModelSettings,
        budget: Budget,
        usage: RunUsage,
        *,
        json_protocol: bool,
    ):
        self.method, self.task, self.env, self.model = method, task, env, model
        self.settings, self.budget, self.usage = settings, budget, usage
        self.json_protocol = json_protocol
        self.pool_shares: dict[str, float] | None = None
        self.pool_planned: dict[str, int] = {}
        self.consumed = 0.0
        self.reserved = {"requests": 0, "tool_calls": 0, "total_tokens": 0}
        if method in {"hybrid_v2", "hybrid_v2_1", "hybrid_v2_2"}:
            shares = env.state.options.hybrid_pool_weights or ROLE_POOLS["hybrid_v2"]
            self.pool_shares = dict(shares)
            self.pool_planned = dict(POOL_PLANNED)
            self.pool_planned["orchestrate"] = (
                2 + env.state.options.hybrid_max_decision_rounds
                if method == "hybrid_v2_2"
                else env.state.options.hybrid_plan_fanout
            )
            self.pool_planned["critique"] = env.state.options.hybrid_plan_fanout + 2
            self.index = 0
        else:
            self.weights = env.state.options.stage_weights or CONCURRENT_WEIGHTS[method]
            if len(self.weights) != len(CONCURRENT_WEIGHTS[method]):
                raise ValueError("concurrent stage_weights length does not match strategy")
            self.index = 0

    async def __call__(self, instruction: str, role: PhaseName = "solve") -> Answer:
        if self.pool_shares is not None:
            key = role if role in self.pool_shares else "task"
            planned = max(1, self.pool_planned.get(key, 1))
            share = self.pool_shares[key] / planned
            if key == "verify":
                # The verifier is the final stage; unallocated budget carries forward to it.
                share = max(share, 1.0 - self.consumed)
                until_fraction = 1.0
                allocations = {
                    resource: getattr(self.budget, resource) - self.reserved[resource]
                    for resource in ("requests", "tool_calls", "total_tokens")
                }
            else:
                until_fraction = min(1.0, self.consumed + share)
                allocations = {
                    resource: int(getattr(self.budget, resource) * share)
                    for resource in ("requests", "tool_calls", "total_tokens")
                }
            if any(value < 1 for value in allocations.values()):
                raise UsageLimitExceeded("concurrent stage has no reserved capacity")
            for resource in allocations:
                self.reserved[resource] += allocations[resource]
            self.consumed = until_fraction
            index = self.index
            self.index += 1
        else:
            if self.method == "hybrid_v1" and role in {"critique", "verify"}:
                self.index = max(self.index, 2 if role == "critique" else 4)
            elif self.method == "speculative" and role == "reconcile":
                self.index = max(self.index, 2)
            index = self.index
            self.index += 1
            if index >= len(self.weights):
                raise UsageLimitExceeded("concurrent stage reservations exhausted")
            weight = self.weights[index]
            # The first two workers overlap and share a wall-clock window, not token allotments.
            wave_end = max(2, index + 1)
            until_fraction = sum(self.weights[:wave_end])
            allocations = {
                key: int(getattr(self.budget, key) * weight)
                for key in ("requests", "tool_calls", "total_tokens")
            }
            if any(value < 1 for value in allocations.values()):
                raise UsageLimitExceeded("concurrent stage has no reserved capacity")
        child = self.env.fork(allocations["tool_calls"])
        child.state.started = self.env.state.started
        child.state.options = self.env.state.options.model_copy(update={"stage_weights": None})
        emit = self.env.state.emit

        def branch_emit(event: dict[str, Any]) -> None:
            if "phase" in event:
                event["phase"]["stage_index"] = index
                if "failure" in event["phase"]:
                    event["phase"]["failure"]["stage_index"] = index
            emit({**event, "branch_index": index})

        child.state.emit = branch_emit
        usage = RunUsage()
        reserve = min(self.env.state.options.return_reserve_seconds, self.budget.seconds / 10)
        budget = self.budget.model_copy(
            update={**allocations, "seconds": (self.budget.seconds - reserve) * until_fraction}
        )
        child.state.options = child.state.options.model_copy(update={"return_reserve_seconds": 0})
        worker = TeamWorker(
            self.method,
            self.task,
            child,
            self.model,
            self.settings,
            budget,
            usage,
            json_protocol=self.json_protocol,
        )
        worker.weights = [1.0]
        worker.method = self.method
        worker.pool_shares = None
        worker.pool_planned = {}
        worker.consumed = 0.0
        worker.evidence_namespace = str(index + 1)
        emit(
            {
                "budget_reservation": {
                    "branch_index": index,
                    "role": role,
                    **allocations,
                    "deadline_seconds": budget.seconds,
                    "policy": "concurrent-v1",
                }
            }
        )
        try:
            return await worker(instruction, role)
        finally:
            self.usage.incr(usage)
            state = self.env.state
            state.phase = role
            if child.state.exhausted_phase:
                state.exhausted_phase = child.state.exhausted_phase
            state.request_attempts += child.state.request_attempts
            state.request_responses += child.state.request_responses
            state.usage_complete &= child.state.usage_complete
            state.unreported_token_reserve += child.state.unreported_token_reserve
            for phase in child.state.phases:
                phase["branch_index"] = index
                phase["stage_index"] = index
                state.phases.append(phase)
            self.env.absorb(child)
            child.close()
