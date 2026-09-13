"""Additional public task families with independent partitions and an explicit join."""

from __future__ import annotations

import random
from typing import Any

from poc.execution.sql_contracts import TaskInput


def dependency_join(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    expected: dict[str, Any] = {}
    last: list[str] = []
    for branch in range(2):
        finish = 0
        previous = None
        for index in range(max(2, size // 2)):
            key = f"branch-{branch}-job-{index:04}"
            duration = rng.randint(1, 35)
            jobs.append({"id": key, "duration": duration, "branch": branch})
            if previous is not None:
                edges.append({"job": key, "requires": previous})
            finish += duration
            expected[key] = finish
            previous = key
        assert previous is not None
        last.append(previous)
    jobs.append({"id": "join", "duration": 1, "branch": 2})
    edges.extend({"job": "join", "requires": key} for key in last)
    expected["join"] = max(expected[key] for key in last) + 1
    rng.shuffle(jobs)
    rng.shuffle(edges)
    return TaskInput(
        prompt="Compute integer earliest FINISH times for EVERY job, starting at zero. "
        "A job starts after ALL prerequisites finish. The two public branches are independent "
        "until the final join; each can be analyzed separately before combining the results. "
        "Return a complete values mapping keyed by the exact job IDs.",
        tables={"jobs": jobs, "dependencies": edges},
    ), expected
