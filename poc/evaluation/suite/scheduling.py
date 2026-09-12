"""Resource-constrained scheduling with a feasible planted witness, not answer matching."""

from __future__ import annotations

import random
from typing import Any

from poc.evaluation.suite.models import Answer, TaskCase, TaskInput


def scheduling(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    count = {80: 32, 240: 64, 800: 128}[size]
    names = [f"task-{i:04}" for i in range(count)]
    rng.shuffle(names)  # IDs do not reveal the planted order.
    machines = [0] * 4
    finishes: list[int] = []
    jobs: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    witness: dict[str, Any] = {}
    for i, name in enumerate(names):
        machine = rng.randrange(4)
        duration = rng.randint(5, 30)
        parents = rng.sample(range(i), min(i, rng.randint(0, 3)))
        release = rng.randint(0, max(machines) // 2)
        start = max(release, machines[machine], max((finishes[p] for p in parents), default=0))
        finish = start + duration
        machines[machine] = finish
        finishes.append(finish)
        witness[name] = start
        jobs.append(
            {
                "id": name,
                "machine": machine,
                "duration": duration,
                "release": release,
                "deadline": finish + rng.randint(0, 25),
            }
        )
        edges.extend({"job": name, "requires": names[p]} for p in parents)
    rng.shuffle(jobs)
    rng.shuffle(edges)
    return TaskInput(
        prompt=(
            "Construct a feasible non-preemptive schedule for ALL jobs on their assigned machines. "
            "Each machine processes at most one job at a time. Respect release times, deadlines "
            "(finish <= deadline) and ALL precedence edges (start >= prerequisite finish). "
            "All start times must be nonnegative integers; finish = start + duration. "
            "Finish all jobs by the horizon in the limits table. Return values mapping job ID "
            "to its start time. Any feasible schedule passes, not only the reference schedule."
        ),
        tables={"schedule_jobs": jobs, "precedence": edges, "limits": [{"horizon": max(finishes)}]},
    ), witness


def grade_schedule(case: TaskCase, answer: Answer) -> dict[str, float]:
    from poc.evaluation.suite.diagnostics import diagnose

    return diagnose(case.input, answer)["scores"]
