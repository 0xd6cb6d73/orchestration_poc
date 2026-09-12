"""Resource-constrained scheduling with a feasible planted witness, not answer matching."""

from __future__ import annotations

import random
from itertools import pairwise
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
    jobs = {row["id"]: row for row in case.input.tables["schedule_jobs"]}
    starts = answer.values
    if set(starts) != set(jobs) or any(type(v) is not int or v < 0 for v in starts.values()):
        return {"exact": 0.0, "fraction_correct": 0.0}
    finishes = {name: starts[name] + row["duration"] for name, row in jobs.items()}
    horizon = case.input.tables["limits"][0]["horizon"]
    checks = [
        starts[n] >= row["release"] and finishes[n] <= min(row["deadline"], horizon)
        for n, row in jobs.items()
    ]
    checks.extend(
        starts[e["job"]] >= finishes[e["requires"]] for e in case.input.tables["precedence"]
    )
    for machine in {row["machine"] for row in jobs.values()}:
        ordered = sorted(
            (starts[n], finishes[n]) for n, row in jobs.items() if row["machine"] == machine
        )
        checks.extend(left[1] <= right[0] for left, right in pairwise(ordered))
    return {"exact": float(all(checks)), "fraction_correct": sum(checks) / len(checks)}
