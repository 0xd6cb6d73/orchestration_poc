"""Capacitated vehicle routing with time windows and independently checked witnesses."""

from __future__ import annotations

import random
from typing import Any

from poc.evaluation.suite.models import Answer, TaskCase, TaskInput


def routing(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    count = {80: 24, 240: 48, 800: 96}[size]
    names = [f"stop-{i:03}" for i in range(count)]
    rng.shuffle(names)
    stops: list[dict[str, Any]] = []
    vehicles: list[dict[str, Any]] = []
    witness: dict[str, Any] = {}
    for vehicle in range(4):
        time, x, y, load = 0, 0, 0, 0
        for position, name in enumerate(names[vehicle::4], 1):
            nx, ny = rng.randint(-30, 30), rng.randint(-30, 30)
            demand, service = rng.randint(1, 9), rng.randint(1, 5)
            time += abs(x - nx) + abs(y - ny)
            opens = max(0, time - rng.randint(0, 30))
            closes = time + rng.randint(0, 25)
            stops.append(
                {
                    "id": name,
                    "x": nx,
                    "y": ny,
                    "demand": demand,
                    "service": service,
                    "opens": opens,
                    "closes": closes,
                }
            )
            witness[name] = {"vehicle": vehicle, "position": position}
            time += service
            load += demand
            x, y = nx, ny
        vehicles.append(
            {
                "id": vehicle,
                "capacity": load + rng.randint(0, 3),
                "return_deadline": time + abs(x) + abs(y) + rng.randint(0, 15),
            }
        )
    rng.shuffle(stops)
    return TaskInput(
        prompt=(
            "Plan delivery routes for ALL stops using four vehicles. Each vehicle starts at "
            "depot (0,0) at time zero and must return by its return_deadline. Travel time is "
            "Manhattan distance abs(dx)+abs(dy). At each stop arrival may wait until opens, "
            "but service must START by closes; then add service time before leaving. "
            "Total demand assigned to a vehicle must not exceed capacity. Visit every stop "
            "exactly once. Return values mapping stop ID to {vehicle: integer ID, position: "
            "integer visit order starting at 1}. Positions must be contiguous per vehicle. "
            "Unused vehicles are allowed. Any feasible routing passes."
        ),
        tables={"stops": stops, "vehicles": vehicles},
    ), witness


def grade_routing(case: TaskCase, answer: Answer) -> dict[str, float]:
    from poc.evaluation.suite.diagnostics import diagnose

    return diagnose(case.input, answer)["scores"]
