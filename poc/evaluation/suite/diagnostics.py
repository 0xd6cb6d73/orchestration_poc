"""Public-input validation. Never reads references or supplies a solving algorithm."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from typing import Any, cast

from poc.evaluation.suite.models import Answer, TaskInput


def diagnose(task: TaskInput, answer: Answer) -> dict[str, Any]:
    tables, values = task.tables, answer.values
    shape: list[dict[str, Any]] = []
    if "schedule_jobs" in tables:
        rows, key, kind = tables["schedule_jobs"], "id", "scheduling"
    elif "stops" in tables and "vehicles" in tables:
        rows, key, kind = tables["stops"], "id", "routing"
    elif "invoices" in tables:
        rows, key, kind = tables["invoices"], "invoice", "ledger"
    elif "jobs" in tables and "dependencies" in tables:
        rows, key, kind = tables["jobs"], "id", "dependencies"
    elif "queries" in tables and "policies" in tables:
        rows, key, kind = tables["queries"], "id", "access"
    else:
        return {"answer_valid": None, "feasible": None, "dimensions": {}, "violations": []}
    ids = {r[key] for r in rows}
    if ids != set(values):
        shape.append(
            {
                "dimension": "coverage",
                "missing": sorted(ids - set(values)),
                "unexpected": sorted(set(values) - ids),
            }
        )
    for name, value in values.items():
        valid = True
        if kind == "routing":
            valid = (
                isinstance(value, dict)
                and set(cast(dict[str, Any], value)) == {"vehicle", "position"}
                and type(cast(dict[str, Any], value)["vehicle"]) is int
                and type(cast(dict[str, Any], value)["position"]) is int
                and value["vehicle"] in {v["id"] for v in tables["vehicles"]}
            )
        elif kind == "access":
            valid = isinstance(value, str) and value in ("allow", "deny")
        else:
            valid = type(value) is int and (kind == "ledger" or value >= 0)
        if not valid:
            shape.append({"dimension": "value_type", "id": name})
    if shape:
        return {
            "answer_valid": False,
            "feasible": False if kind in ("routing", "scheduling") else None,
            "dimensions": {},
            "violations": shape,
            "scores": {"exact": 0.0, "fraction_correct": 0.0},
        }
    dimensions: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    violations: list[dict[str, Any]] = []

    def check(dimension: str, passed: bool, **evidence: Any) -> None:
        dimensions[dimension]["total"] += 1
        dimensions[dimension]["passed"] += int(passed)
        if not passed:
            violations.append({"dimension": dimension, **evidence})

    if kind == "scheduling":
        jobs = {r["id"]: r for r in rows}
        finishes = {n: values[n] + r["duration"] for n, r in jobs.items()}
        horizon = tables["limits"][0]["horizon"]
        for n, r in jobs.items():
            check(
                "job_window",
                values[n] >= r["release"] and finishes[n] <= min(r["deadline"], horizon),
                id=n,
                start=values[n],
                finish=finishes[n],
                release=r["release"],
                deadline=r["deadline"],
                horizon=horizon,
            )
        for e in tables["precedence"]:
            check(
                "precedence",
                values[e["job"]] >= finishes[e["requires"]],
                job=e["job"],
                requires=e["requires"],
                start=values[e["job"]],
                prerequisite_finish=finishes[e["requires"]],
            )
        for m in {r["machine"] for r in rows}:
            ordered = sorted(
                (values[n], finishes[n], n) for n, r in jobs.items() if r["machine"] == m
            )
            for left, right in pairwise(ordered):
                check(
                    "machine_overlap",
                    left[1] <= right[0],
                    machine=m,
                    left=left[2],
                    right=right[2],
                    left_finish=left[1],
                    right_start=right[0],
                )
    elif kind == "routing":
        stops = {r["id"]: r for r in rows}
        for v in tables["vehicles"]:
            route = sorted((a["position"], n) for n, a in values.items() if a["vehicle"] == v["id"])
            check(
                "positions",
                [p for p, _ in route] == list(range(1, len(route) + 1)),
                vehicle=v["id"],
            )
            load = sum(stops[n]["demand"] for _, n in route)
            check(
                "capacity",
                load <= v["capacity"],
                vehicle=v["id"],
                load=load,
                capacity=v["capacity"],
            )
            time, x, y = 0, 0, 0
            for _, n in route:
                s = stops[n]
                time = max(time + abs(x - s["x"]) + abs(y - s["y"]), s["opens"])
                check(
                    "service_window",
                    time <= s["closes"],
                    id=n,
                    service_start=time,
                    closes=s["closes"],
                )
                time += s["service"]
                x, y = s["x"], s["y"]
            returned = time + abs(x) + abs(y)
            check(
                "return_deadline",
                returned <= v["return_deadline"],
                vehicle=v["id"],
                returned=returned,
                deadline=v["return_deadline"],
            )
    total = sum(d["total"] for d in dimensions.values())
    passed = sum(d["passed"] for d in dimensions.values())
    result: dict[str, Any] = {
        "answer_valid": True,
        "feasible": not violations if total else None,
        "dimensions": dict(dimensions),
        "violations": violations,
    }
    if total:
        result["scores"] = {"exact": float(passed == total), "fraction_correct": passed / total}
    return result
