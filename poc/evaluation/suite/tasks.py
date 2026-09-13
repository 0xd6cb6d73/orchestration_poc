from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Any

from poc.evaluation.authorization.generator import authorization
from poc.evaluation.authorization.grading import grade_authorization
from poc.evaluation.suite.coordination_tasks import dependency_join
from poc.evaluation.suite.models import Answer, TaskCase, TaskInput, grade
from poc.evaluation.suite.routing import grade_routing, routing
from poc.evaluation.suite.scheduling import grade_schedule, scheduling

TaskFactory = Callable[[random.Random, int], tuple[TaskInput, dict[str, Any]]]
FACTORIES: dict[str, TaskFactory] = {}
TaskGrader = Callable[[TaskCase, Answer], dict[str, float]]
GRADERS: dict[str, TaskGrader] = {}


def register_task(name: str, factory: TaskFactory, grader: TaskGrader = grade) -> None:
    if name in FACTORIES:
        raise ValueError(f"task already registered: {name}")
    FACTORIES[name] = factory
    GRADERS[name] = grader


def generate(family: str, seed: int, split: str, difficulty: str) -> TaskCase:
    size = {"standard": 80, "hard": 240, "stress": 800}[difficulty]
    # Separate streams across families and splits, independent of Python hash randomization.
    entropy = hashlib.sha256(f"v1:{family}:{seed}:{split}".encode()).digest()
    public, expected = FACTORIES[family](random.Random(entropy), size)
    return TaskCase.model_validate(
        {
            "id": f"{family}-v1-{split}-{difficulty}-{seed}",
            "family": family,
            "seed": seed,
            "split": split,
            "difficulty": difficulty,
            "input": public,
            "expected": expected,
        }
    )


def ledger(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    invoices: list[dict[str, Any]] = []
    payments: list[dict[str, Any]] = []
    expected: dict[str, Any] = {}
    for i in range(size):
        key = f"invoice-{i:04}"
        amount = rng.randint(1000, 90000)
        invoices.append({"invoice": key, "revision": 1, "amount": amount + 317})
        invoices.append({"invoice": key, "revision": 2, "amount": amount})
        paid = 0
        for j in range(rng.randint(1, 5)):
            value = rng.randint(100, 20000)
            status = rng.choice(["settled", "settled", "pending", "void"])
            kind = rng.choice(["payment", "payment", "refund"])
            row = {
                "event": f"event-{i}-{j}",
                "invoice": key,
                "amount": value,
                "status": status,
                "kind": kind,
            }
            payments.append(row)
            if rng.random() < 0.4:
                payments.append(dict(row))
            if status == "settled":
                paid += value if kind == "payment" else -value
        expected[key] = amount - paid
    rng.shuffle(invoices)
    rng.shuffle(payments)
    return TaskInput(
        prompt=(
            "Reconcile EVERY invoice. Return values mapping invoice ID to signed outstanding "
            "integer cents. Use only the highest invoice revision. Deduplicate payments by event "
            "ID before summing; only settled events count. A refund subtracts from paid. "
            "Outstanding = latest amount minus net paid; retain negative and zero balances."
        ),
        tables={"invoices": invoices, "payments": payments},
    ), expected


def dependencies(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    finishes: list[int] = []
    for i in range(size):
        duration = rng.randint(1, 35)
        parents = sorted(rng.sample(range(i), min(i, rng.randint(1, 5))))
        nodes.append({"id": f"job-{i:04}", "duration": duration})
        edges.extend({"job": f"job-{i:04}", "requires": f"job-{p:04}"} for p in parents)
        finishes.append(duration + max((finishes[p] for p in parents), default=0))
    rng.shuffle(nodes)
    rng.shuffle(edges)
    expected: dict[str, Any] = {f"job-{i:04}": finish for i, finish in enumerate(finishes)}
    return TaskInput(
        prompt=(
            "Compute earliest FINISH time for EVERY job in this DAG. Unlimited parallel workers, "
            "time starts at zero, no preemption or overhead. Each job starts only after ALL its "
            "requirements finish. Return values mapping job ID to integer finish time."
        ),
        tables={"jobs": nodes, "dependencies": edges},
    ), expected


def access(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    memberships: list[dict[str, Any]] = []
    nesting: list[dict[str, Any]] = []
    policies: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    groups = [f"group-{i:03}" for i in range(max(12, size // 4))]
    ancestors: dict[str, set[str]] = {}
    for i, group in enumerate(groups):
        parents = rng.sample(groups[:i], min(i, rng.randint(1, 3)))
        ancestors[group] = {group}
        for parent in parents:
            nesting.append({"child": group, "parent": parent})
            ancestors[group].update(ancestors[parent])
        for resource in range(8):
            if rng.random() < 0.4:
                policies.append(
                    {
                        "group_id": group,
                        "resource": f"resource-{resource}",
                        "effect": rng.choice(["allow", "allow", "deny"]),
                    }
                )
    expected: dict[str, Any] = {}
    for i in range(size):
        user = f"user-{i:04}"
        direct = rng.sample(groups, rng.randint(1, 3))
        effective = set[str]().union(*(ancestors[g] for g in direct))
        memberships.extend({"user": user, "group_id": g} for g in direct)
        for j in range(2):
            resource = f"resource-{rng.randrange(8)}"
            key = f"query-{i:04}-{j}"
            queries.append({"id": key, "user": user, "resource": resource})
            effects = {
                p["effect"]
                for p in policies
                if p["group_id"] in effective and p["resource"] == resource
            }
            expected[key] = "allow" if "allow" in effects and "deny" not in effects else "deny"
    return TaskInput(
        prompt=(
            "Audit EVERY access query. A user inherits policies from direct groups and all "
            "transitive parent groups. Any matching deny overrides all allows; absent allow "
            "means deny. Match resource exactly. Return values mapping query ID to 'allow' or 'deny'."
        ),
        tables={
            "memberships": memberships,
            "nesting": nesting,
            "policies": policies,
            "queries": queries,
        },
    ), expected


register_task("ledger", ledger)
register_task("dependencies", dependencies)
register_task("access", access)

register_task("scheduling", scheduling, grade_schedule)

register_task("routing", routing, grade_routing)

register_task("authorization", authorization, grade_authorization)


def partitioned_ledger(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    task, expected = ledger(rng, size)
    ids = sorted({row["invoice"] for row in task.tables["invoices"]})
    return task.model_copy(
        update={
            "prompt": task.prompt + " The public partitions table groups invoices into independent "
            "work units. Combine every partition into one final mapping.",
            "tables": {
                **task.tables,
                "partitions": [
                    {"invoice": key, "partition": index % 4} for index, key in enumerate(ids)
                ],
            },
        }
    ), expected


register_task("partitioned_ledger", partitioned_ledger)
register_task("dependency_join", dependency_join)
