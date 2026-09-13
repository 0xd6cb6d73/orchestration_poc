"""Outcome grading: entitlement restoration, isolation, impact and evidence provenance."""

from __future__ import annotations

from typing import Any, cast

from poc.evaluation.authorization.reference import Entitlements
from poc.evaluation.authorization.simulator import AuthorizationSandbox, Row, parse_changes
from poc.evaluation.suite.models import Answer, TaskCase

SCORE_NAMES = (
    "exact",
    "fraction_correct",
    "restored_access",
    "preserved_access",
    "preserved_denials",
    "tenant_impact",
    "evidence_support",
)


def _strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(type(item) is str for item in cast(list[Any], value))
        and len(set(cast(list[Any], value))) == len(cast(list[Any], value))
    )


def grade_authorization(case: TaskCase, answer: Answer) -> dict[str, float]:
    zero: dict[str, float] = {name: 0.0 for name in SCORE_NAMES}
    values = answer.values
    if set(values) != {"affected_tenants", "changes", "evidence", "explanation"}:
        return zero
    if (
        not _strings(values["affected_tenants"])
        or not isinstance(values["explanation"], str)
        or not values["explanation"].strip()
        or not isinstance(values["evidence"], list)
    ):
        return zero
    engine = AuthorizationSandbox(case.input.tables)
    try:
        configurations = parse_changes(values["changes"], engine.configurations)
    except ValueError:
        return zero
    expected_tenants = set(case.expected["submission"]["affected_tenants"])
    claimed = set(values["affected_tenants"])
    impact = len(expected_tenants & claimed) / max(1, len(expected_tenants | claimed))
    oracle = Entitlements(case.input.tables)
    changes = {r["change_id"]: r for r in case.input.tables["auth_changes"]}
    observations = {r["trace_id"]: r for r in case.input.tables["auth_audit"]}
    supported: set[str] = set()
    invalid = 0
    seen: set[tuple[str, str, str]] = set()
    for raw in cast(list[Any], values["evidence"]):
        if (
            not isinstance(raw, dict)
            or set(cast(Row, raw)) != {"request_id", "policy_id", "change_id"}
            or not all(type(v) is str for v in cast(Row, raw).values())
        ):
            return zero
        link = cast(Row, raw)
        identity = (link["request_id"], link["policy_id"], link["change_id"])
        if identity in seen:
            invalid += 1
            continue
        seen.add(identity)
        req = engine.requests.get(link["request_id"])
        change = changes.get(link["change_id"])
        valid = bool(
            req
            and change
            and req["tenant_id"] in expected_tenants
            and change["tenant_id"] == req["tenant_id"]
            and change["path"] == req["path"]
            and change["component"] == "authorization-adapter"
            and change["before_json"] != change["after_json"]
            and observations[req["trace_id"]]["decision"] == "deny"
            and link["policy_id"] in oracle.supporting_policies(req)
        )
        if valid and req:
            supported.add(req["tenant_id"])
        else:
            invalid += 1
    evidence_score = len(supported) / max(1, len(expected_tenants) + invalid)
    passed = {"regressed": 0, "allow": 0, "deny": 0}
    totals = {"regressed": 0, "allow": 0, "deny": 0}
    for check in case.expected["checks"]:
        expected = check["expected"]
        category = check["category"]
        totals[category] += 1
        passed[category] += (
            engine.evaluate(check["request"], configurations)["decision"] == expected
        )
    restored = passed["regressed"] / max(1, totals["regressed"])
    preserved_access = passed["allow"] / max(1, totals["allow"])
    preserved = passed["deny"] / max(1, totals["deny"])
    dimensions = [impact, evidence_score, restored, preserved_access, preserved]
    return {
        "exact": float(all(d == 1 for d in dimensions)),
        "fraction_correct": sum(dimensions) / len(dimensions),
        "restored_access": restored,
        "preserved_access": preserved_access,
        "preserved_denials": preserved,
        "tenant_impact": impact,
        "evidence_support": evidence_score,
    }
