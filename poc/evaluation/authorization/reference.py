"""Operator-only entitlement oracle, independent of cache and runtime configuration."""

from __future__ import annotations

from collections import defaultdict

from poc.evaluation.authorization.simulator import Row, Tables


class Entitlements:
    def __init__(self, tables: Tables):
        self.tokens = {r["token_id"]: r for r in tables["auth_tokens"]}
        self.issuers = {r["tenant_id"]: r["issuer"] for r in tables["auth_tenants"]}
        self.active = {(r["tenant_id"], r["subject_id"]): r["active"] for r in tables["auth_users"]}
        self.owners = {r["resource_id"]: r["tenant_id"] for r in tables["auth_resources"]}
        ancestors: dict[tuple[str, str], set[str]] = {
            (r["tenant_id"], r["group_id"]): {r["group_id"]} for r in tables["auth_groups"]
        }
        # Fixed-point closure, deliberately independent from the application's traversal.
        changed = True
        while changed:
            changed = False
            for edge in tables["auth_group_edges"]:
                child = ancestors[edge["tenant_id"], edge["child"]]
                before = len(child)
                child.update(ancestors[edge["tenant_id"], edge["parent"]])
                changed |= len(child) != before
        effective: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in tables["auth_memberships"]:
            effective[row["tenant_id"], row["subject_id"]].update(
                ancestors[row["tenant_id"], row["group_id"]]
            )
        self.allowed: dict[tuple[str, str, str], list[str]] = {}
        by_tenant: dict[str, list[Row]] = defaultdict(list)
        for row in tables["auth_policies"]:
            by_tenant[row["tenant_id"]].append(row)
        for (tenant, subject), groups in effective.items():
            matched: dict[str, list[Row]] = defaultdict(list)
            for policy in by_tenant[tenant]:
                if policy["group_id"] in groups:
                    matched[policy["resource_id"]].append(policy)
            for resource, policies in matched.items():
                if all(p["effect"] != "deny" for p in policies):
                    self.allowed[tenant, subject, resource] = sorted(
                        p["policy_id"] for p in policies
                    )

    def supporting_policies(self, request: Row) -> list[str]:
        token = self.tokens[request["token_id"]]
        tenant = request["tenant_id"]
        if (
            token["valid"] != 1
            or token["tenant_id"] != tenant
            or token["issuer"] != self.issuers[tenant]
            or not self.active.get((tenant, token["subject_id"]), False)
            or self.owners[request["resource_id"]] != tenant
        ):
            return []
        return self.allowed.get((tenant, token["subject_id"], request["resource_id"]), [])

    def decision(self, request: Row) -> str:
        return "allow" if self.supporting_policies(request) else "deny"
