"""Public application behavior. No references, grader, filesystem, or model calls.

Replay is deliberately a configuration-repair sandbox, not arbitrary Python execution.
Every replay starts from the captured cache state; experiments never mutate other trials.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, cast

Row = dict[str, Any]
Tables = dict[str, list[Row]]
PATHS = ("interactive", "export", "scheduled")
DEFAULTS: Row = {
    "normalize_cached_groups": True,
    "cache_key": "tenant_subject",
    "check_membership_version": True,
    "fallback": "deny",
    "cache_enabled": True,
}

# This is application documentation exposed to investigators, not generator/grader code.
ADAPTER_SOURCE = """def authorize(request, config, directory, cache, policies):
    token = verify_signature_expiry_and_tenant_issuer(request.token, request.tenant)
    if not token or request.resource.tenant != request.tenant:
        return DENY
    if not directory.active(request.tenant, token.subject):
        return DENY
    key = (request.tenant, token.subject) if config.cache_key == "tenant_subject" else token.subject
    snapshot = cache.get(request.path, key) if config.cache_enabled else None
    if snapshot and config.check_membership_version:
        if snapshot.membership_version != directory.version(request.tenant, token.subject):
            snapshot = None
    if snapshot is None:
        groups = directory.canonical_groups(request.tenant, token.subject)
    elif config.normalize_cached_groups:
        groups = directory.to_canonical(request.tenant, snapshot.groups)
    else:
        groups = snapshot.groups
    groups = transitive_parents(request.tenant, groups)
    effects = policies.match(request.tenant, request.resource, groups)
    if DENY in effects:
        return DENY
    if ALLOW in effects:
        return ALLOW
    return ALLOW if config.fallback == "authenticated" else DENY
"""


def parse_changes(
    changes: Any, configurations: dict[tuple[str, str], Row]
) -> dict[tuple[str, str], Row]:
    """Apply the documented patch language, rejecting ambiguous or unknown settings."""
    if not isinstance(changes, list) or len(cast(list[Any], changes)) > 256:
        raise ValueError("changes must be a list of at most 256 setting changes")
    updated = {key: dict(value) for key, value in configurations.items()}
    seen: set[tuple[str, str, str]] = set()
    for raw in cast(list[Any], changes):
        if not isinstance(raw, dict) or set(cast(Row, raw)) != {
            "tenant_id",
            "path",
            "setting",
            "value",
        }:
            raise ValueError("each change requires tenant_id, path, setting, value")
        change = cast(Row, raw)
        tenant, path, setting, value = (
            change["tenant_id"],
            change["path"],
            change["setting"],
            change["value"],
        )
        if not all(isinstance(s, str) for s in (tenant, path, setting)):
            raise ValueError("tenant_id, path and setting must be strings")
        if (tenant, path) not in updated or setting not in DEFAULTS:
            raise ValueError("unknown configuration scope or setting")
        if (tenant, path, setting) in seen:
            raise ValueError("duplicate configuration change")
        seen.add((tenant, path, setting))
        if type(value) is not type(DEFAULTS[setting]):
            raise ValueError("setting value has the wrong type")
        if setting == "cache_key" and value not in ("subject", "tenant_subject"):
            raise ValueError("cache_key must be subject or tenant_subject")
        if setting == "fallback" and value not in ("deny", "authenticated"):
            raise ValueError("fallback must be deny or authenticated")
        updated[tenant, path][setting] = value
    return updated


class AuthorizationSandbox:
    def __init__(self, tables: Tables):
        self.tenants = {r["tenant_id"]: r for r in tables["auth_tenants"]}
        self.users = {(r["tenant_id"], r["subject_id"]): r for r in tables["auth_users"]}
        self.resources = {r["resource_id"]: r for r in tables["auth_resources"]}
        self.tokens = {r["token_id"]: r for r in tables["auth_tokens"]}
        self.requests = {r["request_id"]: r for r in tables["auth_requests"]}
        self.configurations = {
            (r["tenant_id"], r["path"]): json.loads(r["settings_json"])
            for r in tables["auth_runtime"]
        }
        self.cache = {(r["tenant_id"], r["subject_id"], r["path"]): r for r in tables["auth_cache"]}
        self.memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in tables["auth_memberships"]:
            self.memberships[row["tenant_id"], row["subject_id"]].add(row["group_id"])
        self.aliases: dict[tuple[str, str], str] = {}
        for row in tables["auth_groups"]:
            for column in ("group_id", "legacy_id", "current_id"):
                self.aliases[row["tenant_id"], row[column]] = row["group_id"]
        self.parents: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in tables["auth_group_edges"]:
            self.parents[row["tenant_id"], row["child"]].add(row["parent"])
        self.policies: dict[tuple[str, str], list[Row]] = defaultdict(list)
        for row in tables["auth_policies"]:
            self.policies[row["tenant_id"], row["resource_id"]].append(row)

    def evaluate(self, request: Row, configurations: dict[tuple[str, str], Row]) -> Row:
        tenant = request["tenant_id"]
        token = self.tokens[request["token_id"]]
        resource = self.resources[request["resource_id"]]
        config = configurations[tenant, request["path"]]
        if (
            not token["valid"]
            or token["tenant_id"] != tenant
            or token["issuer"] != self.tenants[tenant]["issuer"]
            or resource["tenant_id"] != tenant
        ):
            return {"decision": "deny", "reason": "identity_or_resource_boundary", "policy_ids": []}
        subject = token["subject_id"]
        user = self.users[tenant, subject]
        if not user["active"]:
            return {"decision": "deny", "reason": "disabled_subject", "policy_ids": []}
        snapshot = None
        owner = tenant if config["cache_key"] == "tenant_subject" else request["prior_tenant_id"]
        if config["cache_enabled"] and request["cache_state"] == "warm":
            snapshot = self.cache.get((owner, subject, request["path"]))
        if (
            snapshot
            and config["check_membership_version"]
            and snapshot["membership_version"] != user["membership_version"]
        ):
            snapshot = None
        if snapshot is None:
            groups = set(self.memberships[tenant, subject])
        else:
            raw: list[str] = json.loads(snapshot["groups_json"])
            groups = {
                self.aliases.get((tenant, g), g) if config["normalize_cached_groups"] else g
                for g in raw
            }
        frontier = list(groups)
        while frontier:
            for parent in self.parents[tenant, frontier.pop()]:
                if parent not in groups:
                    groups.add(parent)
                    frontier.append(parent)
        matching = [
            p for p in self.policies[tenant, resource["resource_id"]] if p["group_id"] in groups
        ]
        effects = {p["effect"] for p in matching}
        decision = (
            "deny"
            if "deny" in effects
            else "allow"
            if "allow" in effects
            else "allow"
            if config["fallback"] == "authenticated"
            else "deny"
        )
        return {
            "decision": decision,
            "reason": "policy" if effects else "no_matching_policy",
            "cache_hit": snapshot is not None,
            "cache_owner": owner if snapshot else None,
            "effective_groups": sorted(groups),
            "policy_ids": sorted(p["policy_id"] for p in matching),
        }

    def replay(self, changes_json: str, request_id: str) -> str:
        """SQLite scalar function: simulate a public request; never report gold or scores."""
        try:
            if request_id not in self.requests:
                raise ValueError("unknown public request_id")
            configurations = parse_changes(json.loads(changes_json), self.configurations)
            result = self.evaluate(self.requests[request_id], configurations)
        except (ValueError, TypeError) as exc:
            result = {"error": str(exc)}
        return json.dumps(result, separators=(",", ":"))
