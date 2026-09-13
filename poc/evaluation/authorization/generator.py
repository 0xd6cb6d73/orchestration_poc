"""Seeded incident worlds. Large evidence is generated, never checked into git."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from poc.evaluation.authorization.reference import Entitlements
from poc.evaluation.authorization.simulator import (
    ADAPTER_SOURCE,
    DEFAULTS,
    PATHS,
    AuthorizationSandbox,
    Row,
    Tables,
)
from poc.execution.sql_contracts import TaskInput


@dataclass(frozen=True)
class Scale:
    tenants: int
    users_per_tenant: int
    groups_per_tenant: int
    audit_requests: int


SCALES = {
    80: Scale(12, 160, 48, 40000),
    240: Scale(24, 320, 96, 120000),
    800: Scale(48, 640, 128, 300000),
}
REPORTS = ("summary", "payroll", "audit", *(f"department-{i}" for i in range(9)))

PROMPT = """Investigate a tenant-specific report authorization incident after an identity migration.
Some enterprise customers cannot open reports they previously used. Determine which tenants
have regressed denials in the captured requests, establish evidence linking symptoms to
entitlements and runtime changes, and submit a configuration repair. Restore intended access
without granting unintended access, including across tenants, expired credentials, explicit
denies, revoked memberships, and both warm and cold caches. Unaffected tenants must keep working.
The directory and tenant-scoped policies are the source of intended entitlement. Report IDs are
globally unique; subject IDs and canonical group IDs are only unique WITHIN a tenant.

You have captured identity, directory, policy, cache, deployment, request and trace sources plus
application source and operational documents in separate SQL tables. Sources use different IDs;
consult auth_documents for their relationships and experiment semantics. Audit outcomes are
observations from the deployed application, not the intended answer. No prescribed workflow.

Tool limits for this task: 2000 rows and 1,048,576 serialized UTF-8 bytes per SQL response;
use selective queries, aggregation and LIMIT/OFFSET. SQL statements are limited to 16KB.
Use authz_replay(changes_json, request_id) in SELECT to replay any PUBLIC captured request
against an isolated proposed configuration. It returns a JSON decision and execution trace,
never correctness feedback. Pass '[]' for the deployed configuration. Experiments reset the
captured cache for each request and have no persistent effects. They consume the SQL tool budget;
at most 2000 replay invocations are allowed per query, including within aggregates.
Example: SELECT request_id, authz_replay('[]', request_id) FROM auth_requests LIMIT 5.

Return values with exactly these keys:
  affected_tenants: list of tenant IDs with observed denials contrary to directory/policy entitlement;
  changes: list of {tenant_id, path, setting, value}; paths are interactive, export, scheduled;
  evidence: list of {request_id, policy_id, change_id}, linking a regressed denial to an applicable
            allowing policy and the runtime configuration change for that same tenant and path;
            provide at least one valid chain per affected tenant, and no unsupported chains;
  explanation: a nonempty explanation of the mechanism, investigation and repair tradeoffs.
Allowed settings: normalize_cached_groups (boolean), cache_key ('subject' or 'tenant_subject'),
check_membership_version (boolean), fallback ('deny' or 'authenticated'), cache_enabled (boolean).
Change only configuration settings, not directory memberships or policies. Alternative repairs,
including bypassing caches, are accepted if their behavior preserves all entitlements. Hidden
replays cover additional request combinations; passing a few visible examples is insufficient.
The explanation is retained for human review; deterministic scoring checks evidence links,
affected tenants and actual repair behavior, not prose keywords or number of agents used.
"""


def _json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def generate_incident(rng: random.Random, scale: Scale) -> tuple[TaskInput, dict[str, Any]]:
    if scale.tenants < 4 or scale.users_per_tenant < 12 or scale.groups_per_tenant < 8:
        raise ValueError("incident scales require >=4 tenants, >=12 users and >=8 groups")
    tables: Tables = {
        name: []
        for name in (
            "auth_tenants",
            "auth_users",
            "auth_groups",
            "auth_group_edges",
            "auth_memberships",
            "auth_resources",
            "auth_policies",
            "auth_tokens",
            "auth_cache",
            "auth_runtime",
            "auth_idp_events",
            "auth_changes",
            "auth_requests",
            "auth_audit",
            "auth_support",
            "auth_documents",
            "auth_source_files",
        )
    }
    tenant_ids = [f"org-{rng.getrandbits(40):010x}" for _ in range(scale.tenants)]
    affected_indices = set(rng.sample(range(scale.tenants), max(2, scale.tenants // 3)))
    repair: list[Row] = []
    runtime_changes: dict[tuple[str, str], str] = {}
    failures: dict[str, str] = {}
    # Different seeds vary the mechanism, path, migrated cohort, identities and noise.
    for ti, tenant in enumerate(tenant_ids):
        issuer = f"https://identity.example.test/{tenant}"
        migrated = ti in affected_indices or rng.random() < 0.6
        tables["auth_tenants"].append(
            {
                "tenant_id": tenant,
                "customer": f"Enterprise {rng.randrange(1000, 9999)}",
                "issuer": issuer,
                "directory_namespace": "oidc" if migrated else "legacy",
                "migration_at": "2026-04-17T08:45:00Z" if migrated else None,
            }
        )
        groups: list[Row] = []
        for gi in range(scale.groups_per_tenant):
            group = {
                "tenant_id": tenant,
                "group_id": f"role-{gi:03}",
                "legacy_id": f"role-{gi:03}",
                "current_id": f"gid-{rng.getrandbits(96):024x}",
                "display_name": f"Department {gi // 4} permission group {gi}",
            }
            groups.append(group)
            tables["auth_groups"].append(group)
            if gi == 3:
                parents = [1]
            elif gi >= 4:
                parents = rng.sample(range(gi), min(gi, rng.randint(1, 2)))
            else:
                parents = []
            tables["auth_group_edges"].extend(
                {"tenant_id": tenant, "child": group["group_id"], "parent": groups[p]["group_id"]}
                for p in parents
            )
            tables["auth_idp_events"].append(
                {
                    "event_id": f"idp-{tenant}-{gi}",
                    "tenant_id": tenant,
                    "timestamp": "2026-04-17T08:45:00Z",
                    "operation": "group_identifier_mapping",
                    "old_external_id": group["legacy_id"],
                    "new_external_id": group["current_id"],
                    "details_json": _json(
                        {
                            "issuer": issuer,
                            "display_name": group["display_name"],
                            "migration_batch": f"batch-{ti % 4}",
                            "status": "applied" if migrated else "prepared",
                        }
                    ),
                }
            )
        for ri, kind in enumerate(REPORTS):
            resource = f"report-{tenant}-{kind}"
            tables["auth_resources"].append(
                {
                    "resource_id": resource,
                    "tenant_id": tenant,
                    "classification": kind,
                    "storage_key": f"reports/{tenant}/{rng.getrandbits(40):010x}",
                }
            )
            assignments = (
                [(0, "allow"), (1, "allow")]
                if ri == 0
                else [(1, "allow"), (2, "deny")]
                if ri == 1
                else [(2, "allow")]
                if ri == 2
                else [(rng.randrange(4, scale.groups_per_tenant), "allow"), (2, "deny")]
            )
            for gi, effect in assignments:
                tables["auth_policies"].append(
                    {
                        "policy_id": f"pol-{rng.getrandbits(64):016x}",
                        "tenant_id": tenant,
                        "resource_id": resource,
                        "group_id": groups[gi]["group_id"],
                        "effect": effect,
                        "revision": 4,
                        "updated_at": "2026-04-10T16:00:00Z",
                    }
                )
        broken_path = rng.choice(PATHS)
        defect = rng.choice(("normalize_cached_groups", "cache_key", "check_membership_version"))
        if ti in affected_indices:
            failures[tenant] = defect
            repair.append(
                {
                    "tenant_id": tenant,
                    "path": broken_path,
                    "setting": defect,
                    "value": DEFAULTS[defect],
                }
            )
        for path in PATHS:
            settings = dict(DEFAULTS)
            # Benign configuration changes prevent "all changed tenants are broken" shortcuts.
            if ti not in affected_indices:
                if not migrated and rng.random() < 0.7:
                    settings["normalize_cached_groups"] = False
                if rng.random() < 0.4:
                    settings["cache_enabled"] = False
                    settings["check_membership_version"] = False
            if ti in affected_indices and path == broken_path:
                settings[defect] = "subject" if defect == "cache_key" else False
            change_id = f"change-{rng.getrandbits(64):016x}"
            runtime_changes[tenant, path] = change_id
            tables["auth_runtime"].append(
                {
                    "tenant_id": tenant,
                    "path": path,
                    "version": f"authz-2.{ti % 3}.1",
                    "settings_json": _json(settings),
                    "change_id": change_id,
                }
            )
            tables["auth_changes"].append(
                {
                    "change_id": change_id,
                    "tenant_id": tenant,
                    "path": path,
                    "timestamp": "2026-04-17T08:50:00Z",
                    "component": "authorization-adapter",
                    "description": "Apply tenant cache compatibility profile after identity rollout",
                    "before_json": _json(DEFAULTS),
                    "after_json": _json(settings),
                }
            )
            tables["auth_changes"].append(
                {
                    "change_id": f"change-{rng.getrandbits(64):016x}",
                    "tenant_id": tenant,
                    "path": path,
                    "timestamp": "2026-04-17T08:55:00Z",
                    "component": "report-renderer",
                    "description": "Renderer font-pack refresh",
                    "before_json": '{"font_pack":7}',
                    "after_json": '{"font_pack":8}',
                }
            )
        for ui in range(scale.users_per_tenant):
            subject = f"user-{ui:05}"
            category = (ui + ti) % 6
            canonical = [[0], [3], [1, 2], [2], [0], []][category]
            if ui >= 12 and category != 5:
                canonical = [*canonical, rng.randrange(4, scale.groups_per_tenant)]
            active = int(ui < 12 or ui % 37 != 0)
            tables["auth_users"].append(
                {
                    "tenant_id": tenant,
                    "subject_id": subject,
                    "directory_id": f"dir-{tenant}-{ui}",
                    "membership_version": 2,
                    "active": active,
                }
            )
            tables["auth_memberships"].extend(
                {
                    "tenant_id": tenant,
                    "subject_id": subject,
                    "group_id": groups[g]["group_id"],
                    "revision": 2,
                }
                for g in canonical
            )
            current_external = [
                groups[g]["current_id" if migrated else "legacy_id"] for g in canonical
            ]
            for valid in (1, 0):
                tables["auth_tokens"].append(
                    {
                        "token_id": f"token-{tenant}-{ui}-{valid}",
                        "tenant_id": tenant,
                        "subject_id": subject,
                        "issuer": issuer,
                        "valid": valid,
                        "claims_json": _json(
                            {
                                "sub": subject,
                                "iss": issuer,
                                "groups": current_external,
                                "aud": "reporting",
                                "session": f"session-{tenant}-{ui}",
                                "expiry_status": "current" if valid else "expired",
                            }
                        ),
                    }
                )
            # Stale snapshots include both missing grants and revoked grants. Correct runtime
            # checks invalidate them. Legacy IDs intentionally collide across tenant boundaries.
            stale = category in (0, 1, 4)
            old = [] if category in (0, 1) else [1] if category == 4 else canonical
            cached = [
                groups[g]["current_id" if migrated else "legacy_id"]
                for g in (old if stale else canonical)
            ]
            for path in PATHS:
                tables["auth_cache"].append(
                    {
                        "cache_id": f"cache-{tenant}-{ui}-{path}",
                        "tenant_id": tenant,
                        "subject_id": subject,
                        "path": path,
                        "groups_json": _json(cached),
                        "membership_version": 1 if stale else 2,
                        "created_at": "2026-04-17T08:46:00Z" if stale else "2026-04-17T08:48:00Z",
                    }
                )
    tables["auth_documents"] = [
        {
            "document_id": "directory-contract",
            "revision": 3,
            "status": "current",
            "body": "auth_users and auth_memberships are authoritative current directory state. Resolve "
            "auth_group_edges transitively within tenant. auth_groups maps external IDs to canonical "
            "group_id. Policies match tenant, resource and effective canonical group. Deny overrides "
            "allow; absent allow means deny. Subject and role identifiers collide between tenants. "
            "auth_tokens.valid summarizes signature/expiry validation; disabled subjects cannot access.",
        },
        {
            "document_id": "capture-map",
            "revision": 1,
            "status": "current",
            "body": "auth_requests is gateway capture; auth_audit is authorization trace linked by trace_id. "
            "Token IDs join auth_tokens, which holds issuer and subject. auth_runtime.change_id joins "
            "auth_changes. Cache snapshots join on tenant_id, subject_id, path. A warm replay primes "
            "the cache with prior_tenant_id's snapshot first, then the target tenant's scoped snapshot. "
            "A subject-only key reuses the first snapshot. Cold replay starts empty. Experiments use "
            "the same capture each time; they neither flush nor update persistent state. All times UTC.",
        },
        {
            "document_id": "cache-operations",
            "revision": 5,
            "status": "current",
            "body": "Settings are per tenant and execution path. normalize_cached_groups translates cached "
            "IDs through that tenant's alias map. cache_key selects subject-only or tenant+subject "
            "isolation. check_membership_version drops snapshots whose generation differs from the "
            "directory. Cold misses read current directory canonical groups. cache_enabled=false "
            "bypasses cache and increases directory load. fallback=authenticated permits requests "
            "without matching policies but does not override explicit denies or identity checks.",
        },
        {
            "document_id": "legacy-oncall",
            "revision": 1,
            "status": "superseded",
            "body": "Before enterprise tenant isolation, an authenticated fallback was used during imports. "
            "This historical workaround is not the current entitlement contract.",
        },
    ]
    tables["auth_source_files"] = [
        {
            "path": "services/authorization/adapter.py",
            "revision": "adapter-contract-v1",
            "language": "python",
            "content": ADAPTER_SOURCE,
        }
    ]
    engine = AuthorizationSandbox(tables)
    oracle = Entitlements(tables)
    affected: set[str] = set()
    evidence: dict[str, Row] = {}

    def request(
        ti: int, ui: int, path: str, resource_index: int, warm: bool, valid: int = 1
    ) -> Row:
        tenant = tenant_ids[ti]
        return {
            "tenant_id": tenant,
            "path": path,
            "resource_id": f"report-{tenant}-{REPORTS[resource_index]}",
            "token_id": f"token-{tenant}-{ui}-{valid}",
            "cache_state": "warm" if warm else "cold",
            "prior_tenant_id": tenant_ids[(ti + 1) % scale.tenants],
        }

    def capture(req: Row) -> None:
        index = len(tables["auth_requests"])
        request_id, trace_id = f"req-{index:08}", f"trace-{rng.getrandbits(128):032x}"
        req = {
            **req,
            "request_id": request_id,
            "trace_id": trace_id,
            "timestamp": f"2026-04-17T09:{index % 60:02}:{(index // 60) % 60:02}Z",
        }
        tables["auth_requests"].append(req)
        observed = engine.evaluate(req, engine.configurations)
        tables["auth_audit"].append(
            {
                "event_id": f"audit-{index:08}",
                "trace_id": trace_id,
                "service": f"report-authz-{req['path']}",
                "decision": observed["decision"],
                "http_status": 200 if observed["decision"] == "allow" else 403,
                "latency_ms": rng.randint(2, 40),
                "details_json": _json(
                    {
                        **observed,
                        "request": {k: req[k] for k in ("resource_id", "path", "cache_state")},
                        "identity": engine.tokens[req["token_id"]],
                        "runtime": {
                            "version": f"authz-2.{tenant_ids.index(req['tenant_id']) % 3}.1",
                            "region": rng.choice(("eu-west", "eu-central", "us-east")),
                            "pod": f"authz-{rng.randrange(64):02}",
                            "trace_sampled": True,
                        },
                    }
                ),
            }
        )
        supporting = oracle.supporting_policies(req)
        if observed["decision"] == "deny" and supporting:
            tenant = req["tenant_id"]
            affected.add(tenant)
            evidence.setdefault(
                tenant,
                {
                    "request_id": request_id,
                    "policy_id": supporting[0],
                    "change_id": runtime_changes[tenant, req["path"]],
                },
            )

    # A coverage floor guarantees observable healthy/affected, allow/deny and warm/cold cohorts.
    for ti in range(scale.tenants):
        for ui in range(12):
            for path in PATHS:
                for ri in range(len(REPORTS)):
                    for warm in (False, True):
                        capture(request(ti, ui, path, ri, warm))
    while len(tables["auth_requests"]) < scale.audit_requests:
        capture(
            request(
                rng.randrange(scale.tenants),
                rng.randrange(scale.users_per_tenant),
                rng.choice(PATHS),
                rng.randrange(len(REPORTS)),
                rng.random() < 0.8,
                int(rng.random() >= 0.03),
            )
        )
    if affected != {tenant_ids[i] for i in affected_indices}:
        raise AssertionError("generated incident must expose every injected regression")
    tables["auth_support"] = [
        {
            "ticket_id": f"support-{i:04}",
            "customer_tenant": tenant,
            "request_reference": evidence[tenant]["request_id"],
            "message": "Report access fails intermittently after this morning's identity migration.",
        }
        for i, tenant in enumerate(sorted(affected)[: max(1, len(affected) // 2)])
    ]
    hidden: list[Row] = []
    for ti in range(scale.tenants):
        subjects = sorted(set(range(12)) | set(rng.sample(range(scale.users_per_tenant), 12)))
        if scale.users_per_tenant > 37:
            subjects = sorted(set(subjects) | {37})  # Include a disabled subject deterministically.
        for ui in subjects:
            for path in PATHS:
                for ri in range(len(REPORTS)):
                    for warm in (False, True):
                        req = request(ti, ui, path, ri, warm)
                        req["prior_tenant_id"] = rng.choice(tenant_ids)
                        hidden.append({"request": req, "expected": oracle.decision(req)})
            # Credential and resource isolation cases are withheld from the public replay API.
            expired = request(ti, ui, "interactive", 0, True, 0)
            foreign_token = {
                **request(ti, ui, "export", 0, True),
                "token_id": f"token-{tenant_ids[(ti + 1) % scale.tenants]}-{ui}-1",
            }
            foreign_resource = {
                **request(ti, ui, "scheduled", 0, False),
                "resource_id": f"report-{tenant_ids[(ti + 1) % scale.tenants]}-summary",
            }
            for req in (expired, foreign_token, foreign_resource):
                hidden.append({"request": req, "expected": oracle.decision(req)})
    for rows in tables.values():
        rng.shuffle(rows)
    for check in hidden:
        baseline = engine.evaluate(check["request"], engine.configurations)["decision"]
        check["category"] = (
            "regressed"
            if check["expected"] == "allow" and baseline == "deny"
            else check["expected"]
        )
    return TaskInput(prompt=PROMPT, tables=tables), {
        "submission": {
            "affected_tenants": sorted(affected),
            "changes": repair,
            "evidence": list(evidence.values()),
            "explanation": "Restore the tenant/path settings changed during migration: "
            + _json(failures),
        },
        "checks": hidden,
    }


def authorization(rng: random.Random, size: int) -> tuple[TaskInput, dict[str, Any]]:
    return generate_incident(rng, SCALES[size])
