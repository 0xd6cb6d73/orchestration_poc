# Tenant authorization incident benchmark

The `authorization` family asks an agent to investigate report-access failures after
an enterprise identity migration, identify affected tenants, link evidence across
sources, and submit an executable configuration repair. It is registered in the
existing evaluation suite and works with single, review and all team strategies,
using native tools or the JSON transport.

This is a generated application simulation. It does not use customer data, require
downloads, or connect to a deployed identity provider. All corpus construction code
is checked in under `poc/evaluation/authorization`; generated evidence belongs under
ignored `var/`. No large fixture files or additional dependencies are needed.

## Task and evidence

The initial question describes intermittent report denials following a migration.
It does not identify the affected execution path, prescribe agent roles, or supply
the investigation plan. The public answer contract asks for:

- `affected_tenants`: all tenants with captured denials contrary to their entitlements;
- `changes`: tenant/path configuration edits;
- `evidence`: request → applicable allowing policy → runtime change links;
- `explanation`: the mechanism, investigation and repair tradeoffs, for human review.

The 17 public tables represent directory users and memberships, nested groups and
identifier mappings, tenant-owned reports and policies, identity-provider events,
token claim captures, cache snapshots, per-path configuration, change history,
gateway requests, authorization traces, support tickets, operational documentation,
and an application adapter source excerpt. They are exposed through one SQL tool,
with different identifiers requiring joins between sources. Only some affected
tenants have support tickets. Audit decisions describe the faulty application's
behavior, not intended access.

Seeds vary tenant identities, migrated cohorts, group hierarchy, memberships,
report policies, affected tenants and paths, failure mechanisms, and background
requests. Faults include:

1. Cached provider group identifiers are used without tenant-specific normalization.
2. Membership generation checks are disabled, retaining missing or revoked grants.
3. Cache keys omit the tenant, allowing one tenant's snapshot to affect another.

An instance can contain different faults in different tenants. Some healthy tenants
also have configuration changes: for example, legacy identifiers already match
canonical identifiers, or version checks are irrelevant because caching is off.
A renderer deployment and a superseded fallback runbook provide plausible competing
leads. Neither "every denial is a regression" nor "every changed tenant is affected"
is a correct rule.

Entitlement always comes from active users, current directory membership, transitive
tenant-scoped groups, and resource policies, with explicit deny precedence and default
deny. Subject and canonical group IDs deliberately collide between tenants. Signature
and expiry verification are represented by a captured validity flag; real cryptographic
token validation is outside this task.

## Data scale

| Tier | Tenants | Users | Groups | Reports | Gateway requests | Authorization traces |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 12 | 1,920 | 576 | 144 | 40,000 | 40,000 |
| Hard | 24 | 7,680 | 2,304 | 288 | 120,000 | 120,000 |
| Stress | 48 | 30,720 | 6,144 | 576 | 300,000 | 300,000 |

Measured seed-0 public JSONL sizes are approximately **61 MB / 190 MB / 498 MB**
for standard / hard / stress (decimal MB; larger-tier measurements use the test split).
Standard dev seed 0 includes 276 policies,
5,760 cache snapshots and 3,840 token captures. Actual bytes and membership/edge counts
vary by seed; the export manifest records exact row counts, byte counts, schemas and
SHA-256 hashes. Seed-0 self-checks across all three tiers exercised approximately
22,000 / 45,000 / 89,000 hidden replay combinations. Volume comes from structured request/identity/execution evidence,
not arbitrary padding. All records are synthetic and use `.test` issuer domains.

Authorization SQL responses allow **2,000 rows and 1 MiB of serialized UTF-8 JSON**.
A query returning 100 complete audit rows typically exceeds the older suite's 64 KB
limit. Oversized responses return an error asking the agent to narrow the query.
SQL retains its 16 KB statement limit and approximately 20 million opcode interrupt
budget. Other families keep their existing 200-row / 64,000-byte limits.

## Repair sandbox

Agents can use the additional SQL function:

```sql
SELECT request_id, authz_replay('[]', request_id)
FROM auth_requests
WHERE cache_state = 'warm'
LIMIT 10;
```

`authz_replay(changes_json, request_id)` executes the public application behavior
with the proposed edits. For example, an edit has the form:

```json
{"tenant_id":"org-...","path":"export","setting":"cache_enabled","value":false}
```

The first function argument is a JSON **list** of edits. Supported settings are
`normalize_cached_groups`, `cache_key`, `check_membership_version`, `fallback`, and
`cache_enabled`; their types and values are documented in the task prompt and
`auth_documents`. Edits target existing tenant/path pairs. Unknown settings, wrong
types and duplicate edits to one setting are rejected. There is no arbitrary code
execution or mutation of policies and memberships.

The result contains the observed decision, matched policies, effective groups and
cache behavior where applicable. It contains no expected decision, pass/fail result
or constraint score. Request IDs must exist in the public capture. Each replay resets
its cache state: warm requests replay captured cache priming, while cold requests
start empty. Experiments cannot change subsequent experiments or other workers.
At most 2,000 replay invocations are allowed per SQL query, including aggregates;
queries and errors consume the ordinary shared tool budget. Forked team environments
receive the same function, data and bounds with their allocated tool allowance.

This interface adds experiments without requiring strategy-specific tools or a new
model transport. It is deliberately a configuration repair task, not a code-patching
or full production-deployment task. Cache bypass is accepted as an alternative
mitigation when it preserves entitlements; the simulator does not grade the resulting
directory load or performance cost. Those tradeoffs belong in the explanation.

## Grading and reproducibility

The hidden reference uses a separate directory/policy oracle that does not consult
cache contents or runtime configuration. Grading applies the submitted repair to
withheld request combinations across tenants, paths, reports, and cold/warm states,
including different cache-priming tenants, expired credentials, disabled subjects,
revoked grants, explicit denies and foreign tokens/resources. These are withheld
combinations over the same captured world, not a hidden second application or secret
external corpus. Dev and test use separate seeded streams.

Each trial records:

- `restored_access`: previously regressed allowed requests that now succeed;
- `preserved_access`: already-working allowed requests that remain allowed;
- `preserved_denials`: forbidden requests that remain or become denied;
- `tenant_impact`: Jaccard agreement on affected tenant IDs;
- `evidence_support`: affected-tenant coverage with valid relational evidence chains,
  penalizing unsupported or duplicate chains.

`fraction_correct` is the mean of these five dimensions. `exact=1` requires all five
to equal one. A single unintended grant prevents full success. Broken answer shapes
or invalid patches score zero. Separating regressed access from healthy access keeps
the many healthy requests from masking a failure to repair the incident. Partial
scores are diagnostic, not evidence that a repair is acceptable.

Alternative patches are executed and judged by behavior, not matched against the
generator's patch. Evidence checks verify real request/policy/change relationships;
the free-form causal explanation is retained but not automatically semantically
graded. The benchmark neither requires a particular tool sequence nor rewards agent
count. A correct final patch is evaluated even if the agent did not call replay.

The case hash includes public data and operator references. Existing report manifests,
implementation digests, resume checks and Phoenix publication work for the new family.
Phoenix publication includes references for operators, as with other families; it
must not be exposed as an investigative data source. Large tiers also increase report
publication volume and per-trial SQLite memory usage.

## Run

Generate an inspectable public corpus, with no model calls:

```bash
uv run python -m poc.evaluation.authorization generate \
  --seed 0 --split dev --difficulty standard \
  --output var/authorization/dev-standard-0
```

This creates `manifest.json` and `public/*.jsonl`, plus `public/task.txt`. The output
directory must not already exist. References are not exported by default. Operators
can explicitly add `--include-reference` to write a separate `private/reference.json`;
never expose that directory to evaluated agents.

Check that the planted repair passes and the deployed configuration fails:

```bash
uv run python -m poc.evaluation.authorization check --seed 0
```

This is an environment/grader self-check, not a competitive agent result. The existing
`sql-baseline` does not solve authorization incidents.

Start a model pilot with one strategy and seed:

```bash
uv run python -m poc.evaluation.suite run \
  --config configs/evaluation-authorization.json \
  --models deepseek-v4-flash-0731 --strategies single --seeds 0 \
  --output var/evaluation/authorization-single-seed0.jsonl
```

The example configuration includes seven native strategies and three dev seeds, for
21 trials if no subset is selected. It uses the repository's existing DeepSeek endpoint
configuration; replace or extend `models` for the desired comparison. Each trial allows
80 requests, 120 tool calls, one million tokens and 900 seconds. These are starting
allowances, not calibrated budgets or quality claims. No external requests are made by
generation, export, grading, or the self-check. The suite generates data directly;
exporting first is optional.

To grade a manually produced answer without model calls:

```bash
uv run python -m poc.evaluation.authorization grade \
  --seed 0 --split dev --difficulty standard --answer /path/to/answer.json
```

The answer file must contain `{"values": {...}}`. Use the same seed, split and tier as
the investigation. This operator command exposes scores and must remain outside the
agent tool boundary.

Before drawing orchestration rankings, calibrate on several dev cases, freeze the
generator and budgets, and compare matched held-out seeds. This first implementation
provides a reproducible workload and objective outcome checks; it does not establish
that any orchestration strategy outperforms a single agent.
