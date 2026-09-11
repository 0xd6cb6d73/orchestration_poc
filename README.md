# Hierarchical OODA incident-investigation PoC

An offline, inspectable implementation of a three-tier agent hierarchy. The main
orchestrator creates domain supervisors, supervisors authorize small workers, and
every worker completes one task through a bounded LangGraph OODA graph. Workflow
DAGs and agent authority are deliberately separate. Sub-orchestrators can also
authorize hierarchical DAG, board-and-claim, managed-pool, or speculative execution.

The demo uses synthetic checkout metrics, logs, deployment records, and a manifest.
It never connects to or modifies a live system. The deterministic model adapter is
intentional: the architecture, checkpoints, validation exchange, authorization, and
trajectory can be exercised without API credentials.

## Run with uv

The repository uses uv for dependency locking, environment management, commands,
and the native `uv_build` package backend.

```bash
uv sync
uv run hierarchical-ooda-poc
```

Open <http://localhost:8000/docs>. A basic demonstration is:

```bash
curl -sS -X POST http://localhost:8000/runs \
  -H 'content-type: application/json' -d '{}'

# Use the returned run id. Approval may include edits and creates a new immutable version.
curl -sS -X POST http://localhost:8000/runs/RUN_ID/approval \
  -H 'content-type: application/json' \
  -d '{"plan_version":1,"edits":{"constraints":["Offline fixtures only","Do not modify live systems","Exclude customer identifiers"]}}'

curl -sS http://localhost:8000/runs/RUN_ID
```

The read-only page is at `/ui/runs/RUN_ID`; completed Markdown artifacts are served
at `/artifacts/ARTIFACT_ID` with immutable ETags.

## Test and build

```bash
uv run pytest
uv build
```

`uv.lock` pins the complete resolution. Runtime, developer, and optional telemetry
dependencies are declared in `pyproject.toml`; install Phoenix instrumentation with
`uv sync --extra telemetry`.

## Phoenix container

```bash
docker compose up --build
```

This starts one application process and a pinned Phoenix 14.0.0 container. Application
state, LangGraph checkpoints, immutable artifacts, and Phoenix data use separate
persistent storage. The application journal is authoritative; telemetry export is
best-effort. Phoenix's server license is Elastic License 2.0, the licensing exception
called out by the specification.

After startup, `docker compose ps` should report the app as `healthy`. The API is at
<http://localhost:8000/docs>, while Phoenix is at <http://localhost:6006>.

## What the demo proves

- Plan approval is version-specific; edits create a new immutable revision.
- Authority checks prevent main-to-worker, supervisor-to-supervisor, and worker spawns.
- Domain tools are restricted to active worker roles and active plan revisions.
- Baseline and incident percentile workers have distinct identities and execute as
  parallel workflow branches.
- The incident worker interrupts on a missing timezone. Its supervisor remains free,
  delegates a one-task manifest workflow, then resumes the saved workflow thread.
- Every task emits a typed `WorkerResult`; dependency adapters bind only selected
  outputs and artifact references.
- Files are content-addressed and writes/tool operations have stable idempotency keys.
- Coordination and all OODA phases are journaled and mirrored to searchable Phoenix
  spans when telemetry is enabled.

## Execution strategies

`ExecutionCoordinator` selects implementations through `StrategyRegistry`; it does
not contain a mode switch. New implementations register a factory against a stable
mode, while strategy-specific configuration stays in `ExecutionPolicy.options`.
All built-in strategies share the same `submit`, `cancel`, and `snapshot` lifecycle.

```python
policy = ExecutionPolicy(
    mode=ExecutionMode.BOARD_CLAIM,
    max_workers=2,
    allowed_roles={"manifest_reader"},
    max_task_attempts=3,
    lease_seconds=30,
)
handle = await runtime.submit_execution(
    run_id=run_id,
    owner_suborchestrator_id=supervisor_id,
    goal_ref="manifest-analysis",
    policy=policy,
)
board = runtime.strategies.get(policy.mode)
board.post_task(
    execution_id=handle.execution_id,
    task_id="read-timezone",
    required_role="manifest_reader",
    task_spec_ref="fixture:manifest-timezone",
)
claim = board.claim_next(execution_id=handle.execution_id, worker_id=worker_id)
```

- Board workers are materialized by a trusted, replaceable `CapacityScheduler` under
  plan role/population limits. Claims then use atomic compare-and-set updates,
  expiring leases, increasing fences, attempt history, and a lease reaper.
- Managed pools enforce population and role quotas, then use deterministic score,
  least-recently-assigned, and slot-ID arbitration. Assignments are fenced leases.
- Speculative groups cap fan-out, retain every candidate, reject invalid candidates,
  and serialize final reconciliation.
- The tool gateway checks the current claim, assignment, or candidate grant against
  canonical storage. Tokens are hashed at rest and omitted from events and reprs;
  speculative grants cannot perform direct external effects.

SQLite keeps the PoC self-contained. The transaction-bounded coordination layer can
be backed by PostgreSQL in a distributed runtime without changing the strategy API.

The PoC intentionally excludes live infrastructure access, arbitrary code execution,
distributed queues, and hot graph mutation. Swarm coordination is implemented and
tested in-process; worker process materialization remains a trusted runtime concern.
