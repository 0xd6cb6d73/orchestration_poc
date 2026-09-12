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

Open <http://localhost:8000/ui> for the web control room. It can propose and approve
missions, cancel active runs, and displays the live parent/child agent hierarchy,
workflow progress, artifacts, and durable trajectory. The JSON API remains available
at <http://localhost:8000/docs>.

The launch form selects the worker runtime and orchestration mode for the run. When
`pydantic_ai` or `semantic_pydantic_ai` is selected, it also requires the provider
and model identifier. That
configuration is stored in the immutable mission plan, submitted to each domain's
execution strategy, propagated to every generated task, and shown on the run page.
The built-in `custom_python` runtime always uses the offline deterministic fixture
model, so its provider/model fields are intentionally disabled.

A basic API demonstration is:

```bash
curl -sS -X POST http://localhost:8000/runs \
  -H 'content-type: application/json' \
  -d '{"agent_runtime":{"backend":"custom_python"},"execution_mode":"managed_pool"}'

# Use the returned run id. Approval may include edits and creates a new immutable version.
curl -sS -X POST http://localhost:8000/runs/RUN_ID/approval \
  -H 'content-type: application/json' \
  -d '{"plan_version":1,"edits":{"constraints":["Offline fixtures only","Do not modify live systems","Exclude customer identifiers"]}}'

curl -sS http://localhost:8000/runs/RUN_ID
```

Completed Markdown artifacts are served at `/artifacts/ARTIFACT_ID` with immutable
ETags.

## Test and build

```bash
uv run pytest
uv build
```

Install and run the repository's formatting, linting, and type-checking hooks with:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

The checked-in `.pre-commit-config.yaml` runs Ruff lint fixes, Ruff formatting, and
Pyright through the versions pinned in `uv.lock`.

`uv.lock` pins the complete resolution. Runtime, developer, and optional telemetry
dependencies are declared in `pyproject.toml`; install Phoenix instrumentation with
`uv sync --extra telemetry`.

## Phoenix container

```bash
docker compose up --build
```

This starts one application process and a pinned Phoenix 20 non-root container. A
short-lived initialization container grants Phoenix's UID access to its named volume
before the server starts. Application
state, LangGraph checkpoints, immutable artifacts, and Phoenix data use separate
persistent storage. The application journal is authoritative; telemetry export is
best-effort. Phoenix's server license is Elastic License 2.0, the licensing exception
called out by the specification.

The image includes the optional OpenAI and Anthropic clients. To use an external
worker model from the web control room, export `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` before `docker compose up`; Compose passes those variables into
the application container without storing them in a mission plan. OpenAI-compatible
endpoints can be selected without exposing their URL through the UI:

```bash
export OPENAI_API_KEY='sk-or-v1-...'
export OPENAI_BASE_URL='https://openrouter.ai/api/v1'
docker compose up --build
```

When using this compatibility path, select `openai` as the provider in the UI and
enter the upstream service's model identifier. Compose also forwards an optional
`ANTHROPIC_BASE_URL`.

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

## Hybrid board swarm

`hybrid_v1` is a versioned policy bundle inside `board_claim`; it is not another
execution backend. Select both values in the web launch form, or create a run with:

```json
{
  "execution_mode": "board_claim",
  "swarm_strategy": "hybrid_v1",
  "hybrid": {
    "proposals_per_round": 3,
    "max_collaboration_rounds": 2,
    "max_critics_per_candidate": 2,
    "initial_candidate_visibility": "sealed_to_round"
  }
}
```

The approved plan pins the resolved allocation, context, communication,
collaboration, acceptance, and completion policy versions. The same board task,
claim, lease, and capacity machinery remains authoritative. A capability router
selects provisionable runtime profiles; a shared context assembler serves both
agent backends and records exact artifact/message inclusion plus output provenance.

Hybrid investigation uses bounded application-controlled rounds: independent
proposals are sealed, released together, versioned into candidate clusters,
challenged against original evidence, scored as a vector, and retained even when
refuted. A separate assurance worker verifies the selected artifact. Typed artifact
requirements distinguish a published candidate from verified evidence and an
accepted deliverable, and the main orchestrator records a deterministic final gate.

State ownership remains deliberately split:

| State | Authoritative owner |
|---|---|
| Approved objective, constraints, strategy, policy versions | Immutable mission plan |
| Task readiness, active owner, lease, fencing generation | Board-claim strategy |
| Worker graph state and validation continuation | LangGraph checkpoint thread |
| Artifact bytes and access policy | Content-addressed artifact store and application DB |
| Claims, contradictions, candidate history, tests | Evidence blackboard and collaboration controller |
| Verification, acceptance, delivery authority | Verification service and completion gate |
| Agent population and profile capacity | Capacity scheduler |
| Observability | Durable event journal; Phoenix is a best-effort projection |

New routers, collaboration policies, verification policies, and completion gates can
be registered behind these contracts without changing LangGraph worker graphs or
telemetry bootstrap registration. The compatibility `board` strategy remains the
default for existing and newly created non-hybrid runs.

## Interchangeable agent backends

Worker execution is selected above the LangGraph and telemetry layers through
`AgentExecutorRegistry`. The built-in `custom_python` executor preserves the existing
LangGraph OODA worker unchanged. The `pydantic_ai` executor builds a Pydantic AI
`Agent`, exposes the same RBAC-protected tool gateway, and converts its validated
structured output into the same `WorkerResult` contract.

An orchestrator can select a backend per workflow task, so both implementations can
run in the same DAG:

```python
TaskSpec(
    id="draft",
    role="claim_drafter",
    agent_backend=AgentBackend.PYDANTIC_AI,
    goal="Draft one evidence-backed claim.",
    output_schema="DraftClaim",
)
```

A swarm scheduler selects the backend for the workers it materializes through the
execution policy:

```python
ExecutionPolicy(
    mode=ExecutionMode.BOARD_CLAIM,
    agent_backend=AgentBackend.PYDANTIC_AI,
    max_workers=2,
    allowed_roles={"manifest_reader"},
)
```

If a task does not select a backend, the role's `RoleSpec.agent_backend` is used.
Every `AgentInstance` persists the resolved backend, and a stable agent identity
cannot silently switch implementations during recovery. Additional implementations
can be registered without changing `WorkerAdapter` or any scheduler.

`pydantic-ai-slim` provides the framework-neutral Pydantic AI core. The `llm`
optional dependency installs the OpenAI and Anthropic clients used by the documented
configuration path:

```bash
uv sync --extra llm
export OPENAI_API_KEY='...'
# Anthropic instead uses ANTHROPIC_API_KEY.
```

Select the provider and model on a run (`openai:gpt-5-mini`, for example), or inject
a `PydanticModelFactory` into `Runtime`. Provider SDKs read credentials from their
standard environment variables; secrets are never written to plans, reports, or the
event journal. Per-run `agent_runtime.options` are passed to Pydantic AI as model
settings, so values such as `temperature` are configurable independently of the
orchestration mode. Tests inject Pydantic AI's `FunctionModel`, so they remain
offline and deterministic.

The `semantic_pydantic_ai` subtype runs the same typed Pydantic AI worker and tool
boundary, then validates the result before creating its output artifact. It first
checks deterministic protocol invariants such as assigned hypothesis identity,
arithmetic consistency, record counts, and exact publication metadata. A second,
tool-free Pydantic AI invocation independently assesses every acceptance criterion,
internal consistency, evidence sufficiency, and schema-specific semantic criteria.
The executor derives acceptance from the structured assessment; the reviewer does
not return the worker's final status directly. On rejection, the executor sends the
validation issues back to the same worker with its original conversation history and
asks for a complete corrected output. One semantic revision is allowed by default;
roles can override `max_semantic_revisions`. Each rejection is recorded as
`agent.semantic_validation_completed`, each retry as `agent.output_revision_requested`,
and a repeatedly invalid result returns a failed `WorkerResult`.

## Agent evaluations

For orchestration comparisons, run the complete incident benchmark. Unlike the two
focused worker regression cases below, this workload is deliberately not a toy: one
trial executes parallel metric and evidence branches, manifest-backed timezone
validation (including pause/resume for the OODA backend), percentile and correlation
analysis, causal claim review, final report
synthesis, and exact artifact lineage. The standard workload has at least 12 worker
tasks across four workflows. `hybrid_v1` additionally runs sealed independent
proposals, critiques, scoring, and independent verification.

The default command runs all four orchestration modes with the same offline worker
implementation and prints completion, quality, latency, tool, worker, and token
metrics:

```bash
uv run hierarchical-ooda-benchmark \
  --repeat 3 \
  --report eval-reports/orchestration-offline.json
```

Use a real LLM after installing the provider clients and exporting its API key:

```bash
uv run --extra llm hierarchical-ooda-benchmark \
  --mode hierarchical_dag \
  --mode board_claim \
  --mode managed_pool \
  --mode speculative \
  --backend pydantic_ai \
  --model openai:gpt-5-mini \
  --model-setting temperature=0 \
  --repeat 3 \
  --report eval-reports/orchestration-openai.json

uv run --extra llm hierarchical-ooda-benchmark \
  --mode board_claim \
  --strategy hybrid_v1 \
  --backend pydantic_ai \
  --model anthropic:claude-sonnet-4-5 \
  --report eval-reports/hybrid-anthropic.json
```

Trials run sequentially by default to avoid latency distortion and provider rate
limits; `--max-concurrency` is available when throughput is the metric of interest.
The machine-readable report includes every quality check plus per-trial counts and
token usage. Generated report text is omitted unless `--include-output` is supplied;
its SHA-256 digest is always retained. A failed quality gate produces a non-zero exit
status.

For focused prompt and tool-use regression tests, the `poc.evaluation` package uses
Pydantic Evals to run individual worker experiments through the same worker boundary.

An evaluation variant can
change the agent backend, provider/model, system prompt, per-role allowed tools, and
execution limits without changing a golden dataset. The standard evaluators grade:

- successful outcome and expected result fields;
- required and forbidden tool use;
- tool argument fragments; and
- tool success and maximum tool-call budgets.

Reports include the effective provider, model, allowed tools, and a SHA-256 prompt
fingerprint for each case. Prompt text itself is not persisted in reports.

Run the built-in offline baseline and save its full report:

```bash
uv run hierarchical-ooda-eval \
  --backend custom_python \
  --name custom-baseline \
  --report eval-reports/custom-baseline.json
```

The deterministic `custom_python` backend is useful as an infrastructure baseline.
Prompt and provider/model experiments should use `pydantic_ai` (or another executor
that consumes those `RoleSpec` fields).

Evaluate a Pydantic AI prompt/model variant and compare it with a previous report:

```bash
uv run --extra llm hierarchical-ooda-eval \
  --backend pydantic_ai \
  --model openai:gpt-5-mini \
  --prompt-file prompts/manifest-worker.txt \
  --tools manifest_reader=read_manifest \
  --name manifest-prompt-v2 \
  --baseline eval-reports/manifest-prompt-v1.json \
  --report eval-reports/manifest-prompt-v2.json
```

The command returns a non-zero exit status if a case, evaluator, or assertion fails,
making it suitable for CI quality gates. Use `--repeat` to measure nondeterministic
variants.
New typed datasets and custom evaluators can be supplied programmatically through
`create_agent_dataset` or registered with `AgentEvaluationDatasetRegistry`.

The PoC intentionally excludes live infrastructure access, arbitrary code execution,
distributed queues, and hot graph mutation. Swarm coordination is implemented and
tested in-process; worker process materialization remains a trusted runtime concern.
