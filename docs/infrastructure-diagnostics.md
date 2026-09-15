# Fast infrastructure diagnostics

Run this after changing SQL workers, provider wrappers, output contracts, budgets, or
coordination policies:

```sh
.venv/bin/python -m poc.evaluation.suite --no-env-file diagnose \
  --output var/evaluation/infrastructure-diagnostic.jsonl
```

The default is fully offline. It executes **121 probes** in roughly **3 seconds** on the
development machine, with a 30-second overall watchdog. Scripted model responses drive
the real adapters, SQL connections, model wrappers, artifact gates, and controllers.
There are no paid requests, seeds, repetitions, optimization tasks, or strategy rankings.
The standalone command `python -m poc.evaluation.diagnostic` runs the same checks.

## Coverage

All built-in orchestration strategies are exercised: single, review, hierarchical DAG,
board claim, managed pool, speculative, and both hybrid strategies. Each runs with native tools and JSON
transport. Team strategies exercise both `reliable-v2` serial execution and
`concurrent-v1`; the SQL baseline also checks the public tool environment. Older
`legacy-v1` and `bounded-v1` controls are outside this diagnostic's acceptance contract.

Successful canaries must reach and complete every required role, run SQL, deliver a
complete artifact, and emit the expected controller events. A plausible final answer
alone cannot pass. Concurrent canaries rendezvous in their first branches, so the scan
can check actual overlap, shared reservations, board contention, and independent SQL
evidence references. The workload contains only three jobs, with valid start times
already supplied in the public release column.

Fault probes check:

- Context admission before provider calls and effective completion caps after wrappers.
- Malformed identifiers and incomplete output contracts at every strategy's handoffs.
- Request ceilings at every strategy, plus request-versus-phase timeout provenance.
- Review retention after an empty revision and speculative retention after reconciliation
  times out, with return time reserved before the hard deadline.
- Hybrid critique contracts, contradictory positive validity/zero-constraint judgments,
  foreign evidence references, negative verifier verdicts, and a failed proposal under an
  explicit one-candidate quorum. Retention never bypasses verification.
- Pool reassignment to a different worker generation, repeated SQL errors, and concurrent
  cancellation with all model callbacks drained.
- A transient 429, reasoning-only output, duplicate JSON keys, multiple actions, and
  explicitly enabled escaped-JSON normalization.

Injected failures count as a pass only when the fault was actually exercised and the
specific expected rejection or recovery occurred. An unrelated exception, a skipped
stage, an accepted malformed artifact, or missing coverage fails the scan. Tests of the
diagnostic deliberately bypass an artifact gate and replace a controller with a no-op
to check that these regressions are detected.

## Short live-provider canaries

Scripted responses cannot establish that a real endpoint accepts a binding or that a
model follows a role schema. To check that boundary, select **one** binding from a model
config:

```sh
.venv/bin/python -m poc.evaluation.suite diagnose \
  --live-config configs/evaluation-reliable-teams.json \
  --model deepseek-v4-flash-0731 \
  --output var/evaluation/infrastructure-live.jsonl
```

This opts into paid model calls. Only the selected model binding is read; the source
matrix's tasks, repetitions, seeds, budgets, architecture options, and other models are
not launched. Multiple bindings require an explicit `--model` selection. These canaries
use the same three-job public lookup task and probe policies, with no fault injection.
All variants produce 25 cases including the SQL baseline, with four active cases, a
20-second per-case watchdog, and a 180-second whole-run deadline. Each case has ceilings
of 60 requests, 40 tools, 200,000 total tokens, and 2,048 completion tokens per request.
These are hard diagnostic allowances, not spending targets; successful canaries typically
need only a query and a submission per role. No result is ranked by time, cost, or solving
quality. A live timeout or role rejection is evidence to investigate, not proof of an
orchestrator defect. Long-context load behavior and model-solving ability require separate
experiments.

To narrow a follow-up or preview the exact cases:

```sh
.venv/bin/python -m poc.evaluation.suite --no-env-file diagnose \
  --strategies review hybrid_v1 --protocols json --policies reliable-v2 --list-probes

.venv/bin/python -m poc.evaluation.suite diagnose \
  --live-config configs/evaluation-reliable-teams.json --model deepseek-v4-flash-0731 \
  --protocols json --case-seconds 15 --suite-seconds 90 --concurrency 4
```

## Evidence and exit status

The JSONL output begins with the declared probe manifest, flushes a record after every
case, and ends with a summary. Each record contains named invariant checks, causal events,
phase records, resource consumption, sanitized exception types and failure scope, and the
submitted artifact. It excludes benchmark grades and candidate feasibility scores.
Existing files are never overwritten; use a new path for each run.

Exit code 0 means all requested probes passed. Exit code 1 means an invariant failed or
coverage is incomplete. Invalid arguments or an existing output file produce exit code 2.
When the suite deadline expires, active work is cancelled and remaining cases appear as
`not_run`; they cannot silently become successful coverage. If the process is interrupted
externally, already-flushed records remain available even without a final summary.

Only rerun the focused checks while developing this evaluation:

```sh
.venv/bin/pytest -q tests/test_infrastructure_diagnostic.py tests/test_evaluation_suite.py
```
