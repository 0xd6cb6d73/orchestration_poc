# Orchestration evaluation suite

The task-independent suite lives in `poc/evaluation/suite`. The existing
`hierarchical-ooda-eval` and `hierarchical-ooda-benchmark` commands remain incident-demo
regression checks. Their report phrase and worker-count gates are not capability scores.
The new suite grades final answers and constraints, without rewarding extra agents or calls.

## Fast infrastructure check

Use [the diagnostic evaluation](infrastructure-diagnostics.md) to exercise every strategy
and known failure/recovery paths in seconds, without running the benchmark matrix:

```sh
.venv/bin/python -m poc.evaluation.suite --no-env-file diagnose
```

It also offers short live-provider canaries for one selected model binding.

## Run

```bash
uv sync --extra llm --extra evaluation --extra telemetry

# Fully offline environment/grader validation, no model calls:
uv run python -m poc.evaluation.suite run \
  --config configs/evaluation-smoke.json --output var/evaluation/smoke.jsonl

# The evaluation CLI loads OPENAI_BASE_URL and OPENAI_API_KEY from .env automatically.
# Exported variables take precedence over .env values.
export PHOENIX_ENABLED=1
export PHOENIX_BASE_URL=http://localhost:6006
export PHOENIX_OTLP_HTTP_ENDPOINT=http://localhost:6006/v1/traces
# For a remote authenticated instance also export PHOENIX_API_KEY.

docker compose up -d phoenix
uv run python -m poc.evaluation.suite run \
  --config configs/evaluation-models.json --output var/evaluation/models.jsonl --max-concurrency 4 --phoenix

# Publish saved results later, without repeating model calls:
uv run python -m poc.evaluation.suite publish var/evaluation/models.jsonl
```

The evaluation CLI loads `.env` from the working directory before creating models,
loading plugins or configuring Phoenix. It parses the file without executing shell commands
or expanding variable references, and preserves existing exported variables. A missing default
`.env` is fine; an explicitly selected missing file is an error. To select another file use
`python -m poc.evaluation.suite --env-file /path/to/models.env run ...`; use `--no-env-file`
to rely only on exported variables. These global flags go before `run` or `publish`.

Trials run concurrently, with **four active trials by default**. Set `max_concurrency`
in the matrix JSON or override it with `run --max-concurrency 8`; use `1` for serial
execution. Each trial has its own database, token/request/tool counters and timeout;
time spent waiting for a worker does not count toward that timeout. Draft and review
stages within one trial remain sequential and share that trial's budget. Completed trials
are flushed immediately in completion order. Cancellation stops active workers and leaves
completed records readable. The configured concurrency is recorded in the manifest.
Progress is printed to stderr at startup and whenever a trial starts or stops, as
`Progress: 3/12 finished, 4 running`. Finished includes terminal failures and results
already saved when resuming; queued trials are not counted as running.

Select a subset of the config with `--models`, `--strategies`, `--families` and `--seeds`:

```bash
uv run python -m poc.evaluation.suite run \
  --config configs/evaluation-smoke.json --output var/evaluation/subset.jsonl \
  --models sql-oracle --strategies sql-baseline --families ledger access --seeds 0 2
```

Each flag accepts space-separated values and can be repeated. Models are selected by
their config `name`, not provider model ID. Omitted flags include the entire configured
axis; unknown values are rejected before execution. Selections preserve config order and
ignore duplicate requests. All configured repetitions run for the selected combinations.
The report records the selected matrix, including only the selected strategies' architecture
options. To resume it, pass the same subset flags; use a new output for a different subset.

Concurrency overlaps model API waits, not CPU-heavy Python or SQLite computation.
Choose a limit suitable for the endpoint's capacity and keep it consistent when comparing
latency results. The concurrency limit controls trials; an adapter can still make multiple
simultaneous requests within one trial. Provider rate limits remain in effect.

New runs use exclusive creation. Terminal trials and candidate/phase events are flushed
and fsynced immediately. `run --resume` requires the same config, case hashes and
implementation digest (including the lockfile and registered plugin source files). It skips
**all recorded terminal trials**, including wrong answers, timeouts and provider failures.
Only trials without a terminal record run again, under a fresh attempt ID. This is recovery
of an interrupted sweep, not retry-until-success. Candidates from interrupted attempts stay
in the journal for inspection and are not supplied to the restarted model.

`read_report` reconstructs summaries and coverage without a final summary record. Resume
repairs only an unterminated, invalid last JSON record, saving the original file first;
interior corruption and duplicate trial keys are rejected. File locks prevent concurrent
writers to the same report or publication receipt. A run that ends with missing terminal
records cannot report successful completion.

Publication uses the report's `.phoenix.json` receipt, saved atomically before remote
writes and after each dataset, experiment and run. Retries reuse the same dataset and
experiments, reconcile remote runs (including writes whose responses were lost), and
upsert evaluations. Conflicting remote output is an error, never an overwrite. Do not
remove the receipt when retrying. The Python `publish` API requires `receipt_path` for
durable retry behavior; calls without a receipt explicitly create a fresh publication.
Legacy receipts are adopted by checking dataset hashes, experiment configuration and
recorded run outputs. Publishing never re-executes models.

```bash
# Resume an interrupted sweep without rerunning terminal failures:
uv run python -m poc.evaluation.suite run --config configs/evaluation-models.json \
  --output var/evaluation/new-models.jsonl --resume

# Inspect local coverage or reconcile with Phoenix, without model calls or remote writes:
uv run python -m poc.evaluation.suite inspect var/evaluation/models.jsonl
uv run python -m poc.evaluation.suite inspect var/evaluation/models.jsonl --phoenix
```

The historical `orchestration-3f4f3fadb4c7-f64816d5434c` report has 27/240 terminal
records locally and in Phoenix, with no final summary. The other 213 results were never
recorded locally. Its older implementation cannot be resumed by this changed harness;
use its original code/environment to continue it, or create a new versioned report.

The full example matrix is **240 trials**: six models × two strategies × two challenge families ×
five seeds × two repetitions. Start calibration with one seed, one repetition and `single`.
Each trial has its own budget; the total sweep can consume the sum of those budgets.
The smoke configuration uses the `baseline` cohort and has **no LLM**; its success proves
that the environment and graders agree, not that any model has solved the benchmark.

## Models and strategies

`configs/evaluation-models.json` contains these exact user-selected model IDs:

| Cohort | Model |
| --- | --- |
| 7B | `meta-llama/llama-3.1-8b-instruct` |
| 7B | `qwen/qwen-2.5-7b-instruct` |
| 27B | `mistralai/mistral-small-3.2-24b-instruct` |
| 27B | `qwen/qwen3.8-27b` |
| 100B | `openai/gpt-oss-120b` |
| frontier-flash | `deepseek/deepseek-v4-flash-0731` |

Cohorts are comparison labels, not exact parameter counts (the selected
Llama is 8B and Mistral Small is 24B). Use exact provider model IDs when extending
the matrix. No harness rewrite is needed.
For an OpenAI-compatible local server, change the environment variable referenced by
`base_url_env` and supply that server's model ID. For Pydantic AI's other providers omit
`base_url_env` and use `provider:model`. Install its provider extra separately if needed.

`single` runs one tool-using agent. `review` runs a drafter and a fresh reviewing agent
that receives the draft and can inspect the same database. Both stages default to the matrix model; explicit execution phase bindings can override it.
They share request/token/tool limits and share one wall-clock timeout. They use native
function calling. `single-json` and `review-json` offer an explicit text JSON action
protocol for endpoints without native tools. Do not pool scores from different protocols.
`sql-baseline` is an independent deterministic reference implementation for the three
analysis families, for validating generators and graders. It does not solve scheduling.

The suite also supports every built-in execution method through generic SQL task bindings
in `poc/execution/sql_orchestration.py`. These invoke the existing scheduler and hybrid
controllers with trial-local authority and storage; they do not run the incident workflow.

| Strategy | SQL task policy (`sql-team-v1`) |
| --- | --- |
| `hierarchical_dag` | LangGraph plan → solve DAG, with a persistent execution authority record |
| `board_claim` | Two workers claim dependency-ordered plan/solve tasks under fenced leases |
| `managed_pool` | Two eligible pool slots bid for plan/solve offers; the pool arbitrates and completes assignments |
| `speculative` | Two authorized independent candidate submissions, followed by a model reconciliation and persisted reconciliation decision |
| `hybrid_v1` | Two sealed board-claimed proposals, fresh critiques, controller selection, independent model verification, and the existing acceptance/delivery gates |
| `hybrid_v2` | An LLM orchestrator decomposes the task into 2–8 small scoped tasks (with per-task scope and definition of done), two sealed plan candidates are judged and deterministically selected, tasks run in dependency waves with per-task critiques, an orchestrator decision loop (accept/revise/add/escalate) drives bounded revisions, and integration flows through the existing independent verification and delivery gates |

Every method also has a `-json` variant. Discover registered strategies (including plugins):

```bash
uv run python -m poc.evaluation.suite list
uv run python -m poc.evaluation.suite run \
  --config configs/evaluation-methods.json --output var/evaluation/methods.jsonl
```

The example compares all seven native model strategies on one model, two families and
one seed (14 trials). Edit the matrix model entry for your endpoint. Subset flags select
entries already present in that matrix; add `-json` names to the matrix to use that protocol.

The SQL team policies are intentionally bounded: candidate generation and worker model
calls within a trial are **serial**, with one shared request/token/tool budget and absolute
deadline. This measures candidate diversity and coordination behavior, not parallel fanout
latency or distributed worker throughput. No candidate sees another sealed proposal.
Hybrid scores and verification are model judgments over public SQL data; the official
grader sees only the final returned answer. Rejected verification fails the trial.

Team workers use the matrix model unless an execution role has a `phase_models`
binding; `phase_models.solve` remains the fallback for unbound team roles.
Review-specific options remain exclusive to `review`/`review-json`. Team adapters currently
reject finalization reserves; configure those only for single/review strategies. Prompts,
fanout (two), and hybrid round count (one) are versioned in the implementation, not supplied
by the grader. Scheduler events are saved under `event.orchestration`, including the method
and policy version. The implementation digest includes the runtime components, so scheduler
changes invalidate resume just as strategy changes do.

## Tasks and difficulty

The new opt-in [`authorization` incident family](authorization-evaluation.md) adds
multi-source investigation and executable configuration repair. Standard cases contain
roughly 61 MB of generated public evidence, with SQL replay experiments and up to 1 MiB
tool responses. Use `configs/evaluation-authorization.json`; scheduling/routing defaults
and their existing data/tool limits remain unchanged.

All built-in data is generated in Python. No external evaluation corpus is committed or
needed. Generated data and reports belong in ignored `var/`, not source directories.
Cases identify generator version, seed, split, difficulty, and SHA-256. `dev` and `test`
use separate random streams; they are reproducible splits, not a secret benchmark.
Never send references, generator code, grader code, or Phoenix credentials to evaluated
agents. Trusted Python plugins run in-process; this is not a sandbox against a malicious
adapter author. Model actions have access only to an in-memory public-data database.

| Family | Required work | Standard / hard / stress |
| --- | --- | --- |
| Ledger | Reconcile revisions, duplicate events, payment status and refunds, preserve signed balances | 80 / 240 / 800 invoices |
| Dependencies | Compute earliest finishes through a branching prerequisite DAG | 80 / 240 / 800 jobs |
| Access | Join users, transitive group inheritance and resource policies with deny precedence | 160 / 480 / 1600 queries |
| Routing | Assign and order deliveries under vehicle capacities, travel times, delivery windows and return deadlines | 24 / 48 / 96 stops, four vehicles |
| Scheduling | Find a non-preemptive schedule satisfying assigned machine capacity, release times, deadlines, precedence and a horizon | 32 / 64 / 128 jobs, four machines |

The default matrix runs scheduling and routing. Ledger, dependencies and access are
diagnostic/calibration tasks; add their names to `families` when useful. The first pilot
found ledger solvable by two 27B-cohort models, so it is not a default challenge.

The first three require complete keyed answers; `exact` grades typed equality and
`fraction_correct` gives partial credit while penalizing missing and extra keys.
Scheduling and routing check constraints and accept alternative feasible solutions. Their partial
score is the fraction of constraints satisfied and is not numerically comparable with
answer-entry fractions from other families. Feasible witnesses are planted by the generator
and independently checked; optimality is not claimed or required.

SQL supports joins, windows and recursive CTEs, with 200-row pagination, 64KB result/SQL
value limits, a 16KB SQL statement limit and a 20-million-opcode query interrupt budget.
Writes, PRAGMA, ATTACH and extension loading are denied by SQLite's authorizer. Failed
queries consume tool budget. No arbitrary Python, shell, filesystem, or network execution
is exposed to models. This deliberately covers analysis and planning rather than browser,
repository repair or environment deployment skills.

Large generated tables alone do not guarantee difficulty: a capable model can solve the
analysis tasks with a good query. Scheduling supplies a constraint-search target. Calibrate
on dev seeds: examine exact success, partial credit, calls and time for each family; move
easy families to harder tiers or add constraints, then freeze the version and test seeds.
A single pilot seed cannot establish that a task is suitably hard for the entire 27B class.
Use at least 20–50 held-out seeds per family before making ranking claims. Repetitions
measure within-case variability, not independent additional tasks. Compare matched cases
and use a paired bootstrap over case IDs (not individual repetitions) for confidence
intervals. No significance claim or confidence interval is inferred by this initial CLI.

## Measurements and Phoenix

The JSONL manifest records the complete matrix, budgets, task IDs, content hashes, and
a SHA-256 of the built-in suite implementation. Keep the lockfile and custom adapter
revision with each published result for reproducibility.
Trial records contain exact and partial scores, status, elapsed time, requests, input/output
tokens, query trajectory, answer, trace ID and optional estimated cost. Configure both
`input_usd_per_million` and `output_usd_per_million` to compute nominal token cost; missing
prices stay null. Cache discounts, provider fees and failed requests without returned usage
are not included. Token ceilings are checked against reported usage; a response can cross
the limit before it can be stopped. Wall-clock timeout and model output caps bound that risk.

Summaries are per model, strategy and family, including failures in the denominator.
An incorrect answer is an ordinary benchmark result; timeout/provider/budget errors produce
a nonzero CLI exit code. The harness does not silently retry whole trials or discard failures.
Provider-side retries and limits should be kept consistent across comparisons.

Phoenix receives versioned datasets, one experiment per model/strategy, linked runs, and
`CODE` evaluations for both scores. With telemetry enabled, trial and SQL spans and
Pydantic AI model spans link the experiments to trajectories. Phoenix reference outputs
are never forwarded to an adapter. Dataset uploads contain generated public tables and
private gold in separate input/output fields. Access to the Phoenix instance is therefore
reserved for benchmark operators, not evaluated agents.

The integration uses the [Phoenix client experiments API](https://arize-phoenix.readthedocs.io/projects/client/api/experiments.html)
and [dataset API](https://arize-phoenix.readthedocs.io/projects/client/api/datasets.html).

## Extend without changing the runner

Create an importable, trusted module and register a factory and (optionally) grader:

```python
from poc.evaluation.suite.tasks import register_task

# factory(random.Random, size) -> (TaskInput, reference_values)
# grader(TaskCase, Answer) -> {"exact": float, "fraction_correct": float}
register_task("my-task", my_factory, my_grader)
```

Tables must be nonempty, use safe SQL identifiers, and contain SQLite scalar values.
Use the seeded RNG supplied by the runner. Bump the case/generator version when task
semantics change; the content hash additionally detects drift. A custom grader can use
private metadata inside the case's expected mapping, never in its public input.

To benchmark another orchestration system:

```python
from poc.evaluation.suite.adapters import register_adapter


async def solve(public_task, environment, model, settings, budget, usage):
    # Coordinate your agents, exposing only environment.query as the task tool.
    # Every model invocation must update the shared RunUsage and enforce UsageLimits.
    # Return Answer(values=...). Do not reset budgets between workers.
    ...


register_adapter("my-orchestrator", solve)
```

Add the names to matrix `families` / `strategies`, then run with
`python -m poc.evaluation.suite --plugin my_package.evaluations run ...`.
Use the same plugin when publishing a saved report so cases can be regenerated and hashed.

For a future externally sourced task, include a download script that pins the upstream
revision, verifies a checksum, records source/license/split provenance, and downloads to a
cache outside the repository. Commit the script and metadata only. Never include downloaded
corpora in a plugin module or in fixtures. Tests can use generated miniature examples.


## Benchmark boundary and execution strategies

The harness provides public task inputs, the same read-only SQL tool, hard trial resource
limits, observations, and final grading. It never chooses a final answer, rescues a timed-out
strategy, feeds constraint scores back to a model, or chooses a diagnostic best candidate.
Task generation, reference answers, difficulty, SQL limits and official grading are unchanged.

`poc/execution/sql_strategy.py` implements the reusable single/review strategies;
`poc/evaluation/suite/adapters.py` only binds them to the benchmark registry. Execution
contracts and ports have no imports from the evaluation package. The local SQL oracle
remains a harness self-check, not a competitive model strategy.

The execution layer owns prompts, internal phase allocations, model selection, retries,
and submission/failure policies. `ArchitectureOptions` and `Budget` are persisted execution
configuration, re-exported by the suite for compatibility; internal allocations do not
increase the harness's hard total budget. Single/review remain small reference strategies.
The generic team adapters exercise the existing hierarchical, board, pool, speculative
and hybrid components using the explicit SQL policies above. Results describe those policies;
they are not measurements of the incident workflow or concurrent/distributed scheduling.

### Observations are not assistance

Candidate snapshots may carry offline diagnostics for analysis; only a checkpoint
acknowledgement (`{"recorded": true}`) is returned to an agent. There is no
`validate_candidate` environment tool and no grader-based output repair. Legacy
`output_validation` and `constraint_feedback` fields are retained so archived reports can
be inspected and published, but new runs reject either field set to true.

`best_candidate` remains diagnostic only. Invalid outputs receive the same zero-score
semantics; a full but infeasible answer is scored normally. A fully feasible answer is
still required to pass scheduling/routing. Fractional scores count constraint checks, not
independently solved tasks. No unknown/private-reference correctness is inferred.

### Execution-owned review failure policy

`poc/execution/review.py::review_submission` accepts a typed submitted draft and a reviewer
callback. It has no task tables, grader, expected answer, or model dependency. Its default
is to propagate review errors. With `review_failure_policy=return_submitted_draft`, the SQL
strategy retains its submitted draft after an explicitly recoverable internal review error.
Retention is based on submission provenance, never on benchmark correctness or coverage.
An empty or incorrect typed draft can therefore be retained and still score zero. A normal
review response takes precedence; this controller is not a correctness selector.

The harness's outer timeout and external cancellation always remain failures. Strategies
must reserve enough time to handle internal deadlines and return before the hard deadline.
An unfinished draft checkpoint cannot be passed off as a completed draft. Finalization,
when configured, requires a new model submission using the latest checkpoint, never the
best-by-score candidate. Per-request, request-count and token limits are cumulative.

A returned fallback is recorded as `answer_source=draft_fallback`, `recovered=true`, with
its reviewer error class and phase. This is execution telemetry, not a harness decision.
Summaries and Phoenix expose review completion and recovery separately from answer scores.
`configs/evaluation-review-recovery.json` is an experimental allocation, not a recommended
setting: the seed-0 pilot found its 60% draft fraction too short, and a 90% follow-up did
not produce a draft either. No historical reports are rescored under this revised boundary.

### Heterogeneous phase bindings

By default every phase uses the matrix model and settings. Assign another model explicitly:

```json
{
  "architecture_options": {
    "review": {
      "review_failure_policy": "return_submitted_draft",
      "phase_models": {
        "review": {
          "name": "independent-reviewer",
          "model_class": "27B",
          "model": "qwen/qwen3.8-27b",
          "base_url_env": "OPENAI_BASE_URL",
          "api_key_env": "OPENAI_API_KEY",
          "settings": {"temperature": 0}
        }
      }
    }
  }
}
```

This illustrates configuration, not a recommendation for that model. Bindings support
`draft`, `review`, `finalize` for review strategies, and `solve`, `finalize` for single
strategies. Unspecified phases retain the matrix model. Each binding has its own provider,
settings and optional prices. Every phase records its binding and token deltas; totals
share one `RunUsage`. Cost sums each phase using its actual binding's prices and remains
unknown if any price or usage is missing. SDK-internal retries are not separately counted.

Phase bindings are included in saved configuration and Phoenix metadata. Source digests
include the execution implementation, and changed assignments/configuration invalidate
resume. Give each model team a separate report/configuration when comparing teams; the
matrix model label alone no longer describes all models involved. Role-wide settings or
provider overrides must not silently relabel an existing cohort.

See [the orchestration investigation](orchestration-model-routing.md) for existing
per-task model support, strategy-specific gaps, and a fair experimental design.

### Explicit review decisions

Set `architecture_options.review.review_protocol` to `"decision-v1"` (also supported
by `review-json`) to evaluate the execution-owned accept/revise/decline protocol.
The default `"replace"` preserves the earlier unconditional replacement strategy for
controlled comparisons. `configs/evaluation-review-decision.json` selects the same
seed-0 Qwen 2.5/Qwen 3.8 scheduling/routing pilot, with four concurrent trials.

A reviewer returns `action`, a nonempty `reason`, and an optional `replacement`:

- `accept` returns the exact retained draft; no replacement is allowed.
- `revise` requires a complete replacement answer. It is returned even if it is
  empty, incomplete, or worse than the draft; no grader selects between them.
- `decline` returns the retained draft and records that review was declined. This
  is not approval, an exception recovery, or evidence that the draft is correct.

For example, `{"action":"accept","reason":"Checked the submission","replacement":null}`
accepts the artifact without copying its values into a new model output. A revision
uses `{"action":"revise","reason":"Changed the assignment","replacement":{"values":{...}}}`.
The reason is reviewer-reported text, not a verified explanation.

Malformed decisions receive at most `output_retries` protocol retries. Existing
`review_failure_policy` controls recovery after those retries fail or after an
eligible internal reviewer error. External cancellation and the benchmark's hard
cutoff still propagate. All requests and tools share the original budgets.

The generic controller is `poc/execution/review.py`; SQL prompt/schema integration
is `poc/execution/sql_strategy.py`. Neither accesses evaluation diagnostics.
No evidence-handoff, progress-detection, model-escalation, or coverage-preservation
policy is bundled into this protocol. An explicit bad revision remains possible.

Trial records include the raw `review_decision`, and review phases include
`review_action`. `answer_source` distinguishes `draft_accept`, `draft_decline`,
`review` (revision), and `draft_fallback` (exception recovery). `review_outcome`
continues to describe phase execution, so a completed decline must not be interpreted
as reviewer endorsement. Phoenix receives the same fields in the trial output.

The [limited decision-protocol verification](../var/evaluation/review-decision-seed0-analysis.md)
repeated all12 pilot trials. Qwen2.5 returned six valid `revise` decisions but no
feasible answer; Qwen3.8 timed out before review in all six trials. The live run did
not exercise accept, decline, or protocol-error recovery; deterministic tests cover
those branches. The protocol remains opt-in, with no demonstrated task-success gain
from this small comparison.

### Bounded SQL team workers

Team strategies now default to `architecture_options.<strategy>.team_policy="bounded-v1"`
(`sql-team-v2` in events). `legacy-v1` retains the earlier shared solve worker for
policy comparisons; the common malformed-JSON retry fix still applies. To compare
against the exact pre-fix implementation, run checkpoint `d5acbfc` separately.

The execution worker supplies typed plan, critique, and verification outputs in both
native and JSON transports. Final answers retain the public task's answer mapping;
structural conformance does not imply semantic correctness. Intermediate plans and
scores are no longer recorded as candidate task answers. Malformed actions receive
at most `output_retries` retries. Repeating the same SQL error more than
`output_retries + 1` times without a successful query stops the worker.

Every stage has a cumulative fraction of the original wall-clock, token, request,
and tool ceilings. Unused earlier allocation carries forward; later allocation
cannot be borrowed. Default weights are:

| Methods | Stages | Weights |
|---|---|---|
| DAG, board, pool | plan, solve | .20, .80 |
| speculative | proposal, proposal, reconcile | .35, .35, .30 |
| hybrid | proposal, proposal, critique, critique, verify | .30, .30, .12, .12, .16 |
| hybrid_v2 | orchestrate, task, critique, integrate, verify (role pools) | .24, .30, .34, .05, .07 |

An explicit `stage_weights` list may override these; it must match the stage count
and sum to one. These defaults are hypotheses to evaluate, not tuned success claims.
Bounded workers default to a 60s request timeout and completion caps of 4096 tokens
for plan/critique/verify and 16384 for other roles. Explicit budget settings override
these defaults. Cancelled requests retain incomplete usage accounting.

`hybrid_v2` replaces the positional stage array with per-role budget pools because its
call count is adaptive (plan repairs and decision rounds). Each call consumes
`pool(role) / planned(role)` of the trial budget, with unused allocation carrying
forward and the verifier (the final stage) inheriting all unallocated budget.
`POOL_PLANNED` fixes the expected calls per role: orchestrate = plan fan-out,
task = 8, critique = fan-out + 2, integrate and verify = 1. An explicit
`hybrid_pool_weights` mapping may override the pools; it must cover all five roles
and sum to one.

A `ModelSpec.context_window` enables conservative input-plus-completion admission
control, including schemas. The estimate uses serialized UTF-8 bytes plus framing
allowance, not the provider's exact tokenizer. It can stop earlier than necessary;
it never truncates or summarizes task inputs. Bounded workers also reserve estimated
input tokens before setting each request's completion cap. Provider usage remains
the accounting authority and hard-budget overshoots remain failures.

`phase_models` accepts `plan`/`solve` for DAG, board and pool; `proposal`/`reconcile`
for speculation; and `proposal`/`critique`/`verify` for hybrid. `hybrid_v2` adds
`orchestrate`, `task` and `integrate` bindings. Repeated roles share a
binding. A legacy `solve` binding is the fallback for unbound team roles. These change
model invocation, not just worker labels. Phase records include the role, stage
index, actual model, cumulative limits and measured usage. The shipped v2 tuning
config binds `orchestrate` (the plan author and decision loop) to a frontier-class
model; per-role overrides remain user-editable through the same `phase_models`
mechanism.

Speculative execution supports an explicit `speculative_failure_policy` of
`return_first_submitted`. After an eligible later-stage failure it can return the
first controller-committed candidate within the original hard limits, recording
`first_submitted_fallback` and the failure. It never picks by benchmark score.
The default is `fail`. External cancellation propagates. Hybrid always requires
selection and accepted verification; speculative retention does not bypass its gate.

The [DeepSeek tuning matrix](../configs/evaluation-deepseek-role-tuning.json) is a
separate, single-seed configuration experiment. It raises request timeouts to 180s,
allows up to 65536 completion tokens for planning, 32768 for solving/proposals/
reconciliation, and 16384 for critique/verification. DAG, board and pool allocate
70% cumulatively to planning; hybrid uses .30, .30, .20, .10, .10. The total remains
300000 tokens and 300s. These allowances include provider-reported completion usage,
which can greatly exceed the size of the visible structured output. The public task,
SQL tools, role contracts and grader are unchanged. Treat this as a tuning candidate,
not a generally validated model profile.

The [300k-token seed-0 comparison](../var/evaluation/deepseek-bounded-300k-20260912/analysis.md)
measured 1/80 feasible team finals with the initial bounded profile versus 9/80 with
the exact pre-fix control. Typed role outputs and finite retry handling address
protocol defects, but the initial time/completion limits also cut off productive
DeepSeek calls. These defaults are not a demonstrated quality improvement. Keep
execution completion, final feasibility and incomplete usage separate when comparing
profiles; larger allowances and heterogeneous bindings require their own trials.
