# Orchestration evaluation suite

The task-independent suite lives in `poc/evaluation/suite`. The existing
`hierarchical-ooda-eval` and `hierarchical-ooda-benchmark` commands remain incident-demo
regression checks. Their report phrase and worker-count gates are not capability scores.
The new suite grades final answers and constraints, without rewarding extra agents or calls.

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
| 7B | `mistralai/mistral-nemo` |
| 7B | `meta-llama/llama-3.1-8b-instruct` |
| 7B | `qwen/qwen-2.5-7b-instruct` |
| 27B | `mistralai/mistral-small-3.2-24b-instruct` |
| 27B | `qwen/qwen3.8-27b` |
| 27B | `google/gemma-3-27b-it` |

Cohorts are comparison labels, not exact parameter counts (Nemo is 12B; the selected
Llama is 8B and Mistral Small is 24B). Add `100B` or `frontier-flash` entries with exact
provider model IDs when extending the matrix. No harness rewrite is needed.
For an OpenAI-compatible local server, change the environment variable referenced by
`base_url_env` and supply that server's model ID. For Pydantic AI's other providers omit
`base_url_env` and use `provider:model`. Install its provider extra separately if needed.

`single` runs one tool-using agent. `review` runs a drafter and a fresh reviewing agent
that receives the draft and can inspect the same database. Both stages use the same
model, share request/token/tool limits and share one wall-clock timeout. They use native
function calling. `single-json` and `review-json` offer an explicit text JSON action
protocol for endpoints without native tools. Do not pool scores from different protocols.
`sql-baseline` is an independent deterministic reference implementation for the three
analysis families, for validating generators and graders. It does not solve scheduling.

These are reference orchestration adapters, not aliases for the incident runtime's
board-claim, managed-pool, or hybrid modes. Those modes still operate on incident-specific
roles/tools. Connecting them to these new tasks requires a generic-task adapter; the
public adapter interface is intentionally independent of those roles and workflow shapes.

## Tasks and difficulty

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


## Reliability, diagnostics, and architecture interventions

Task generators, task prompts and official scoring rules are unchanged. A fully feasible
final answer is still required to pass scheduling/routing. The evaluator does not correct
answers, infer candidates from arbitrary SQL or reasoning text, promote checkpoints to
passes, shorten SQL error loops, or add solver hints. All models/cohorts use the same
rules. Model size is a reporting label, not a control-flow condition.

Every terminal record contains separate execution `status`, `diagnostics.answer_valid`,
`diagnostics.feasible`, official `scores`, and per-constraint passed/total counts with
violation evidence. Invalid shape gates retain the original zero-score semantics.
Scheduling's `job_window` count combines release, deadline and horizon exactly as before;
its overlap score still counts adjacent intervals. Routing still checks contiguous
positions, capacities, service-start windows and return deadlines. Partial fractions are
not a percentage of independently solved tasks. Diagnostics for private-reference families
do not disclose correct values. Unregistered public schemas have unknown validity rather
than an invented validator.

The built-in adapters checkpoint structured outputs (including drafts and rejected output
attempts) for operators. `best_candidate` is a **diagnostic** selection by public feasibility,
constraint fraction and shape, never the official answer and never a private-reference
oracle. Candidate and phase events are also written before the terminal trial, so an
interrupted process does not erase them. A timeout/reviewer error still receives official
zero scores even if a feasible candidate was saved. To checkpoint an intermediate answer
before it is returned, an architecture must explicitly enable and use candidate submission.

Model-visible assistance is opt-in per strategy and recorded in both report and Phoenix
experiment metadata. For example, this matrix fragment defines an assisted review
architecture; it is not equivalent to the default `review`:

```json
{
  "architecture_options": {
    "review": {
      "candidate_submission": true,
      "output_validation": true,
      "constraint_feedback": true,
      "output_retries": 1
    }
  }
}
```

- `candidate_submission` exposes `submit_candidate(values)`. Its default response only
  acknowledges storage, revealing neither validity nor constraint results.
- `output_validation` rejects incomplete or malformed final mappings with structural
  feedback and a bounded number of model repair attempts. It does not reject a properly
  shaped answer for being infeasible. With candidate submission it also reports shape.
- `constraint_feedback` exposes a budgeted `validate_candidate(values)` tool using public
  constraints only. It includes the same feedback in candidate submission and validates
  the draft before review, charging a tool call. No reference answer or solving algorithm
  enters the prompt.
- SQL, candidate submission and validation consume one shared tool-call budget. Repeated
  queries and repeated SQL errors are still permitted up to the ordinary budget, with no
  special retry suppression or targeted SQL guidance.

The text-JSON adapters expose the equivalent explicit actions `{"candidate": {...}}`
and `{"validate": {...}}` only when those options are enabled. Default `single`, `review`,
`single-json` and `review-json` retain their original tool surfaces and model-visible
feedback. Do not pool results from different architecture options just because the strategy
names match. Built-in options affect the reference adapters; custom plugins must explicitly
implement the same contracts if they want comparable interventions.

Optional budget fields apply to built-in adapters:

```json
{
  "budget": {
    "seconds": 300,
    "requests": 40,
    "tool_calls": 80,
    "total_tokens": 100000,
    "request_timeout_seconds": 90,
    "max_output_tokens": 8192,
    "draft_fraction": 0.6,
    "finalization_seconds": 20,
    "finalization_tokens": 5000,
    "finalization_requests": 2
  }
}
```

These values are an example policy, not new recommended model limits. The defaults are no
request/output cap, `draft_fraction=1`, and zero finalization reserves, retaining the original
allocation. Set policies before comparing architectures; do not tune them after seeing test
answers. `draft_fraction` partitions the non-finalization wall time, reported-token ceiling
and request allowance between draft and review. Limits remain cumulative across phases.
When a work phase exhausts its allocation and a checkpoint exists, finalization can use the
reserved budget for a **new model submission** with no tools. It receives the latest
checkpoint, not the evaluator's best-by-score selection. If that submission fails, the
checkpoint remains diagnostic only. A draft that exhausts its reserved portion can pass its
latest checkpoint to review. Without an explicit checkpoint it cannot fabricate a draft.

Reported token ceilings are checked after responses; an unexpectedly large response or
prompt can overshoot a phase reserve. These are not hard provider-side admission guarantees.
Output caps bound generation, and every phase remains inside the whole-trial limits.
Per-request timeouts have a distinct `request_timeout` status. Trials record solve/draft/
review/finalize phase timing, phase outcome, whether review started, and the exhausted phase.

The suite disables SDK-level instrumentation when configuring telemetry, retaining Pydantic
AI model spans as the single model-call instrumentation layer. Trial usage remains the
canonical aggregate. Request attempts/responses and `usage_complete` distinguish missing
usage after cancellations/provider errors from genuinely zero usage; costs remain null when
usage is incomplete. Legacy usage completeness is unknown. Provider-internal retries are
not separately counted as adapter requests.

Reports show expected/recorded coverage, including zero-run cells, and matched case/repetition
single-versus-review comparisons. Unmatched trials are counted and excluded from deltas.
No significance claim is generated. Empty or incomplete coverage is not a model ranking.
