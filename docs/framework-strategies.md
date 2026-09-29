# Native framework strategy POC

The SQL evaluation suite can load three optional strategies with `--plugin
poc.frameworks.evaluation`: `pydantic_background`, `strands_background`, and
`llamaindex_workflows`. The adapter passes each candidate the public task, the
read-only SQL tool, a model specification, and the suite budget. The existing
dataset generator and graders remain in `poc.evaluation.suite`.

## Install and run

Each candidate has a pinned optional dependency set. Run one candidate per
process with `max_concurrency: 1` in the matrix. The LlamaIndex package requires
OpenAI SDK 2.x and the Pydantic AI `llm` extra requires 3.x; `uv` marks these
extras as conflicting and resolves them in separate environments.

```sh
uv run --extra pydantic-orchestration --extra llm python -m poc.evaluation.suite \
  --plugin poc.frameworks.evaluation run \
  --config configs/evaluation-framework-pydantic.json \
  --output var/framework-pydantic.jsonl --max-concurrency 1

uv run --extra strands-orchestration --extra llm python -m poc.evaluation.suite \
  --plugin poc.frameworks.evaluation run \
  --config configs/evaluation-framework-strands.json \
  --output var/framework-strands.jsonl --max-concurrency 1

uv run --extra llamaindex-orchestration python -m poc.evaluation.suite \
  --plugin poc.frameworks.evaluation run \
  --config configs/evaluation-framework-llamaindex.json \
  --output var/framework-llamaindex.jsonl --max-concurrency 1
```

Set `OPENAI_BASE_URL` and `OPENAI_API_KEY` to the same OpenAI-compatible model
endpoint for each run. The example matrices use the same task family, model ID,
seed and budget. They are evaluation examples, not recorded live-model results.
The matrices select the suite's scheduling and routing benchmark families by
default, with the same five seeds, two repetitions and trial budget as
`configs/evaluation-models.json`. Other existing task families can be selected
in each matrix or on the CLI. Each matrix runs one model and one strategy in its
own optional SDK environment; keep the model endpoint and model ID the same
when comparing their reports.

`tests/test_framework_benchmark_cli.py` runs the real benchmark command for both
families against a local OpenAI-compatible fixture. It checks SQL tool access,
exact grading, report coverage, request counts and token usage for each installed
backend. The fixture supplies a known feasible answer, so passing this test
verifies the evaluation path, not live-model task performance. Run it in each
backend's optional dependency environment before a live sweep.

## Headless app request

The same backend can be called through `poc.frameworks.app.start()` or the JSON CLI.
The current registered profile is `sql_readonly`; it accepts public tables and a
read-only SQL tool bundle. For example, save this as `request.json`:

```json
{
  "schema_version": 1,
  "caller_request_id": "example-1",
  "backend": "pydantic_background",
  "profile_id": "sql_readonly",
  "tool_bundle_ref": "sql_readonly",
  "input_messages": ["Find the n value for x"],
  "tables": {"items": [{"id": "x", "n": 7}]},
  "model_spec": {
    "name": "framework-model",
    "model_class": "7B",
    "model": "meta-llama/llama-3.1-8b-instruct",
    "base_url_env": "OPENAI_BASE_URL"
  },
  "limits": {"requests": 40, "tool_calls": 80, "total_tokens": 100000, "seconds": 300}
}
```

```sh
uv run --extra pydantic-orchestration --extra llm python -m poc.frameworks \
  --request request.json --output var/framework-run
```

The output directory receives `manifest.json`, `result.json`, `events.jsonl`, and
`transcripts.jsonl`. The Python run handle also exposes `events()`, `result()`, and
`cancel(reason)`. A mock grant, denial, or timeout can be selected with
`approval_mode` in the request for synthetic leaf-tool probes.

The adapters are optional imports. Existing strategies and evaluations work
without installing any of these extras. `framework_max_tasks` in
`architecture_options` bounds the number of agent task instances; the default
is 16. The suite enforces its deadline and final usage/tool limits. The shared
SQL bridge checks tool calls before execution and records canonical task IDs,
attempt IDs and causal events. Each backend owns completion delivery and parent
activation. The mock approval client is available for synthetic leaf-tool
conformance; it is not presented as the repo's plan approval API.

## Capability evidence and current gate

| Candidate | Native tool/answer smoke | Active parent: B result drives C while A runs | Idle parent: child completion wakes it | Status |
| --- | --- | --- | --- | --- |
| Pydantic AI 2.51.0 + Harness 0.36.0 | Pass | Not separately tested | Pass | Further conformance required |
| LlamaIndex Core 0.14.25 + Workflows 2.25.0 | Pass against local OpenAI-compatible model | Pass | Pass | Further conformance required |
| Strands 1.57.1 | Pass against local OpenAI-compatible model | Pass | Fail | Further conformance required |

The active-parent probe matches the clarified requirement: the root has a model call
in flight while B finishes, performs an independent SQL query, then sees B's nonce
at the next model boundary and dispatches C before A finishes. Strands passes this
probe. The idle-parent probe is deliberately stronger: after the root says it is
waiting, it requires B's completion alone to wake the root before A finishes.
Strands fails that variant, recorded as a strict expected failure in
`tests/test_framework_strategies.py`. With `wait_for_completion=True`, Strands
waits for the remaining background work when an invocation ends; with it
disabled, an invocation can return while children remain active. The idle
behavior is a documented limitation for this composition, not a failure of the
clarified active-orchestrator requirement.

The tests also check exact-call mock approval binding, duplicate decisions,
denial, timeout and sibling tool progress. The approval service is synthetic;
container execution, real per-tool approval API integration, full nested
conformance, global token admission under concurrent load for the Strands and
LlamaIndex clients, and live-model quality comparison remain unverified. These
are separate release gates, not implicit passes.

## Ownership ledger

| Responsibility | Pydantic AI | Strands | LlamaIndex |
| --- | --- | --- | --- |
| Agent runtime and tool loop | Native `Agent` | Native `Agent` | Native `FunctionAgent` |
| Child result delivery | Harness `BackgroundTools` | Native background tasks, barrier observed | Backend-local Workflow events and parent mailbox |
| Parent continuation | Harness | Native background facility, barrier observed | Backend-local serialized activation step |
| Role and tool policy | Shared `RunBridge` checks | Same | Same |
| Mock approval and effect deduplication | Shared approval client and tool boundary | Same | Same |
| Evaluation adapter | `poc.frameworks.evaluation` | Same | Same |

No framework code imports the existing LangGraph or custom SQL orchestration
controllers. The existing SQL environment and graders are reused at the public
evaluation boundary. The extra LlamaIndex mailbox and activation code is kept
in its backend module so its maintenance cost stays visible.
