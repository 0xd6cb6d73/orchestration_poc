# Authorization evaluation investigation

This investigation uses the `authorization-v1-dev-standard-0` case, the same
`deepseek/deepseek-v4-flash-0731` model endpoint at temperature 0, and each
framework's native agent loop. The report files are under ignored
`var/evaluation/authorization-framework-*.jsonl`. The deterministic local-model
fixture passes the real CLI, SQL tool, report, and exact grader path for all three
backends. That fixture supplies the generator's answer; it does not measure
live-model problem solving.

The first live pilots used 80 model requests, 120 SQL calls, 1 million tokens,
and 900 seconds. Token admission stopped Pydantic at 20 requests and 37 SQL
calls. Strands stopped after 26 requests and 63 SQL calls with the underlying
budget error wrapped as `EventLoopException`; the adapter now unwraps it.
LlamaIndex stopped after 10 requests and 20 SQL calls with the budget error
wrapped as `RuntimeError`; the adapter now preserves typed budget failures.
These pilots did not establish an adequate task budget.

The larger pilots allowed 300 model requests, 400 SQL calls, 100 million tokens,
and 1,800 seconds. These are diagnostic runs on one seed, not a representative
comparison of framework success rates. After finding a valid Pydantic
submission stranded behind an extra model turn, I started a one-hour Pydantic
rerun with immediate accepted-finish termination. LlamaIndex also received a
one-hour rerun after its 30-minute wide-output run reached the wall deadline.
Every live authorization run observed so far used the root agent only; none
dispatched a child. The separate conformance tests cover parent activity and
child-result delivery.

| Run | Model requests | SQL calls | Terminal result | Main finding |
| --- | ---: | ---: | --- | --- |
| Pydantic, direct-output fix | 71 attempts, 70 responses | 134 | 1,800-second timeout; no submission | Investigated a plausible repair, then continued checking scope and controls. |
| Pydantic, accepted-finish stop fix | 56 | 91 | Completed in 1,267 seconds; exact 1.0 | Submitted four repairs with supported evidence and stopped on acceptance. |
| Strands, larger budget | 140 attempts, 139 reported usages | 288 | 1,800-second timeout; no submission | Repeated the same replay queries near the end. |
| LlamaIndex, wider output | 14 attempts, 13 responses | 28 | 1,800-second timeout; no submission | Found the mechanisms and began targeted replays, but a long model call was still active at the deadline. |
| LlamaIndex, one-hour follow-up | 114 | 209 | Error after 2,989 seconds; `MissingFinishTask` in trace | Found four repairs, then repeated reasoning until the 65,536-token output cap. |

## Pydantic AI Harness

The root read operational documents, tenant runtime changes, request and audit
captures, directory memberships, policies, cache snapshots, and authorization
source. It formed the correct two-part repair mechanism: normalize migrated
cached group IDs on two tenant paths, and use a tenant-scoped cache key on two
other paths. It used `authz_replay` to test visible requests and controls such as
explicit denies, invalid tokens, and inactive users. In the direct-output run,
it also considered three version-check changes for tenants without observed
regressed denials. At 30 minutes it had made 134 distinct SQL calls and was
still reasoning about whether to include those extra changes. No `finish_task`
call occurred in that run. Two broad replay queries hit the per-query 2,000
invocation limit, after which the agent narrowed them.

An earlier diagnostic run exposed a separate adapter issue: the old
`finish_task` validator rejected direct answer objects and accepted only a
`{"values": ...}` wrapper. The model eventually submitted a wrapped answer
with four affected tenants, four configuration changes, and four evidence
chains. `finish_task` accepted it, but the Pydantic agent immediately started
another model request. The run was intentionally interrupted before a terminal
trial report. Offline grading of that exact model-produced candidate against
the seed-0 case returned `exact=1.0` and 1.0 for all component metrics. This is
a candidate inspection, not a completed benchmark score. The candidate is
saved in ignored
`var/evaluation/authorization-framework-pydantic-diagnostic-candidate.json`.
The adapter now accepts direct answer objects and uses Pydantic's node iterator
to stop before another model request after an accepted `finish_task`. The
one-hour rerun finished in 1,267 seconds with 56 model requests, 91 SQL calls,
and `exact=1.0`; all six component metrics were also 1.0. It submitted a
direct JSON answer with four path-specific repairs and nine evidence chains,
then emitted `task.finish_accepted`, `model.run_stopped`, and `task.succeeded`
without another model request. This is a completed benchmark result for this
one seed.

## Strands

The root initially explored the same documents, runtime changes, request
captures, and policy tables. It identified a cache-key repair to test, then
spent most of the run replaying tiny samples around low-numbered request IDs.
The final 88 SQL calls alternated between the same two replay queries; each
returned the same one captured request. The model received successful tool
results on each turn, but issued the pair again. It never called `finish_task`. Strands
also logged warnings that `reasoningContent` is unsupported in multi-turn Chat
Completions conversations. Its OpenAI adapter filters those reasoning blocks
from subsequent messages. That is a plausible continuity factor, while the
trace only establishes the repeated replay loop. The 30-minute deadline
stopped the run after about 31.7 million reported tokens. More token allowance
alone would not resolve this observed loop.

## LlamaIndex Workflows

The initial larger-budget run reached exactly 20 model iterations, the native
`FunctionAgent` default, and failed with `WorkflowRuntimeError` after 38 SQL
calls. The workflow now passes the suite request limit as `max_iterations`.
The next run read documents and source and began comparing deployed behavior
with intended entitlements. Its fifth model call spent 751 seconds and used
exactly 32,768 output tokens on a thinking block, leaving no tool call or final
answer; the workflow ended with `MissingFinishTask`. The configuration now
allows 65,536 output tokens and the adapter enforces a total per-request timeout.
The wider-output rerun reached 14 request attempts and 28 SQL calls in 30
minutes. It read source code and operational documents, identified the group
normalization and tenant-key mechanisms, and began targeted replays. Its final
request was still in flight at the wall-clock deadline. It had 747,969 reported
input tokens and 26,214 output tokens; the unfinished request has unknown
usage. The shared bridge now records an outstanding reservation as unknown
when the native workflow closes, so subsequent reports will not call this
usage complete.

The one-hour follow-up made 114 model requests and 209 SQL calls in 2,989
seconds. It identified the same four affected tenant paths as the successful
Pydantic run and tested the repair against cold-cache directory and policy
decisions. It also corrected an intermediate mistake: newly allowed requests
under the cache-key repair were intended restorations, not unintended grants.
The final request took 886 seconds and returned a thinking-only response of
65,536 output tokens. In that response, the model repeated the same two
already-completed verification steps 851 times each, with no `finish_task` or
other tool call. The native workflow then ended with `MissingFinishTask`;
reported usage was complete at 18.4 million input and 158,029 output tokens.
This is a model continuation loop visible in the response, after the earlier
native iteration and output-cap integration limits had been fixed.

The one-hour follow-up started before the later LlamaIndex `return_direct`
change. The current adapter stops on an accepted `finish_task`, and fixture
tests confirm rejected submissions can be retried. This run failed before
that stop path because the model never called `finish_task`.

## Interpretation

A missing submission receives zero on the authorization grader even when its
trace shows useful investigation. The completed Pydantic rerun establishes an
exact repair on this one seed. The earlier Pydantic and Strands pilots show that
continuation and termination behavior materially affect measured outcomes.
The LlamaIndex follow-up shows that increasing the wall budget alone did not
produce a submission: the final model response exhausted its output cap in a
repetition loop. These runs do not establish a framework quality ranking or a
reliable authorization budget.
