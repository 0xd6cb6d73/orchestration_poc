# SQL execution reliability policies

The default team policy is `reliable-v2`. It uses typed role contracts, public artifact
validation, role budget profiles when supplied, immutable candidate snapshots, and
explicit hybrid decisions with SQL evidence provenance. `concurrent-v1` enables
representative coordination behavior. `bounded-v1` and `legacy-v1` remain serial controls.
For an old permissive output control also set `artifact_contract: legacy-v1`; to reproduce
an old deadline with no return reserve set `return_reserve_seconds: 0`.
These controls are explicit in the regression fixtures. Existing reports are not rewritten.

## Admission and accounting

`ModelSpec.endpoint_context_window` overrides generic `context_window` metadata for the
configured endpoint. `completion_limit` is a separate endpoint completion ceiling. The
adapter caps completion by those limits, a conservative input estimate, the declared
request cap, and remaining phase tokens. An unspecified completion allowance defaults
to 16,384, never all remaining context. Inputs and schemas are not truncated. Native
schemas count toward input admission. `request_settings` events report the admitted
cap and effective endpoint limits; `response_shape` records part kinds and finish reason.

Unknown usage after a failed request retains a conservative token reservation, separately
from provider-reported usage and cost. Retries consume request allowance. A 429 can be
retried without reserving generated tokens; ambiguous failures reserve their entire
admitted input/output allowance. Incomplete provider usage keeps cost unknown.
`provider_retries` defaults to zero and is limited to three; retries cover 429/502/503/504
and remain inside phase deadlines and budgets. External cancellation propagates.

## Public contracts and review

`public-v1` checks complete canonical ID coverage, value types, and referenced vehicle
IDs for the existing public scheduling, routing, ledger, dependency, and access schemas.
Unknown schemas require a nonempty mapping. It does not repair IDs, fill entries, check
schedule feasibility, or access evaluation diagnostics. Complete but infeasible answers
remain eligible for submission and posthoc grading. Checks run at worker handoff and
before accepting review replacements. Native and JSON output retries are bounded.

Speculation's `return_first_submitted` recovery returns an immutable copy of the first
controller-committed submission on a later recoverable failure. It never picks the best
graded candidate. Review's `return_submitted_draft` similarly retains its submitted draft.
Malformed reconciliation or revision is an explicit recoverable protocol failure.
Diagnostic checkpoints are not promoted automatically. A small configurable return
reserve lets execution recovery finish before the harness hard deadline.

Hybrid uses `supported-v1`: `structural_validity`, `feasibility` (`supported`, `unsupported`,
`abstain`), `reason`, `evidence_refs`, and the five evaluation scores. Supporting a
candidate requires a structurally valid artifact, a supported decision, observed SQL
references, and positive validity/evidence/constraint-satisfaction scores. A positive
validity score alone cannot authorize selection. The verifier receives the critic's
record and must supply its own SQL references and boolean verdict. References establish
observation provenance, not semantic correctness. Model judgments can still be wrong.

Failed hybrid proposals are abandoned on the board, preventing critics from claiming
them. Failed critiques abstain for that candidate. `hybrid_proposal_quorum` defaults to
two and can explicitly be set to one. Surviving candidates must still pass criticism,
selection, verification and the completion gate; timeouts never bypass these gates.

`hybrid_v2` extends the collaboration engine with an LLM orchestrator that decomposes
the task into 2–8 scoped tasks (`poc/hybrid/planning.py`). Plans are data, never
authority: two sealed plan candidates are judged with the same evidence-based critique
protocol and selected deterministically by score; the selected plan passes hard
validation (scope, definition of done, acyclic dependencies, 2–8 tasks) with one
bounded repair attempt before it becomes the round's plan of record. Tasks run in
dependency waves under per-task goal contracts, receive targeted critiques, and an
orchestrator decision loop accepts, revises, adds or escalates — bounded by
`max_collaboration_rounds`. Revised and added tasks become derived candidates through
`revise_candidate`. Integration, independent verification and the completion gate are
unchanged from `hybrid_v1`.

## Measured role profiles

Each model can specify `role_profiles`, with a version, `measurement_source`,
`max_output_tokens` (including provider reasoning tokens), and `request_timeout_seconds`.
Architecture-level role profiles take precedence over model profiles. Explicit global
budget caps remain upper bounds. Serial stage allocations carry unused capacity forward
without borrowing from future stages. Stage weights and profiles are separate: increasing
a request ceiling does not increase that stage's cumulative allocation.

`poc.evaluation.suite.budget_profiles.measured_profiles` derives initial profiles from
observed phase output totals and elapsed time, with declared headroom. It includes failed
phases, excludes unknown token measurements, and never filters by grader success. Phase
totals conservatively bound individual requests; these profiles are starting ceilings,
not optimized policies. The checked-in reliable/concurrent configs were calibrated from
the 364 recorded trials with 1.25 headroom and reserve later stages explicitly. The
Mistral Nemo binding uses the observed endpoint's 128,000 context limit. No claim is made
that live provider behavior or success rates have been validated by the offline tests.

## Concurrent behavior

`concurrent-v1` reserves disjoint token, request, and tool shares before launching each
worker, creates a separate SQLite connection, and merges observations once after it ends.
Unused branch shares are not borrowed. The first pair overlaps in a shared wall-clock
window; downstream stages retain their own reservations. Cancellation drains all child
coroutines before closing scheduler state.

- DAG: two independent planning predecessors join before solving.
- Board: three workers contend for two ready planning tasks; a dependent solve waits for both.
- Pool: two plans run concurrently; a third planning allocation either reassigns one failed
  plan to another worker or checks the plans. Two failed plans exhaust this retry policy.
  Failed assignment generations are fenced before the offer is reopened.
- Speculation: two candidates run concurrently before reconciliation.
- Hybrid: two sealed proposals run concurrently before the critic and verifier stages.

`partitioned_ledger` and `dependency_join` add independent-work and join workloads without
changing the original task factories. New configs use held-out test seeds 101 and 102.
Coordination fixtures assert overlapping worker spans, claim contention, reassignment,
shared reservations, candidate identity, and cleanup. These establish mechanism coverage,
not evidence of better end-to-end solving or scalability.

## Reporting and running

For fast regression detection, use [infrastructure diagnostics](infrastructure-diagnostics.md):
`python -m poc.evaluation.suite --no-env-file diagnose`. It exercises all strategies and
injected failures with a small public task, without launching the full matrix.

Trial failures include sanitized exception type, request/phase/trial scope, stage/index,
thresholds, and consumption. Phase timeouts use `phase_timeout`; failed submissions are
counted in `no_submissions`, separate from malformed returned `invalid_answers`.
The batch helpers count only verification paths named by the batch manifest and validate
all configurations against their target implementations before any batch launches.
The archived 364-trial monitor's filename-count defect is also corrected locally.

Run the new matrices explicitly when live evaluation is desired:

```sh
.venv/bin/python -m poc.evaluation.suite run --config configs/evaluation-reliable-teams.json --output var/reliable-v2.jsonl
.venv/bin/python -m poc.evaluation.suite run --config configs/evaluation-concurrent-teams.json --output var/concurrent-v1.jsonl
```

These commands make paid provider requests. Use new output files; implementation/config
changes correctly invalidate resume against historical reports. Calibration and acceptance
fixtures run offline and do not submit or publish evaluation data.

The read-only batch monitor is available as:

```sh
.venv/bin/python -m poc.evaluation.suite --no-env-file monitor var/evaluation/three-model-900s-1m-20260913
.venv/bin/python -m poc.evaluation.suite --no-env-file calibrate path/to/report.jsonl --source my-calibration-run
```
