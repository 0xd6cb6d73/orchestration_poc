# Preserve a submitted draft when review cannot finish

Status: superseded by the benchmark-boundary correction in
[orchestration-model-routing.md](orchestration-model-routing.md). The original plan below
is historical: the harness no longer performs recovery or grader-based eligibility checks.
Execution owns draft retention, and a hard harness deadline remains a failure.

## Evidence and current contract

Dataset `orchestration-3f4f3fadb4c7-ea28f9d7a983` contains three Qwen 3.8 review
runs with a feasible draft and an empty, zero-scored final answer after timeout.
Trace `a003b3d458d4c8907733903ca1819924` completed its draft at about 222 seconds
and was cancelled during review at the 300-second trial deadline.

This is an explicit baseline contract, not accidental checkpoint deletion:

- `adapters.py::_orchestrate` returns the review result, not the draft. With zero
  finalization reserves, exhausted review propagates an exception.
- `runner.py::run_trial` starts with an empty answer and only assigns it when the
  adapter returns. Its outer timeout can cancel the adapter before inner recovery.
  Non-completed trials receive zero official scores.
- `runtime.py::best_candidate` is diagnostic, selected by evaluator scores. It
  must not become the source of a fallback answer.
- `docs/evaluation.md` and `tests/test_evaluation_reliability.py` explicitly enforce
  no automatic checkpoint promotion and model-submitted finalization.

## Proposed intervention

Add an opt-in `review_failure_policy` to `ArchitectureOptions`, with values
`fail` (existing default) and `return_submitted_draft`. Reject its use with
non-review strategies. Apply the same policy to native and JSON review adapters.
Publish it through the existing architecture metadata, digest, resume, and
comparison machinery. Keep baseline results and new intervention results separate.

The fallback is the immutable answer returned by a successfully completed draft
phase, never `best_candidate`, a rejected output, SQL text, or an intermediate
`submit_candidate` checkpoint. Shape validation must pass before it is eligible;
reuse public coverage/type diagnostics, without ranking candidates or consulting
private reference answers. Unknown validation is ineligible. Feasibility and
official scoring remain the grader's job: returning an infeasible draft is not a pass.

## Implementation steps

1. **Make the committed draft explicit.** In `TrialState`, retain a deep copy of
   the draft only after the draft phase returns successfully. Keep it separate
   from diagnostic candidates. Record its eligibility and submission provenance.

2. **Recover at a single terminal boundary.** In `run_trial`, after known review
   deadline/request-timeout, shared-budget exhaustion, or provider/protocol errors,
   select the committed draft only when the policy is enabled, review began, and
   the draft is eligible. This boundary also catches the outer timeout. Do not
   catch external cancellation, `KeyboardInterrupt`, or arbitrary programming
   exceptions as successful recovery. Check a draft produced before the deadline;
   do not initiate a new model call after budget exhaustion.

3. **Separate answer availability from review failure.** A recovered answer is
   returned and graded normally, while recording `answer_source=draft_fallback`,
   `review_outcome`, the original failure kind, and `recovered=true`. Keep the
   interrupted review phase and incomplete usage visible. Define terminal status
   consistently (proposed: `completed` with recovery metadata and no terminal
   error). Update summaries and Phoenix evaluations to expose recovery rate so
   recovered answers cannot masquerade as completed reviews. Ordinary baselines
   retain their existing status/error/score contract.

4. **Allocate time for actual review.** Add an intervention matrix starting with
   `draft_fraction=0.6`, `finalization_seconds=20`, `finalization_tokens=5000`, and
   `finalization_requests=2` within the existing 300-second/100k-token/40-request
   budget. These are starting values, not measured optima. For the model-finalization
   path, preserve the requirement for a fresh model submission. Use one absolute
   deadline originating at the runner; inner work deadlines must precede it.
   Align request/token ceilings with cumulative usage. Recovery itself performs
   no additional tool/model work.

5. **Retain benchmark auditability.** Update the documented baseline/intervention
   distinction, report metadata, resume validation and Phoenix publishing tests.
   Do not modify or retroactively rescore the original experiment. The three
   feasible drafts are evidence for this change, not three newly claimed passes.

## Acceptance tests

Use deterministic `FunctionModel` responses and bounded async delays in
`tests/test_evaluation_reliability.py`:

- Policy disabled: existing timeout/checkpoint/reviewer-error tests remain unchanged.
- Valid submitted draft followed by reviewer timeout: recover its exact mapping,
  expose recovery provenance, and preserve the interrupted phase and usage flag.
- Exercise both the inner phase deadline and the runner's earlier outer deadline.
- Reviewer request timeout, cumulative request/token/tool exhaustion and known
  provider/protocol errors: recover without an extra request or exceeded budget.
- Draft failed, absent, structurally invalid, unknown-validity, or merely checkpointed:
  no fallback. Include a better-scoring diagnostic checkpoint to prove it is ignored.
- Reviewer succeeds: return its answer; do not silently choose the grader's favorite.
- External cancellation or application bug: propagate rather than claim completion.
- Mutating reviewer inputs cannot alter the saved draft; native and JSON paths agree.
- An eligible but infeasible draft remains a graded failure.
- Report/resume and Phoenix round trips preserve policy and recovery metadata.

## Evaluation

First rerun Qwen 3.8 on the same ten cases and two repetitions, then the full matrix.
Compare the existing baseline, reserved-budget review, and reserved-budget review
with explicit draft fallback. Keep total budgets fixed. Report official feasibility,
draft feasibility, review-start/completion rates, recovery rate, answer source,
latency, and usage completeness, with paired case results. This separates benefits
from better budget allocation from benefits due to the changed submission policy.
