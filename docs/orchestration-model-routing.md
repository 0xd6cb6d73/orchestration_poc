# Orchestration failures and heterogeneous model teams

## What the pilot establishes

The seed-0 pilot ran Qwen 3.8 27B and Qwen 2.5 7B on scheduling and routing with
concurrency four. Four Qwen 3.8 runs exhausted a 168-second draft window; a two-run
follow-up exhausted a 252-second window. Each had received one tool-reading response
and was waiting for its second model response. No candidate was submitted. We cannot
distinguish useful reasoning, provider delay, or an unproductive generation from the
incomplete response/usage records. A different reviewer cannot solve missing drafting.

Qwen 2.5's routing trials returned empty outputs without reading the tables. Scheduling
outputs were infeasible; in the tuning check a completed reviewer replaced a full,
infeasible draft with an empty answer. Error recovery does not address that normal-return
regression. No live trial tested a submitted draft followed by reviewer failure.

These are failures to expose and measure, not reasons to add hints, graders, or repair
tools to the benchmark. The old run also showed a feasible draft lost during review:
that motivates execution-owned submission policy, not evaluator-selected fallback.
The [pilot report](../var/evaluation/review-recovery-seed0-analysis.md) preserves the evidence.

## Boundary restored in this change

- SQL agent implementations and their configuration/ports now live under `poc/execution`.
  They have no evaluation imports. The benchmark registry calls these implementations.
- Review retention is a generic execution controller with separate reviewer callback,
  deep-copied submission, explicit failure policy, and no correctness oracle.
- The harness never selects a fallback, including after its hard deadline. It grades
  only a returned answer. Invalid/incorrect retained drafts still fail.
- Grader-feedback tools are removed. Historical feedback flags remain readable but
  cannot be enabled for new runs. Candidate observations are not model feedback.
- Static per-phase model assignments are supported, recorded, and priced independently
  while sharing total limits. No routing policy uses model-size labels or grader scores.

## Existing runtime support and remaining gaps

| Strategy/component | Existing heterogeneous support | Next integration step |
|---|---|---|
| Single/review SQL agents | This change adds independent solve/draft/review/finalize bindings and records per-phase usage. | Compare fixed model teams; add an execution-owned accept/revise review protocol before dynamic routing. |
| Hierarchical DAG/workflows | `TaskSpec.agent_provider`, `agent_model`, and `agent_options` override `RoleSpec`; `WorkerAdapter.execute` resolves these into the spawned agent. `PydanticAIAgentExecutor` uses that binding. | Declare planner, domain worker and reviewer bindings in versioned workflow configuration; benchmark the real workflow path rather than relabeling the SQL review adapter as hierarchical. |
| Board claims | Tasks have required roles; workers carry provider/model identity. Capacity is reconciled per role. | Use distinct roles for model specializations and avoid a global `ExecutionPolicy.agent_model` override that homogenizes workers. Add explicit per-role bindings to policy if roles should remain model-independent. |
| Managed pool | Role quotas, eligible-role offers and worker identities already constrain assignment. | Include model capability/provider/latency information in offers and deterministic assignment policy. Prices/latency estimates must come from declared configuration or prior observations, never current-task gold scores. |
| Speculative execution | Candidate groups authorize multiple roles/workers; fanout is bounded and reconciliation is explicit. | Allocate model-diverse candidates and bind the reconciler separately. Account for all candidates, including cancelled/losing ones; retain the existing effect and reconciliation boundaries. |
| Semantic Pydantic executor | `semantic_model_factory` can differ from the worker factory. | Make reviewer binding explicit and correct telemetry/settings: `_run_semantic_review` currently copies the worker role, applies worker `provider_options`, and reports the worker model as validator. A different factory alone can hide the actual reviewer identity. This is investigated, not changed here. |

Relevant code: `worker_adapter.py::execute`, `pydantic_ai_executor.py::execute`,
`capacity.py::reconcile`, `managed_pool.py::publish_offer`,
`speculative.py::authorize_candidate`, and
`semantic_pydantic_ai_executor.py::_run_semantic_review` under `poc/execution`.
Roles declare authority and tools as well as model defaults: changing a model must not
accidentally expand a role's tool permissions or access to private task information.

## Candidate solutions owned by orchestration

1. **Missing draft: bounded escalation, not a longer reviewer timeout.** A strategy
   can partition its total budget into an initial drafting attempt and an alternate-model
   attempt when no submission arrives. Cancel the first attempt, record incomplete usage,
   and admit the replacement only within the remaining shared budget. Do not restart the
   global clock or claim the cancelled generation was free. Test generation limits and
   model assignments independently; an output cap may cause truncation rather than help.

2. **Review regression: change the review protocol.** Prefer a structured decision
   such as accept the exact submission, propose a replacement, or decline. An accept
   decision should reference the immutable submitted artifact, avoiding accidental empty
   rewrites. Protocol/schema failures can be handled by execution policy. Any semantic
   checks must be part of the strategy's declared implementation over ordinary inputs,
   not calls into the benchmark grader. The generic controller added here retains the
   simpler replace-on-success behavior; it does not claim to solve semantic regression.

3. **Reviewer failure: retain an execution-owned submission.** Implemented through
   `review_submission`. Recovery must complete inside an internal deadline preceding
   the benchmark cutoff. Returning a poor draft remains a poor result. Never select
   whichever checkpoint has the highest benchmark score.

4. **Role mismatch: bind models by measured role performance.** A model that generates
   good drafts need not be the best verifier or planner. Conversely, a smaller reviewer
   is not automatically adequate: Qwen 2.5's empty outputs argue against assuming it is
   a safe cheap reviewer. Measure each role before choosing a default model team.

## Representative evaluation design

Keep public tasks, tool contracts, seeds, task difficulty and grading fixed. Compare:

- Each drafter alone.
- Each drafter with same-model review.
- Each drafter with a different reviewer, both pair directions where meaningful.
- The same model teams with explicitly versioned recovery/escalation policies.

Use the same total request/tool/time limits and report actual per-model usage and cost.
Equal token totals are not equivalent monetary cost across models, so keep both visible.
Count provider/protocol failures and no-submission rates alongside final feasibility;
do not drop failures to make team quality appear better. Report draft-to-final changes
offline, review completion, recovery, cancelled usage, and per-phase latency.

Saved-draft replay is useful to isolate reviewer behavior, with identical evidence supplied
to every reviewer. Label it a component evaluation: it excludes drafting costs and success
rates and cannot replace an end-to-end orchestration benchmark. Evaluate real hierarchical,
board, pool and speculative implementations through thin adapters before comparing them.

The next bounded experiment should compare a small fixed set of model teams and one
execution policy at a time. Do not add automatic constraint feedback or retune the task
until the runs pass. Mixed models, escalation, and a structured review protocol are
hypotheses to measure; this change does not assert that any particular model team wins.
