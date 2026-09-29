# hybrid_v2.2 SQL strategy

`hybrid_v2_2` and `hybrid_v2_2-json` are separate SQL benchmark strategies. The
v2 and v2.1 adapters keep their prior planning behavior.

The planner receives the original request and a short specialist capability
summary. Its worker has no query tool, and the adapter does not append the SQL
schema or table contents to its prompt. One model call proposes 2–8 conceptual
work packets: a distinct question, scope and exclusions, report criteria, and
necessary dependencies. The planner does not choose SQL tables, queries,
evidence requirements, budgets, or output schema. A malformed plan gets one
structural repair attempt. Planning has no model judge or SQL feasibility probes.
The plan artifact and selection event record the validated delegation directly.

The dispatcher assigns `TaskEvidence` output contracts and positive per-task
limits from the run budget. It grants each specialist access to the public SQL
schema and tables so the specialist can choose its own investigative method.
The scheduler executes dependency waves, audits task evidence before releasing
dependents, and sends accepted prerequisite findings to dependent specialists.
If a task fails its audit, the orchestrator sees only the task's conceptual
contract and an abstract failure category. A revision receives fresh dispatch
limits; revised prerequisites invalidate their dependents.

Task evidence, task audits, integration, and independent final verification use
the v2.1 evidence and answer-status contracts. Task audits and final verification
may query public SQL. The planning stage does not claim to establish factual
feasibility or semantic task independence: those questions remain for specialist
execution and downstream review. Exact duplicate task objectives and invalid
dependency graphs are rejected structurally. All specialists can access all
public tables in this version; isolation is by task objective and evidence
review rather than a per-task table allowlist.

This variant currently covers the SQL benchmark adapter. The live runtime still
uses its existing `hybrid_v2` workflow. Deterministic tests and infrastructure
diagnostics cover v2.2; no paid model benchmark result is available yet.
