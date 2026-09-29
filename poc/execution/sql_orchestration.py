"""Generic SQL task bindings for the persistent orchestration components.

Policies live here, outside grading. Serial workers share usage and an environment;
concurrent workers reserve disjoint allowances and use isolated connections.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, cast

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.blackboard.service import BlackboardService
from poc.control.spawn_policy import SpawnPolicy
from poc.execution.board_claim import BoardClaimStrategy
from poc.execution.capacity import CapacityScheduler, InProcessWorkerMaterializer
from poc.execution.hierarchical_strategy import HierarchicalDAGStrategy
from poc.execution.managed_pool import ManagedPoolStrategy
from poc.execution.speculative import SpeculativeStrategy
from poc.execution.sql_artifacts import CommittedCandidates, validate_artifact
from poc.execution.sql_concurrency import ConcurrentTeamWorker, gather_branches
from poc.execution.sql_contracts import Answer, ArchitectureOptions, Budget, PhaseName, TaskInput
from poc.execution.sql_ports import (
    PhaseTimeout,
    RequestTimeout,
    TaskEnvironment,
    ToolBudgetExceeded,
    failure_details,
)
from poc.execution.sql_strategy import (
    ContextAdmissionExceeded,
    single,
    single_json,
)
from poc.execution.sql_team_worker import (
    SupportedCritiqueValues,
    TaskEvidenceValues,
    TeamWorker,
    V21Answer,
    V21VerificationValues,
    scoped_task,
)
from poc.hybrid.collaboration_controller import CollaborationController, CollaborationError
from poc.hybrid.communication import CommunicationService
from poc.hybrid.completion_gate import CompletionGate, DeliveryBlocked
from poc.hybrid.contracts import (
    Candidate,
    CollaborationPhase,
    ContextManifest,
    EvaluationVector,
    GoalContract,
    ProvenanceRecord,
)
from poc.hybrid.planning import (
    MAX_PLAN_TASKS,
    OrchestratorPlan,
    PlannedTask,
    PlanValidationError,
    apply_decision,
    parse_decision,
    parse_plan,
    plan_problems,
    validate_decision,
    wave_order,
)
from poc.hybrid.policies import SwarmPolicyResolver
from poc.hybrid.verification_service import VerificationService
from poc.models import (
    AgentInstance,
    EventRecord,
    ExecutionMode,
    ExecutionPolicy,
    HybridConfig,
    MissionPlan,
    PlanArea,
    SwarmStrategy,
    Tier,
)
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry
from poc.services.artifact_store import ArtifactStore

METHODS = (*[mode.value for mode in ExecutionMode], "hybrid_v1", "hybrid_v2", "hybrid_v2_1")
Solve = Callable[[str, PhaseName], Awaitable[Answer]]
RECOVERABLE = (
    PhaseTimeout,
    RequestTimeout,
    UsageLimitExceeded,
    ToolBudgetExceeded,
    ModelAPIError,
    UnexpectedModelBehavior,
)


class OrchestratorEscalated(CollaborationError):
    """Typed attribution: the orchestrator escalated a task instead of revising it.

    hybrid_v2 defers escalations (the task is blocked, the run continues), so the
    decision loop no longer raises this; it stays part of the typed attribution
    surface so any escalation that must fail fast still maps to
    "orchestrator_escalated" instead of the quorum catch-all.
    """

    def __init__(self, task_id: str, rationale: str):
        self.task_id = task_id
        self.rationale = rationale
        super().__init__(f"orchestrator escalated task {task_id!r}: {rationale}")
        self.failure_details: dict[str, Any] = {
            "type": type(self).__name__,
            "scope": "phase",
            "stage": "orchestrate",
            "stage_index": None,
            "threshold": {},
            "consumption": {},
            "task_id": task_id,
            "rationale": rationale,
        }


class PlanningRejected(CollaborationError):
    """Typed attribution: hybrid_v2 planning ended without a selectable plan candidate.

    Raised when the bounded planning repair still fails plan validation, or when
    every sealed plan candidate and the bounded repair candidate was refuted by
    the plan judges. Select would otherwise raise a generic CollaborationError
    that the adapter mislabels as "proposal_quorum_not_met".
    """

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(
            "hybrid_v2 planning rejected: "
            + ("; ".join(self.problems) or "no viable plan candidate")
        )
        self.failure_details: dict[str, Any] = {
            "type": type(self).__name__,
            "scope": "phase",
            "stage": "orchestrate",
            "stage_index": None,
            "threshold": {},
            "consumption": {"problems": "; ".join(self.problems)},
        }


class DecisionRoundsExhausted(CollaborationError):
    """Typed attribution: decision rounds ended with no viable task to integrate."""

    def __init__(
        self,
        message: str = "hybrid_v2 decision rounds exhausted before integration",
        blocked: dict[str, dict[str, Any]] | None = None,
    ):
        self.blocked = dict(blocked or {})
        super().__init__(message)
        self.failure_details: dict[str, Any] = {
            "type": type(self).__name__,
            "scope": "phase",
            "stage": "orchestrate",
            "stage_index": None,
            "threshold": {},
            "consumption": {},
            "blocked": {task_id: record.get("reason") for task_id, record in self.blocked.items()},
        }


class TaskContextInfeasible(CollaborationError):
    """Typed attribution: a scoped prompt cannot fit the model's context window.

    Raised by the pre-flight check before any model request is made, so an
    infeasible task costs nothing; the decision loop defers it like an escalation.
    A task whose request-time admission also failed deterministically maps here
    with the observed error instead of prompt/window numbers.
    """

    def __init__(
        self,
        *,
        prompt_chars: int,
        estimated_tokens: int,
        context_window: int,
        model: str,
        reserve_tokens: int,
        runtime_detail: str | None = None,
    ):
        self.prompt_chars = prompt_chars
        self.estimated_tokens = estimated_tokens
        self.context_window = context_window
        self.model = model
        self.reserve_tokens = reserve_tokens
        self.runtime_detail = runtime_detail
        if runtime_detail is not None:
            message = (
                f"a task on {model} failed request admission deterministically: {runtime_detail}"
            )
        else:
            message = (
                f"scoped prompt of {prompt_chars} characters (~{estimated_tokens} estimated tokens) "
                f"plus a {reserve_tokens}-token output reserve exceeds the {context_window}-token "
                f"context window of {model}"
            )
        super().__init__(message)
        self.failure_details: dict[str, Any] = {
            "type": type(self).__name__,
            "scope": "phase",
            "stage": "task",
            "stage_index": None,
            "threshold": {
                "context_window": context_window,
                "max_output_tokens": reserve_tokens,
            },
            "consumption": {
                "prompt_chars": prompt_chars,
                "estimated_tokens": estimated_tokens,
                "model": model,
            },
        }


class StageBudgetExhausted(CollaborationError):
    """Typed attribution: the run-level budget is spent; no further stage may claim."""

    def __init__(
        self,
        *,
        stage: str,
        tokens: int,
        requests: int,
        token_limit: int,
        request_limit: int,
    ):
        self.stage = stage
        self.tokens = tokens
        self.requests = requests
        self.token_limit = token_limit
        self.request_limit = request_limit
        super().__init__(
            f"run budget exhausted before stage {stage!r}: {tokens}/{token_limit} tokens, "
            f"{requests}/{request_limit} requests"
        )
        self.failure_details: dict[str, Any] = {
            "type": type(self).__name__,
            "scope": "phase",
            "stage": stage,
            "stage_index": None,
            "threshold": {"tokens": token_limit, "requests": request_limit},
            "consumption": {"tokens": tokens, "requests": requests},
        }


class SQLTeam:
    """Trial-local authority and durable scheduler state, containing only public inputs."""

    def __init__(
        self,
        root: Path,
        method: str,
        task: TaskInput,
        budget: Budget,
        *,
        backend: str,
        model_name: str,
        options: ArchitectureOptions | None = None,
        env_schema: dict[str, list[str]] | None = None,
        usage: RunUsage | None = None,
        context_window: int | None = None,
    ):
        self.method = method
        self.options = options or ArchitectureOptions()
        self.policy_version = self.options.team_policy
        self.reliable = self.options.team_policy in {"reliable-v2", "concurrent-v1"}
        self.concurrent = self.options.team_policy == "concurrent-v1"
        self.task = task
        self.backend = backend
        self.model_name = model_name
        self.budget = budget
        # Shared run-level usage reference; budget-aware stage checks read it live.
        self.usage = usage
        # Declared context window of the resolved model; scoped-prompt pre-flight
        # admission uses it when no phase-model binding carries one.
        self.context_window = context_window
        self.env_schema = env_schema or {}
        self.task_pool_planned: dict[str, int] | None = None
        self.db = Database(root / "scheduler.sqlite")
        self.emitted = 0
        self.submitted = CommittedCandidates()
        self.artifacts = ArtifactStore(root / "artifacts", self.db)
        self.mode = (
            ExecutionMode.BOARD_CLAIM
            if method in {"hybrid_v1", "hybrid_v2", "hybrid_v2_1"}
            else ExecutionMode(method)
        )
        swarm = {
            "hybrid_v1": SwarmStrategy.HYBRID_V1,
            "hybrid_v2": SwarmStrategy.HYBRID_V2,
            "hybrid_v2_1": SwarmStrategy.HYBRID_V2,
        }.get(method, SwarmStrategy.BOARD)
        policies = SwarmPolicyResolver().resolve(swarm)
        plan = MissionPlan(
            plan_id="sql-plan",
            run_id="sql-run",
            objective=task.prompt,
            execution_mode=self.mode,
            swarm_strategy=swarm,
            policy_set=policies,
            constraints=[],
            permitted_sources=["public SQL tables"],
            permitted_tools=["query"],
            completion_criteria=["submit an answer"],
            areas=[
                PlanArea(
                    id="sql",
                    goal=task.prompt,
                    owner_role="sql-supervisor",
                    execution_mode=self.mode,
                    allowed_execution_modes=frozenset({self.mode}),
                )
            ],
            budgets={"max_workers": MAX_PLAN_TASKS + 2 if method == "hybrid_v2_1" else 3},
        )
        self.db.create_run("sql-run", task.prompt, "sql-main", plan)
        self.db.approve_plan(plan, plan.model_copy(update={"status": "approved"}))
        self.owner = self.agent("sql-supervisor", Tier.SUB, "sql-main")
        self.workers = [
            self.agent(f"sql-worker-{i}", Tier.WORKER, self.owner.agent_instance_id)
            for i in range(3)
        ]
        self.policy = ExecutionPolicy(
            mode=self.mode,
            agent_backend=backend,
            agent_model=model_name,
            swarm_strategy=swarm,
            policy_set=policies,
            max_workers=MAX_PLAN_TASKS + 2 if method == "hybrid_v2_1" else 3,
            allowed_roles=frozenset({"sql-worker"}),
            speculative_fanout=2 if self.mode == ExecutionMode.SPECULATIVE else 1,
            lease_seconds=max(1, int(budget.seconds) + 1),
        )
        self.strategy = {
            ExecutionMode.HIERARCHICAL_DAG: HierarchicalDAGStrategy,
            ExecutionMode.BOARD_CLAIM: BoardClaimStrategy,
            ExecutionMode.MANAGED_POOL: ManagedPoolStrategy,
            ExecutionMode.SPECULATIVE: SpeculativeStrategy,
        }[self.mode](self.db)

    def commit_candidate(self, answer: Answer, source: str) -> None:
        if self.method == "hybrid_v2_1" and isinstance(answer, V21Answer):
            if answer.status != "inconclusive":
                validate_artifact(
                    self.task,
                    answer,
                    self.options.artifact_contract,
                    allow_partial=answer.status == "partial",
                )
        else:
            validate_artifact(self.task, answer, self.options.artifact_contract)
        self.submitted.append(answer)
        serialized = answer.model_dump_json()
        self.db.record_event(
            EventRecord(
                run_id="sql-run",
                event_type="sql.candidate_committed",
                data={
                    "source": source,
                    "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                    "answer": answer.model_dump(),
                },
            )
        )

    def model_for(self, role: PhaseName) -> str:
        binding = self.options.phase_models.get(role) or self.options.phase_models.get("solve")
        return binding.model if binding else self.model_name

    def agent(self, name: str, tier: Tier, parent: str) -> AgentInstance:
        agent = AgentInstance(
            agent_instance_id=name,
            run_id="sql-run",
            parent_agent_id=parent,
            tier=tier,
            role_id="sql-worker" if tier == Tier.WORKER else name,
            role_version=1,
            plan_version=1,
            agent_backend=self.backend,
            agent_model=self.model_name,
        )
        self.db.put_agent(agent)
        return agent

    def flush(self, emit: Callable[[dict[str, Any]], None], method: str) -> None:
        events = self.db.events("sql-run")
        for event in events[self.emitted :]:
            emit({"orchestration": {"method": method, "policy": self.policy_version, **event}})
        self.emitted = len(events)

    async def start(self) -> None:
        self.handle = await self.strategy.submit(
            run_id="sql-run",
            owner_suborchestrator_id=self.owner.agent_instance_id,
            goal_ref="sql",
            policy=self.policy,
        )

    def stage_budget_error(self, stage: str) -> StageBudgetExhausted | None:
        """Typed exhaustion signal when the run-level budget is already spent.

        Checked before a stage attempts or refreshes a board claim: re-claiming
        with no token or request headroom only produces zero-token phase records.
        """
        if self.usage is None:
            return None
        tokens, requests = self.usage.total_tokens, self.usage.requests
        if tokens >= self.budget.total_tokens or requests >= self.budget.requests:
            return StageBudgetExhausted(
                stage=stage,
                tokens=tokens,
                requests=requests,
                token_limit=self.budget.total_tokens,
                request_limit=self.budget.requests,
            )
        return None

    def require_context_feasible(self, prompt: str, role: PhaseName) -> None:
        """Pre-flight admission for scoped prompts; raises before any model request.

        Estimated from the assembled prompt text (chars // 4, the codebase's
        conservative char-to-token ratio) against the bound model's declared
        context window minus the output reserve the stage would use. The window
        comes from a phase-model binding when one is wired, else from the
        ContextBoundModel the adapter resolved; with neither, an unknown window
        still fails at request time.
        """
        binding = self.options.phase_models.get(role) or self.options.phase_models.get("solve")
        window = (binding.endpoint_context_window or binding.context_window) if binding else None
        if window is None:
            window = self.context_window
        if window is None:
            return
        profile = self.options.role_profiles.get(role) or (
            binding.role_profiles.get(role) if binding else None
        )
        reserve = self.budget.max_output_tokens or (profile.max_output_tokens if profile else 16384)
        if profile:
            reserve = min(reserve, profile.max_output_tokens)
        estimated_tokens = len(prompt) // 4
        if estimated_tokens + reserve > window:
            raise TaskContextInfeasible(
                prompt_chars=len(prompt),
                estimated_tokens=estimated_tokens,
                context_window=window,
                model=self.model_for(role),
                reserve_tokens=reserve,
            )

    async def bounded_board_work(
        self,
        index: int,
        task_id: str,
        solve: Solve,
        prompt: str,
        dependencies: tuple[str, ...] = (),
        role: PhaseName = "solve",
    ) -> Answer:
        """Board work with one same-task retry: a recoverable failure releases the task
        back to ready and the retry re-claims the SAME id, so downstream dependencies
        observe a single completed task per stage."""
        exhausted = self.stage_budget_error(task_id)
        if exhausted is not None:
            raise exhausted
        board = cast(BoardClaimStrategy, self.strategy)
        worker = self.workers[index].model_copy(update={"agent_model": self.model_for(role)})
        self.db.put_agent(worker)
        capacity = CapacityScheduler(
            self.db, InProcessWorkerMaterializer(SpawnPolicy(self.db, RoleRegistry()))
        )
        capacity.activate_existing(self.handle, worker)
        board.post_task(
            execution_id=self.handle.execution_id,
            task_id=task_id,
            required_role=worker.role_id,
            task_spec_ref=task_id,
            dependency_ids=dependencies,
        )
        claim = board.claim_next(
            execution_id=self.handle.execution_id, worker_id=worker.agent_instance_id
        )
        if claim is None or claim.task_id != task_id:
            raise RuntimeError("SQL worker could not claim its ready task")
        try:
            answer = await solve(prompt, role)
            board.complete(claim, result_ref=answer.model_dump_json())
            return answer
        except RECOVERABLE as exc:
            if not self.reliable:
                board.release(claim, reason="SQL worker interrupted", abandon=False)
                raise
            exhausted = self.stage_budget_error(task_id)
            if exhausted is not None:
                # No headroom for the retry solve: abandon the claim instead of
                # spinning through zero-token re-claim/solve cycles.
                board.release(claim, reason="run budget exhausted", abandon=True)
                raise exhausted from exc
            self.db.record_event(
                EventRecord(
                    run_id="sql-run",
                    event_type="sql.stage_retried",
                    data={"stage": task_id, "error": type(exc).__name__},
                )
            )
            board.release(claim, reason="stage retry", abandon=False)
            retry_claim = board.claim_task(
                execution_id=self.handle.execution_id,
                task_id=task_id,
                worker_id=worker.agent_instance_id,
            )
            if retry_claim is None:
                raise RuntimeError(f"stage retry could not re-claim task {task_id!r}") from exc
            try:
                retry_answer = await solve(prompt, role)
                board.complete(retry_claim, result_ref=retry_answer.model_dump_json())
                return retry_answer
            except BaseException:
                board.release(retry_claim, reason="SQL worker interrupted", abandon=True)
                raise
        except BaseException:
            board.release(claim, reason="SQL worker interrupted", abandon=self.reliable)
            raise

    async def board_work(
        self,
        index: int,
        task_id: str,
        solve: Solve,
        prompt: str,
        dependencies: tuple[str, ...] = (),
        role: PhaseName = "solve",
    ) -> Answer:
        board = cast(BoardClaimStrategy, self.strategy)
        worker = self.workers[index].model_copy(update={"agent_model": self.model_for(role)})
        self.db.put_agent(worker)
        capacity = CapacityScheduler(
            self.db, InProcessWorkerMaterializer(SpawnPolicy(self.db, RoleRegistry()))
        )
        capacity.activate_existing(self.handle, worker)
        board.post_task(
            execution_id=self.handle.execution_id,
            task_id=task_id,
            required_role=worker.role_id,
            task_spec_ref=task_id,
            dependency_ids=dependencies,
        )
        claim = board.claim_next(
            execution_id=self.handle.execution_id, worker_id=worker.agent_instance_id
        )
        if claim is None or claim.task_id != task_id:
            raise RuntimeError("SQL worker could not claim its ready task")
        try:
            answer = await solve(prompt, role)
            board.complete(claim, result_ref=answer.model_dump_json())
            return answer
        except BaseException:
            board.release(claim, reason="SQL worker interrupted", abandon=self.reliable)
            raise

    async def run(self, method: str, solve: Solve) -> Answer:
        plan_prompt = (
            "Develop a solution plan using the public SQL tables. Return "
            'your plan as {"values":{"plan":"your reasoning and proposed queries"}}. '
            "This is an intermediate planning task, not the final answer."
        )

        def final_prompt(plan: Answer) -> str:
            return (
                "Solve the original task completely. Use this worker's plan critically: "
                + plan.model_dump_json()
            )

        if self.concurrent and method == "hierarchical_dag":
            # Two independent predecessors join before the final solver may start.
            plans = await gather_branches(
                [solve(plan_prompt + f" Independent branch {i}.", "plan") for i in range(2)]
            )
            return await solve(
                "Solve using these independent plans critically: "
                + json.dumps([p.model_dump() for p in plans]),
                "solve",
            )
        if self.concurrent and method == "board_claim":
            return await self.contended_board(solve, plan_prompt)
        if self.concurrent and method == "managed_pool":
            return await self.reassigning_pool(solve, plan_prompt)

        if method == "hierarchical_dag":

            async def plan_node(state: dict[str, Any]) -> dict[str, Any]:
                return {"plan": await solve(plan_prompt, "plan")}

            async def solve_node(state: dict[str, Any]) -> dict[str, Any]:
                return {"answer": await solve(final_prompt(state["plan"]), "solve")}

            graph = cast(Any, StateGraph(cast(Any, dict)))
            graph.add_node("plan", plan_node)
            graph.add_node("solve", solve_node)
            graph.add_edge(START, "plan")
            graph.add_edge("plan", "solve")
            graph.add_edge("solve", END)
            result = await graph.compile().ainvoke({})
            return cast(Answer, result["answer"])
        if method == "board_claim":
            plan = await self.board_work(0, "plan", solve, plan_prompt, role="plan")
            return await self.board_work(1, "solve", solve, final_prompt(plan), ("plan",))
        if method == "managed_pool":
            pool = cast(ManagedPoolStrategy, self.strategy)
            slots = [
                pool.start_slot(
                    execution_id=self.handle.execution_id, worker_id=w.agent_instance_id
                )
                for w in self.workers[:2]
            ]

            async def assigned(task_id: PhaseName, prompt: str) -> Answer:
                offer = pool.publish_offer(
                    execution_id=self.handle.execution_id,
                    task_id=task_id,
                    eligible_roles=["sql-worker"],
                    bid_deadline=datetime.now(UTC) + timedelta(seconds=1),
                )
                for slot in slots:
                    pool.submit_bid(
                        execution_id=self.handle.execution_id, offer_id=offer, slot_id=slot
                    )
                assignment = pool.arbitrate(execution_id=self.handle.execution_id, offer_id=offer)
                answer = await solve(prompt, task_id)
                pool.complete(assignment, result_ref=answer.model_dump_json())
                return answer

            plan = await assigned("plan", plan_prompt)
            return await assigned("solve", final_prompt(plan))
        if method == "speculative":
            speculative = cast(SpeculativeStrategy, self.strategy)
            group = speculative.start_group(
                execution_id=self.handle.execution_id, logical_task_id="solve"
            )
            candidates: list[Answer] = []
            grants = [
                speculative.authorize_candidate(
                    execution_id=self.handle.execution_id,
                    group_id=group,
                    worker_id=w.agent_instance_id,
                )
                for w in self.workers[:2]
            ]

            async def propose(index: int) -> Answer:
                grant = grants[index]
                answer = await solve(
                    f"Independently solve the complete task. You are candidate {index + 1}.",
                    "proposal",
                )
                speculative.complete_candidate(grant, result_ref=answer.model_dump_json())
                self.commit_candidate(answer, f"proposal-{index}")
                return answer

            if self.concurrent:
                candidates = await gather_branches([propose(i) for i in range(2)])
            else:
                for i in range(2):
                    candidates.append(await propose(i))
            answer = await solve(
                "Reconcile these independent candidate answers. Inspect SQL as needed "
                "and return your complete final answer: "
                + json.dumps([a.model_dump() for a in candidates]),
                "reconcile",
            )
            speculative.reconcile(
                execution_id=self.handle.execution_id,
                group_id=group,
                accepted_candidate_ids=[g.candidate_id for g in grants],
                final_result_ref=answer.model_dump_json(),
                reason="SQL model reconciled all submitted candidates; validity denotes schema only",
            )
            return answer
        if method in {"hybrid_v2", "hybrid_v2_1"}:
            return await self.hybrid_v2(solve, version_2_1=method == "hybrid_v2_1")
        return await self.hybrid(solve)

    async def contended_board(self, solve: Solve, prompt: str) -> Answer:
        board = cast(BoardClaimStrategy, self.strategy)
        capacity = CapacityScheduler(
            self.db, InProcessWorkerMaterializer(SpawnPolicy(self.db, RoleRegistry()))
        )
        for worker in self.workers:
            capacity.activate_existing(self.handle, worker)
        for i in range(2):
            board.post_task(
                execution_id=self.handle.execution_id,
                task_id=f"plan-{i}",
                required_role="sql-worker",
                task_spec_ref=f"plan-{i}",
            )

        async def claim(index: int) -> Answer | None:
            grant = board.claim_next(
                execution_id=self.handle.execution_id,
                worker_id=self.workers[index].agent_instance_id,
            )
            if grant is None:
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run", event_type="sql.claim_contended", data={"worker": index}
                    )
                )
                return None
            try:
                answer = await solve(prompt + f" Independent task {grant.task_id}.", "plan")
                board.complete(grant, result_ref=answer.model_dump_json())
                return answer
            except BaseException:
                board.release(grant, reason="SQL worker interrupted")
                raise

        plans = await gather_branches([claim(i) for i in range(3)])
        return await self.board_work(
            0,
            "solve",
            solve,
            "Solve using these plans critically: "
            + json.dumps([p.model_dump() for p in plans if p is not None]),
            ("plan-0", "plan-1"),
        )

    async def reassigning_pool(self, solve: Solve, prompt: str) -> Answer:
        pool = cast(ManagedPoolStrategy, self.strategy)
        slots = [
            pool.start_slot(execution_id=self.handle.execution_id, worker_id=w.agent_instance_id)
            for w in self.workers
        ]
        failed: list[tuple[str, str]] = []

        def assign(offer: str):
            for slot in slots:
                row = self.db.conn.execute(
                    "SELECT status FROM swarm_pool_slots WHERE slot_id=?", (slot,)
                ).fetchone()
                if row[0] == "idle":
                    pool.submit_bid(
                        execution_id=self.handle.execution_id, offer_id=offer, slot_id=slot
                    )
            return pool.arbitrate(execution_id=self.handle.execution_id, offer_id=offer)

        def offer(task: str) -> str:
            return pool.publish_offer(
                execution_id=self.handle.execution_id,
                task_id=task,
                eligible_roles=["sql-worker"],
                bid_deadline=datetime.now(UTC) + timedelta(days=1),
            )

        async def plan(index: int) -> Answer | None:
            task = f"plan-{index}"
            offer_id = offer(task)
            assignment = assign(offer_id)
            try:
                answer = await solve(prompt + f" Independent branch {index}.", "plan")
                pool.complete(assignment, result_ref=answer.model_dump_json())
                return answer
            except RECOVERABLE:
                pool.fail_assignment(assignment)
                failed.append((task, offer_id))
                return None

        plans = await gather_branches([plan(0), plan(1)])
        if len(failed) > 1:
            raise UnexpectedModelBehavior("pool retry allocation exhausted: two failed branches")
        # One reserved third planning allocation: reassignment or an independent check.
        assignment = assign(failed[0][1] if failed else offer("plan-check"))
        try:
            extra = await solve(prompt + " Check and complete the planning work.", "plan")
            pool.complete(assignment, result_ref=extra.model_dump_json())
        except BaseException:
            pool.fail_assignment(assignment)
            raise
        final_assignment = assign(offer("solve"))
        try:
            answer = await solve(
                "Solve using these plans critically: "
                + json.dumps([p.model_dump() for p in [*plans, extra] if p is not None]),
                "solve",
            )
        except BaseException:
            pool.fail_assignment(final_assignment)
            raise
        pool.complete(final_assignment, result_ref=answer.model_dump_json())
        return answer

    async def hybrid(self, solve: Solve) -> Answer:
        controller = CollaborationController(
            self.db, self.artifacts, BlackboardService(self.db), CommunicationService(self.db)
        )
        round_ = controller.frame(
            run_id="sql-run",
            domain_id="sql",
            steward=self.owner,
            member_agent_ids=tuple(w.agent_instance_id for w in self.workers),
            config=HybridConfig(proposals_per_round=2),
        )
        answers: dict[str, Answer] = {}
        critic_records: dict[str, dict[str, Any]] = {}

        async def propose(index: int) -> None:
            worker = self.workers[index]
            try:
                answer = await self.board_work(
                    index,
                    f"proposal-{index}",
                    solve,
                    f"Independently solve the complete task. You are sealed proposer {index + 1}.",
                    role="proposal",
                )
                validate_artifact(
                    self.task,
                    answer,
                    "public-v1" if self.reliable else self.options.artifact_contract,
                )
            except RECOVERABLE as exc:
                if not self.reliable:
                    raise
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="sql.branch_failed",
                        data={"stage": "proposal", "index": index, "error": type(exc).__name__},
                    )
                )
                return
            artifact = self.artifacts.write(
                "sql-run",
                {"proposal": index, "answer": answer.model_dump()},
                producer_task_id=f"proposal-{index}",
            )
            goal = GoalContract(
                run_id="sql-run",
                plan_version=1,
                parent_objective=self.task.prompt,
                local_scope="Independently solve the complete SQL task",
                global_invariants=("read-only SQL",),
                output_schema="Answer",
                permissions=("query",),
            )
            self.db.put_goal_contract(goal)
            manifest = ContextManifest(
                run_id="sql-run",
                plan_version=1,
                agent_instance_id=worker.agent_instance_id,
                task_id=f"proposal-{index}",
                attempt_id=f"proposal-{index}-1",
                context_policy_version=self.policy_version,
                goal_contract_id=goal.goal_contract_id,
            )
            self.db.put_context_manifest(manifest)
            self.db.put_provenance(
                ProvenanceRecord(
                    run_id="sql-run",
                    artifact_id=artifact.artifact_id,
                    producer_agent_id=worker.agent_instance_id,
                    task_id=f"proposal-{index}",
                    attempt_id=manifest.attempt_id,
                    role_id=worker.role_id,
                    role_version=1,
                    prompt_fingerprint=hashlib.sha256(self.task.prompt.encode()).hexdigest(),
                    agent_backend=self.backend,
                    model=self.model_for("proposal"),
                    context_manifest_id=manifest.context_manifest_id,
                )
            )
            candidate = controller.submit_candidate(
                round_id=round_.round_id,
                author=worker,
                hypothesis_key=f"proposal-{index}",
                artifact_id=artifact.artifact_id,
                evidence_refs=(),
            )
            answers[candidate.candidate_id] = answer.model_copy(deep=True)
            self.commit_candidate(answer, candidate.candidate_id)

        if self.concurrent:
            await gather_branches([propose(i) for i in range(2)])
        else:
            for i in range(2):
                await propose(i)
        candidates = controller.release(
            round_.round_id,
            minimum_proposals=self.options.hybrid_proposal_quorum if self.reliable else None,
        )
        controller.cluster_candidates(round_.round_id)
        for index, candidate in enumerate(candidates):
            refs: tuple[str, ...] = ()
            try:
                critique = await self.board_work(
                    2,
                    f"critique-{index}",
                    solve,
                    "Critique this proposed answer using SQL. "
                    + (
                        "Use supported-v1: structural_validity, feasibility (supported/unsupported/abstain), "
                        "evidence_refs from your SQL queries, reason, and evaluation with five 0..5 scores. "
                        if self.reliable
                        else 'Return {"values":{"validity":0,"evidence":0,"usefulness":0,"novelty":0,"constraint_satisfaction":0}} with integer scores 0..5. '
                    )
                    + "Candidate: "
                    + answers[candidate.candidate_id].model_dump_json(),
                    role="critique",
                )
                if self.reliable:
                    decision = SupportedCritiqueValues.model_validate(critique.values)
                    evaluation = decision.evaluation
                    refs = tuple(decision.evidence_refs)
                    passed = (
                        decision.structural_validity
                        and decision.feasibility == "supported"
                        and bool(refs)
                        and evaluation.constraint_satisfaction > 0
                        and evaluation.validity > 0
                        and evaluation.evidence > 0
                    )
                else:
                    evaluation = EvaluationVector.model_validate(critique.values)
                    passed = evaluation.validity > 0
                critique_data = critique.model_dump()
            except RECOVERABLE as exc:
                if not self.reliable:
                    raise
                # A failed critique abstains for this candidate; other branches can proceed.
                evaluation = EvaluationVector(
                    validity=0, evidence=0, usefulness=0, novelty=0, constraint_satisfaction=0
                )
                passed = False
                critique_data = {"abstention": type(exc).__name__}
            artifact = self.artifacts.write(
                "sql-run",
                {"candidate": candidate.candidate_id, "decision": critique_data},
                producer_task_id=f"critique-{index}",
            )
            critic_records[candidate.candidate_id] = {
                "artifact_id": artifact.artifact_id,
                "decision": critique_data,
            }
            controller.record_evaluation(
                candidate_id=candidate.candidate_id,
                critic=self.workers[2],
                critique_artifact_id=artifact.artifact_id,
                passed=passed,
                findings=(
                    "SQL model critique; supported-v1" if self.reliable else "SQL model critique",
                ),
                evidence_refs=(artifact.artifact_id,) if refs else (),
                evaluation=evaluation,
            )
        for phase in (
            CollaborationPhase.CRITIQUED,
            CollaborationPhase.TESTED,
            CollaborationPhase.RECOMBINED,
        ):
            controller.advance(round_.round_id, phase)
        selected = controller.select(round_.round_id)
        # Selection is the execution controller's policy over model judgments. No
        # benchmark score, private reference or diagnostic is available here.
        verification = VerificationService(self.db)
        request = verification.request(
            run_id="sql-run",
            subject_artifact_id=selected.artifact_id,
            producer_agent_id=selected.author_agent_id,
            schema="Answer",
            required_checks=("answer_supported",),
            evidence_refs=(),
        )
        check = await self.board_work(
            2,
            "verify",
            solve,
            "Independently verify this selected answer against the SQL data and all task constraints. "
            + (
                "Use supported-v1: answer_supported, evidence_refs from your own SQL queries, and reason. "
                if self.reliable
                else 'Return {"values":{"answer_supported":true}} if supported, or false otherwise. '
            )
            + "Critic evidence references: "
            + json.dumps(critic_records[selected.candidate_id])
            + " This is your judgment, not a benchmark score. Answer: "
            + answers[selected.candidate_id].model_dump_json(),
            role="verify",
        )
        if (not self.reliable and set(check.values) != {"answer_supported"}) or type(
            check.values["answer_supported"]
        ) is not bool:
            raise ValueError("hybrid verification must return a boolean answer_supported")
        verdict = verification.complete(
            verification_id=request.verification_id,
            verifier=self.workers[2],
            checks={"answer_supported": check.values["answer_supported"]},
            findings=("Independent SQL model verification",),
            evidence_refs=(),
            limitations=("Model judgment; no private reference or grading access",),
        )
        verification.accept(
            request=request,
            verdict=verdict,
            policy_version=self.policy_version,
            downstream_uses=("submit_answer",),
        )
        controller.advance(round_.round_id, CollaborationPhase.VERIFIED_AND_SCORED)
        gate = CompletionGate(self.db)
        plan = self.db.get_plan("sql-run")
        assert plan is not None
        gate.require(
            gate.evaluate(
                plan=plan,
                deliverable_artifact_id=selected.artifact_id,
                required_accepted_artifacts=(selected.artifact_id,),
            )
        )
        controller.complete(round_.round_id)
        return answers[selected.candidate_id].model_copy(deep=True)

    async def hybrid_v2(self, solve: Solve, *, version_2_1: bool = False) -> Answer:
        """LLM-orchestrated decomposition: plan fan-out, scoped tasks, decision loop."""
        event_prefix = "hybrid_v2_1" if version_2_1 else "hybrid_v2"
        controller = CollaborationController(
            self.db, self.artifacts, BlackboardService(self.db), CommunicationService(self.db)
        )
        fanout = self.options.hybrid_plan_fanout
        workers = self.workers
        plan_schema = json.dumps(
            {
                "parent_objective": "the original task, restated",
                "tasks": [
                    {
                        "task_id": "t1",
                        "objective": "what this task accomplishes",
                        "local_scope": "exactly what is in scope: tables, artifacts, queries",
                        **(
                            {"allowed_tables": ["public table names this task may query"]}
                            if version_2_1
                            else {}
                        ),
                        "out_of_scope": ["explicit exclusions"],
                        "definition_of_done": ["verifiable completion criteria"],
                        "dependencies": ["ids of prerequisite tasks"],
                        "output_schema": "TaskEvidence" if version_2_1 else "Answer",
                        **(
                            {
                                "budgets": {
                                    "requests": max(1, self.budget.requests // MAX_PLAN_TASKS),
                                    "tool_calls": max(1, self.budget.tool_calls // MAX_PLAN_TASKS),
                                    "total_tokens": max(
                                        1, self.budget.total_tokens // MAX_PLAN_TASKS
                                    ),
                                    "seconds": max(1, int(self.budget.seconds // MAX_PLAN_TASKS)),
                                }
                            }
                            if version_2_1
                            else {}
                        ),
                    }
                ],
                "integration_definition_of_done": ["criteria for combining all task outputs"],
            }
        )
        plan_prompt = (
            "Read the task and decompose it into between 2 and 8 small, independently "
            "verifiable tasks over the public SQL tables. Each task must have a narrow "
            "local scope, explicit out_of_scope exclusions, verifiable definition_of_done "
            "criteria, and dependencies only on earlier task ids. Prefer tasks that can be "
            "critiqued against SQL evidence. Do not investigate the data now and do not "
            "run SQL: the schema summary in this prompt is authoritative for what tables "
            "and columns exist; the executing workers and the plan judges verify claims "
            "later. Respond with the plan directly. Return your plan as "
            '{"values":' + plan_schema + "}. This is an orchestration task, not the final answer."
        )
        if version_2_1:
            plan_prompt += (
                " Plan investigations that can establish or narrow an unknown answer. "
                "Name assumptions and evidence needed in each definition_of_done. "
                "For every task set allowed_tables to the exact public tables it may query, "
                "output_schema to TaskEvidence, and positive integer limits in budgets "
                "for requests, tool_calls, total_tokens, and seconds. An exploratory "
                "task is valid when its evidence can be checked after execution."
            )

        def validate_v2_1_plan(candidate: OrchestratorPlan) -> OrchestratorPlan:
            if not version_2_1:
                return candidate
            problems = plan_problems(candidate)
            problems.extend(
                f"task {task.task_id}: allowed_tables must name existing public tables"
                for task in candidate.tasks
                if not task.allowed_tables or not set(task.allowed_tables) <= set(self.env_schema)
            )
            problems.extend(
                f"task {task.task_id}: budgets must set requests, tool_calls, total_tokens, and seconds"
                for task in candidate.tasks
                if set(task.budgets) != {"requests", "tool_calls", "total_tokens", "seconds"}
            )
            problems.extend(
                f"task {task.task_id}: output_schema must be TaskEvidence"
                for task in candidate.tasks
                if task.output_schema != "TaskEvidence"
            )
            if problems:
                raise PlanValidationError(problems)
            return candidate

        plans: dict[str, OrchestratorPlan] = {}

        planning = controller.frame(
            run_id="sql-run",
            domain_id="sql",
            steward=self.owner,
            member_agent_ids=tuple(w.agent_instance_id for w in workers),
            config=HybridConfig(proposals_per_round=fanout),
            team_id="team:sql-run:planning",
            round_id="round:sql-run:planning",
        )

        async def propose_plan(index: int) -> None:
            worker = workers[index % len(workers)]
            instruction = plan_prompt + f" You are sealed plan candidate {index + 1}."
            try:
                answer = await self.bounded_board_work(
                    index % 2, f"plan-draft-{index}", solve, instruction, role="orchestrate"
                )
                try:
                    plan = validate_v2_1_plan(parse_plan(answer.values))
                except PlanValidationError as exc:
                    answer = await self.bounded_board_work(
                        index % 2,
                        f"plan-draft-{index}-repair",
                        solve,
                        instruction
                        + " Your previous plan was rejected: "
                        + "; ".join(exc.problems)
                        + " Return one corrected plan. This is your only repair attempt.",
                        role="orchestrate",
                    )
                    plan = validate_v2_1_plan(parse_plan(answer.values))
            except RECOVERABLE as exc:
                if not self.reliable:
                    raise
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="sql.branch_failed",
                        data={
                            "stage": "plan_draft",
                            "index": index,
                            "error": type(exc).__name__,
                        },
                    )
                )
                return
            except PlanValidationError as exc:
                # An unparseable plan submission fails that branch only; the quorum
                # or the bounded planning repair absorbs it.
                if not self.reliable:
                    raise
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="sql.branch_failed",
                        data={
                            "stage": "plan_draft",
                            "index": index,
                            "error": type(exc).__name__,
                            "detail": "; ".join(exc.problems)[:200],
                        },
                    )
                )
                return
            artifact = self.artifacts.write(
                "sql-run", {"plan": plan.model_dump()}, producer_task_id=f"plan-draft-{index}"
            )
            candidate = controller.submit_candidate(
                round_id=planning.round_id,
                author=worker,
                hypothesis_key=f"plan-draft-{index}",
                artifact_id=artifact.artifact_id,
                evidence_refs=(),
            )
            plans[candidate.candidate_id] = plan

        if self.concurrent:
            await gather_branches([propose_plan(index) for index in range(fanout)])
        else:
            for index in range(fanout):
                await propose_plan(index)
        plan_candidates = controller.release(
            planning.round_id,
            minimum_proposals=self.options.hybrid_proposal_quorum if self.reliable else None,
        )
        controller.cluster_candidates(planning.round_id)
        controller.advance(planning.round_id, CollaborationPhase.CRITIQUED)

        async def judge_plan(
            candidate: Candidate, board_task_id: str
        ) -> tuple[bool, EvaluationVector, dict[str, Any]]:
            refs: tuple[str, ...] = ()
            try:
                critique = await self.bounded_board_work(
                    2,
                    board_task_id,
                    solve,
                    "Judge this proposed decomposition of the original task. "
                    + (
                        "For exploratory work, abstain on feasibility when data inspection is "
                        "required; reserve unsupported for a demonstrable contradiction. "
                        if version_2_1
                        else ""
                    )
                    + "Review the PLAN, never perform it: verify that every referenced table and column "
                    "exists, that each task has one narrow scope with no overlapping work, "
                    "that dependencies form a sound acyclic order, and that every "
                    "definition_of_done criterion is objectively checkable by the task that "
                    "owns it. Do not investigate the incident yourself and do not run the "
                    "plan's queries end-to-end; schema and row-existence checks are enough "
                    "to ground feasibility, and you should stop as soon as every task has a "
                    "justified verdict. If a query fails or returns nothing useful, adjust "
                    "it once using the SQL schema summary above and never repeat a statement "
                    "that already failed; if a few checks cannot ground feasibility, return "
                    "feasibility='abstain' with your reason instead of investigating further. "
                    + (
                        "Use supported-v1: structural_validity, feasibility (supported/unsupported/abstain), "
                        "evidence_refs copied exactly from the evidence_ref values your query results returned, "
                        "reason, and evaluation with five 0..5 scores. "
                        if self.reliable
                        else 'Return {"values":{"validity":0,"evidence":0,"usefulness":0,"novelty":0,"constraint_satisfaction":0}} with integer scores 0..5. '
                    )
                    + "Candidate plan: "
                    + plans[candidate.candidate_id].model_dump_json(),
                    role="critique",
                )
                if self.reliable:
                    judge = SupportedCritiqueValues.model_validate(critique.values)
                    evaluation = judge.evaluation
                    refs = tuple(judge.evidence_refs)
                    passed = (
                        judge.structural_validity
                        and (
                            judge.feasibility != "unsupported"
                            if version_2_1
                            else judge.feasibility == "supported"
                        )
                        and (version_2_1 or bool(refs))
                        and evaluation.constraint_satisfaction > 0
                        and evaluation.validity > 0
                        and (version_2_1 or evaluation.evidence > 0)
                    )
                else:
                    evaluation = EvaluationVector.model_validate(critique.values)
                    passed = evaluation.validity > 0
                critique_data = critique.model_dump()
            except RECOVERABLE as exc:
                if not self.reliable:
                    raise
                evaluation = EvaluationVector(
                    validity=0, evidence=0, usefulness=0, novelty=0, constraint_satisfaction=0
                )
                passed = False
                critique_data = {"abstention": type(exc).__name__}
            judge_artifact = self.artifacts.write(
                "sql-run",
                {"candidate": candidate.candidate_id, "decision": critique_data},
                producer_task_id=board_task_id,
            )
            controller.record_evaluation(
                candidate_id=candidate.candidate_id,
                critic=workers[2],
                critique_artifact_id=judge_artifact.artifact_id,
                passed=passed,
                findings=(
                    ("orchestrator plan critique; supported-v1",)
                    if self.reliable
                    else ("orchestrator plan critique",)
                ),
                evidence_refs=(judge_artifact.artifact_id,) if refs else (),
                evaluation=evaluation,
            )
            return passed, evaluation, critique_data

        plan_findings: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(plan_candidates):
            _, _, decision = await judge_plan(candidate, f"plan-judge-{index}")
            plan_findings[candidate.candidate_id] = decision
        controller.advance(planning.round_id, CollaborationPhase.TESTED)

        def viable_plans() -> list[str]:
            return [
                candidate.candidate_id
                for candidate in controller.release(planning.round_id)
                if candidate.state == "viable"
            ]

        if not viable_plans():
            # One bounded re-plan when every sealed plan candidate was rejected: the
            # orchestrator sees the recorded findings and rebuilds once, judged like
            # any other plan candidate.
            findings = json.dumps(
                {
                    "rejected": [
                        {
                            "candidate": candidate.candidate_id,
                            "decision": plan_findings.get(candidate.candidate_id, {}),
                        }
                        for candidate in plan_candidates
                    ]
                }
            )
            repair_answer = await self.bounded_board_work(
                0,
                "plan-repair",
                solve,
                plan_prompt + " Both sealed plan candidates were rejected by the judge with these "
                "findings: "
                + findings
                + " Rebuild one corrected decomposition that addresses the findings. "
                "This is your only repair attempt.",
                role="orchestrate",
            )
            try:
                repair_plan = validate_v2_1_plan(parse_plan(repair_answer.values))
            except PlanValidationError as exc:
                # A completed repair draft whose answer still fails plan validation
                # must not escape untyped: reject the planning round typed.
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type=f"{event_prefix}.planning_rejected",
                        data={
                            "stage": "plan_repair",
                            "error": type(exc).__name__,
                            "problems": "; ".join(exc.problems),
                        },
                    )
                )
                raise PlanningRejected(list(exc.problems)) from exc
            repair_artifact = self.artifacts.write(
                "sql-run", {"plan": repair_plan.model_dump()}, producer_task_id="plan-repair"
            )
            repair_candidate = controller.revise_candidate(
                round_id=planning.round_id,
                author=workers[0],
                hypothesis_key="plan-repair",
                artifact_id=repair_artifact.artifact_id,
            )
            plans[repair_candidate.candidate_id] = repair_plan
            plan_candidates = (*plan_candidates, repair_candidate)
            await judge_plan(repair_candidate, "plan-repair-judge")
        if not viable_plans():
            # Every sealed plan candidate and the bounded repair (when it ran) was
            # refuted: reject planning typed instead of letting select raise a
            # generic CollaborationError the adapter mislabels as a quorum failure.
            # The round phase here is TESTED, from which release is legal: release
            # only transitions out of COLLECTING_SEALED_PROPOSALS and returns the
            # recorded candidates from any later phase, so re-reading viability
            # neither changes the phase nor re-releases anything.
            self.db.record_event(
                EventRecord(
                    run_id="sql-run",
                    event_type=f"{event_prefix}.planning_rejected",
                    data={
                        "stage": "plan_select",
                        "problems": (
                            f"all {len(plan_candidates)} plan candidates were refuted by the judges"
                        ),
                        "findings": {
                            candidate.candidate_id: plan_findings.get(candidate.candidate_id, {})
                            for candidate in plan_candidates
                        },
                    },
                )
            )
            raise PlanningRejected(["every plan candidate was refuted by the plan judges"])
        controller.advance(planning.round_id, CollaborationPhase.RECOMBINED)
        selected_plan = controller.select(planning.round_id)
        plan = plans[selected_plan.candidate_id]
        if version_2_1 and self.task_pool_planned is not None:
            # Reserve for the selected task count plus bounded revision rounds.
            # Small plans need useful per-task SQL capacity; large plans retain
            # the existing eight-task conservative ceiling.
            self.task_pool_planned["task"] = min(
                MAX_PLAN_TASKS, len(plan.tasks) + self.options.hybrid_max_decision_rounds
            )
        self.db.record_event(
            EventRecord(
                run_id="sql-run",
                event_type=f"{event_prefix}.plan_selected",
                data={"plan": plan.model_dump(), "candidate_id": selected_plan.candidate_id},
            )
        )

        task_answers: dict[str, Answer] = {}
        task_artifacts: dict[str, str] = {}
        task_board_ids: dict[str, str] = {}
        task_candidates: dict[str, str] = {}
        task_states: dict[str, str] = {}
        task_findings: dict[str, dict[str, Any]] = {}
        task_attempts: dict[str, int] = {}
        task_block_reasons: dict[str, dict[str, Any]] = {}
        # Per-task failure budget keyed by the plan's LOGICAL task id (e.g. "t3").
        # Derivation: execute_task receives PlannedTask objects whose task_id is the
        # logical plan id; board ids are per attempt ("task-t3" for the first
        # execution, "task-t3-rev4" for decision revisions) because apply_decision
        # replaces a revised task in place under the SAME task_id. Keying on
        # task.task_id therefore groups every revision of one logical task into a
        # single budget. One counter entry equals one failed bounded_board_work
        # pair (initial attempt + its one same-task retry).
        task_failure_pairs: dict[str, int] = {}

        def block_task(task_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
            """Mark a task blocked: it cannot run this run, and dependents inherit it.

            Escalations and context-infeasible prompts defer here instead of killing
            the run; viable tasks still reach integration, and a run where nothing
            stays viable fails via the post-loop check carrying these reasons.
            """
            task_states[task_id] = "blocked"
            task_block_reasons[task_id] = {"reason": reason, **(details or {})}

        # The execution round's proposal population equals the task count, so its member
        # roster must scale with the selected plan instead of the fixed worker pool.
        execution_members = workers if version_2_1 else list(workers)
        while len(execution_members) < (
            MAX_PLAN_TASKS + 2 if version_2_1 else max(2, len(plan.tasks))
        ):
            execution_members.append(
                self.agent(
                    f"sql-worker-{len(execution_members)}",
                    Tier.WORKER,
                    self.owner.agent_instance_id,
                )
            )
        execution = controller.frame(
            run_id="sql-run",
            domain_id="sql",
            steward=self.owner,
            member_agent_ids=tuple(m.agent_instance_id for m in execution_members),
            config=HybridConfig(proposals_per_round=max(2, len(plan.tasks))),
            team_id="team:sql-run:execution",
            round_id="round:sql-run:execution",
        )

        def task_worker_index(task: PlannedTask) -> int:
            index = next(i for i, item in enumerate(plan.tasks) if item.task_id == task.task_id)
            return index if index < 2 else index + 1

        def task_prompt(task: PlannedTask) -> str:
            return (
                f"Execute ONLY this scoped task.\nTask id: {task.task_id}\n"
                f"Objective: {task.objective}\nLocal scope: {task.local_scope}\n"
                f"Out of scope: {'; '.join(task.out_of_scope) or 'nothing beyond the local scope'}\n"
                + (f"Allowed SQL tables: {', '.join(task.allowed_tables)}\n" if version_2_1 else "")
                + (
                    "Accepted prerequisite outputs: "
                    + json.dumps({dep: task_answers[dep].values for dep in task.dependencies})
                    + "\n"
                    if version_2_1
                    and task.dependencies
                    and all(dep in task_answers for dep in task.dependencies)
                    else ""
                )
                + "Definition of done:\n- "
                + "\n- ".join(task.definition_of_done)
                + "\nDo not work outside the local scope; other tasks cover the rest of "
                "the objective. Return this task's output, not the whole answer."
            )

        async def execute_task(task: PlannedTask, index: int, attempt: int) -> None:
            worker = workers[index] if version_2_1 else workers[index % 2]
            task_attempts[task.task_id] = attempt
            task_id = (
                f"task-{task.task_id}" if attempt == 1 else f"task-{task.task_id}-rev{attempt}"
            )
            missing = [
                dep
                for dep in task.dependencies
                if (task_states.get(dep) != "viable" if version_2_1 else dep not in task_artifacts)
            ]
            if missing:
                # A failed dependency blocks execution; the decision loop re-plans it.
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="sql.task_blocked",
                        data={"task": task.task_id, "missing_dependencies": missing},
                    )
                )
                return
            # The worker embeds the whole task input and schema in the scoped prompt;
            # pre-flight the exact assembled text so an over-window prompt fails here,
            # before any claim or request, instead of grinding through decision rounds.
            scoped_prompt = (
                self.task.prompt
                + "\nRole task: "
                + task_prompt(task)
                + "\nSQL schema: "
                + json.dumps(
                    {name: self.env_schema[name] for name in task.allowed_tables}
                    if version_2_1
                    else self.env_schema
                )
            )
            try:
                self.require_context_feasible(scoped_prompt, "task")
            except TaskContextInfeasible as exc:
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="sql.task_context_infeasible",
                        data={"task": task.task_id, **exc.failure_details["consumption"]},
                    )
                )
                block_task(
                    task.task_id,
                    "context_infeasible",
                    {
                        "prompt_chars": exc.prompt_chars,
                        "estimated_tokens": exc.estimated_tokens,
                        "context_window": exc.context_window,
                        "model": exc.model,
                        "reserve_tokens": exc.reserve_tokens,
                    },
                )
                return
            dependencies = tuple(task_board_ids[dep] for dep in task.dependencies)

            def record_goal_manifest() -> ContextManifest:
                goal = GoalContract(
                    run_id="sql-run",
                    plan_version=plan.version,
                    parent_objective=self.task.prompt,
                    local_scope=f"{task.objective} — {task.local_scope}",
                    global_invariants=("read-only SQL",),
                    input_artifact_refs=tuple(
                        task_artifacts[dep] for dep in task.dependencies if dep in task_artifacts
                    ),
                    output_schema=task.output_schema,
                    evidence_requirements=task.evidence_requirements,
                    definition_of_done=task.definition_of_done,
                    budgets=task.budgets,
                    permissions=("query",),
                )
                self.db.put_goal_contract(goal)
                manifest = ContextManifest(
                    run_id="sql-run",
                    plan_version=plan.version,
                    agent_instance_id=worker.agent_instance_id,
                    task_id=task_id,
                    attempt_id=f"{task_id}-1",
                    context_policy_version=self.policy_version,
                    goal_contract_id=goal.goal_contract_id,
                )
                self.db.put_context_manifest(manifest)
                return manifest

            manifest = record_goal_manifest() if version_2_1 else None
            try:
                with scoped_task(task) if version_2_1 else nullcontext():
                    answer = await self.bounded_board_work(
                        index if version_2_1 else index % 2,
                        task_id,
                        solve,
                        task_prompt(task),
                        dependencies,
                        role="task",
                    )
            except RECOVERABLE as exc:
                if not self.reliable:
                    raise
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="sql.branch_failed",
                        data={"stage": "task", "index": task.task_id, "error": type(exc).__name__},
                    )
                )
                if isinstance(exc, ContextAdmissionExceeded):
                    # Admission is a deterministic property of this task's prompt and
                    # gathered evidence on this model: retrying cannot shrink it.
                    # Defer the task like an infeasible pre-flight instead of letting
                    # the decision loop re-execute it round after round. Excluded
                    # from the failure budget: it blocks immediately and cannot
                    # converge through revisions.
                    block_task(
                        task.task_id,
                        "context_infeasible",
                        {
                            "stage": "task",
                            "error": type(exc).__name__,
                            "detail": str(exc)[:200],
                        },
                    )
                    return
                failed_pairs = task_failure_pairs.get(task.task_id, 0) + 1
                task_failure_pairs[task.task_id] = failed_pairs
                if version_2_1 and failed_pairs < 2:
                    task_states[task.task_id] = "failed"
                if failed_pairs >= 2:
                    # Two failed execution pairs for the same logical task: defer it
                    # instead of re-executing revised versions of a task that never
                    # converges across the remaining decision rounds.
                    block_task(
                        task.task_id,
                        "task_failures_exhausted",
                        {
                            "failed_pairs": failed_pairs,
                            "attempts": failed_pairs * 2,
                            "error": type(exc).__name__,
                        },
                    )
                return
            artifact = self.artifacts.write(
                "sql-run",
                {"task": task.task_id, "attempt": attempt, "answer": answer.model_dump()},
                producer_task_id=task_id,
            )
            if manifest is None:
                manifest = record_goal_manifest()
            self.db.put_provenance(
                ProvenanceRecord(
                    run_id="sql-run",
                    artifact_id=artifact.artifact_id,
                    producer_agent_id=worker.agent_instance_id,
                    task_id=task_id,
                    attempt_id=manifest.attempt_id,
                    role_id=worker.role_id,
                    role_version=1,
                    prompt_fingerprint=hashlib.sha256(
                        (self.task.prompt + task.task_id + str(attempt)).encode()
                    ).hexdigest(),
                    agent_backend=self.backend,
                    model=self.model_for("task"),
                    context_manifest_id=manifest.context_manifest_id,
                )
            )
            if attempt == 1 and not (version_2_1 and initial_released):
                candidate = controller.submit_candidate(
                    round_id=execution.round_id,
                    author=worker,
                    hypothesis_key=task.task_id,
                    artifact_id=artifact.artifact_id,
                    evidence_refs=(),
                )
            else:
                candidate = controller.revise_candidate(
                    round_id=execution.round_id,
                    author=worker,
                    hypothesis_key=task.task_id,
                    artifact_id=artifact.artifact_id,
                    parent_candidate_ids=(
                        (task_candidates[task.task_id],) if task.task_id in task_candidates else ()
                    ),
                )
            task_answers[task.task_id] = answer.model_copy(deep=True)
            task_artifacts[task.task_id] = artifact.artifact_id
            task_board_ids[task.task_id] = task_id
            task_candidates[task.task_id] = candidate.candidate_id
            if attempt != 1:
                critiqued.discard(task.task_id)

        def critique_prompt(task: PlannedTask) -> str:
            output = task_answers.get(task.task_id)
            return (
                "Audit this scoped task output against its OWN contract, not the whole "
                "task: verify each definition_of_done criterion and look for work beyond "
                "local_scope. "
                + (
                    "Distinguish a disproved finding from one not yet checked. "
                    if version_2_1
                    else ""
                )
                + "Do not re-derive, extend, or improve the task's result; a "
                "few targeted checks that show whether the stated criteria are met are "
                "enough. If a query fails or returns nothing useful, adjust it once using "
                "the SQL schema summary above and never repeat a statement that already "
                "failed; if you cannot ground the audit, record the unmet criteria and "
                "reason instead of investigating further. "
                + (
                    "Use supported-v1: structural_validity, feasibility (supported/unsupported/abstain), "
                    "evidence_refs from your SQL queries, reason, and evaluation with five 0..5 scores. "
                    if self.reliable
                    else 'Return {"values":{"validity":0,"evidence":0,"usefulness":0,"novelty":0,"constraint_satisfaction":0}} with integer scores 0..5. '
                )
                + "Task contract: "
                + json.dumps(
                    {
                        "task_id": task.task_id,
                        "objective": task.objective,
                        "local_scope": task.local_scope,
                        "out_of_scope": list(task.out_of_scope),
                        "definition_of_done": list(task.definition_of_done),
                    }
                )
                + " Output: "
                + (
                    json.dumps(output.values)
                    if output is not None
                    else json.dumps({"missing": "the task produced no committed output"})
                )
            )

        critiqued: set[str] = set()
        deferred_evaluations: list[dict[str, Any]] = []

        async def critique_pass() -> None:
            for task in plan.tasks:
                if task.task_id in critiqued or task.task_id not in task_candidates:
                    continue
                critique_task_id = (
                    f"critique-task-{task.task_id}-{task_attempts.get(task.task_id, 1)}"
                )
                critique_instruction = critique_prompt(task)
                refs: tuple[str, ...] = ()
                try:
                    critique = await self.bounded_board_work(
                        2,
                        critique_task_id,
                        solve,
                        critique_instruction,
                        role="critique",
                    )
                    if self.reliable:
                        decision = SupportedCritiqueValues.model_validate(critique.values)
                        evaluation = decision.evaluation
                        refs = tuple(decision.evidence_refs)
                        passed = (
                            decision.structural_validity
                            and decision.feasibility == "supported"
                            and bool(refs)
                            and evaluation.constraint_satisfaction > 0
                            and evaluation.validity > 0
                            and evaluation.evidence > 0
                        )
                    else:
                        evaluation = EvaluationVector.model_validate(critique.values)
                        passed = evaluation.validity > 0
                    critique_data = critique.model_dump()
                except RECOVERABLE as exc:
                    if not self.reliable:
                        raise
                    evaluation = EvaluationVector(
                        validity=0, evidence=0, usefulness=0, novelty=0, constraint_satisfaction=0
                    )
                    passed = False
                    critique_data = {"abstention": type(exc).__name__}
                artifact = self.artifacts.write(
                    "sql-run",
                    {
                        "task": task.task_id,
                        "candidate": task_candidates[task.task_id],
                        "decision": critique_data,
                    },
                    producer_task_id=(
                        f"critique-task-{task.task_id}-{task_attempts.get(task.task_id, 1)}"
                    ),
                )
                evaluation_args: dict[str, Any] = dict(
                    candidate_id=task_candidates[task.task_id],
                    critic=workers[2],
                    critique_artifact_id=artifact.artifact_id,
                    passed=passed,
                    findings=(
                        ("scoped task critique; supported-v1",)
                        if self.reliable
                        else ("scoped task critique",)
                    ),
                    evidence_refs=(artifact.artifact_id,) if refs else (),
                    evaluation=evaluation,
                )
                if version_2_1 and not initial_released:
                    deferred_evaluations.append(evaluation_args)
                else:
                    controller.record_evaluation(**evaluation_args)
                task_states[task.task_id] = "viable" if passed else "refuted"
                task_findings[task.task_id] = critique_data
                critiqued.add(task.task_id)

        initial_released = False
        waves = wave_order(plan)
        for wave in waves:
            if self.concurrent:
                await gather_branches(
                    [
                        execute_task(task, task_worker_index(task) if version_2_1 else index, 1)
                        for index, task in enumerate(wave)
                    ]
                )
            else:
                for index, task in enumerate(wave):
                    await execute_task(task, task_worker_index(task) if version_2_1 else index, 1)
            if version_2_1:
                # A prerequisite is usable only after its own audit accepted it.
                await critique_pass()
        controller.release(execution.round_id, minimum_proposals=1 if task_candidates else 0)
        controller.cluster_candidates(execution.round_id)
        controller.advance(execution.round_id, CollaborationPhase.CRITIQUED)
        initial_released = True
        if version_2_1:
            for evaluation_args in deferred_evaluations:
                controller.record_evaluation(**evaluation_args)
        elif task_candidates:
            await critique_pass()
        controller.advance(execution.round_id, CollaborationPhase.TESTED)

        def invalidate_dependents(task_id: str) -> None:
            stale = {task_id}
            while True:
                descendants = {
                    task.task_id
                    for task in plan.tasks
                    if any(dep in stale for dep in task.dependencies)
                }
                if descendants <= stale:
                    break
                stale.update(descendants)
            stale.discard(task_id)
            for descendant in stale:
                task_states[descendant] = "stale"
                task_answers.pop(descendant, None)
                task_artifacts.pop(descendant, None)
                task_board_ids.pop(descendant, None)
                critiqued.discard(descendant)
                self.db.record_event(
                    EventRecord(
                        run_id="sql-run",
                        event_type="hybrid_v2_1.task_invalidated",
                        data={"task": descendant, "changed_dependency": task_id},
                    )
                )

        async def rerun_ready_dependents() -> None:
            for wave in wave_order(plan):
                ready = [
                    task
                    for task in wave
                    if (
                        task_states.get(task.task_id) == "stale"
                        or (task_states.get(task.task_id) is None and bool(task.dependencies))
                    )
                    and all(task_states.get(dep) == "viable" for dep in task.dependencies)
                ]
                for task in ready:
                    next_attempt = (
                        task_attempts.get(task.task_id, 0) + 1
                        if task.task_id in task_candidates
                        else 1
                    )
                    await execute_task(task, task_worker_index(task), next_attempt)
                if ready:
                    await critique_pass()

        def decision_prompt(task: PlannedTask, findings: dict[str, Any], suffix: str) -> str:
            return (
                "One scoped task failed its critique. Choose exactly one decision: "
                '"revise" (re-issue the same task id with tightened or corrected scope), '
                '"add_task" (only when the critique exposed a gap no existing task covers; '
                "the added task may depend only on accepted tasks), or "
                '"escalate" (only when no revision can fix it). Never accept a refuted output. '
                'Return {"values": {"decision": ..., "rationale": ..., "revision": <revised task '
                'object>, "added_task": <new task object>}}; omit fields that do not apply. '
                "Task contract: "
                + task.model_dump_json()
                + " Critique findings: "
                + json.dumps(findings)
                + suffix
            )

        decision_rounds = 0
        while decision_rounds < self.options.hybrid_max_decision_rounds:
            # Blocked tasks poison their dependents: a task whose dependency is
            # blocked can never run, so it is blocked explicitly (skipped without a
            # decision request) instead of re-entering the decision loop forever.
            # Transitive closure runs each round; it terminates because only tasks
            # that are neither viable nor blocked are ever marked.
            while True:
                newly_blocked = [
                    task
                    for task in plan.tasks
                    if task_states.get(task.task_id) not in {"viable", "blocked"}
                    and any(task_states.get(dep) == "blocked" for dep in task.dependencies)
                ]
                if not newly_blocked:
                    break
                for task in newly_blocked:
                    dependency = next(
                        dep for dep in task.dependencies if task_states.get(dep) == "blocked"
                    )
                    self.db.record_event(
                        EventRecord(
                            run_id="sql-run",
                            event_type="sql.task_blocked",
                            data={"task": task.task_id, "blocked_dependency": dependency},
                        )
                    )
                    block_task(
                        task.task_id,
                        "blocked_dependency",
                        {
                            "dependency": dependency,
                            "dependency_reason": task_block_reasons.get(dependency, {}).get(
                                "reason"
                            ),
                        },
                    )
            pending = [
                task
                for task in plan.tasks
                if task_states.get(task.task_id) not in {"viable", "blocked"}
                and (
                    not version_2_1
                    or all(task_states.get(dep) == "viable" for dep in task.dependencies)
                )
            ]
            if not pending:
                break
            decision_rounds += 1
            for task in pending:
                findings = task_findings.get(
                    task.task_id, {"missing": task.task_id not in task_candidates}
                )
                decision_instruction = decision_prompt(
                    task, findings, f" Decision round {decision_rounds}."
                )
                decision_task_id = f"decision-{task.task_id}-{decision_rounds}"
                try:
                    decision_answer = await self.bounded_board_work(
                        2,
                        decision_task_id,
                        solve,
                        decision_instruction,
                        role="orchestrate",
                    )
                    decision = parse_decision(decision_answer.values)
                    validate_decision(
                        decision, plan, pending_task_id=task.task_id if version_2_1 else None
                    )
                    if version_2_1:
                        validate_v2_1_plan(apply_decision(plan, decision))
                except RECOVERABLE as acquisition_exc:
                    # A failed decision consumes the round for this task; the next
                    # decision round re-plans it.
                    if not self.reliable:
                        raise
                    self.db.record_event(
                        EventRecord(
                            run_id="sql-run",
                            event_type=f"{event_prefix}.decision_unusable",
                            data={
                                "task": task.task_id,
                                "attempted": decision_rounds,
                                "error": type(acquisition_exc).__name__,
                            },
                        )
                    )
                    continue
                except PlanValidationError:
                    self.db.record_event(
                        EventRecord(
                            run_id="sql-run",
                            event_type=f"{event_prefix}.decision_unusable",
                            data={
                                "task": task.task_id,
                                "attempted": decision_rounds,
                                "error": "PlanValidationError",
                            },
                        )
                    )
                    continue
                try:
                    if decision.added_task is not None and any(
                        dep not in task_states or task_states[dep] != "viable"
                        for dep in decision.added_task.dependencies
                    ):
                        raise PlanValidationError(
                            ["added task dependencies must reference accepted tasks"]
                        )
                    if (
                        decision.decision == "revise"
                        and decision.revision is not None
                        and any(
                            dep not in task_states or task_states[dep] != "viable"
                            for dep in decision.revision.dependencies
                        )
                    ):
                        raise PlanValidationError(
                            ["revision dependencies must reference accepted tasks"]
                        )
                except PlanValidationError as exc:
                    repair_answer: Answer | None = None
                    try:
                        decision_answer = await self.bounded_board_work(
                            2,
                            f"decision-{task.task_id}-{decision_rounds}-repair",
                            solve,
                            decision_prompt(task, findings, "")
                            + " Your previous decision was rejected: "
                            + "; ".join(exc.problems)
                            + " Return one corrected decision.",
                            role="orchestrate",
                        )
                        decision = parse_decision(decision_answer.values)
                        validate_decision(
                            decision, plan, pending_task_id=task.task_id if version_2_1 else None
                        )
                        if version_2_1:
                            validate_v2_1_plan(apply_decision(plan, decision))
                        repair_answer = decision_answer
                    except PlanValidationError as repair_exc:
                        # An unusable decision consumes the round for this task;
                        # the next decision round re-plans it.
                        self.db.record_event(
                            EventRecord(
                                run_id="sql-run",
                                event_type=f"{event_prefix}.decision_unusable",
                                data={
                                    "task": task.task_id,
                                    "attempted": decision_rounds,
                                    "error": type(repair_exc).__name__,
                                },
                            )
                        )
                        continue
                    except RECOVERABLE:
                        raise
                if decision.decision == "escalate":
                    self.db.record_event(
                        EventRecord(
                            run_id="sql-run",
                            event_type=f"{event_prefix}.escalated",
                            data={"task": task.task_id, "rationale": decision.rationale},
                        )
                    )
                    # Deferred, not fatal: an escalation blocks its own task and the
                    # decision loop continues, so other viable tasks still reach
                    # integration. Dependents are blocked by the propagation above;
                    # a run with nothing left viable fails at the post-loop check.
                    block_task(task.task_id, "escalated", {"rationale": decision.rationale})
                    continue
                if decision.decision == "revise" and decision.revision is not None:
                    plan = apply_decision(plan, decision)
                    if version_2_1:
                        invalidate_dependents(decision.revision.task_id)
                    await execute_task(
                        decision.revision,
                        task_worker_index(decision.revision) if version_2_1 else 0,
                        decision_rounds + 1,
                    )
                if decision.decision == "add_task" and decision.added_task is not None:
                    plan = apply_decision(plan, decision)
                    await execute_task(
                        decision.added_task,
                        task_worker_index(decision.added_task) if version_2_1 else 1,
                        decision_rounds + 1,
                    )
            await critique_pass()
            if version_2_1:
                await rerun_ready_dependents()
        if not any(task_states.get(task.task_id) == "viable" for task in plan.tasks):
            # Nothing executed and survived critique: no integration is possible.
            # The typed cause is the recorded block reasons, so the failure code says
            # why (escalations, context-infeasible prompts) instead of the quorum
            # catch-all. A context-infeasible task is a hard, permanent condition,
            # so it takes precedence over other block reasons; a pre-flight block
            # carries the prompt/window numbers, a runtime admission block only the
            # observed error, and both map to the same typed surface.
            preflight = [
                record
                for record in task_block_reasons.values()
                if record["reason"] == "context_infeasible" and "context_window" in record
            ]
            if preflight:
                raise TaskContextInfeasible(
                    prompt_chars=preflight[0]["prompt_chars"],
                    estimated_tokens=preflight[0]["estimated_tokens"],
                    context_window=preflight[0]["context_window"],
                    model=preflight[0]["model"],
                    reserve_tokens=preflight[0]["reserve_tokens"],
                )
            if any(
                record["reason"] == "context_infeasible" for record in task_block_reasons.values()
            ):
                runtime = next(
                    record
                    for record in task_block_reasons.values()
                    if record["reason"] == "context_infeasible" and "context_window" not in record
                )
                raise TaskContextInfeasible(
                    prompt_chars=0,
                    estimated_tokens=0,
                    context_window=0,
                    model=self.model_for("task"),
                    reserve_tokens=0,
                    runtime_detail=str(runtime.get("detail") or runtime.get("error")),
                )
            raise DecisionRoundsExhausted(
                "hybrid_v2 decision rounds exhausted before integration", task_block_reasons
            )

        controller.advance(execution.round_id, CollaborationPhase.RECOMBINED)
        # Only accepted (viable) tasks reach integration; blocked tasks contributed
        # no output and their dependents were blocked with them. The verifier still
        # sees every task's state below, so uncovered scope stays visible.
        accepted_tasks = [
            task
            for task in plan.tasks
            if task_states.get(task.task_id) == "viable" and task.task_id in task_answers
        ]
        integration_prompt = (
            "Combine these scoped task outputs into the complete final answer for the "
            "original task: "
            + self.task.prompt
            + ". Integration definition of done: "
            + "; ".join(plan.integration_definition_of_done)
            + ". Resolve conflicts between task outputs; never drop covered work. "
            + (
                " State every remaining gap; return status supported, partial, or inconclusive "
                "with claims, assumptions, and unresolved_questions. "
                if version_2_1
                else ""
            )
            + "Task outputs: "
            + json.dumps(
                [
                    {"task": task.task_id, "answer": task_answers[task.task_id].model_dump()}
                    for task in accepted_tasks
                ]
            )
        )
        integrated = await self.bounded_board_work(
            0,
            "integrate",
            solve,
            integration_prompt,
            tuple(
                task_board_ids[task.task_id]
                for task in accepted_tasks
                if task.task_id in task_board_ids
            ),
            role="integrate",
        )
        if (
            version_2_1
            and V21Answer.model_validate(integrated).status == "supported"
            and any(task_states.get(task.task_id) != "viable" for task in plan.tasks)
        ):
            raise DeliveryBlocked(
                "supported answer requires accepted coverage of every planned task"
            )
        self.commit_candidate(integrated, "integrate")
        integration_artifact = self.artifacts.write(
            "sql-run", {"integration": integrated.model_dump()}, producer_task_id="integrate"
        )
        integration_goal = GoalContract(
            run_id="sql-run",
            plan_version=plan.version,
            parent_objective=self.task.prompt,
            local_scope="Integrate accepted scoped task outputs into the complete answer",
            global_invariants=("read-only SQL",),
            input_artifact_refs=tuple(
                task_artifacts[task.task_id]
                for task in accepted_tasks
                if task.task_id in task_artifacts
            ),
            output_schema="Answer",
            definition_of_done=plan.integration_definition_of_done,
            permissions=("query",),
        )
        self.db.put_goal_contract(integration_goal)
        integration_manifest = ContextManifest(
            run_id="sql-run",
            plan_version=plan.version,
            agent_instance_id=workers[0].agent_instance_id,
            task_id="integrate",
            attempt_id="integrate-1",
            context_policy_version=self.policy_version,
            goal_contract_id=integration_goal.goal_contract_id,
        )
        self.db.put_context_manifest(integration_manifest)
        self.db.put_provenance(
            ProvenanceRecord(
                run_id="sql-run",
                artifact_id=integration_artifact.artifact_id,
                producer_agent_id=workers[0].agent_instance_id,
                task_id="integrate",
                attempt_id=integration_manifest.attempt_id,
                role_id=workers[0].role_id,
                role_version=1,
                prompt_fingerprint=hashlib.sha256(
                    (self.task.prompt + "integrate" + str(plan.version)).encode()
                ).hexdigest(),
                agent_backend=self.backend,
                model=self.model_for("integrate"),
                context_manifest_id=integration_manifest.context_manifest_id,
            )
        )

        verification = VerificationService(self.db)
        integrated_v21 = V21Answer.model_validate(integrated) if version_2_1 else None
        accepted_evidence = (
            {
                task.task_id: TaskEvidenceValues.model_validate(task_answers[task.task_id].values)
                for task in accepted_tasks
            }
            if version_2_1
            else {}
        )
        task_refs = tuple(
            dict.fromkeys(
                ref
                for output in accepted_evidence.values()
                for finding in output.findings
                for ref in finding.evidence_refs
            )
        )
        integration_refs = (
            tuple(
                dict.fromkeys(ref for claim in integrated_v21.claims for ref in claim.evidence_refs)
            )
            if integrated_v21 is not None
            else ()
        )
        required_refs = tuple(dict.fromkeys((*task_refs, *integration_refs)))
        request = verification.request(
            run_id="sql-run",
            subject_artifact_id=integration_artifact.artifact_id,
            producer_agent_id=workers[0].agent_instance_id,
            schema="V21Answer" if version_2_1 else "Answer",
            required_checks=("claims_supported", "gaps_disclosed")
            if version_2_1
            else ("answer_supported",),
            evidence_refs=required_refs,
        )
        check = await self.bounded_board_work(
            2,
            "verify",
            solve,
            "Independently verify this integrated answer against the SQL data and all task "
            "constraints. "
            + (
                "For v2.1, verify each asserted claim and whether gaps are disclosed. "
                "Return status (supported/partial/inconclusive), claims_supported, "
                "gaps_disclosed, evidence_refs from your own SQL queries, and reason. "
                if version_2_1
                else "Use supported-v1: answer_supported, evidence_refs from your own SQL queries, and reason. "
                if self.reliable
                else 'Return {"values":{"answer_supported":true}} if supported, or false otherwise. '
            )
            + "Scoped task critique records: "
            + json.dumps({task.task_id: task_states.get(task.task_id) for task in plan.tasks})
            + (
                " Accepted task findings and gaps: "
                + json.dumps(
                    {
                        task_id: {
                            "findings": [finding.model_dump() for finding in output.findings],
                            "assumptions": output.assumptions,
                            "unresolved_questions": output.unresolved_questions,
                        }
                        for task_id, output in accepted_evidence.items()
                    }
                )
                if version_2_1
                else ""
            )
            + " This is your judgment, not a benchmark score. Answer: "
            + integrated.model_dump_json(),
            role="verify",
        )
        if version_2_1:
            assert integrated_v21 is not None
            assessed = V21VerificationValues.model_validate(check.values)
            checks = {
                "claims_supported": assessed.claims_supported
                and assessed.status == integrated_v21.status,
                "gaps_disclosed": assessed.gaps_disclosed
                and (
                    integrated_v21.status == "supported"
                    or bool(integrated_v21.unresolved_questions)
                ),
            }
            verdict_refs = tuple(dict.fromkeys((*required_refs, *assessed.evidence_refs)))
            findings = (assessed.reason,)
        else:
            if (not self.reliable and set(check.values) != {"answer_supported"}) or type(
                check.values["answer_supported"]
            ) is not bool:
                raise ValueError("hybrid verification must return a boolean answer_supported")
            checks = {"answer_supported": check.values["answer_supported"]}
            verdict_refs = ()
            findings = ("Independent SQL model verification of the integrated answer",)
        verdict = verification.complete(
            verification_id=request.verification_id,
            verifier=workers[2],
            checks=checks,
            findings=findings,
            evidence_refs=verdict_refs,
            limitations=("Model judgment; no private reference or grading access",),
        )
        verification.accept(
            request=request,
            verdict=verdict,
            policy_version=self.policy_version,
            downstream_uses=("submit_answer",),
        )
        if version_2_1:
            assert integrated_v21 is not None
            self.db.record_event(
                EventRecord(
                    run_id="sql-run",
                    event_type="hybrid_v2_1.answer_assessed",
                    data={
                        "status": integrated_v21.status,
                        "unresolved_questions": integrated_v21.unresolved_questions,
                        "evidence_refs": list(verdict_refs),
                    },
                )
            )
        controller.advance(execution.round_id, CollaborationPhase.VERIFIED_AND_SCORED)
        gate = CompletionGate(self.db)
        mission = self.db.get_plan("sql-run")
        assert mission is not None
        gate.require(
            gate.evaluate(
                plan=mission,
                deliverable_artifact_id=integration_artifact.artifact_id,
                required_accepted_artifacts=(integration_artifact.artifact_id,),
            )
        )
        controller.complete(execution.round_id)
        return integrated.model_copy(deep=True)


def adapter(method: str, *, json_protocol: bool = False):
    if method not in METHODS:
        raise ValueError(f"unknown SQL orchestration method: {method}")

    async def run(
        task: TaskInput,
        env: TaskEnvironment,
        model: Model | str,
        settings: ModelSettings,
        budget: Budget,
        usage: RunUsage,
    ) -> Answer:
        if (
            budget.finalization_seconds
            or budget.finalization_tokens
            or budget.finalization_requests
        ):
            raise ValueError("SQL runtime adapters do not support finalization reserves")
        worker = single_json if json_protocol else single

        worker_type = (
            ConcurrentTeamWorker if env.state.options.team_policy == "concurrent-v1" else TeamWorker
        )
        bounded_worker = worker_type(
            method, task, env, model, settings, budget, usage, json_protocol=json_protocol
        )

        async def solve(instruction: str, role: PhaseName) -> Answer:
            team.flush(env.state.emit, method)
            if env.state.options.team_policy != "legacy-v1":
                return await bounded_worker(instruction, role)
            options = env.state.options
            env.state.options = options.model_copy(update={"artifact_contract": "legacy-v1"})
            try:
                answer = await worker(
                    task.model_copy(update={"prompt": task.prompt + "\n" + instruction}),
                    env,
                    model,
                    settings,
                    budget,
                    usage,
                )
            finally:
                env.state.options = options
            if role not in {"plan", "critique", "verify", "orchestrate", "task"}:
                validate_artifact(task, answer, options.artifact_contract)
            return answer

        with TemporaryDirectory(prefix="sql-orchestration-") as directory:
            binding = env.state.options.phase_models.get("solve")
            model_name = (
                binding.model
                if binding
                else (model if isinstance(model, str) else model.model_name)
            )
            team = SQLTeam(
                Path(directory),
                method,
                task,
                budget,
                backend="sql-json" if json_protocol else "sql-native",
                model_name=model_name,
                options=env.state.options,
                env_schema=env.schema,
                usage=usage,
                context_window=getattr(model, "context_window", None),
            )
            if method == "hybrid_v2_1":
                team.task_pool_planned = bounded_worker.pool_planned
            try:
                await team.start()
                try:
                    return await team.run(method, solve)
                except (CollaborationError, DeliveryBlocked) as exc:
                    details = failure_details(
                        exc,
                        scope="phase",
                        stage=env.state.phase,
                        threshold={
                            "proposal_quorum": env.state.options.hybrid_proposal_quorum,
                            "verification_required": True,
                        },
                        consumption={"committed_candidates": len(team.submitted)},
                    )
                    # Typed attribution in precedence order; only genuinely
                    # untyped CollaborationErrors (quorum, validator, controller
                    # conditions) fall through to the quorum heuristic.
                    if isinstance(exc, DeliveryBlocked):
                        details["code"] = "verification_rejected"
                    elif isinstance(exc, OrchestratorEscalated):
                        details["code"] = "orchestrator_escalated"
                    elif isinstance(exc, PlanningRejected):
                        details["code"] = "planning_rejected"
                    elif isinstance(exc, TaskContextInfeasible):
                        details["code"] = "task_context_infeasible"
                    elif isinstance(exc, StageBudgetExhausted):
                        details["code"] = "stage_budget_exhausted"
                    elif isinstance(exc, DecisionRoundsExhausted):
                        details["code"] = "decision_rounds_exhausted"
                    else:
                        details["code"] = (
                            "proposal_quorum_not_met"
                            if len(team.submitted) < env.state.options.hybrid_proposal_quorum
                            else "no_admissible_candidate"
                        )
                    env.state.emit({"delivery_blocked": details})
                    raise
                except (
                    PhaseTimeout,
                    RequestTimeout,
                    UsageLimitExceeded,
                    ToolBudgetExceeded,
                    ModelAPIError,
                    UnexpectedModelBehavior,
                ) as exc:
                    if not (
                        method == "speculative"
                        and env.state.options.speculative_failure_policy == "return_first_submitted"
                        and team.submitted
                        and perf_counter() < env.state.started + budget.seconds
                        and usage.total_tokens <= budget.total_tokens
                        and usage.requests <= budget.requests
                    ):
                        raise
                    # First committed artifact, never the grader's best candidate. Hybrid's
                    # verification gate is not bypassed by this speculative-only policy.
                    env.state.answer_source = "first_submitted_fallback"
                    env.state.recovery = {"error": type(exc).__name__, "phase": env.state.phase}
                    env.state.emit({"recovery": env.state.recovery})
                    return team.submitted[0].model_copy(deep=True)
            finally:
                try:
                    team.flush(env.state.emit, method)
                finally:
                    team.db.close()

    return run
