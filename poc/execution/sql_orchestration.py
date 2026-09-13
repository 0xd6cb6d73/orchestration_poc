"""Generic SQL task bindings for the persistent orchestration components.

Policies live here, outside grading. Serial workers share usage and an environment;
concurrent workers reserve disjoint allowances and use isolated connections.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
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
from poc.execution.sql_strategy import single, single_json
from poc.execution.sql_team_worker import SupportedCritiqueValues, TeamWorker
from poc.hybrid.collaboration_controller import CollaborationController, CollaborationError
from poc.hybrid.communication import CommunicationService
from poc.hybrid.completion_gate import CompletionGate, DeliveryBlocked
from poc.hybrid.contracts import (
    CollaborationPhase,
    ContextManifest,
    EvaluationVector,
    GoalContract,
    ProvenanceRecord,
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

METHODS = (*[mode.value for mode in ExecutionMode], "hybrid_v1")
Solve = Callable[[str, PhaseName], Awaitable[Answer]]
RECOVERABLE = (
    PhaseTimeout,
    RequestTimeout,
    UsageLimitExceeded,
    ToolBudgetExceeded,
    ModelAPIError,
    UnexpectedModelBehavior,
)


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
    ):
        self.options = options or ArchitectureOptions()
        self.policy_version = self.options.team_policy
        self.reliable = self.options.team_policy in {"reliable-v2", "concurrent-v1"}
        self.concurrent = self.options.team_policy == "concurrent-v1"
        self.task = task
        self.backend = backend
        self.model_name = model_name
        self.db = Database(root / "scheduler.sqlite")
        self.emitted = 0
        self.submitted = CommittedCandidates()
        self.artifacts = ArtifactStore(root / "artifacts", self.db)
        self.mode = ExecutionMode.BOARD_CLAIM if method == "hybrid_v1" else ExecutionMode(method)
        swarm = SwarmStrategy.HYBRID_V1 if method == "hybrid_v1" else SwarmStrategy.BOARD
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
            budgets={"max_workers": 3},
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
            max_workers=3,
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
            if role not in {"plan", "critique", "verify"}:
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
            )
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
                    details["code"] = (
                        "verification_rejected"
                        if isinstance(exc, DeliveryBlocked)
                        else "proposal_quorum_not_met"
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
