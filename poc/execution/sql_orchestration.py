"""Generic SQL task bindings for the persistent orchestration components.

Policies live here, outside grading. Workers share one environment, usage counter and
absolute deadline. Candidate generation is serial to preserve strict shared accounting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
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
from poc.execution.sql_contracts import Answer, Budget, TaskInput
from poc.execution.sql_ports import TaskEnvironment
from poc.execution.sql_strategy import single, single_json
from poc.hybrid.collaboration_controller import CollaborationController
from poc.hybrid.communication import CommunicationService
from poc.hybrid.completion_gate import CompletionGate
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
Solve = Callable[[str], Awaitable[Answer]]


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
    ):
        self.task = task
        self.backend = backend
        self.model_name = model_name
        self.db = Database(root / "scheduler.sqlite")
        self.emitted = 0
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
            emit({"orchestration": {"method": method, "policy": "sql-team-v1", **event}})
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
    ) -> Answer:
        board = cast(BoardClaimStrategy, self.strategy)
        worker = self.workers[index]
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
            answer = await solve(prompt)
            board.complete(claim, result_ref=answer.model_dump_json())
            return answer
        except BaseException:
            board.release(claim, reason="SQL worker interrupted")
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

        if method == "hierarchical_dag":

            async def plan_node(state: dict[str, Any]) -> dict[str, Any]:
                return {"plan": await solve(plan_prompt)}

            async def solve_node(state: dict[str, Any]) -> dict[str, Any]:
                return {"answer": await solve(final_prompt(state["plan"]))}

            graph = cast(Any, StateGraph(cast(Any, dict)))
            graph.add_node("plan", plan_node)
            graph.add_node("solve", solve_node)
            graph.add_edge(START, "plan")
            graph.add_edge("plan", "solve")
            graph.add_edge("solve", END)
            result = await graph.compile().ainvoke({})
            return cast(Answer, result["answer"])
        if method == "board_claim":
            plan = await self.board_work(0, "plan", solve, plan_prompt)
            return await self.board_work(1, "solve", solve, final_prompt(plan), ("plan",))
        if method == "managed_pool":
            pool = cast(ManagedPoolStrategy, self.strategy)
            slots = [
                pool.start_slot(
                    execution_id=self.handle.execution_id, worker_id=w.agent_instance_id
                )
                for w in self.workers[:2]
            ]

            async def assigned(task_id: str, prompt: str) -> Answer:
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
                answer = await solve(prompt)
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
            for index, grant in enumerate(grants):
                answer = await solve(
                    f"Independently solve the complete task. You are candidate {index + 1}."
                )
                candidates.append(answer)
                speculative.complete_candidate(grant, result_ref=answer.model_dump_json())
            answer = await solve(
                "Reconcile these independent candidate answers. Inspect SQL as needed "
                "and return your complete final answer: "
                + json.dumps([a.model_dump() for a in candidates])
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
        for index, worker in enumerate(self.workers[:2]):
            answer = await self.board_work(
                index,
                f"proposal-{index}",
                solve,
                f"Independently solve the complete task. You are sealed proposer {index + 1}.",
            )
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
                context_policy_version="sql-team-v1",
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
                    model=self.model_name,
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
            answers[candidate.candidate_id] = answer
        candidates = controller.release(round_.round_id)
        controller.cluster_candidates(round_.round_id)
        for index, candidate in enumerate(candidates):
            critique = await self.board_work(
                2,
                f"critique-{index}",
                solve,
                'Critique this proposed answer using SQL. Return {"values":{"validity":0,'
                '"evidence":0,"usefulness":0,"novelty":0,"constraint_satisfaction":0}} '
                "with each score an integer 0..5 (5 strongest). These are your judgments, "
                "not benchmark feedback. Candidate: "
                + answers[candidate.candidate_id].model_dump_json(),
            )
            evaluation = EvaluationVector.model_validate(critique.values)
            artifact = self.artifacts.write(
                "sql-run",
                {"candidate": candidate.candidate_id, "evaluation": evaluation.model_dump()},
                producer_task_id=f"critique-{index}",
            )
            controller.record_evaluation(
                candidate_id=candidate.candidate_id,
                critic=self.workers[2],
                critique_artifact_id=artifact.artifact_id,
                passed=evaluation.validity > 0,
                findings=("SQL model critique",),
                evidence_refs=(),
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
            'Return {"values":{"answer_supported":true}} if supported, or false otherwise. '
            "This is your judgment, not a benchmark score. Answer: "
            + answers[selected.candidate_id].model_dump_json(),
        )
        if (
            set(check.values) != {"answer_supported"}
            or type(check.values["answer_supported"]) is not bool
        ):
            raise ValueError("hybrid verification must return a boolean answer_supported")
        verdict = verification.complete(
            verification_id=request.verification_id,
            verifier=self.workers[2],
            checks=check.values,
            findings=("Independent SQL model verification",),
            evidence_refs=(),
            limitations=("Model judgment; no private reference or grading access",),
        )
        verification.accept(
            request=request,
            verdict=verdict,
            policy_version="sql-team-v1",
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
        return answers[selected.candidate_id]


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

        async def solve(instruction: str) -> Answer:
            team.flush(env.state.emit, method)
            return await worker(
                task.model_copy(update={"prompt": task.prompt + "\n" + instruction}),
                env,
                model,
                settings,
                budget,
                usage,
            )

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
            )
            try:
                await team.start()
                return await team.run(method, solve)
            finally:
                try:
                    team.flush(env.state.emit, method)
                finally:
                    team.db.close()

    return run
