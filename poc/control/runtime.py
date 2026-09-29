from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from poc.blackboard.models import EpistemicStatus, RecordType
from poc.blackboard.service import BlackboardService
from poc.control.spawn_policy import SpawnPolicy
from poc.control.supervisor_actor import SupervisorActor
from poc.execution.agent_executor import AgentExecutorRegistry, CustomPythonAgentExecutor
from poc.execution.board_claim import BoardClaimStrategy
from poc.execution.capacity import CapacityScheduler, InProcessWorkerMaterializer
from poc.execution.hierarchical_strategy import HierarchicalDAGStrategy
from poc.execution.managed_pool import ManagedPoolStrategy
from poc.execution.ooda_graph import OODAHarness
from poc.execution.pydantic_ai_executor import PydanticAIAgentExecutor, PydanticModelFactory
from poc.execution.semantic_pydantic_ai_executor import SemanticPydanticAIAgentExecutor
from poc.execution.speculative import SpeculativeStrategy
from poc.execution.strategy import ExecutionCoordinator, StrategyRegistry
from poc.execution.worker_adapter import WorkerAdapter
from poc.execution.workflow_compiler import WorkflowCompiler
from poc.execution.workflow_runner import WorkflowRunner
from poc.hybrid.capability_router import CapabilityRouter
from poc.hybrid.collaboration_controller import CollaborationController
from poc.hybrid.communication import CommunicationService
from poc.hybrid.completion_gate import CompletionGate
from poc.hybrid.context_assembler import ContextAssembler
from poc.hybrid.contracts import Candidate, CollaborationPhase, EvaluationVector
from poc.hybrid.policies import SwarmPolicyResolver
from poc.hybrid.verification_service import VerificationService
from poc.models import (
    HYBRID_STRATEGIES,
    AgentInstance,
    ApprovalRequest,
    DependencyRequirement,
    EventRecord,
    ExecutionHandle,
    ExecutionMode,
    ExecutionPolicy,
    HybridConfig,
    InputBinding,
    MissionPlan,
    PlanArea,
    RunCreate,
    RunStatus,
    SwarmStrategy,
    TaskSpec,
    Tier,
    WorkflowSpec,
    new_id,
)
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry
from poc.services.artifact_store import ArtifactStore
from poc.services.model_adapter import DeterministicModelAdapter
from poc.services.rbac_adapter import RBACAdapter
from poc.services.tool_gateway import ToolGateway


class Runtime:
    def __init__(
        self,
        data_dir: str | Path,
        fixture_root: str | Path | None = None,
        *,
        roles: RoleRegistry | None = None,
        pydantic_model_factory: PydanticModelFactory | None = None,
        semantic_pydantic_model_factory: PydanticModelFactory | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(self.data_dir / "application.sqlite")
        self.artifacts = ArtifactStore(self.data_dir / "artifacts", self.db)
        self.roles = roles or RoleRegistry()
        self.spawns = SpawnPolicy(self.db, self.roles)
        self.capacity = CapacityScheduler(self.db, InProcessWorkerMaterializer(self.spawns))
        self.strategies = StrategyRegistry(self.db)
        self.strategies.register(HierarchicalDAGStrategy.mode, HierarchicalDAGStrategy)
        self.strategies.register(BoardClaimStrategy.mode, BoardClaimStrategy)
        self.strategies.register(ManagedPoolStrategy.mode, ManagedPoolStrategy)
        self.strategies.register(SpeculativeStrategy.mode, SpeculativeStrategy)
        self.executions = ExecutionCoordinator(self.strategies)
        self.rbac = RBACAdapter(self.db, self.roles)
        fixture_root = fixture_root or Path(__file__).parents[1] / "fixtures" / "incident_example"
        self.tools = ToolGateway(self.db, self.rbac, self.artifacts, fixture_root)
        self.ooda = OODAHarness(
            self.db, self.roles, DeterministicModelAdapter(), self.tools, self.artifacts
        )
        self.agent_executors = AgentExecutorRegistry()
        self.agent_executors.register(CustomPythonAgentExecutor(self.ooda))
        self.agent_executors.register(
            PydanticAIAgentExecutor(
                self.db,
                self.roles,
                self.tools,
                self.artifacts,
                pydantic_model_factory,
            )
        )
        self.agent_executors.register(
            SemanticPydanticAIAgentExecutor(
                self.db,
                self.roles,
                self.tools,
                self.artifacts,
                pydantic_model_factory,
                semantic_model_factory=semantic_pydantic_model_factory,
            )
        )
        self.policy_resolver = SwarmPolicyResolver()
        self.context = ContextAssembler(self.db, self.artifacts, self.roles)
        self.blackboard = BlackboardService(self.db)
        self.communications = CommunicationService(self.db)
        self.capabilities = CapabilityRouter(self.db)
        self.collaboration = CollaborationController(
            self.db, self.artifacts, self.blackboard, self.communications
        )
        self.verification = VerificationService(self.db)
        self.completion_gate = CompletionGate(self.db)
        board_strategy = self.strategies.get(ExecutionMode.BOARD_CLAIM)
        if not isinstance(board_strategy, BoardClaimStrategy):
            raise TypeError("board_claim registry entry must be BoardClaimStrategy")
        self.adapter = WorkerAdapter(
            self.db,
            self.spawns,
            self.agent_executors,
            self.context,
            self.capacity,
            board_strategy,
        )
        self.compiler = WorkflowCompiler(self.adapter, self._get_agent)
        self.runner = WorkflowRunner(self.db, self.compiler, self.data_dir / "checkpoints.sqlite")
        self.actors: dict[str, SupervisorActor] = {}
        self.run_tasks: dict[str, asyncio.Task[None]] = {}

    def create_run(self, request: RunCreate) -> tuple[dict[str, Any], MissionPlan]:
        self.agent_executors.get(request.agent_runtime.backend)
        self.strategies.get(request.execution_mode)
        run_id = new_id("run")
        main = AgentInstance(
            agent_instance_id=f"main-{run_id}",
            run_id=run_id,
            parent_agent_id=None,
            tier=Tier.MAIN,
            role_id="main_orchestrator",
            role_version=1,
            plan_version=1,
        )
        plan = self._mission_plan(run_id, request)
        self.db.create_run(run_id, request.objective, main.agent_instance_id, plan)
        self.db.put_agent(main)
        self.actors[main.agent_instance_id] = SupervisorActor(main, self.db)
        return self.db.get_run(run_id) or {}, plan

    async def approve(
        self, run_id: str, request: ApprovalRequest, *, start: bool = True
    ) -> MissionPlan:
        run = self._require_run(run_id)
        if run["status"] != RunStatus.AWAITING_APPROVAL:
            raise ValueError(f"run is not awaiting approval (status={run['status']})")
        current = self.db.get_plan(run_id)
        if not current or request.plan_version != current.version:
            raise ValueError("approval must target the active immutable plan version")
        edits = request.edits
        if edits and any(value is not None for value in edits.model_dump().values()):
            approved = current.model_copy(deep=True)
            approved.version += 1
            approved.status = "approved"
            for key, value in edits.model_dump(exclude_none=True).items():
                setattr(approved, key, value)
            self.db.record_event(
                EventRecord(
                    run_id=run_id,
                    event_type="plan.edited",
                    data={"from_version": current.version, "to_version": approved.version},
                )
            )
        else:
            approved = current.model_copy(update={"status": "approved"})
        self.db.approve_plan(current, approved)
        if approved.version != 1:
            main = self._get_agent(run["main_agent_id"])
            updated_main = main.model_copy(update={"plan_version": approved.version})
            self.db.conn.execute(
                "UPDATE agents SET plan_version=?,payload=? WHERE agent_instance_id=?",
                (approved.version, updated_main.model_dump_json(), updated_main.agent_instance_id),
            )
            self.db.conn.commit()
            self.actors[updated_main.agent_instance_id] = SupervisorActor(updated_main, self.db)
        if start:
            self.run_tasks[run_id] = asyncio.create_task(self._execute_run(run_id, approved))
        return approved

    async def wait(self, run_id: str) -> None:
        task = self.run_tasks.get(run_id)
        if task:
            await task

    async def submit_execution(
        self,
        *,
        run_id: str,
        owner_suborchestrator_id: str,
        goal_ref: str,
        policy: ExecutionPolicy,
    ) -> ExecutionHandle:
        """Public extension point for an approved sub-orchestrator execution."""
        self.agent_executors.get(policy.agent_backend)
        return await self.executions.submit(
            run_id=run_id,
            owner_suborchestrator_id=owner_suborchestrator_id,
            goal_ref=goal_ref,
            policy=policy,
        )

    async def recover(self) -> list[str]:
        """Resume runs left active after a process stop using durable plans/checkpoints."""
        recovered: list[str] = []
        for run in self.db.list_runs():
            if run["status"] != RunStatus.RUNNING or run["cancellation_requested"]:
                continue
            plan = self.db.get_plan(run["run_id"])
            if not plan or plan.status != "approved":
                continue
            self.db.record_event(
                EventRecord(
                    run_id=run["run_id"],
                    event_type="recovery.started",
                    data={"plan_version": plan.version},
                )
            )
            self.run_tasks[run["run_id"]] = asyncio.create_task(
                self._execute_run(run["run_id"], plan)
            )
            recovered.append(run["run_id"])
        return recovered

    async def close(self) -> None:
        pending = [task for task in self.run_tasks.values() if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self.runner.close()
        self.db.close()

    def status(self, run_id: str) -> dict[str, Any]:
        run = self._require_run(run_id)
        plan = self.db.get_plan(run_id)
        return {
            "run": run,
            "plan": plan.model_dump(mode="json") if plan else None,
            "agents": self.db.list_agents(run_id),
            "workflows": self.db.list_workflows(run_id),
            "executions": [
                {**dict(row), "policy": json.loads(row["policy"])}
                for row in self.db.conn.execute(
                    "SELECT * FROM executions WHERE run_id=? ORDER BY created_at", (run_id,)
                )
            ],
            "artifacts": [a.model_dump(mode="json") for a in self.db.list_artifacts(run_id)],
            "events": self.db.events(run_id),
            "hybrid": self.db.hybrid_status(run_id),
        }

    async def _execute_run(self, run_id: str, plan: MissionPlan) -> None:
        try:
            main = self._get_agent(self._require_run(run_id)["main_agent_id"])
            main_actor = self._actor(main)
            assignments: list[dict[str, Any]] = []
            supervisors: dict[str, AgentInstance] = {}
            for area in plan.areas:
                child = self.spawns.spawn(
                    run_id=run_id,
                    parent=main,
                    child_role=area.owner_role,
                    plan_version=plan.version,
                    stable_key=f"{run_id}-{area.id}",
                )
                supervisors[area.id] = child
                self.actors[child.agent_instance_id] = SupervisorActor(child, self.db)
                await self.submit_execution(
                    run_id=run_id,
                    owner_suborchestrator_id=child.agent_instance_id,
                    goal_ref=area.id,
                    policy=ExecutionPolicy(
                        mode=area.execution_mode,
                        swarm_strategy=plan.swarm_strategy,
                        policy_set=plan.policy_set,
                        hybrid=plan.hybrid,
                        agent_backend=plan.agent_runtime.backend,
                        agent_provider=plan.agent_runtime.provider,
                        agent_model=plan.agent_runtime.model,
                        agent_options=plan.agent_runtime.options,
                        max_workers=plan.budgets["max_workers"],
                        allowed_roles=frozenset(
                            self.roles.get(area.owner_role).allowed_child_roles
                        ),
                        options=plan.execution_options,
                    ),
                )
                assignments.append(
                    {
                        "command_id": f"assign:{area.id}:v{plan.version}",
                        "kind": "assign_goal",
                        "sub_orchestrator": child.agent_instance_id,
                        "goal": area.goal,
                        "input_artifacts": [],
                        "acceptance_criteria": area.acceptance_criteria,
                    }
                )
            await main_actor.turn(
                {"type": "plan.approved", "plan_version": plan.version}, assignments
            )

            metrics_spec = self._metrics_workflow(
                run_id, plan, supervisors["metrics"].agent_instance_id
            )
            evidence_spec = self._evidence_workflow(
                run_id, plan, supervisors["evidence"].agent_instance_id
            )
            metrics_actor = self._actor(supervisors["metrics"])
            evidence_actor = self._actor(supervisors["evidence"])
            await asyncio.gather(
                metrics_actor.turn({"type": "goal.assigned"}, [self._submit_command(metrics_spec)]),
                evidence_actor.turn(
                    {"type": "goal.assigned"}, [self._submit_command(evidence_spec)]
                ),
            )
            metrics_result, evidence_result = await asyncio.gather(
                self._run_metrics_with_validation(metrics_spec, supervisors["metrics"], plan),
                self.runner.start(evidence_spec),
            )
            self._ensure_success(metrics_result, metrics_spec)
            self._ensure_success(evidence_result, evidence_spec)
            self.db.record_event(
                EventRecord(
                    run_id=run_id,
                    event_type="result.accepted",
                    actor_id=main.agent_instance_id,
                    data={"workflow_id": metrics_spec.workflow_id},
                )
            )
            self.db.record_event(
                EventRecord(
                    run_id=run_id,
                    event_type="result.accepted",
                    actor_id=main.agent_instance_id,
                    data={"workflow_id": evidence_spec.workflow_id},
                )
            )

            metrics = metrics_result["results"]["compare"]
            evidence = evidence_result["results"]["match_deployment"]
            source_artifacts = list(
                dict.fromkeys(metrics["evidence_artifacts"] + evidence["evidence_artifacts"])
            )
            selected_claim: dict[str, Any] | None = None
            selected_candidate_artifact_id: str | None = None
            required_accepted_artifacts: tuple[str, ...] = ()
            if plan.swarm_strategy == SwarmStrategy.HYBRID_V1:
                selected_claim, selected_candidate = await self._run_hybrid_search(
                    run_id=run_id,
                    plan=plan,
                    reporting_supervisor=supervisors["reporting"],
                    assurance_supervisor=supervisors["assurance"],
                    metrics=metrics["result"]["content"],
                    evidence=evidence["result"],
                    source_artifacts=source_artifacts,
                )
                source_artifacts.append(selected_candidate.artifact_id)
                selected_candidate_artifact_id = selected_candidate.artifact_id
                required_accepted_artifacts = (selected_candidate.artifact_id,)
            elif plan.swarm_strategy == SwarmStrategy.HYBRID_V2:
                selected_claim, selected_candidate = await self._run_orchestrated_plan(
                    run_id=run_id,
                    plan=plan,
                    reporting_supervisor=supervisors["reporting"],
                    assurance_supervisor=supervisors["assurance"],
                    metrics=metrics["result"]["content"],
                    evidence=evidence["result"],
                    source_artifacts=source_artifacts,
                )
                source_artifacts.append(selected_candidate.artifact_id)
                selected_candidate_artifact_id = selected_candidate.artifact_id
                required_accepted_artifacts = (selected_candidate.artifact_id,)
            report_spec = self._report_workflow(
                run_id,
                plan,
                supervisors["reporting"].agent_instance_id,
                metrics["result"]["content"],
                evidence["result"],
                source_artifacts,
                selected_claim,
                selected_candidate_artifact_id,
            )
            await self._actor(supervisors["reporting"]).turn(
                {"type": "dependencies.accepted"}, [self._submit_command(report_spec)]
            )
            report_result = await self.runner.start(report_spec)
            self._ensure_success(report_result, report_spec)
            final = report_result["results"]["assemble"]
            report_artifact = final["result"]["published"]["artifact_id"]
            if plan.swarm_strategy in HYBRID_STRATEGIES:
                delivery = self.completion_gate.evaluate(
                    plan=plan,
                    deliverable_artifact_id=report_artifact,
                    required_accepted_artifacts=required_accepted_artifacts,
                )
                self.completion_gate.require(delivery)
            commands = [
                {
                    "command_id": f"deliver:{report_artifact}",
                    "kind": "deliver_artifact",
                    "artifact_id": report_artifact,
                    "recipient_id": "user",
                }
            ]
            await main_actor.turn({"type": "domain_outputs.accepted"}, commands)
            self.db.record_event(
                EventRecord(
                    run_id=run_id,
                    event_type="deliverable.accepted",
                    actor_id=main.agent_instance_id,
                    data={"artifact_id": report_artifact, "evidence_artifacts": source_artifacts},
                )
            )
            self.db.update_run_status(run_id, RunStatus.COMPLETED)
        except asyncio.CancelledError:
            self.db.update_run_status(run_id, RunStatus.CANCELLED)
            raise
        except BaseException as exc:
            self.db.update_run_status(run_id, RunStatus.FAILED, str(exc))

    async def _run_metrics_with_validation(
        self, spec: WorkflowSpec, supervisor: AgentInstance, plan: MissionPlan
    ) -> dict[str, Any]:
        result = await self.runner.start(spec)
        interrupts = result.get("__interrupt__", ())
        if not interrupts:
            # Executors that do not expose LangGraph's interrupt primitive still
            # receive the same independent manifest check. This keeps the mission
            # contract backend-neutral instead of silently allowing an LLM worker
            # to assume a timezone that the deterministic OODA worker must validate.
            lookup = self._manifest_workflow(spec.run_id, plan, supervisor.agent_instance_id)
            await self._actor(supervisor).turn(
                {"type": "validation.required"}, [self._submit_command(lookup)]
            )
            lookup_result = await self.runner.start(lookup)
            self._ensure_success(lookup_result, lookup)
            self.db.record_event(
                EventRecord(
                    run_id=spec.run_id,
                    event_type="validation.completed",
                    actor_id=supervisor.agent_instance_id,
                    data={"workflow_id": lookup.workflow_id, "source": "manifest"},
                )
            )
            return result
        interrupt_item = interrupts[0]
        request = interrupt_item.value
        self.db.record_event(
            EventRecord(
                run_id=spec.run_id,
                event_type="workflow.paused",
                actor_id=supervisor.agent_instance_id,
                data={
                    "workflow_id": spec.workflow_id,
                    "interrupt_id": interrupt_item.id,
                    "request_id": request["request_id"],
                },
            )
        )
        # Yield after dispatch: other domain workflows remain runnable while this one is paused.
        await asyncio.sleep(0)
        # The supervisor does not inspect the manifest. It dispatches a supplementary one-task workflow.
        lookup = self._manifest_workflow(spec.run_id, plan, supervisor.agent_instance_id)
        await self._actor(supervisor).turn(
            {"type": "validation.requested", "request": request}, [self._submit_command(lookup)]
        )
        lookup_result = await self.runner.start(lookup)
        self._ensure_success(lookup_result, lookup)
        fact_result = lookup_result["results"]["read_manifest"]
        resolution = {
            "request_id": request["request_id"],
            "approved": True,
            "response": fact_result["result"]["fact"],
            "evidence_artifacts": fact_result["evidence_artifacts"],
        }
        await self._actor(supervisor).turn(
            {"type": "supplementary_result.completed"},
            [
                {
                    "command_id": f"resolve:{request['request_id']}",
                    "kind": "resolve_validation",
                    **resolution,
                }
            ],
        )
        self.db.record_event(
            EventRecord(
                run_id=spec.run_id,
                event_type="workflow.resumed",
                actor_id=supervisor.agent_instance_id,
                data={"workflow_id": spec.workflow_id, "request_id": request["request_id"]},
            )
        )
        return await self.runner.resume(spec, resolution)

    def _get_agent(self, agent_id: str) -> AgentInstance:
        agent = self.db.get_agent(agent_id)
        if not agent:
            raise KeyError(agent_id)
        return agent

    def _actor(self, agent: AgentInstance) -> SupervisorActor:
        return self.actors.setdefault(agent.agent_instance_id, SupervisorActor(agent, self.db))

    def _require_run(self, run_id: str) -> dict[str, Any]:
        run = self.db.get_run(run_id)
        if not run:
            raise KeyError(run_id)
        return run

    def _mission_plan(self, run_id: str, request: RunCreate) -> MissionPlan:
        return MissionPlan(
            plan_id=new_id("plan"),
            run_id=run_id,
            objective=request.objective,
            agent_runtime=request.agent_runtime,
            execution_mode=request.execution_mode,
            execution_options=request.execution_options,
            swarm_strategy=request.swarm_strategy,
            policy_set=self.policy_resolver.resolve(request.swarm_strategy),
            hybrid=request.hybrid,
            constraints=[
                "Offline fixture access only",
                "Do not modify live systems",
                "Exclude customer identifiers",
            ],
            permitted_sources=["metrics.csv", "logs.jsonl", "deployments.json", "manifest.json"],
            permitted_tools=[
                "read_metric_slice",
                "calculate_percentile",
                "read_log_slice",
                "count_log_pattern",
                "read_deployment_record",
                "read_manifest",
                "write_artifact",
            ],
            areas=[
                PlanArea(
                    id="metrics",
                    owner_role="metrics_supervisor",
                    goal="Measure regression with comparable windows",
                    execution_mode=request.execution_mode,
                    allowed_execution_modes=frozenset(ExecutionMode),
                    acceptance_criteria=[
                        "p95 values have units and evidence",
                        "timestamp ambiguity is resolved",
                    ],
                ),
                PlanArea(
                    id="evidence",
                    owner_role="evidence_supervisor",
                    goal="Correlate logs and deployment changes",
                    execution_mode=request.execution_mode,
                    allowed_execution_modes=frozenset(ExecutionMode),
                    acceptance_criteria=[
                        "misleading patterns are qualified",
                        "deployment window is explicit",
                    ],
                ),
                PlanArea(
                    id="reporting",
                    owner_role="reporting_supervisor",
                    goal="Produce an evidence-backed report",
                    execution_mode=request.execution_mode,
                    allowed_execution_modes=frozenset(ExecutionMode),
                    depends_on=["metrics", "evidence"],
                    acceptance_criteria=[
                        "claims cite accepted artifacts",
                        "follow-up tests are non-destructive",
                    ],
                ),
                *(
                    [
                        PlanArea(
                            id="assurance",
                            owner_role="assurance_supervisor",
                            goal="Independently verify candidate evidence before synthesis",
                            execution_mode=request.execution_mode,
                            allowed_execution_modes=frozenset(ExecutionMode),
                            depends_on=["metrics", "evidence"],
                            acceptance_criteria=[
                                "producer and verifier identities differ",
                                "all required evidence checks are explicit",
                            ],
                        )
                    ]
                    if request.swarm_strategy in HYBRID_STRATEGIES
                    else []
                ),
            ],
            completion_criteria=[
                "all domain outputs accepted",
                "report artifact has exact evidence lineage",
            ],
        )

    @staticmethod
    def _submit_command(spec: WorkflowSpec) -> dict[str, Any]:
        return {
            "command_id": f"submit:{spec.workflow_id}:r{spec.revision}",
            "kind": "submit_workflow",
            "workflow_spec": spec.model_dump(mode="json"),
            "authorized_worker_roles": spec.authorized_worker_roles,
            "max_workers": spec.max_workers,
            "budget": {"ooda_cycles": 3, "tool_calls": 2},
        }

    @staticmethod
    def _ensure_success(result: dict[str, Any], spec: WorkflowSpec) -> None:
        if result.get("__interrupt__"):
            raise RuntimeError(f"workflow {spec.workflow_id} remained paused")
        failed = [
            task
            for task, item in result.get("results", {}).items()
            if item["outcome"] != "succeeded"
        ]
        if failed:
            raise RuntimeError(f"workflow {spec.workflow_id} failed tasks: {failed}")

    def _metrics_workflow(self, run_id: str, plan: MissionPlan, owner: str) -> WorkflowSpec:
        return self._configured_workflow(
            WorkflowSpec(
                workflow_id=f"{run_id}-metrics",
                run_id=run_id,
                owner=owner,
                approved_plan_version=plan.version,
                authorized_worker_roles=[
                    "window_selector",
                    "percentile_calculator",
                    "metric_comparator",
                ],
                tasks=[
                    TaskSpec(
                        id="select_windows",
                        role="window_selector",
                        goal="Select comparable baseline and incident windows.",
                        static_inputs={
                            "analysis_range": {
                                "start": "2026-04-17T12:00:00Z",
                                "end": "2026-04-17T13:09:00Z",
                            },
                            "suspected_incident_start": "2026-04-17T13:00:00",
                            "comparison_window_minutes": 10,
                        },
                        output_schema="WindowSelection",
                        acceptance_criteria=["two non-overlapping windows selected"],
                    ),
                    TaskSpec(
                        id="baseline_p95",
                        role="percentile_calculator",
                        goal="Compute baseline checkout p95.",
                        depends_on=["select_windows"],
                        input_bindings=[
                            InputBinding(
                                source_task="select_windows", field="baseline", target="window"
                            )
                        ],
                        output_schema="PercentileResult",
                        acceptance_criteria=["p95 has milliseconds and sample count"],
                    ),
                    TaskSpec(
                        id="incident_p95",
                        role="percentile_calculator",
                        goal="Compute incident checkout p95.",
                        depends_on=["select_windows"],
                        input_bindings=[
                            InputBinding(
                                source_task="select_windows", field="incident", target="window"
                            )
                        ],
                        output_schema="PercentileResult",
                        acceptance_criteria=["timezone assumption is validated"],
                    ),
                    TaskSpec(
                        id="compare",
                        role="metric_comparator",
                        goal="Compare baseline and incident p95 with units.",
                        depends_on=["baseline_p95", "incident_p95"],
                        input_bindings=[
                            InputBinding(
                                source_task="baseline_p95", field="result", target="baseline"
                            ),
                            InputBinding(
                                source_task="incident_p95", field="result", target="incident"
                            ),
                        ],
                        output_schema="MetricComparison",
                        acceptance_criteria=["absolute and relative change present"],
                    ),
                ],
            ),
            plan,
        )

    def _evidence_workflow(self, run_id: str, plan: MissionPlan, owner: str) -> WorkflowSpec:
        return self._configured_workflow(
            WorkflowSpec(
                workflow_id=f"{run_id}-evidence",
                run_id=run_id,
                owner=owner,
                approved_plan_version=plan.version,
                authorized_worker_roles=[
                    "log_slice_selector",
                    "log_pattern_counter",
                    "deployment_matcher",
                ],
                tasks=[
                    TaskSpec(
                        id="select_log_slice",
                        role="log_slice_selector",
                        goal="Select baseline and incident log evidence.",
                        static_inputs={
                            "analysis_range": {
                                "start": "2026-04-17T12:00:00Z",
                                "end": "2026-04-17T13:10:00Z",
                            },
                            "suspected_incident_start": "2026-04-17T13:00:00Z",
                        },
                        output_schema="LogSlice",
                        acceptance_criteria=["bounded slice returned"],
                    ),
                    TaskSpec(
                        id="count_error_pattern",
                        role="log_pattern_counter",
                        goal="Count patterns and flag misleading correlation.",
                        depends_on=["select_log_slice"],
                        input_bindings=[
                            InputBinding(
                                source_task="select_log_slice",
                                field="log_slice",
                                target="log_slice",
                            )
                        ],
                        output_schema="PatternCount",
                        acceptance_criteria=["all patterns counted"],
                    ),
                    TaskSpec(
                        id="match_deployment",
                        role="deployment_matcher",
                        goal="Match checkout deployment to incident window.",
                        depends_on=["count_error_pattern"],
                        static_inputs={
                            "incident_start": "2026-04-17T13:00:00Z",
                            "lookback_minutes": 30,
                        },
                        input_bindings=[
                            InputBinding(
                                source_task="count_error_pattern",
                                field="pattern_counts",
                                target="pattern_counts",
                            )
                        ],
                        output_schema="DeploymentMatch",
                        acceptance_criteria=["proximity in minutes present"],
                    ),
                ],
            ),
            plan,
        )

    def _manifest_workflow(self, run_id: str, plan: MissionPlan, owner: str) -> WorkflowSpec:
        return self._configured_workflow(
            WorkflowSpec(
                workflow_id=f"{run_id}-manifest-lookup",
                run_id=run_id,
                owner=owner,
                approved_plan_version=plan.version,
                authorized_worker_roles=["manifest_reader"],
                max_workers=1,
                tasks=[
                    TaskSpec(
                        id="read_manifest",
                        role="manifest_reader",
                        goal="Read only the fixture timezone declaration.",
                        output_schema="ManifestFact",
                        acceptance_criteria=["timezone statement is explicit"],
                    ),
                ],
            ),
            plan,
        )

    async def _run_hybrid_search(
        self,
        *,
        run_id: str,
        plan: MissionPlan,
        reporting_supervisor: AgentInstance,
        assurance_supervisor: AgentInstance,
        metrics: dict[str, Any],
        evidence: dict[str, Any],
        source_artifacts: list[str],
    ) -> tuple[dict[str, Any], Candidate]:
        focuses = ["nearby_deployment", "payment_timeout", "cache_queueing"]
        focuses.extend(
            f"independent_alternative_{index}"
            for index in range(max(0, plan.hybrid.proposals_per_round - len(focuses)))
        )
        focuses = focuses[: plan.hybrid.proposals_per_round]
        proposal_workflow_id = f"{run_id}-hybrid-proposals-r1"
        critique_workflow_id = f"{run_id}-hybrid-critiques-r1"
        verification_workflow_id = f"{run_id}-hybrid-verification-r1"
        proposal_agents = [
            self._pre_spawn_worker(
                run_id=run_id,
                plan=plan,
                parent=reporting_supervisor,
                role="claim_drafter",
                workflow_id=proposal_workflow_id,
                task_id=f"propose_{index}",
            )
            for index in range(len(focuses))
        ]
        critic_agents = [
            self._pre_spawn_worker(
                run_id=run_id,
                plan=plan,
                parent=reporting_supervisor,
                role="claim_checker",
                workflow_id=critique_workflow_id,
                task_id=f"critique_{index}",
            )
            for index in range(len(focuses))
        ]
        verifier = self._pre_spawn_worker(
            run_id=run_id,
            plan=plan,
            parent=assurance_supervisor,
            role="evidence_verifier",
            workflow_id=verification_workflow_id,
            task_id="verify_selected_candidate",
        )
        round_ = self.collaboration.frame(
            run_id=run_id,
            domain_id="reporting",
            steward=reporting_supervisor,
            member_agent_ids=tuple(
                agent.agent_instance_id for agent in [*proposal_agents, *critic_agents, verifier]
            ),
            config=plan.hybrid,
            team_id=f"team:{run_id}:hybrid:r1",
            round_id=f"round:{run_id}:hybrid:r1",
        )
        proposal_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=proposal_workflow_id,
                run_id=run_id,
                owner=reporting_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["claim_drafter"],
                max_workers=plan.hybrid.proposals_per_round,
                tasks=[
                    TaskSpec(
                        id=f"propose_{index}",
                        role="claim_drafter",
                        goal="Produce one independent falsifiable incident hypothesis.",
                        static_inputs={
                            "metrics": metrics,
                            "evidence": evidence,
                            "hypothesis_focus": focus,
                        },
                        static_artifact_refs=source_artifacts,
                        output_schema="DraftClaim",
                        acceptance_criteria=[
                            "one falsifiable hypothesis",
                            "alternatives remain explicit",
                        ],
                        domain_id="reporting",
                        team_id=round_.team_id,
                        collaboration_round_id=round_.round_id,
                        artifact_visibility="sealed_to_round",
                    )
                    for index, focus in enumerate(focuses)
                ],
            ),
            plan,
        )
        await self._actor(reporting_supervisor).turn(
            {"type": "collaboration.proposals_requested"},
            [self._submit_command(proposal_spec)],
        )
        proposal_result = await self.runner.start(proposal_spec)
        self._ensure_success(proposal_result, proposal_spec)
        candidate_content: dict[str, dict[str, Any]] = {}
        for index, focus in enumerate(focuses):
            result = proposal_result["results"][f"propose_{index}"]
            content = result["result"]["content"]
            author = self._get_agent(result["agent_instance_id"])
            candidate = self.collaboration.submit_candidate(
                round_id=round_.round_id,
                author=author,
                hypothesis_key=str(content.get("hypothesis_key", focus)),
                artifact_id=result["output_artifact"],
                evidence_refs=tuple(source_artifacts),
            )
            candidate_content[candidate.candidate_id] = content
        released = self.collaboration.release(round_.round_id)
        self.collaboration.cluster_candidates(round_.round_id)

        critique_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=critique_workflow_id,
                run_id=run_id,
                owner=reporting_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["claim_checker"],
                max_workers=min(len(released), plan.hybrid.max_critics_per_candidate + 1),
                tasks=[
                    TaskSpec(
                        id=f"critique_{index}",
                        role="claim_checker",
                        goal="Challenge one candidate assumption against original evidence.",
                        static_inputs={
                            "draft": candidate_content[candidate.candidate_id],
                            "hypothesis_key": candidate.hypothesis_key,
                        },
                        static_artifact_refs=[candidate.artifact_id, *source_artifacts],
                        artifact_requirements={
                            candidate.artifact_id: DependencyRequirement.CANDIDATE_PUBLISHED
                        },
                        output_schema="CheckedClaim",
                        acceptance_criteria=["assumption and evidence check are explicit"],
                        domain_id="reporting",
                        team_id=round_.team_id,
                        collaboration_round_id=round_.round_id,
                        artifact_visibility="domain",
                    )
                    for index, candidate in enumerate(released)
                ],
            ),
            plan,
        )
        await self._actor(reporting_supervisor).turn(
            {"type": "collaboration.candidates_released"},
            [self._submit_command(critique_spec)],
        )
        critique_result = await self.runner.start(critique_spec)
        self._ensure_success(critique_result, critique_spec)
        for index, candidate in enumerate(released):
            result = critique_result["results"][f"critique_{index}"]
            content = result["result"]["content"]
            supported = bool(content["supported"])
            critic = self._get_agent(result["agent_instance_id"])
            self.collaboration.record_evaluation(
                candidate_id=candidate.candidate_id,
                critic=critic,
                critique_artifact_id=result["output_artifact"],
                passed=supported,
                findings=tuple(str(item) for item in content.get("checks", [])),
                evidence_refs=tuple(source_artifacts),
                evaluation=EvaluationVector(
                    validity=5 if supported else 1,
                    evidence=5 if supported else 1,
                    usefulness=5 if supported else 2,
                    novelty=5 if candidate.hypothesis_key == "cache_queueing" else 2,
                    constraint_satisfaction=5,
                ),
            )
        self.collaboration.advance(round_.round_id, CollaborationPhase.CRITIQUED)
        self.collaboration.advance(round_.round_id, CollaborationPhase.TESTED)
        self.collaboration.advance(round_.round_id, CollaborationPhase.RECOMBINED)
        selected = self.collaboration.select(round_.round_id)
        selected_content = candidate_content[selected.candidate_id]

        required_checks = (
            "claim_matches_original_evidence",
            "misleading_signals_qualified",
            "constraints_preserved",
        )
        verification_request = self.verification.request(
            run_id=run_id,
            subject_artifact_id=selected.artifact_id,
            producer_agent_id=selected.author_agent_id,
            schema="DraftClaim",
            required_checks=required_checks,
            evidence_refs=tuple(source_artifacts),
        )
        verification_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=verification_workflow_id,
                run_id=run_id,
                owner=assurance_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["evidence_verifier"],
                max_workers=1,
                tasks=[
                    TaskSpec(
                        id="verify_selected_candidate",
                        role="evidence_verifier",
                        goal="Independently verify the selected claim against original evidence.",
                        static_inputs={
                            "candidate": selected_content,
                            "subject_artifact_id": selected.artifact_id,
                        },
                        static_artifact_refs=[selected.artifact_id, *source_artifacts],
                        artifact_requirements={
                            selected.artifact_id: DependencyRequirement.CANDIDATE_PUBLISHED
                        },
                        output_schema="VerificationResult",
                        acceptance_criteria=list(required_checks),
                        domain_id="reporting",
                        team_id=round_.team_id,
                        collaboration_round_id=round_.round_id,
                        artifact_visibility="domain",
                    )
                ],
            ),
            plan,
        )
        await self._actor(assurance_supervisor).turn(
            {"type": "verification.requested"},
            [self._submit_command(verification_spec)],
        )
        verification_result = await self.runner.start(verification_spec)
        self._ensure_success(verification_result, verification_spec)
        verified = verification_result["results"]["verify_selected_candidate"]
        verification_content = verified["result"]["content"]
        checks = {str(key): bool(value) for key, value in verification_content["checks"].items()}
        verdict = self.verification.complete(
            verification_id=verification_request.verification_id,
            verifier=self._get_agent(verified["agent_instance_id"]),
            checks=checks,
            findings=tuple(str(item) for item in verification_content["findings"]),
            evidence_refs=tuple(source_artifacts),
        )
        acceptance = self.verification.accept(
            request=verification_request,
            verdict=verdict,
            policy_version=plan.policy_set.acceptance_policy,
            downstream_uses=("report_synthesis", "final_delivery"),
        )
        if not acceptance.accepted:
            raise RuntimeError("selected hybrid candidate did not pass independent verification")
        verifier_agent = self._get_agent(verified["agent_instance_id"])
        self.blackboard.publish(
            actor=verifier_agent,
            domain_id="reporting",
            record_type=RecordType.DECISION,
            statement=f"Candidate {selected.candidate_id} independently verified",
            supporting_artifacts=(selected.artifact_id, *source_artifacts),
            status=EpistemicStatus.SUPPORTED,
            visibility_ref="reporting",
        )
        self.collaboration.advance(round_.round_id, CollaborationPhase.VERIFIED_AND_SCORED)
        self.collaboration.complete(round_.round_id)
        return selected_content, selected

    async def _run_orchestrated_plan(
        self,
        *,
        run_id: str,
        plan: MissionPlan,
        reporting_supervisor: AgentInstance,
        assurance_supervisor: AgentInstance,
        metrics: dict[str, Any],
        evidence: dict[str, Any],
        source_artifacts: list[str],
    ) -> tuple[dict[str, Any], Candidate]:
        """hybrid_v2: LLM-planned decomposition with scoped task waves and a decision loop."""
        from poc.hybrid.planning import (
            PlannedTask,
            PlanValidationError,
            apply_decision,
            parse_decision,
            parse_plan,
            validate_decision,
        )

        fanout = plan.hybrid.proposals_per_round
        planning_round_id = f"round:{run_id}:hybrid-v2:planning:r1"
        planning_team_id = f"team:{run_id}:hybrid-v2:planning:r1"
        planning_workflow_id = f"{run_id}-hybrid-v2-plans-r1"
        planner_agents = [
            self._pre_spawn_worker(
                run_id=run_id,
                plan=plan,
                parent=reporting_supervisor,
                role="orchestrator_planner",
                workflow_id=planning_workflow_id,
                task_id=f"plan_{index}",
            )
            for index in range(fanout)
        ]
        planning = self.collaboration.frame(
            run_id=run_id,
            domain_id="reporting",
            steward=reporting_supervisor,
            member_agent_ids=tuple(agent.agent_instance_id for agent in planner_agents),
            config=plan.hybrid,
            team_id=planning_team_id,
            round_id=planning_round_id,
        )
        planning_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=planning_workflow_id,
                run_id=run_id,
                owner=reporting_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["orchestrator_planner"],
                max_workers=fanout,
                tasks=[
                    TaskSpec(
                        id=f"plan_{index}",
                        role="orchestrator_planner",
                        goal="Decompose the objective into small scoped tasks.",
                        static_inputs={
                            "objective": plan.objective,
                            "candidate_index": index,
                            "metrics": metrics,
                            "evidence": evidence,
                        },
                        static_artifact_refs=source_artifacts,
                        output_schema="OrchestratorPlan",
                        acceptance_criteria=[
                            "between two and eight scoped tasks",
                            "every task states scope and definition of done",
                        ],
                        domain_id="reporting",
                        team_id=planning.team_id,
                        collaboration_round_id=planning.round_id,
                        artifact_visibility="sealed_to_round",
                    )
                    for index in range(fanout)
                ],
            ),
            plan,
        )
        await self._actor(reporting_supervisor).turn(
            {"type": "orchestration.plans_requested"},
            [self._submit_command(planning_spec)],
        )
        planning_result = await self.runner.start(planning_spec)
        self._ensure_success(planning_result, planning_spec)
        drafts: list[tuple[str, dict[str, Any], str]] = []
        for index in range(fanout):
            plan_result = planning_result["results"][f"plan_{index}"]
            author = self._get_agent(plan_result["agent_instance_id"])
            candidate = self.collaboration.submit_candidate(
                round_id=planning.round_id,
                author=author,
                hypothesis_key=f"plan_{index}",
                artifact_id=plan_result["output_artifact"],
                evidence_refs=tuple(source_artifacts),
            )
            drafts.append(
                (
                    candidate.candidate_id,
                    plan_result["result"]["content"],
                    plan_result["output_artifact"],
                )
            )
        self.collaboration.release(planning.round_id)
        self.collaboration.cluster_candidates(planning.round_id)
        self.collaboration.advance(planning.round_id, CollaborationPhase.CRITIQUED)

        judge_workflow_id = f"{run_id}-hybrid-v2-plan-judges-r1"
        for index in range(len(drafts)):
            self._pre_spawn_worker(
                run_id=run_id,
                plan=plan,
                parent=reporting_supervisor,
                role="claim_checker",
                workflow_id=judge_workflow_id,
                task_id=f"judge_{index}",
            )
        round_ = self.db.get_collaboration_round(planning_round_id)
        assert round_ is not None
        judge_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=judge_workflow_id,
                run_id=run_id,
                owner=reporting_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["claim_checker"],
                max_workers=len(drafts),
                tasks=[
                    TaskSpec(
                        id=f"judge_{index}",
                        role="claim_checker",
                        goal="Judge one orchestration plan candidate against the objective.",
                        static_inputs={"draft": draft, "hypothesis_key": candidate_id},
                        static_artifact_refs=[artifact_id, *source_artifacts],
                        artifact_requirements={
                            artifact_id: DependencyRequirement.CANDIDATE_PUBLISHED
                        },
                        output_schema="CheckedClaim",
                        acceptance_criteria=["plan coverage and scope sharpness judged"],
                        domain_id="reporting",
                        team_id=round_.team_id,
                        collaboration_round_id=round_.round_id,
                        artifact_visibility="domain",
                    )
                    for index, (candidate_id, draft, artifact_id) in enumerate(drafts)
                ],
            ),
            plan,
        )
        await self._actor(reporting_supervisor).turn(
            {"type": "orchestration.plans_released"},
            [self._submit_command(judge_spec)],
        )
        judge_result = await self.runner.start(judge_spec)
        self._ensure_success(judge_result, judge_spec)
        judged: list[tuple[str, dict[str, Any], str, bool]] = []
        for index, (candidate_id, draft, artifact_id) in enumerate(drafts):
            judge = judge_result["results"][f"judge_{index}"]
            content = judge["result"]["content"]
            critic = self._get_agent(judge["agent_instance_id"])
            supported = bool(content["supported"])
            self.collaboration.record_evaluation(
                candidate_id=candidate_id,
                critic=critic,
                critique_artifact_id=judge["output_artifact"],
                passed=supported,
                findings=tuple(str(item) for item in content.get("checks", [])),
                evidence_refs=tuple(source_artifacts),
                evaluation=EvaluationVector(
                    validity=5 if supported else 1,
                    evidence=5 if supported else 1,
                    usefulness=5 if supported else 2,
                    novelty=3,
                    constraint_satisfaction=5,
                ),
            )
            judged.append((candidate_id, draft, artifact_id, supported))
        self.collaboration.advance(planning.round_id, CollaborationPhase.TESTED)
        self.collaboration.advance(planning.round_id, CollaborationPhase.RECOMBINED)
        selected_plan_candidate = self.collaboration.select(planning.round_id)
        selected = next(
            (item for item in judged if item[0] == selected_plan_candidate.candidate_id), None
        )
        if selected is None:
            raise RuntimeError("selected orchestration plan candidate was not found")
        _, selected_content, selected_artifact_id, _ = selected
        try:
            orchestrator_plan = parse_plan(selected_content)
        except PlanValidationError:
            repair_workflow_id = f"{run_id}-hybrid-v2-plan-repair-r1"
            repair_spec = self._configured_workflow(
                WorkflowSpec(
                    workflow_id=repair_workflow_id,
                    run_id=run_id,
                    owner=reporting_supervisor.agent_instance_id,
                    approved_plan_version=plan.version,
                    authorized_worker_roles=["orchestrator_planner"],
                    max_workers=1,
                    tasks=[
                        TaskSpec(
                            id="plan_repair",
                            role="orchestrator_planner",
                            goal="Repair the rejected orchestration plan.",
                            static_inputs={
                                "objective": plan.objective,
                                "metrics": metrics,
                                "evidence": evidence,
                            },
                            static_artifact_refs=[selected_artifact_id, *source_artifacts],
                            output_schema="OrchestratorPlan",
                            acceptance_criteria=["the repaired plan passes hard validation"],
                            domain_id="reporting",
                        )
                    ],
                ),
                plan,
            )
            await self._actor(reporting_supervisor).turn(
                {"type": "orchestration.plan_repair_requested"},
                [self._submit_command(repair_spec)],
            )
            repair_result = await self.runner.start(repair_spec)
            self._ensure_success(repair_result, repair_spec)
            repair = repair_result["results"]["plan_repair"]
            try:
                orchestrator_plan = parse_plan(repair["result"]["content"])
            except PlanValidationError as failures:
                raise RuntimeError(
                    "orchestrator plan failed hard validation after the repair attempt: "
                    + "; ".join(failures.problems)
                ) from failures
        self.db.record_event(
            EventRecord(
                run_id=run_id,
                event_type="hybrid_v2.plan_selected",
                actor_id=reporting_supervisor.agent_instance_id,
                data={
                    "plan": orchestrator_plan.model_dump(),
                    "candidate_id": selected_plan_candidate.candidate_id,
                },
            )
        )
        self.collaboration.advance(planning.round_id, CollaborationPhase.VERIFIED_AND_SCORED)
        self.collaboration.complete(planning.round_id)

        execution_workflow_id = f"{run_id}-hybrid-v2-tasks-r1"
        integration_workflow_id = f"{run_id}-hybrid-v2-integration-r1"
        drafter_agents = [
            self._pre_spawn_worker(
                run_id=run_id,
                plan=plan,
                parent=reporting_supervisor,
                role="claim_drafter",
                workflow_id=execution_workflow_id,
                task_id=f"task_{task.task_id}",
            )
            for task in orchestrator_plan.tasks
        ]
        assembler_agent = self._pre_spawn_worker(
            run_id=run_id,
            plan=plan,
            parent=reporting_supervisor,
            role="report_assembler",
            workflow_id=integration_workflow_id,
            task_id="integrate",
        )
        execution = self.collaboration.frame(
            run_id=run_id,
            domain_id="reporting",
            steward=reporting_supervisor,
            member_agent_ids=(
                reporting_supervisor.agent_instance_id,
                *(agent.agent_instance_id for agent in drafter_agents),
                assembler_agent.agent_instance_id,
            ),
            config=HybridConfig(
                proposals_per_round=max(2, len(orchestrator_plan.tasks)),
                max_collaboration_rounds=plan.hybrid.max_collaboration_rounds,
            ),
            team_id=f"team:{run_id}:hybrid-v2:execution:r1",
            round_id=f"round:{run_id}:hybrid-v2:execution:r1",
        )
        task_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=execution_workflow_id,
                run_id=run_id,
                owner=reporting_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["claim_drafter"],
                max_workers=len(orchestrator_plan.tasks),
                tasks=[
                    TaskSpec(
                        id=f"task_{task.task_id}",
                        role="claim_drafter",
                        goal=f"Execute scoped task {task.task_id} within its stated scope.",
                        static_inputs={
                            "task": {**task.model_dump(), "attempt": 1},
                            "objective": orchestrator_plan.parent_objective,
                            "metrics": metrics,
                            "evidence": evidence,
                        },
                        static_artifact_refs=source_artifacts,
                        depends_on=[f"task_{dependency}" for dependency in task.dependencies],
                        output_schema="DraftClaim",
                        acceptance_criteria=list(task.definition_of_done),
                        domain_id="reporting",
                        team_id=execution.team_id,
                        collaboration_round_id=execution.round_id,
                        artifact_visibility="sealed_to_round",
                    )
                    for task in orchestrator_plan.tasks
                ],
            ),
            plan,
        )
        await self._actor(reporting_supervisor).turn(
            {"type": "orchestration.tasks_requested"},
            [self._submit_command(task_spec)],
        )
        task_result = await self.runner.start(task_spec)
        self._ensure_success(task_result, task_spec)
        task_outputs: dict[str, dict[str, Any]] = {}
        for task in orchestrator_plan.tasks:
            produced = task_result["results"][f"task_{task.task_id}"]
            author = self._get_agent(produced["agent_instance_id"])
            candidate = self.collaboration.submit_candidate(
                round_id=execution.round_id,
                author=author,
                hypothesis_key=task.task_id,
                artifact_id=produced["output_artifact"],
                evidence_refs=tuple(source_artifacts),
            )
            task_outputs[task.task_id] = {
                "candidate_id": candidate.candidate_id,
                "content": produced["result"]["content"],
                "artifact_id": produced["output_artifact"],
            }
        self.collaboration.release(execution.round_id, minimum_proposals=1)
        self.collaboration.cluster_candidates(execution.round_id)
        self.collaboration.advance(execution.round_id, CollaborationPhase.CRITIQUED)
        task_states: dict[str, str] = {}
        task_findings: dict[str, dict[str, Any]] = {}

        async def critique_task(task: PlannedTask, attempt: int, artifact: dict[str, Any]) -> bool:
            round_ = self.db.get_collaboration_round(execution.round_id)
            assert round_ is not None
            workflow_id = f"{run_id}-hybrid-v2-critique-{attempt}-{task.task_id}"
            spec = self._configured_workflow(
                WorkflowSpec(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    owner=reporting_supervisor.agent_instance_id,
                    approved_plan_version=plan.version,
                    authorized_worker_roles=["claim_checker"],
                    max_workers=1,
                    tasks=[
                        TaskSpec(
                            id=f"critique_{task.task_id}",
                            role="claim_checker",
                            goal="Critique one scoped task output against its task contract.",
                            static_inputs={
                                "task": {**task.model_dump(), "attempt": attempt},
                                "draft": artifact["content"],
                                "hypothesis_key": task.task_id,
                                "critique_attempt": attempt,
                            },
                            static_artifact_refs=[artifact["artifact_id"], *source_artifacts],
                            artifact_requirements={
                                artifact["artifact_id"]: DependencyRequirement.CANDIDATE_PUBLISHED
                            },
                            output_schema="CheckedClaim",
                            acceptance_criteria=["scope creep and unmet criteria are explicit"],
                            domain_id="reporting",
                            team_id=round_.team_id,
                            collaboration_round_id=round_.round_id,
                            artifact_visibility="domain",
                        )
                    ],
                ),
                plan,
            )
            await self._actor(reporting_supervisor).turn(
                {"type": "orchestration.task_critique_requested"},
                [self._submit_command(spec)],
            )
            result = await self.runner.start(spec)
            self._ensure_success(result, spec)
            critique = result["results"][f"critique_{task.task_id}"]
            content = critique["result"]["content"]
            critic = self._get_agent(critique["agent_instance_id"])
            supported = bool(content["supported"])
            self.collaboration.record_evaluation(
                candidate_id=artifact["candidate_id"],
                critic=critic,
                critique_artifact_id=critique["output_artifact"],
                passed=supported,
                findings=tuple(str(item) for item in content.get("checks", [])),
                evidence_refs=tuple(source_artifacts),
                evaluation=EvaluationVector(
                    validity=5 if supported else 1,
                    evidence=5 if supported else 1,
                    usefulness=5 if supported else 2,
                    novelty=3,
                    constraint_satisfaction=5,
                ),
            )
            task_states[task.task_id] = "viable" if supported else "refuted"
            task_findings[task.task_id] = {
                "checks": list(content.get("checks", [])),
                "caveat": str(content.get("caveat", "")),
                "missing": False,
            }
            return supported

        for task in orchestrator_plan.tasks:
            await critique_task(task, 1, task_outputs[task.task_id])
        self.collaboration.advance(execution.round_id, CollaborationPhase.TESTED)

        decision_rounds = 0
        while decision_rounds < plan.hybrid.max_collaboration_rounds:
            pending = [
                task
                for task in orchestrator_plan.tasks
                if task_states.get(task.task_id) != "viable"
            ]
            if not pending:
                break
            decision_rounds += 1
            decision_workflow_id = f"{run_id}-hybrid-v2-decisions-r{decision_rounds}"
            decision_spec = self._configured_workflow(
                WorkflowSpec(
                    workflow_id=decision_workflow_id,
                    run_id=run_id,
                    owner=reporting_supervisor.agent_instance_id,
                    approved_plan_version=plan.version,
                    authorized_worker_roles=["orchestrator_planner"],
                    max_workers=len(pending),
                    tasks=[
                        TaskSpec(
                            id=f"decision_{task.task_id}",
                            role="orchestrator_planner",
                            goal="Decide the next action for one critiqued scoped task.",
                            static_inputs={
                                "decision_request": {
                                    "pending_task": task.model_dump(),
                                    "findings": task_findings.get(task.task_id, {}),
                                    "accepted_task_ids": [
                                        task_id
                                        for task_id, state in task_states.items()
                                        if state == "viable"
                                    ],
                                },
                                "objective": orchestrator_plan.parent_objective,
                                "metrics": metrics,
                                "evidence": evidence,
                            },
                            static_artifact_refs=source_artifacts,
                            output_schema="OrchestratorDecision",
                            acceptance_criteria=["decision is revise, add_task, or escalate"],
                            domain_id="reporting",
                        )
                        for task in pending
                    ],
                ),
                plan,
            )
            await self._actor(reporting_supervisor).turn(
                {"type": "orchestration.decisions_requested"},
                [self._submit_command(decision_spec)],
            )
            decision_result = await self.runner.start(decision_spec)
            self._ensure_success(decision_result, decision_spec)
            for task in pending:
                decision = parse_decision(
                    decision_result["results"][f"decision_{task.task_id}"]["result"]["content"]
                )
                validate_decision(decision, orchestrator_plan, pending_task_id=task.task_id)
                if decision.decision == "escalate":
                    self.db.record_event(
                        EventRecord(
                            run_id=run_id,
                            event_type="hybrid_v2.escalated",
                            actor_id=reporting_supervisor.agent_instance_id,
                            data={"task": task.task_id, "rationale": decision.rationale},
                        )
                    )
                    raise RuntimeError(
                        f"orchestrator escalated task {task.task_id!r}: {decision.rationale}"
                    )
                revised = (
                    decision.revision if decision.decision == "revise" else decision.added_task
                )
                assert revised is not None
                orchestrator_plan = apply_decision(orchestrator_plan, decision)
                revision_workflow_id = (
                    f"{run_id}-hybrid-v2-revision-{decision_rounds}-{revised.task_id}"
                )
                revision_agent = self._pre_spawn_worker(
                    run_id=run_id,
                    plan=plan,
                    parent=reporting_supervisor,
                    role="claim_drafter",
                    workflow_id=revision_workflow_id,
                    task_id=f"task_{revised.task_id}",
                )
                del revision_agent
                revision_spec = self._configured_workflow(
                    WorkflowSpec(
                        workflow_id=revision_workflow_id,
                        run_id=run_id,
                        owner=reporting_supervisor.agent_instance_id,
                        approved_plan_version=plan.version,
                        authorized_worker_roles=["claim_drafter"],
                        max_workers=1,
                        tasks=[
                            TaskSpec(
                                id=f"task_{revised.task_id}",
                                role="claim_drafter",
                                goal=f"Re-execute scoped task {revised.task_id} after the decision.",
                                static_inputs={
                                    "task": {
                                        **revised.model_dump(),
                                        "attempt": decision_rounds + 1,
                                    },
                                    "objective": orchestrator_plan.parent_objective,
                                    "metrics": metrics,
                                    "evidence": evidence,
                                },
                                static_artifact_refs=source_artifacts,
                                output_schema="DraftClaim",
                                acceptance_criteria=list(revised.definition_of_done),
                                domain_id="reporting",
                                team_id=execution.team_id,
                                collaboration_round_id=execution.round_id,
                                artifact_visibility="sealed_to_round",
                            )
                        ],
                    ),
                    plan,
                )
                await self._actor(reporting_supervisor).turn(
                    {"type": "orchestration.revision_requested"},
                    [self._submit_command(revision_spec)],
                )
                revision_result = await self.runner.start(revision_spec)
                self._ensure_success(revision_result, revision_spec)
                revision = revision_result["results"][f"task_{revised.task_id}"]
                parent_candidate_id = task_outputs.get(revised.task_id, {}).get("candidate_id")
                revision_candidate = self.collaboration.revise_candidate(
                    round_id=execution.round_id,
                    author=reporting_supervisor,
                    hypothesis_key=revised.task_id,
                    artifact_id=revision["output_artifact"],
                    parent_candidate_ids=(
                        (parent_candidate_id,) if parent_candidate_id is not None else ()
                    ),
                )
                task_outputs[revised.task_id] = {
                    "candidate_id": revision_candidate.candidate_id,
                    "content": revision["result"]["content"],
                    "artifact_id": revision["output_artifact"],
                }
                await critique_task(revised, decision_rounds + 1, task_outputs[revised.task_id])
        if any(task_states.get(task.task_id) != "viable" for task in orchestrator_plan.tasks):
            raise RuntimeError("hybrid_v2 decision rounds exhausted before integration")

        integration_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=integration_workflow_id,
                run_id=run_id,
                owner=reporting_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["report_assembler"],
                max_workers=1,
                tasks=[
                    TaskSpec(
                        id="integrate",
                        role="report_assembler",
                        goal="Integrate accepted scoped task outputs into the final report.",
                        static_inputs={
                            "task_outputs": [
                                task_outputs[task.task_id]["content"]
                                for task in orchestrator_plan.tasks
                            ],
                            "integration_definition_of_done": list(
                                orchestrator_plan.integration_definition_of_done
                            ),
                            "metrics": metrics,
                            "evidence": evidence,
                        },
                        static_artifact_refs=[
                            task_outputs[task.task_id]["artifact_id"]
                            for task in orchestrator_plan.tasks
                        ],
                        output_schema="FinalReport",
                        acceptance_criteria=list(orchestrator_plan.integration_definition_of_done),
                        domain_id="reporting",
                        team_id=execution.team_id,
                        collaboration_round_id=execution.round_id,
                        artifact_visibility="released_to_team",
                    )
                ],
            ),
            plan,
        )
        await self._actor(reporting_supervisor).turn(
            {"type": "orchestration.integration_requested"},
            [self._submit_command(integration_spec)],
        )
        integration_result = await self.runner.start(integration_spec)
        self._ensure_success(integration_result, integration_spec)
        integrated = integration_result["results"]["integrate"]
        integration_author = self._get_agent(integrated["agent_instance_id"])
        integration_candidate = self.collaboration.revise_candidate(
            round_id=execution.round_id,
            author=integration_author,
            hypothesis_key="integration",
            artifact_id=integrated["output_artifact"],
        )
        self.collaboration.advance(execution.round_id, CollaborationPhase.RECOMBINED)

        required_checks = (
            "claim_matches_original_evidence",
            "misleading_signals_qualified",
            "constraints_preserved",
        )
        verification_request = self.verification.request(
            run_id=run_id,
            subject_artifact_id=integration_candidate.artifact_id,
            producer_agent_id=integration_author.agent_instance_id,
            schema="FinalReport",
            required_checks=required_checks,
            evidence_refs=tuple(source_artifacts),
        )
        _verifier = self._pre_spawn_worker(
            run_id=run_id,
            plan=plan,
            parent=assurance_supervisor,
            role="evidence_verifier",
            workflow_id=f"{run_id}-hybrid-v2-verification-r1",
            task_id="verify_integration",
        )
        verification_spec = self._configured_workflow(
            WorkflowSpec(
                workflow_id=f"{run_id}-hybrid-v2-verification-r1",
                run_id=run_id,
                owner=assurance_supervisor.agent_instance_id,
                approved_plan_version=plan.version,
                authorized_worker_roles=["evidence_verifier"],
                max_workers=1,
                tasks=[
                    TaskSpec(
                        id="verify_integration",
                        role="evidence_verifier",
                        goal="Independently verify the integrated report against original evidence.",
                        static_inputs={
                            "integration": True,
                            "candidate": integrated["result"]["content"],
                            "subject_artifact_id": integration_candidate.artifact_id,
                        },
                        static_artifact_refs=[
                            integration_candidate.artifact_id,
                            *source_artifacts,
                        ],
                        artifact_requirements={
                            integration_candidate.artifact_id: (
                                DependencyRequirement.CANDIDATE_PUBLISHED
                            )
                        },
                        output_schema="VerificationResult",
                        acceptance_criteria=list(required_checks),
                        domain_id="reporting",
                    )
                ],
            ),
            plan,
        )
        await self._actor(assurance_supervisor).turn(
            {"type": "verification.requested"},
            [self._submit_command(verification_spec)],
        )
        verification_result = await self.runner.start(verification_spec)
        self._ensure_success(verification_result, verification_spec)
        verified = verification_result["results"]["verify_integration"]
        verification_content = verified["result"]["content"]
        checks = {str(key): bool(value) for key, value in verification_content["checks"].items()}
        verdict = self.verification.complete(
            verification_id=verification_request.verification_id,
            verifier=self._get_agent(verified["agent_instance_id"]),
            checks=checks,
            findings=tuple(str(item) for item in verification_content["findings"]),
            evidence_refs=tuple(source_artifacts),
        )
        acceptance = self.verification.accept(
            request=verification_request,
            verdict=verdict,
            policy_version=plan.policy_set.acceptance_policy,
            downstream_uses=("report_synthesis", "final_delivery"),
        )
        if not acceptance.accepted:
            raise RuntimeError("integrated hybrid_v2 report did not pass independent verification")
        verifier_agent = self._get_agent(verified["agent_instance_id"])
        self.collaboration.record_evaluation(
            candidate_id=integration_candidate.candidate_id,
            critic=verifier_agent,
            critique_artifact_id=verified["output_artifact"],
            passed=True,
            findings=tuple(str(item) for item in verification_content["findings"]),
            evidence_refs=tuple(source_artifacts),
            evaluation=EvaluationVector(
                validity=5,
                evidence=5,
                usefulness=5,
                novelty=3,
                constraint_satisfaction=5,
            ),
        )
        self.blackboard.publish(
            actor=verifier_agent,
            domain_id="reporting",
            record_type=RecordType.DECISION,
            statement="Integrated orchestrated report independently verified",
            supporting_artifacts=(integration_candidate.artifact_id, *source_artifacts),
            status=EpistemicStatus.SUPPORTED,
            visibility_ref="reporting",
        )
        self.collaboration.advance(execution.round_id, CollaborationPhase.VERIFIED_AND_SCORED)
        self.collaboration.complete(execution.round_id)
        content = integrated["result"]["content"]
        claim = cast(dict[str, Any], content if isinstance(content, dict) else {"report": content})
        return claim, integration_candidate

    def _pre_spawn_worker(
        self,
        *,
        run_id: str,
        plan: MissionPlan,
        parent: AgentInstance,
        role: str,
        workflow_id: str,
        task_id: str,
    ) -> AgentInstance:
        return self.spawns.spawn(
            run_id=run_id,
            parent=parent,
            child_role=role,
            plan_version=plan.version,
            stable_key=f"{workflow_id}-r1-{task_id}",
            agent_backend=plan.agent_runtime.backend,
            agent_provider=plan.agent_runtime.provider,
            agent_model=plan.agent_runtime.model,
        )

    def _report_workflow(
        self,
        run_id: str,
        plan: MissionPlan,
        owner: str,
        metrics: dict[str, Any],
        evidence: dict[str, Any],
        evidence_artifacts: list[str],
        selected_claim: dict[str, Any] | None = None,
        selected_candidate_artifact_id: str | None = None,
    ) -> WorkflowSpec:
        return self._configured_workflow(
            WorkflowSpec(
                workflow_id=f"{run_id}-report",
                run_id=run_id,
                owner=owner,
                approved_plan_version=plan.version,
                authorized_worker_roles=[
                    "claim_drafter",
                    "claim_checker",
                    "section_renderer",
                    "report_assembler",
                ],
                tasks=[
                    TaskSpec(
                        id="draft_claim",
                        role="claim_drafter",
                        goal="Draft one causal claim with alternatives.",
                        static_inputs={
                            "metrics": metrics,
                            "evidence": evidence,
                            **(
                                {"selected_candidate": selected_claim}
                                if selected_claim is not None
                                else {}
                            ),
                        },
                        static_artifact_refs=evidence_artifacts,
                        artifact_requirements=(
                            {
                                selected_candidate_artifact_id: (
                                    DependencyRequirement.ARTIFACT_VERIFIED
                                )
                            }
                            if selected_candidate_artifact_id is not None
                            else {}
                        ),
                        output_schema="DraftClaim",
                        acceptance_criteria=["claim separates evidence from inference"],
                    ),
                    TaskSpec(
                        id="check_claim",
                        role="claim_checker",
                        goal="Check the claim against exact evidence.",
                        depends_on=["draft_claim"],
                        input_bindings=[
                            InputBinding(source_task="draft_claim", field="content", target="draft")
                        ],
                        output_schema="CheckedClaim",
                        acceptance_criteria=["caveat and misleading signal addressed"],
                    ),
                    TaskSpec(
                        id="render_section",
                        role="section_renderer",
                        goal="Render findings and follow-up tests.",
                        depends_on=["check_claim"],
                        input_bindings=[
                            InputBinding(
                                source_task="check_claim", field="content", target="checked"
                            )
                        ],
                        output_schema="ReportSection",
                        acceptance_criteria=["non-destructive tests included"],
                    ),
                    TaskSpec(
                        id="assemble",
                        role="report_assembler",
                        goal="Assemble final report with lineage.",
                        depends_on=["render_section"],
                        static_inputs={"evidence_artifacts": evidence_artifacts},
                        input_bindings=[
                            InputBinding(
                                source_task="render_section", field="content", target="section"
                            )
                        ],
                        output_schema="FinalReport",
                        acceptance_criteria=["evidence artifact ids included"],
                    ),
                ],
            ),
            plan,
        )

    def _configured_workflow(self, spec: WorkflowSpec, plan: MissionPlan) -> WorkflowSpec:
        runtime = plan.agent_runtime
        tasks = [
            task.model_copy(
                update={
                    "agent_backend": runtime.backend,
                    "agent_provider": runtime.provider,
                    "agent_model": runtime.model,
                    "agent_options": runtime.options,
                }
            )
            for task in spec.tasks
        ]
        configured = spec.model_copy(
            update={
                "execution_mode": plan.execution_mode,
                "execution_options": plan.execution_options,
                "tasks": tasks,
            }
        )
        if plan.swarm_strategy in HYBRID_STRATEGIES:
            profiles = [
                self.capabilities.profile_for_role(
                    self.roles.get(role_id),
                    runtime,
                    task_classes=frozenset(self.roles.get(role_id).output_schemas),
                )
                for role_id in configured.authorized_worker_roles
            ]
            for task in configured.tasks:
                role = self.roles.get(task.role)
                self.capabilities.route(
                    run_id=configured.run_id,
                    task_id=f"{configured.workflow_id}:{task.id}",
                    task_class=task.output_schema,
                    required_tools=frozenset(role.allowed_tools),
                    profiles=profiles,
                )
        return configured
