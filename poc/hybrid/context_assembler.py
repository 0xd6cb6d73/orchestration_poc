from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from poc.hybrid.contracts import ContextManifest, GoalContract, ProvenanceRecord
from poc.models import (
    AgentInstance,
    DependencyRequirement,
    MessageEnvelope,
    MissionPlan,
    TaskSpec,
)
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry
from poc.services.artifact_store import ArtifactStore


@dataclass(frozen=True)
class AssembledContext:
    inputs: dict[str, Any]
    input_artifacts: list[str]
    goal_contract: GoalContract
    manifest: ContextManifest


class DependencyNotReady(RuntimeError):
    pass


class ContextAssembler:
    """Builds the bounded, reconstructable context shared by every executor."""

    def __init__(self, db: Database, artifacts: ArtifactStore, roles: RoleRegistry):
        self.db = db
        self.artifacts = artifacts
        self.roles = roles

    def assemble(
        self,
        *,
        workflow_id: str,
        workflow_revision: int,
        task: TaskSpec,
        plan: MissionPlan,
        agent: AgentInstance,
        attempt_id: str,
        inputs: dict[str, Any],
        input_artifacts: list[str],
    ) -> AssembledContext:
        contract = GoalContract(
            goal_contract_id=f"goal:{workflow_id}:r{workflow_revision}:{task.id}",
            run_id=agent.run_id,
            plan_version=plan.version,
            parent_objective=plan.objective,
            local_scope=task.goal,
            global_invariants=tuple(plan.constraints),
            input_artifact_refs=tuple(input_artifacts),
            output_schema=task.output_schema,
            evidence_requirements=tuple(task.acceptance_criteria),
            definition_of_done=tuple(task.acceptance_criteria),
            budgets={
                "ooda_cycles": plan.budgets.get("ooda_cycles", 3),
                "tool_calls": plan.budgets.get("tool_calls", 2),
            },
            permissions=tuple(self.roles.get(agent.role_id).allowed_tools),
        )
        self.db.put_goal_contract(contract)
        included: dict[str, Any] = {}
        retrieved: list[str] = []
        decisions: list[str] = []
        for artifact_id in dict.fromkeys(input_artifacts):
            requirement = task.artifact_requirements.get(artifact_id)
            if requirement is not None:
                self._require_dependency(agent.run_id, artifact_id, requirement)
            try:
                record, content = self.artifacts.read_for(
                    agent,
                    artifact_id,
                    attempt_id=attempt_id,
                    round_id=task.collaboration_round_id,
                    team_id=task.team_id,
                    domain_id=task.domain_id,
                )
            except (KeyError, PermissionError) as exc:
                decisions.append(f"excluded {artifact_id}: {exc}")
                continue
            retrieved.append(artifact_id)
            excerpt = content[:4096]
            if record.media_type == "application/json":
                try:
                    included[artifact_id] = json.loads(excerpt)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    included[artifact_id] = excerpt.decode(errors="replace")
            else:
                included[artifact_id] = excerpt.decode(errors="replace")
            decisions.append(f"included {artifact_id}: first {len(excerpt)} bytes")
        included_messages = self._permitted_messages(agent, task)
        manifest = ContextManifest(
            context_manifest_id=f"context:{attempt_id}",
            run_id=agent.run_id,
            plan_version=plan.version,
            agent_instance_id=agent.agent_instance_id,
            task_id=task.id,
            attempt_id=attempt_id,
            context_policy_version=plan.policy_set.context_policy,
            goal_contract_id=contract.goal_contract_id,
            available_artifact_refs=tuple(input_artifacts),
            retrieved_artifact_refs=tuple(retrieved),
            included_artifact_refs=tuple(included),
            included_message_ids=tuple(message.message_id for message in included_messages),
            retrieval_decisions=tuple(decisions),
        )
        self.db.put_context_manifest(manifest)
        assembled_inputs = {
            **inputs,
            "_delegation": {
                "goal_contract": contract.model_dump(mode="json"),
                "artifact_excerpts": included,
                "messages": [
                    {
                        "message_id": message.message_id,
                        "message_type": message.message_type,
                        "sender_id": message.sender_id,
                        "payload": message.payload,
                        "payload_ref": message.payload_ref,
                        "trust": "untrusted_collaboration_input",
                    }
                    for message in included_messages
                ],
                "context_manifest_id": manifest.context_manifest_id,
            },
        }
        return AssembledContext(assembled_inputs, retrieved, contract, manifest)

    def _permitted_messages(self, agent: AgentInstance, task: TaskSpec) -> list[MessageEnvelope]:
        if task.team_id is None:
            return []
        team = self.db.get_team(task.team_id)
        if team is None or agent.agent_instance_id not in team.member_agent_ids:
            return []
        messages: list[MessageEnvelope] = []
        for row in self.db.conn.execute(
            "SELECT payload FROM messages WHERE run_id=? AND recipient_id=? "
            "AND delivered_at IS NOT NULL ORDER BY rowid LIMIT 20",
            (agent.run_id, agent.agent_instance_id),
        ):
            message = MessageEnvelope.model_validate_json(row[0])
            if (
                message.payload.get("team_id") == task.team_id
                and message.message_type in team.allowed_message_types
                and message.plan_version == agent.plan_version
            ):
                messages.append(message)
        return messages

    def _require_dependency(
        self,
        run_id: str,
        artifact_id: str,
        requirement: DependencyRequirement,
    ) -> None:
        if requirement == DependencyRequirement.CANDIDATE_PUBLISHED:
            ready = (
                self.db.get_artifact(artifact_id) is not None
                and self.db.get_provenance(artifact_id) is not None
            )
        elif requirement == DependencyRequirement.ARTIFACT_VERIFIED:
            ready = self.db.accepted_artifact(run_id, artifact_id)
        else:
            ready = self.db.delivery_permitted(run_id, artifact_id)
        if not ready:
            raise DependencyNotReady(
                f"artifact {artifact_id!r} does not satisfy {requirement.value!r}"
            )

    def record_provenance(
        self,
        *,
        artifact_id: str,
        task: TaskSpec,
        agent: AgentInstance,
        attempt_id: str,
        input_artifacts: list[str],
        manifest: ContextManifest,
    ) -> ProvenanceRecord:
        operation_ids = tuple(
            str(row[0])
            for row in self.db.conn.execute(
                "SELECT operation_id FROM operations WHERE run_id=? AND operation_id LIKE ? "
                "AND status='succeeded' ORDER BY started_at",
                (agent.run_id, f"{attempt_id}:%"),
            )
        )
        validation_refs = tuple(
            str(row[0])
            for row in self.db.conn.execute(
                "SELECT request_id FROM validations WHERE run_id=? AND task_id=? "
                "AND status='resolved' ORDER BY rowid",
                (agent.run_id, task.id),
            )
        )
        role = self.roles.get(agent.role_id)
        record = ProvenanceRecord(
            run_id=agent.run_id,
            artifact_id=artifact_id,
            producer_agent_id=agent.agent_instance_id,
            task_id=task.id,
            attempt_id=attempt_id,
            input_artifact_refs=tuple(input_artifacts),
            tool_operation_ids=operation_ids,
            validation_refs=validation_refs,
            role_id=agent.role_id,
            role_version=agent.role_version,
            prompt_fingerprint=hashlib.sha256(role.system_prompt.encode()).hexdigest(),
            agent_backend=agent.agent_backend,
            provider=agent.agent_provider,
            model=agent.agent_model,
            context_manifest_id=manifest.context_manifest_id,
        )
        self.db.put_provenance(record)
        return record
