from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from poc.models import WorkflowSpec


class AssignGoal(BaseModel):
    command_id: str
    kind: Literal["assign_goal"] = "assign_goal"
    sub_orchestrator: str
    goal: str
    input_artifacts: list[str]
    acceptance_criteria: list[str]


class SubmitWorkflow(BaseModel):
    command_id: str
    kind: Literal["submit_workflow"] = "submit_workflow"
    workflow_spec: WorkflowSpec
    authorized_worker_roles: list[str]
    max_workers: int
    budget: dict[str, int]


class ResolveValidation(BaseModel):
    command_id: str
    kind: Literal["resolve_validation"] = "resolve_validation"
    request_id: str
    approved: bool
    response: str
    evidence_artifacts: list[str] = Field(default_factory=list)


class DeliverArtifact(BaseModel):
    command_id: str
    kind: Literal["deliver_artifact"] = "deliver_artifact"
    artifact_id: str
    recipient_id: str


class CancelWorkflow(BaseModel):
    command_id: str
    kind: Literal["cancel_workflow"] = "cancel_workflow"
    workflow_id: str
    reason: str


class SupervisorDecision(BaseModel):
    summary: str
    commands: list[dict[str, Any]] = Field(default_factory=list)
    waiting_for: list[str] = Field(default_factory=list)

