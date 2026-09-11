from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, UsageLimits

from poc.execution.agent_executor import AgentExecutionRequest
from poc.execution.pydantic_ai_executor import (
    PydanticAgentDependencies,
    PydanticAIAgentExecutor,
    PydanticModelFactory,
)
from poc.models import AgentBackend, RoleSpec


class SemanticValidationError(ValueError):
    """Raised when a typed worker result does not satisfy its semantic contract."""

    def __init__(self, issues: list[str], *, retryable: bool = True) -> None:
        self.issues = issues
        self.retryable = retryable
        super().__init__(_failure_message(issues))


class SemanticCriterionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str = Field(min_length=1)
    status: Literal["satisfied", "violated", "not_established"]
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class SemanticValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internally_consistent: bool
    criteria: list[SemanticCriterionAssessment]
    contradictions: list[str] = Field(default_factory=list)


_SCHEMA_CRITERIA: dict[str, tuple[str, ...]] = {
    "DraftClaim": (
        "The claim is a falsifiable explanation of the assigned hypothesis focus, not merely "
        "a restatement of an observed symptom.",
        "The confidence and alternatives distinguish correlation from established causation.",
    ),
    "CheckedClaim": (
        "The supported verdict, evidence checks, caveat, and completion summary are mutually "
        "consistent.",
        "The verdict distinguishes evidence that contradicts the claim from evidence that is "
        "merely insufficient to establish causation.",
    ),
    "VerificationResult": (
        "The verification verdict follows from the individual required checks and findings.",
    ),
    "ReportSection": ("Material claims are qualified to the strength of the supplied evidence.",),
    "FinalReport": ("Material claims are qualified to the strength of the supplied evidence.",),
}


class SemanticPydanticAIAgentExecutor(PydanticAIAgentExecutor):
    """Pydantic AI executor with deterministic and model-based semantic validation."""

    backend: str = AgentBackend.SEMANTIC_PYDANTIC_AI

    def __init__(
        self,
        *args: Any,
        semantic_model_factory: PydanticModelFactory | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.semantic_model_factory = semantic_model_factory or self.model_factory

    def _validate_semantics(
        self,
        role: RoleSpec,
        request: AgentExecutionRequest,
        result: dict[str, Any],
        completion_summary: str,
        dependencies: PydanticAgentDependencies,
    ) -> list[dict[str, Any]]:
        protocol_issues = _protocol_issues(request, result, dependencies)
        if protocol_issues:
            self._record_semantic_result(
                request,
                role,
                valid=False,
                stage="deterministic",
                issues=protocol_issues,
            )
            raise SemanticValidationError(protocol_issues)

        criteria = list(
            dict.fromkeys(
                [
                    *request.task.acceptance_criteria,
                    *_SCHEMA_CRITERIA.get(request.task.output_schema, ()),
                ]
            )
        )
        try:
            report, usage = self._run_semantic_review(
                role,
                request,
                result,
                completion_summary,
                criteria,
            )
        except Exception as exc:
            issue = f"semantic reviewer failed: {exc}"
            self._record_semantic_result(
                request,
                role,
                valid=False,
                stage="model_review",
                issues=[issue],
            )
            raise SemanticValidationError([issue], retryable=False) from exc

        issues = _review_issues(criteria, report)
        valid = not issues
        self._record_semantic_result(
            request,
            role,
            valid=valid,
            stage="model_review",
            issues=issues,
            report=report,
            usage=usage,
        )
        if issues:
            raise SemanticValidationError(issues)
        assessments = {assessment.criterion: assessment for assessment in report.criteria}
        return [
            {
                "criterion": criterion,
                "passed": True,
                "semantic_status": assessments[criterion].status,
                "evidence_refs": assessments[criterion].evidence_refs,
                "rationale": assessments[criterion].rationale,
            }
            for criterion in criteria
        ]

    def _max_output_attempts(self, role: RoleSpec) -> int:
        return 1 + role.execution_limits.get("max_semantic_revisions", 1)

    def _output_revision_feedback(
        self,
        role: RoleSpec,
        request: AgentExecutionRequest,
        error: Exception,
        *,
        output_attempt: int,
        max_output_attempts: int,
    ) -> str | None:
        del role
        if (
            not isinstance(error, SemanticValidationError)
            or not error.retryable
            or output_attempt >= max_output_attempts
        ):
            return None
        return json.dumps(
            {
                "instruction": (
                    "Your previous typed output was denied by semantic validation. Reassess the "
                    "original task and evidence, correct every issue below, and return a complete "
                    "replacement output. Do not merely explain the corrections. Preserve the "
                    "assigned task and hypothesis identities. If the result has a published field "
                    "and write_artifact is still available, publish the corrected content and use "
                    "the exact tool response."
                ),
                "task_id": request.task.id,
                "output_schema": request.task.output_schema,
                "semantic_validation_issues": error.issues,
                "revision": output_attempt,
            },
            sort_keys=True,
        )

    def _run_semantic_review(
        self,
        role: RoleSpec,
        request: AgentExecutionRequest,
        result: dict[str, Any],
        completion_summary: str,
        criteria: list[str],
    ) -> tuple[SemanticValidationReport, dict[str, int]]:
        reviewer_role = role.model_copy(
            update={
                "role_id": f"{role.role_id}:semantic_validator",
                "system_prompt": "Independently validate one completed worker result.",
                "allowed_tools": [],
            }
        )
        reviewer = Agent(
            self.semantic_model_factory(reviewer_role),
            output_type=SemanticValidationReport,
            instructions=(
                "Evaluate the worker result against every supplied criterion and the original "
                "inputs. Do not redo the task and do not infer facts absent from the supplied "
                "evidence. Return exactly one assessment for each criterion using its exact text. "
                "Use `violated` when the result contradicts a criterion, and `not_established` "
                "when the available evidence cannot establish it. Mark internally_consistent false "
                "when structured fields, explanations, or the completion summary conflict. Cite "
                "only artifact identifiers present in the input."
            ),
            model_settings=cast(Any, role.provider_options or None),
        )
        prompt = json.dumps(
            {
                "goal": request.task.goal,
                "output_schema": request.task.output_schema,
                "criteria": criteria,
                "inputs": request.inputs,
                "input_artifacts": request.input_artifacts,
                "worker_result": result,
                "completion_summary": completion_summary,
            },
            sort_keys=True,
            default=str,
        )

        def run() -> tuple[SemanticValidationReport, dict[str, int]]:
            reviewed = reviewer.run_sync(
                prompt,
                infer_name=False,
                usage_limits=UsageLimits(request_limit=3, tool_calls_limit=0),
            )
            usage = reviewed.usage
            return reviewed.output, {
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
            }

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return run()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-validator") as pool:
            return pool.submit(run).result()

    def _record_semantic_result(
        self,
        request: AgentExecutionRequest,
        role: RoleSpec,
        *,
        valid: bool,
        stage: str,
        issues: list[str],
        report: SemanticValidationReport | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "stage": stage,
            "valid": valid,
            "validator_provider": role.provider,
            "validator_model": role.model,
            "issues": issues,
        }
        if report is not None:
            data["internally_consistent"] = report.internally_consistent
            data["criteria"] = [item.model_dump(mode="json") for item in report.criteria]
            data["contradictions"] = report.contradictions
        if usage is not None:
            data["usage"] = usage
        self._record(request, "agent.semantic_validation_completed", data)


def _protocol_issues(
    request: AgentExecutionRequest,
    result: dict[str, Any],
    dependencies: PydanticAgentDependencies,
) -> list[str]:
    issues: list[str] = []
    schema = request.task.output_schema

    if schema == "DraftClaim":
        expected = request.inputs.get("hypothesis_focus")
        actual = result["content"]["hypothesis_key"]
        if isinstance(expected, str) and actual != expected:
            issues.append(f"hypothesis_key {actual!r} does not match assigned focus {expected!r}")
    elif schema == "CheckedClaim":
        expected_key = request.inputs.get("hypothesis_key")
        actual_content = result["content"]
        if isinstance(expected_key, str) and actual_content["hypothesis_key"] != expected_key:
            issues.append("checked hypothesis_key does not match the assigned candidate")
        draft = request.inputs.get("draft")
        if isinstance(draft, dict) and actual_content["claim"] != cast(dict[str, Any], draft).get(
            "claim"
        ):
            issues.append("checked claim does not preserve the candidate claim")
    elif schema == "VerificationResult":
        expected_subject = request.inputs.get("subject_artifact_id")
        actual_subject = result["content"]["subject_artifact_id"]
        if isinstance(expected_subject, str) and actual_subject != expected_subject:
            issues.append("verification subject does not match the assigned artifact")

    if schema == "MetricComparison":
        content = result["content"]
        baseline = content["baseline_p95_ms"]
        incident = content["incident_p95_ms"]
        if not math.isclose(content["absolute_increase_ms"], incident - baseline, abs_tol=0.01):
            issues.append("absolute latency increase does not match the supplied values")
        if baseline == 0 or not math.isclose(content["ratio"], incident / baseline, abs_tol=0.01):
            issues.append("latency ratio does not match the supplied values")
    elif schema == "LogSlice":
        log_slice = result["log_slice"]
        if log_slice["count"] != len(log_slice["rows"]):
            issues.append("log slice count does not match its rows")
    elif schema == "PatternCount":
        pattern_counts = result["pattern_counts"]
        if pattern_counts["total"] != sum(pattern_counts["counts"].values()):
            issues.append("pattern total does not match the individual counts")
    elif schema == "DeploymentMatch":
        deployment_match = result["deployment_match"]
        if deployment_match["count"] != len(deployment_match["matches"]):
            issues.append("deployment count does not match its records")

    published = result.get("published")
    if isinstance(published, dict):
        publication_calls = [
            item for item in dependencies.tool_results if item["tool_name"] == "write_artifact"
        ]
        if not publication_calls:
            issues.append("result declares a publication without calling write_artifact")
        else:
            latest_publication = publication_calls[-1]
            if published != latest_publication["result"]:
                issues.append("published metadata does not match the write_artifact response")
            published_content = cast(dict[str, Any], latest_publication["arguments"]).get("content")
            if published_content != result.get("content"):
                issues.append("published artifact does not contain the returned result content")
    return issues


def _review_issues(expected_criteria: list[str], report: SemanticValidationReport) -> list[str]:
    issues: list[str] = []
    returned = [assessment.criterion for assessment in report.criteria]
    if Counter(returned) != Counter(expected_criteria):
        issues.append("semantic reviewer did not return exactly one assessment per criterion")
    if not report.internally_consistent:
        issues.append("worker result is internally inconsistent")
    issues.extend(f"contradiction: {item}" for item in report.contradictions)
    expected = set(expected_criteria)
    for assessment in report.criteria:
        if assessment.criterion in expected and assessment.status != "satisfied":
            issues.append(
                f"criterion {assessment.criterion!r} is {assessment.status}: {assessment.rationale}"
            )
    return issues


def _failure_message(issues: list[str]) -> str:
    return "semantic validation failed: " + "; ".join(issues)
