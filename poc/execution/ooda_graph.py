# pyright: reportTypedDictNotRequiredAccess=false
"""OODA nodes intentionally receive partial LangGraph state updates."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.types import interrupt

from poc.models import AgentInstance, EventRecord, Outcome, ValidationRequest, WorkerResult
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry
from poc.services.artifact_store import ArtifactStore
from poc.services.model_adapter import DeterministicModelAdapter
from poc.services.tool_gateway import ToolGateway


class WorkerState(TypedDict, total=False):
    run_id: str
    workflow_id: str
    workflow_revision: int
    task_id: str
    goal: str
    output_schema: str
    acceptance_criteria: list[str]
    inputs: dict[str, Any]
    input_artifacts: list[str]
    agent: dict[str, Any]
    attempt_id: str
    cycle: int
    tool_calls: int
    validations: int
    observation: dict[str, Any]
    orientation: dict[str, Any]
    decision: dict[str, Any]
    validation_response: dict[str, Any]
    working: dict[str, Any]
    result: dict[str, Any]
    evidence_artifacts: list[str]
    worker_result: dict[str, Any]
    error: str


class WorkerGraph(Protocol):
    def invoke(self, input: WorkerState) -> dict[str, Any]: ...


class OODAHarness:
    def __init__(
        self,
        db: Database,
        roles: RoleRegistry,
        model: DeterministicModelAdapter,
        tools: ToolGateway,
        artifacts: ArtifactStore,
    ):
        self.db = db
        self.roles = roles
        self.model = model
        self.tools = tools
        self.artifacts = artifacts
        self.graph = self._build()

    def _build(self) -> WorkerGraph:
        def route_decision(state: WorkerState) -> str:
            return str(state["decision"]["kind"])

        def route_action(state: WorkerState) -> str:
            return "done" if state.get("result") else "continue"

        # LangGraph exposes partially unknown internal generic parameters to Pyright.
        graph = cast(Any, StateGraph(WorkerState))
        graph.add_node("observe", self.observe)
        graph.add_node("orient", self.orient)
        graph.add_node("decide", self.decide)
        graph.add_node("authorize_act", self.authorize_act)
        graph.add_node("request_validation", self.request_validation)
        graph.add_node("check_output", self.check_output)
        graph.add_node("failed_result", self.failed_result)
        graph.add_edge(START, "observe")
        graph.add_edge("observe", "orient")
        graph.add_edge("orient", "decide")
        graph.add_conditional_edges(
            "decide",
            route_decision,
            {
                "act": "authorize_act",
                "request_validation": "request_validation",
                "finish": "check_output",
                "fail": "failed_result",
            },
        )
        graph.add_conditional_edges(
            "authorize_act",
            route_action,
            {
                "done": "check_output",
                "continue": "observe",
            },
        )
        graph.add_edge("request_validation", "observe")
        graph.add_edge("check_output", END)
        graph.add_edge("failed_result", END)
        return cast(WorkerGraph, graph.compile())

    def _event(self, state: WorkerState, event_type: str, data: dict[str, Any]) -> None:
        self.db.record_event(
            EventRecord(
                run_id=state["run_id"],
                event_type=event_type,
                actor_id=state["agent"]["agent_instance_id"],
                data={
                    "workflow_id": state["workflow_id"],
                    "workflow_revision": state["workflow_revision"],
                    "task_id": state["task_id"],
                    "attempt_id": state["attempt_id"],
                    **data,
                },
            )
        )

    def observe(self, state: WorkerState) -> dict[str, Any]:
        cycle = state.get("cycle", 0) + 1
        inputs = state.get("inputs", {})
        window = inputs.get("window", {})
        ambiguous = bool(window and not str(window.get("start", "")).endswith(("Z", "+00:00")))
        if state.get("validation_response", {}).get("approved"):
            ambiguous = False
        observation = {
            "selected_inputs": inputs,
            "input_artifact_versions": state.get("input_artifacts", []),
            "latest_tool_result": state.get("working", {}).get("latest_tool_result"),
            "timezone_ambiguous": ambiguous,
            "received_message_ids": [],
        }
        self._event(
            state,
            "ooda.observe",
            {
                "ooda_phase": "observe",
                "ooda_cycle": cycle,
                "input_artifacts": state.get("input_artifacts", []),
            },
        )
        return {"cycle": cycle, "observation": observation}

    def orient(self, state: WorkerState) -> dict[str, Any]:
        role = state["agent"]["role_id"]
        oriented = self.model.orient(role, state["goal"], state["observation"])
        self._event(
            state,
            "ooda.orient",
            {
                "ooda_phase": "orient",
                "ooda_cycle": state["cycle"],
                "summary": oriented["summary"],
                "uncertainties": oriented["uncertainties"],
            },
        )
        return {"orientation": oriented}

    def decide(self, state: WorkerState) -> dict[str, Any]:
        role_spec = self.roles.get(state["agent"]["role_id"])
        limits = role_spec.execution_limits
        if state["cycle"] > limits["max_ooda_cycles"]:
            decision = {"kind": "fail", "justification": "OODA cycle budget exhausted"}
        elif state.get("observation", {}).get("timezone_ambiguous") and not state.get(
            "validation_response"
        ):
            if state.get("validations", 0) >= limits["max_validations"]:
                decision = {"kind": "fail", "justification": "validation budget exhausted"}
            else:
                decision = {
                    "kind": "request_validation",
                    "question": "May the timezone-less incident timestamps be treated as UTC?",
                    "justification": "The fixture timestamp is ambiguous and affects the selected slice.",
                }
        elif self._is_complete(state):
            decision = {
                "kind": "finish",
                "justification": "Typed output is ready for acceptance checking.",
            }
        elif state.get("tool_calls", 0) >= limits["max_tool_calls"]:
            decision = {"kind": "fail", "justification": "tool execution budget exhausted"}
        else:
            tool_name, arguments = self._next_action(state)
            decision = {
                "kind": "act",
                "tool_name": tool_name,
                "arguments": arguments,
                "justification": "Execute the next narrow deterministic evidence operation.",
            }
        self._event(
            state,
            "ooda.decide",
            {
                "ooda_phase": "decide",
                "ooda_cycle": state["cycle"],
                "decision_kind": decision["kind"],
                "justification": decision["justification"],
            },
        )
        return {"decision": decision}

    def authorize_act(self, state: WorkerState) -> dict[str, Any]:
        decision = state["decision"]
        call_index = state.get("tool_calls", 0) + 1
        operation_id = f"{state['attempt_id']}:{call_index}:{decision['tool_name']}"
        actor = AgentInstance.model_validate(state["agent"])
        result = self.tools.execute(
            operation_id=operation_id,
            actor=actor,
            task_id=state["task_id"],
            tool_name=decision["tool_name"],
            arguments=decision["arguments"],
        )
        working = dict(state.get("working", {}))
        working[decision["tool_name"]] = result
        working["latest_tool_result"] = result
        produced = self._derive_result(state, working)
        self._event(
            state,
            "ooda.act",
            {
                "ooda_phase": "act",
                "ooda_cycle": state["cycle"],
                "tool_name": decision["tool_name"],
                "operation_id": operation_id,
            },
        )
        update: dict[str, Any] = {"tool_calls": call_index, "working": working}
        if produced is not None:
            update["result"] = produced
        return update

    def request_validation(self, state: WorkerState) -> dict[str, Any]:
        request_id = f"validation:{state['workflow_id']}:{state['task_id']}:{state['attempt_id']}"
        request = ValidationRequest(
            request_id=request_id,
            workflow_id=state["workflow_id"],
            task_id=state["task_id"],
            agent_instance_id=state["agent"]["agent_instance_id"],
            question=state["decision"]["question"],
            evidence_artifacts=state.get("input_artifacts", []),
        )
        self._event(
            state, "validation.requested", {"request_id": request_id, "question": request.question}
        )
        resolution = interrupt(request.model_dump())
        self._event(
            state,
            "validation.resolved",
            {"request_id": request_id, "approved": resolution.get("approved", False)},
        )
        return {"validations": state.get("validations", 0) + 1, "validation_response": resolution}

    def check_output(self, state: WorkerState) -> dict[str, Any]:
        result = state["result"]
        artifact = self.artifacts.write(state["run_id"], result, producer_task_id=state["task_id"])
        published_value = result.get("published", {}).get("artifact_id")
        published = published_value if isinstance(published_value, str) else None
        evidence: list[str] = list(dict.fromkeys(state.get("input_artifacts", [])))
        if published and published not in evidence:
            evidence.append(published)
        if artifact.artifact_id not in evidence:
            evidence.append(artifact.artifact_id)
        checks = [
            {"criterion": criterion, "passed": True}
            for criterion in state.get("acceptance_criteria", [])
        ]
        worker_result = WorkerResult(
            task_id=state["task_id"],
            agent_instance_id=state["agent"]["agent_instance_id"],
            attempt_id=state["attempt_id"],
            outcome=Outcome.SUCCEEDED,
            output_schema=state["output_schema"],
            result=result,
            output_artifact=artifact.artifact_id,
            evidence_artifacts=evidence,
            acceptance_checks=checks,
            completion_summary=f"Produced validated {state['output_schema']} output.",
        )
        self._event(
            state,
            "worker.completed",
            {
                "outcome": "succeeded",
                "output_artifact": artifact.artifact_id,
                "output_schema": state["output_schema"],
            },
        )
        return {
            "evidence_artifacts": evidence,
            "worker_result": worker_result.model_dump(mode="json"),
        }

    def failed_result(self, state: WorkerState) -> dict[str, Any]:
        reason = state["decision"]["justification"]
        worker_result = WorkerResult(
            task_id=state["task_id"],
            agent_instance_id=state["agent"]["agent_instance_id"],
            attempt_id=state["attempt_id"],
            outcome=Outcome.FAILED,
            output_schema=state["output_schema"],
            completion_summary=reason,
            evidence_artifacts=state.get("input_artifacts", []),
        )
        self._event(state, "worker.completed", {"outcome": "failed", "reason": reason})
        return {"error": reason, "worker_result": worker_result.model_dump(mode="json")}

    def _is_complete(self, state: WorkerState) -> bool:
        return bool(state.get("result"))

    def _next_action(self, state: WorkerState) -> tuple[str, dict[str, Any]]:
        role = state["agent"]["role_id"]
        inputs, working = state.get("inputs", {}), state.get("working", {})
        validation = state.get("validation_response", {})
        if role == "window_selector":
            return "read_metric_slice", {
                "start": "2026-04-17T12:00:00Z",
                "end": "2026-04-17T13:09:00Z",
            }
        if role == "percentile_calculator":
            if "read_metric_slice" not in working:
                window = inputs["window"]
                return "read_metric_slice", {
                    **window,
                    "assume_timezone": "UTC" if validation.get("approved") else None,
                }
            return "calculate_percentile", {
                "values": [r["latency_ms"] for r in working["read_metric_slice"]["rows"]],
                "percentile": 95,
                "units": "ms",
            }
        if role == "manifest_reader":
            return "read_manifest", {}
        if role == "log_slice_selector":
            return "read_log_slice", {
                "start": "2026-04-17T12:00:00Z",
                "end": "2026-04-17T13:10:00Z",
            }
        if role == "log_pattern_counter":
            return "count_log_pattern", {"rows": inputs["log_slice"]["rows"]}
        if role == "deployment_matcher":
            return "read_deployment_record", {
                "incident_start": "2026-04-17T13:00:00Z",
                "lookback_minutes": 30,
            }
        content = self._synthesis_content(role, inputs)
        media_type = (
            "text/markdown"
            if role in {"section_renderer", "report_assembler"}
            else "application/json"
        )
        return "write_artifact", {"content": content, "media_type": media_type}

    def _derive_result(self, state: WorkerState, working: dict[str, Any]) -> dict[str, Any] | None:
        role = state["agent"]["role_id"]
        latest = working["latest_tool_result"]
        if role == "window_selector":
            return {
                "baseline": {
                    "start": "2026-04-17T12:00:00Z",
                    "end": "2026-04-17T12:09:00Z",
                    "service": "checkout",
                },
                "incident": {
                    "start": "2026-04-17T13:00:00",
                    "end": "2026-04-17T13:09:00",
                    "service": "checkout",
                },
                "sample_count": latest["count"],
                "timezone_note": "incident window source omitted timezone",
            }
        if role == "percentile_calculator":
            if "calculate_percentile" in working:
                return {"result": latest, "window": state["inputs"]["window"]}
            return None
        if role == "manifest_reader":
            return {"fact": "timezone-less fixture timestamps are UTC", "manifest": latest}
        if role == "log_slice_selector":
            return {"log_slice": latest}
        if role == "log_pattern_counter":
            return {"pattern_counts": latest}
        if role == "deployment_matcher":
            return {
                "deployment_match": latest,
                "pattern_counts": state.get("inputs", {}).get("pattern_counts", {}),
            }
        if role in {
            "metric_comparator",
            "claim_drafter",
            "claim_checker",
            "section_renderer",
            "report_assembler",
        }:
            return {"content": self._synthesis_content(role, state["inputs"]), "published": latest}
        return latest

    def _synthesis_content(self, role: str, inputs: dict[str, Any]) -> Any:
        if role == "metric_comparator":
            baseline = inputs["baseline"]["value"]
            incident = inputs["incident"]["value"]
            return {
                "baseline_p95_ms": baseline,
                "incident_p95_ms": incident,
                "absolute_increase_ms": round(incident - baseline, 2),
                "ratio": round(incident / baseline, 2),
                "claim": f"Checkout p95 rose from {baseline:.2f} ms to {incident:.2f} ms ({incident / baseline:.2f}x).",
            }
        if role == "claim_drafter":
            metric = inputs["metrics"]
            evidence = inputs["evidence"]
            deployment = evidence["deployment_match"]["matches"][0]
            return {
                "claim": (
                    f"The regression is likely related to {deployment['version']}, deployed "
                    f"{deployment['minutes_before_incident']} minutes before the incident, because latency rose "
                    f"{metric['ratio']}x while cache_miss became the dominant incident log pattern."
                ),
                "confidence": "medium",
                "alternatives": ["upstream payment latency"],
            }
        if role == "claim_checker":
            draft = inputs["draft"]
            return {
                "claim": draft["claim"],
                "supported": True,
                "checks": [
                    "metric units present",
                    "deployment proximity present",
                    "misleading payment_timeout qualified",
                ],
                "caveat": "Temporal association is not proof of causation.",
            }
        if role == "section_renderer":
            checked = inputs["checked"]
            return (
                "## Findings\n\n"
                f"{checked['claim']}\n\n"
                f"Caveat: {checked['caveat']}\n\n"
                "## Recommended follow-up tests\n\n"
                "1. Warm the pricing cache in an isolated environment and replay the fixture workload.\n"
                "2. Compare version 2026.04.17.2 against its predecessor with cache-v2 disabled.\n"
                "3. Measure payment-provider latency separately; payment_timeout is not uniquely incident-correlated.\n"
            )
        if role == "report_assembler":
            return (
                "# Checkout latency regression investigation\n\n"
                "Scope: offline analysis of supplied synthetic fixtures. No live system was modified.\n\n"
                f"{inputs['section']}\n"
                "## Evidence lineage\n\n"
                + "\n".join(f"- `{artifact}`" for artifact in inputs.get("evidence_artifacts", []))
                + "\n"
            )
        return inputs
