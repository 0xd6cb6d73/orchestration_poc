"""Deterministic provider responses exercising real SQL workers and controllers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ThinkingPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from poc.execution.sql_contracts import TaskInput

SQL = "SELECT id, release FROM schedule_jobs ORDER BY id"
TASK = TaskInput(
    prompt="Infrastructure canary: inspect the public tables using SQL, then return a complete "
    "values mapping from job ID to integer start time. The release column already contains "
    "a feasible start time for each job; copy it exactly. Every worker should query the tables. "
    "Use one action per response. Planners: give a short plan. Critics and verifiers: follow "
    "your role schema and cite evidence_ref values from your own SQL results. No optimization "
    "or search is needed. Dependencies must finish first, jobs use one machine, and all work "
    "must finish by the public horizon.",
    tables={
        "schedule_jobs": [
            {"id": name, "release": index * 2, "duration": 1, "deadline": 10, "machine": 0}
            for index, name in enumerate(("job-A", "job-B", "job-C"))
        ],
        "precedence": [
            {"job": "job-B", "requires": "job-A"},
            {"job": "job-C", "requires": "job-B"},
        ],
        "limits": [{"horizon": 10}],
    },
)
TARGET = {row["id"]: row["release"] for row in TASK.tables["schedule_jobs"]}


def query_result(messages: list[ModelMessage]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, ToolReturnPart) and part.tool_name == "query":
                    content: Any = part.content
                    return (
                        dict(cast(dict[str, Any], content))
                        if isinstance(content, dict)
                        else json.loads(str(content))
                    )
                if (
                    isinstance(part, UserPromptPart)
                    and isinstance(part.content, str)
                    and part.content.startswith("SQL result: ")
                ):
                    return json.loads(part.content.removeprefix("SQL result: "))
    return None


def role_of(messages: list[ModelMessage], info: AgentInfo, method: str) -> str:
    instructions = info.instructions or ""
    for noun, role in (("planner", "plan"), ("critic", "critique"), ("verifier", "verify")):
        if f"Your role is {noun}" in instructions:
            return role
    prompt = str(messages[0])
    if "Reconcile these independent" in prompt:
        return "reconcile"
    if "sealed proposer" in prompt or "You are candidate" in prompt:
        return "proposal"
    if "Review this submitted draft" in prompt:
        return "review"
    return "draft" if method == "review" else "solve"


class ScriptedProvider:
    def __init__(self, method: str, scenario: str, concurrent: bool):
        self.method, self.scenario, self.concurrent = method, scenario, concurrent
        self.calls = 0
        self.injected = 0
        self.active = 0
        self.initial_workers: set[str] = set()
        self.both_started = asyncio.Event()
        self.model = FunctionModel(self.respond, model_name="infrastructure-fixture-v1")

    def response(
        self, payload: dict[str, Any], info: AgentInfo, *, sql: bool = False
    ) -> ModelResponse:
        if info.output_tools:
            part = ToolCallPart("query" if sql else info.output_tools[0].name, payload)
            return ModelResponse(
                parts=[part], usage=RequestUsage(input_tokens=32, output_tokens=32)
            )
        text = json.dumps(payload)
        if self.scenario == "escaped_json":
            self.injected += 1
            text = json.dumps(text)
        return ModelResponse(
            parts=[TextPart(text)], usage=RequestUsage(input_tokens=32, output_tokens=32)
        )

    async def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls += 1
        self.active += 1
        try:
            return await self._respond(messages, info)
        finally:
            self.active -= 1

    async def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        role = role_of(messages, info, self.method)
        scenario = self.scenario
        if scenario == "rate_limit" and self.calls == 1:
            self.injected += 1
            raise ModelHTTPError(429, "diagnostic-fixture", {"redacted": True})
        if scenario in {"reasoning_only", "duplicate_keys", "multiple_actions"}:
            self.injected += 1
            if scenario == "reasoning_only":
                return ModelResponse(parts=[ThinkingPart('{"sql":"SELECT 1"}')])
            raw = '{"values":{"job-A":0,"job-A":2}}' if scenario == "duplicate_keys" else "{}{}"
            if info.output_tools:
                return ModelResponse(
                    parts=[ToolCallPart("query", '{"sql":"SELECT 1","sql":"SELECT 2"}')]
                )
            return ModelResponse(parts=[TextPart(raw)])
        if scenario in {"request_timeout", "phase_timeout", "cancellation"} or (
            scenario == "reconcile_timeout" and role == "reconcile"
        ):
            self.injected += 1
            await asyncio.Event().wait()
            raise AssertionError("cancelled fixture resumed")
        if scenario == "pool_reassign" and self.calls == 1:
            self.injected += 1
            raise UnexpectedModelBehavior("injected worker failure")
        if scenario == "proposal_loss" and "sealed proposer 1" in str(messages[0]):
            self.injected += 1
            raise UnexpectedModelBehavior("injected proposal failure")
        # A rendezvous makes overlap a tested behavior, rather than a timing coincidence.
        if (
            self.concurrent
            and scenario == "healthy"
            and role in {"plan", "proposal"}
            and len(self.initial_workers) < 2
        ):
            self.initial_workers.add(str(messages[0]))
            if len(self.initial_workers) == 2:
                self.both_started.set()
            await self.both_started.wait()
        result = query_result(messages)
        if result is None or scenario == "sql_no_progress":
            if scenario == "sql_no_progress":
                self.injected += 1
            return self.response(
                {"sql": "SELECT absent FROM absent" if scenario == "sql_no_progress" else SQL},
                info,
                sql=True,
            )
        if "error" in result:
            raise RuntimeError("diagnostic fixture SQL unexpectedly failed")
        values = {str(row[0]): row[1] for row in result["rows"]}
        if (
            scenario == "reconcile_timeout"
            and role == "proposal"
            and "You are candidate 1" in str(messages[0])
        ):
            # Intentionally infeasible but structurally valid. Retention must follow commit
            # provenance, not prefer the other candidate's better answer.
            values = {key: value + 101 for key, value in values.items()}
        if scenario == "malformed_artifact" and role in {"solve", "draft", "proposal", "reconcile"}:
            self.injected += 1
            return self.response({"values": {"0": 0}}, info)
        if role == "plan":
            values = {"plan": "Read id and release from schedule_jobs and return all IDs."}
        elif role == "critique":
            if scenario == "critic_contract":
                self.injected += 1
                return self.response({"values": {"plan": "not a critique"}}, info)
            evaluation: dict[str, int] = dict.fromkeys(
                ("validity", "evidence", "usefulness", "novelty", "constraint_satisfaction"), 4
            )
            refs = [result["evidence_ref"]]
            if scenario == "false_approval":
                self.injected += 1
                evaluation.update(validity=1, constraint_satisfaction=0)
            if scenario == "foreign_evidence":
                self.injected += 1
                refs = ["query:foreign-worker:0:1"]
            values = {
                "structural_validity": True,
                "feasibility": "supported",
                "reason": "Read the public start times",
                "evidence_refs": refs,
                "evaluation": evaluation,
            }
        elif role == "verify":
            if scenario == "verifier_reject":
                self.injected += 1
            values = {
                "answer_supported": scenario != "verifier_reject",
                "reason": "Checked public starts",
                "evidence_refs": [result["evidence_ref"]],
            }
        elif role == "review":
            if scenario == "empty_revision":
                self.injected += 1
                return self.response(
                    {
                        "action": "revise",
                        "reason": "injected empty revision",
                        "replacement": {"values": {}},
                    },
                    info,
                )
            return self.response(
                {"action": "accept", "reason": "The public starts match", "replacement": None}, info
            )
        return self.response({"values": values}, info)
