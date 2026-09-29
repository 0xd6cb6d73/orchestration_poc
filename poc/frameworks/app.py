"""Headless application boundary for one selected native orchestration backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import AsyncIterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import Field, model_validator
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import Answer, Budget, ModelSpec, StrictModel, TaskInput
from poc.execution.sql_strategy import resolve_model
from poc.frameworks.approval import ApprovalRequest, MockApprovalClient
from poc.frameworks.common import RunBridge

BackendName = Literal["pydantic_background", "strands_background", "llamaindex_workflows"]


class RunRequest(StrictModel):
    schema_version: Literal[1] = 1
    caller_request_id: str = Field(min_length=1)
    backend: BackendName
    profile_id: Literal["sql_readonly"] = "sql_readonly"
    tool_bundle_ref: Literal["sql_readonly"] = "sql_readonly"
    input_messages: list[str] = Field(min_length=1)
    tables: dict[str, list[dict[str, Any]]]
    model_spec: ModelSpec
    model_settings: dict[str, Any] = Field(default_factory=dict)
    limits: Budget = Field(default_factory=Budget)
    workspace_refs: list[str] = Field(default_factory=list)
    tags: dict[str, Any] = Field(default_factory=dict)
    approval_mode: Literal["none", "mock_grant", "mock_deny", "mock_timeout"] = "none"
    max_tasks: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def supported_inputs(self) -> RunRequest:
        if self.workspace_refs:
            raise ValueError("workspace references are unsupported by the sql_readonly profile")
        if any(not message.strip() for message in self.input_messages):
            raise ValueError("input messages must contain nonempty text")
        return self


class RunResult(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    caller_request_id: str
    status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    answer: Answer | None
    artifact_refs: list[str] = Field(default_factory=list)
    error_category: str | None = None
    error_message: str | None = None
    usage: dict[str, Any]
    trace_location: str
    effective_manifest: str


class AutoApprovalClient(MockApprovalClient):
    def __init__(self, approved: bool) -> None:
        super().__init__()
        self.approved = approved

    def request(
        self,
        run_id: str,
        task_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ApprovalRequest:
        request = super().request(run_id, task_id, tool_call_id, tool_name, arguments)
        self.decide(
            request.approval_id, request.tool_call_id, request.argument_digest, self.approved
        )
        return request


def _package_versions() -> dict[str, str | None]:
    names = (
        "pydantic-ai-slim",
        "pydantic-ai-harness",
        "strands-agents",
        "llama-index-core",
        "llama-index-workflows",
        "llama-index-llms-openai-like",
    )
    found: dict[str, str | None] = {}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def _manifest(request: RunRequest, run_id: str, settings: ModelSettings) -> dict[str, Any]:
    commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        or None
    )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "caller_request_id": request.caller_request_id,
        "backend": request.backend,
        "profile_id": request.profile_id,
        "tool_bundle_ref": request.tool_bundle_ref,
        "application_commit": commit,
        "dependencies": _package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "container_image_digest": os.environ.get("FRAMEWORK_IMAGE_DIGEST"),
        "model_spec": request.model_spec.model_dump(mode="json"),
        "model_settings": settings,
        "prompt_sha256": hashlib.sha256("\n".join(request.input_messages).encode()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(b"sql_readonly:v1:query_sql").hexdigest(),
        "limits": request.limits.model_dump(mode="json"),
        "max_tasks": request.max_tasks,
        "approval_mode": request.approval_mode,
        "retry_policy": {"model": 0, "effectful_tool": 0},
        "tags": request.tags,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


class RunHandle:
    def __init__(
        self,
        run_id: str,
        task: asyncio.Task[RunResult],
        queue: asyncio.Queue[dict[str, Any]],
        cancellation: dict[str, str],
    ):
        self.run_id = run_id
        self._task = task
        self._queue = queue
        self._cancellation = cancellation

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            if self._task.done() and self._queue.empty():
                return
            try:
                yield await asyncio.wait_for(self._queue.get(), 0.1)
            except TimeoutError:
                continue

    async def result(self) -> RunResult:
        return await self._task

    async def cancel(self, reason: str) -> None:
        self._cancellation["reason"] = reason
        # Let execute enter its try/finally so even an immediate cancel writes result.json.
        await asyncio.sleep(0)
        if not self._task.done():
            self._task.cancel()
        await self._task


def start(
    request: RunRequest,
    output_dir: Path,
    *,
    model_override: Model | str | None = None,
    event_buffer: int = 4096,
) -> RunHandle:
    """Start one backend. The queue is observational; JSONL is the audit source."""
    if event_buffer < 1:
        raise ValueError("event_buffer must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    settings = cast(ModelSettings, {**request.model_spec.settings, **request.model_settings})
    manifest = _manifest(request, run_id, settings)
    _atomic_json(output_dir / "manifest.json", manifest)
    task_input = TaskInput(prompt="\n".join(request.input_messages), tables=request.tables)
    environment = TaskEnvironment(task_input, request.limits.tool_calls)
    environment.state.options = environment.state.options.model_copy(
        update={"framework_max_tasks": request.max_tasks}
    )
    environment.state.model_spec = request.model_spec
    usage = RunUsage()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=event_buffer)
    overflow = False
    events_file = (output_dir / "events.jsonl").open("a", encoding="utf-8")
    transcript_file = (output_dir / "transcripts.jsonl").open("a", encoding="utf-8")

    def emit(event: dict[str, Any]) -> None:
        nonlocal overflow
        events_file.write(json.dumps(event, default=str) + "\n")
        events_file.flush()
        kind = event.get("framework", {}).get("event_type")
        if kind in {"model.request", "model.response", "tool.started", "tool.completed"}:
            transcript_file.write(json.dumps(event, default=str) + "\n")
            transcript_file.flush()
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            overflow = True

    environment.state.emit = emit
    approvals: MockApprovalClient | None = None
    if request.approval_mode == "mock_grant":
        approvals = AutoApprovalClient(True)
    elif request.approval_mode == "mock_deny":
        approvals = AutoApprovalClient(False)
    elif request.approval_mode == "mock_timeout":
        approvals = MockApprovalClient(timeout=0.01)
    bridge = RunBridge(
        task_input,
        environment,
        request.limits,
        usage,
        settings,
        request.model_spec,
        approvals=approvals,
        backend=request.backend,
    )
    bridge.run_id = run_id
    cancellation: dict[str, str] = {}
    bridge.event("run.started")

    async def execute() -> RunResult:
        answer: Answer | None = None
        status: Literal["succeeded", "failed", "cancelled", "timed_out"] = "failed"
        error: str | None = None
        error_message: str | None = None
        try:
            async with asyncio.timeout(request.limits.seconds):
                if request.backend == "pydantic_background":
                    from poc.frameworks.pydantic_background import run

                    model = model_override or resolve_model(request.model_spec)
                    answer = await run(bridge, model)
                elif request.backend == "strands_background":
                    from poc.frameworks.strands_background import run

                    answer = await run(bridge)
                else:
                    from poc.frameworks.llamaindex_workflows import run

                    answer = await run(bridge)
                status = "succeeded"
        except asyncio.CancelledError:
            status, error = "cancelled", "CancelledError"
            error_message = cancellation.get("reason")
        except TimeoutError as exc:
            status, error = "timed_out", "TimeoutError"
            error_message = str(exc) or "run deadline exceeded"
        except Exception as exc:
            status, error = "failed", type(exc).__name__
            error_message = str(exc)
        finally:
            if overflow or queue.full():
                status, error, answer = "failed", "EventBufferOverflow", None
                error_message = "event consumer fell behind the configured buffer"
            bridge.event(f"run.{status}", error_category=error, error_message=error_message)
            environment.close()
            events_file.close()
            transcript_file.close()
            result = RunResult(
                run_id=run_id,
                caller_request_id=request.caller_request_id,
                status=status,
                answer=answer,
                error_category=error,
                error_message=error_message,
                usage={
                    "requests": usage.requests,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "tool_calls": environment.tool_calls,
                    "usage_complete": environment.state.usage_complete,
                },
                trace_location=str(output_dir / "events.jsonl"),
                effective_manifest=str(output_dir / "manifest.json"),
            )
            _atomic_json(output_dir / "result.json", result.model_dump(mode="json"))
        return result

    task = asyncio.create_task(execute())
    return RunHandle(run_id, task, queue, cancellation)
