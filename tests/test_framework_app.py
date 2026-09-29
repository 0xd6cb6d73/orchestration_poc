"""Headless native framework application contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from poc.execution.sql_contracts import Budget, ModelSpec
from poc.frameworks.app import RunRequest, start


@pytest.fixture
def run_request() -> RunRequest:
    return RunRequest(
        caller_request_id="fixture-1",
        backend="pydantic_background",
        input_messages=["Find x"],
        tables={"items": [{"id": "x", "n": 7}]},
        model_spec=ModelSpec(name="fixture", model_class="baseline", model="fixture"),
        limits=Budget(seconds=5),
    )


async def test_headless_run_writes_manifest_events_transcript_and_result(
    tmp_path: Path, run_request: RunRequest
) -> None:
    pytest.importorskip("pydantic_ai_harness")

    async def respond(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        history = str(messages)
        if "SELECT * FROM items" not in history:
            return ModelResponse(parts=[ToolCallPart("query_sql", {"sql": "SELECT * FROM items"})])
        if "accepted" not in history:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "finish_task", {"output": '{"values":{"x":7}}', "used_result_ids": []}
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    output = tmp_path / "run"
    handle = start(run_request, output, model_override=FunctionModel(respond))
    streamed = [event async for event in handle.events()]
    result = await handle.result()
    assert result.status == "succeeded"
    assert result.answer is not None and result.answer.values == {"x": 7}
    assert result.caller_request_id == run_request.caller_request_id
    assert result.usage["tool_calls"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    persisted = json.loads((output / "result.json").read_text())
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    transcript = (output / "transcripts.jsonl").read_text().splitlines()
    assert manifest["run_id"] == result.run_id == persisted["run_id"]
    assert manifest["dependencies"]["pydantic-ai-slim"] == "2.51.0"
    assert streamed == events
    assert transcript
    assert any(e["framework"]["event_type"] == "model.request" for e in events)
    assert not list(output.glob("*.tmp"))


async def test_immediate_cancel_writes_terminal_result(
    tmp_path: Path, run_request: RunRequest
) -> None:
    pytest.importorskip("pydantic_ai_harness")
    wait = asyncio.Event()

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        await wait.wait()
        return ModelResponse(parts=[TextPart("done")])

    output = tmp_path / "cancelled"
    handle = start(run_request, output, model_override=FunctionModel(respond))
    await handle.cancel("test")
    result = await handle.result()
    assert result.status == "cancelled"
    assert result.error_message == "test"
    assert json.loads((output / "result.json").read_text())["status"] == "cancelled"


async def test_bounded_stream_reports_overflow(tmp_path: Path, run_request: RunRequest) -> None:
    pytest.importorskip("pydantic_ai_harness")

    async def respond(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        if "accepted" not in str(messages):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "finish_task", {"output": '{"values":{"x":7}}', "used_result_ids": []}
                    )
                ]
            )
        return ModelResponse(parts=[TextPart("done")])

    output = tmp_path / "overflow"
    result = await start(
        run_request, output, model_override=FunctionModel(respond), event_buffer=1
    ).result()
    assert result.status == "failed"
    assert result.error_category == "EventBufferOverflow"
    assert json.loads((output / "result.json").read_text())["status"] == "failed"
    assert len((output / "events.jsonl").read_text().splitlines()) > 1


async def test_deadline_writes_timed_out_result(tmp_path: Path, run_request: RunRequest) -> None:
    pytest.importorskip("pydantic_ai_harness")
    wait = asyncio.Event()

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        await wait.wait()
        return ModelResponse(parts=[TextPart("done")])

    timed_request = run_request.model_copy(update={"limits": Budget(seconds=0.01)})
    output = tmp_path / "deadline"
    result = await start(timed_request, output, model_override=FunctionModel(respond)).result()
    assert result.status == "timed_out"
    assert result.error_category == "TimeoutError"
    assert json.loads((output / "result.json").read_text())["status"] == "timed_out"
