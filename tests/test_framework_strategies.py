"""Native runtime smoke and early-completion probes for optional POC strategies."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from contextlib import suppress
from typing import Any, cast

import pytest
from pydantic_ai import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Matrix, TaskCase
from poc.evaluation.suite.runner import run_trial
from poc.execution.sql_contracts import Budget, ModelSpec, TaskInput

TASK = TaskInput(prompt="Find x", tables={"items": [{"id": "x", "n": 7}]})


@pytest.fixture(scope="module", autouse=True)
def clean_optional_registration() -> Any:
    yield
    for name in ("pydantic_background", "strands_background", "llamaindex_workflows"):
        ADAPTERS.pop(name, None)


@pytest.mark.parametrize("strategy", ["strands_background", "llamaindex_workflows"])
async def test_native_strategy_runs_evaluation_tool_boundary(
    strategy: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("strands" if strategy == "strands_background" else "llama_index")
    pytest.importorskip("aiohttp")
    from aiohttp import web

    from poc.frameworks import evaluation

    calls = 0

    async def completion(request: Any) -> Any:
        nonlocal calls
        body = await request.json()
        calls += 1
        messages = body["messages"]
        tool_results = [m for m in messages if m["role"] == "tool"]
        name: str | None
        args: dict[str, Any] | None
        if not tool_results:
            name, args = "query_sql", {"sql": "SELECT * FROM items"}
        elif not any("accepted" in str(m["content"]) for m in tool_results):
            name, args = "finish_task", {"output": '{"values":{"x":7}}', "used_result_ids": []}
        else:
            name, args = None, None
        if name:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{calls}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "done"}
        return web.json_response(
            {
                "id": f"cmpl-{calls}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if name else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = cast(tuple[str, int], runner.addresses[0])[1]
    monkeypatch.setenv("POC_TEST_OPENAI_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("POC_TEST_OPENAI_KEY", "fixture")
    spec = ModelSpec(
        name="fixture",
        model_class="baseline",
        model="fixture",
        base_url_env="POC_TEST_OPENAI_URL",
        api_key_env="POC_TEST_OPENAI_KEY",
    )
    env = TaskEnvironment(TASK, 10)
    env.state.model_spec = spec
    try:
        selected = getattr(evaluation, strategy)
        answer = await asyncio.wait_for(
            selected(TASK, env, "fixture", {}, Budget(), RunUsage()), 10
        )
        assert answer.values == {"x": 7}
        assert env.tool_calls == 1
        assert calls >= 3
        case = TaskCase(
            id="framework-fixture",
            family="ledger",
            seed=0,
            split="dev",
            difficulty="standard",
            input=TASK,
            expected={"x": 7},
        )
        matrix = Matrix(
            models=[spec],
            strategies=[strategy],
            families=["ledger"],
            seeds=[0],
            repetitions=1,
            max_concurrency=1,
        )
        trial = await run_trial(case, spec, strategy, 0, matrix)
        assert trial["status"] == "completed"
        assert trial["scores"]["exact"] == 1
    finally:
        env.close()
        await runner.cleanup()


async def test_pydantic_background_replans_on_early_result() -> None:
    pytest.importorskip("pydantic_ai_harness")
    from poc.frameworks.evaluation import pydantic_background

    env = TaskEnvironment(TASK, 10)
    events: list[dict[str, Any]] = []
    release_slow = asyncio.Event()
    steps: defaultdict[str, int] = defaultdict(int)

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        detail = event.get("framework", {})
        if detail.get("event_type") == "task.dispatched" and detail.get("instruction") == "C":
            release_slow.set()

    env.state.emit = emit

    async def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        run_id = messages[0].run_id
        steps[run_id] += 1
        n = steps[run_id]
        first = str(messages[0])
        role = (
            "root"
            if "You are a root" in (info.instructions or "")
            else next((name for name in ("A", "B", "C") if f"Task: {name}" in first), "unknown")
        )
        history = str(messages)
        if role == "root":
            if n == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart("delegate", {"role_key": "specialist", "task": "A"}),
                        ToolCallPart("delegate", {"role_key": "specialist", "task": "B"}),
                    ]
                )
            if "N123" in history and not release_slow.is_set():
                return ModelResponse(
                    parts=[ToolCallPart("delegate", {"role_key": "specialist", "task": "C"})]
                )
            complete = sum(
                event.get("framework", {}).get("event_type") == "task.succeeded"
                and event["framework"].get("parent_id") == root_id()
                for event in events
            )
            if complete == 3:
                if "accepted" in history:
                    return ModelResponse(parts=[TextPart("done")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "finish_task", {"output": '{"values":{"x":7}}', "used_result_ids": []}
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("waiting")])
        if role == "A" and n == 1:
            await release_slow.wait()
        if "accepted" in history:
            return ModelResponse(parts=[TextPart("done")])
        output = '{"values":{"nonce":"N123"}}' if role == "B" else '{"values":{"x":7}}'
        return ModelResponse(
            parts=[ToolCallPart("finish_task", {"output": output, "used_result_ids": []})]
        )

    def root_id() -> str | None:
        return next(
            (
                e["framework"]["task_id"]
                for e in events
                if e.get("framework", {}).get("event_type") == "task.dispatched"
                and e["framework"].get("role") == "root"
            ),
            None,
        )

    try:
        answer = await asyncio.wait_for(
            pydantic_background(TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()),
            5,
        )
        assert answer.values == {"x": 7}
        records = [e["framework"] for e in events if "framework" in e]
        c_dispatch = next(i for i, e in enumerate(records) if e.get("instruction") == "C")
        a_id = next(e["task_id"] for e in records if e.get("instruction") == "A")
        a_terminal = next(
            i
            for i, e in enumerate(records)
            if e["event_type"] == "task.succeeded" and e["task_id"] == a_id
        )
        assert c_dispatch < a_terminal
        assert any(
            record["event_type"] == "model.request"
            and record["task_id"] == root_id()
            and "N123" in record["input"]
            for record in records[:c_dispatch]
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("strategy", "root_work"),
    [
        pytest.param(
            "strands_background",
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Strands 1.57.1 waits for all background tools when the parent is idle",
            ),
        ),
        ("strands_background", True),
        ("llamaindex_workflows", False),
        ("llamaindex_workflows", True),
    ],
)
async def test_native_parent_dispatches_after_early_child_result(
    strategy: str, root_work: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("strands" if strategy == "strands_background" else "llama_index")
    pytest.importorskip("aiohttp")
    from aiohttp import web

    from poc.frameworks import evaluation

    events: list[dict[str, Any]] = []
    model_inputs: list[tuple[str, str]] = []
    release_slow = asyncio.Event()
    b_completed = asyncio.Event()
    steps: defaultdict[str, int] = defaultdict(int)
    c_sent = False

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        detail = event.get("framework", {})
        if detail.get("event_type") == "task.dispatched" and detail.get("instruction") == "C":
            release_slow.set()
        if detail.get("event_type") == "task.succeeded" and "N123" in str(detail.get("result")):
            b_completed.set()

    async def completion(request: Any) -> Any:
        nonlocal c_sent
        body = await request.json()
        messages = body["messages"]
        history = json.dumps(messages)
        system = str(messages[0].get("content", ""))
        role = "root" if "You are a root" in system else "specialist"
        label = (
            "root"
            if role == "root"
            else next((name for name in ("A", "B", "C") if f"Task: {name}" in history), "unknown")
        )
        steps[label] += 1
        model_inputs.append((label, history))
        calls: list[tuple[str, dict[str, Any]]] = []
        if label == "root":
            if steps[label] == 1:
                calls = [
                    ("delegate", {"role_key": "specialist", "task": "A"}),
                    ("delegate", {"role_key": "specialist", "task": "B"}),
                ]
            elif "N123" in history and not c_sent:
                c_sent = True
                calls = [("delegate", {"role_key": "specialist", "task": "C"})]
            elif root_work and steps[label] == 2:
                # The root continues an independent model/tool turn while B finishes.
                await asyncio.wait_for(b_completed.wait(), 2)
                calls = [("query_sql", {"sql": "SELECT 1"})]
            else:
                root_id = next(
                    (
                        e["framework"]["task_id"]
                        for e in events
                        if e.get("framework", {}).get("role") == "root"
                        and e["framework"].get("event_type") == "task.dispatched"
                    ),
                    None,
                )
                completed = sum(
                    e.get("framework", {}).get("event_type") == "task.succeeded"
                    and e["framework"].get("parent_id") == root_id
                    for e in events
                )
                if completed == 3 and "accepted" not in history:
                    calls = [
                        (
                            "finish_task",
                            {"output": '{"values":{"x":7}}', "used_result_ids": []},
                        )
                    ]
        else:
            if label == "A" and steps[label] == 1:
                with suppress(TimeoutError):
                    await asyncio.wait_for(release_slow.wait(), 3)
            if "accepted" not in history:
                output = '{"values":{"nonce":"N123"}}' if label == "B" else '{"values":{"x":7}}'
                calls = [("finish_task", {"output": output, "used_result_ids": []})]
        if calls:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{label}_{steps[label]}_{i}",
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(arguments)},
                    }
                    for i, (tool, arguments) in enumerate(calls)
                ],
            }
        else:
            message = {"role": "assistant", "content": "waiting" if label == "root" else "done"}
        return web.json_response(
            {
                "id": f"cmpl-{len(model_inputs)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = cast(tuple[str, int], runner.addresses[0])[1]
    monkeypatch.setenv("POC_TEST_OPENAI_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("POC_TEST_OPENAI_KEY", "fixture")
    spec = ModelSpec(
        name="fixture",
        model_class="baseline",
        model="fixture",
        base_url_env="POC_TEST_OPENAI_URL",
        api_key_env="POC_TEST_OPENAI_KEY",
    )
    env = TaskEnvironment(TASK, 10)
    env.state.model_spec = spec
    env.state.emit = emit
    try:
        selected = getattr(evaluation, strategy)
        answer = await asyncio.wait_for(
            selected(TASK, env, "fixture", {}, Budget(seconds=15), RunUsage()), 12
        )
        assert answer.values == {"x": 7}
        records = [e["framework"] for e in events if "framework" in e]
        root_id = next(e["task_id"] for e in records if e.get("role") == "root")
        a_id = next(e["task_id"] for e in records if e.get("instruction") == "A")
        b_id = next(e["task_id"] for e in records if e.get("instruction") == "B")
        b_terminal = next(
            i
            for i, e in enumerate(records)
            if e["task_id"] == b_id and e["event_type"] == "task.succeeded"
        )
        a_terminal = next(
            i
            for i, e in enumerate(records)
            if e["task_id"] == a_id and e["event_type"] == "task.succeeded"
        )
        nonce_request = next(
            i
            for i, e in enumerate(records)
            if e["event_type"] == "model.request"
            and e["task_id"] == root_id
            and "N123" in e["input"]
        )
        c_dispatch = next(i for i, e in enumerate(records) if e.get("instruction") == "C")
        witness = {
            "B completed": b_terminal,
            "root model saw B nonce": nonce_request,
            "C dispatched": c_dispatch,
            "A completed": a_terminal,
        }
        assert b_terminal < a_terminal, witness
        assert b_terminal < nonce_request < c_dispatch < a_terminal, witness
        assert any(label == "root" and "N123" in history for label, history in model_inputs)
        if root_work:
            root_requests_without_nonce = [
                i
                for i, e in enumerate(records)
                if e["event_type"] == "model.request"
                and e["task_id"] == root_id
                and "N123" not in e["input"]
            ]
            assert len(root_requests_without_nonce) >= 2
            assert root_requests_without_nonce[1] < b_terminal, witness
            assert any(
                e["event_type"] == "tool.completed"
                and e["task_id"] == root_id
                and e.get("tool") == "query_sql"
                for e in records[b_terminal:c_dispatch]
            )
    finally:
        env.close()
        await runner.cleanup()


@pytest.mark.parametrize("decision", ["grant", "deny", "timeout"])
async def test_mock_approval_binds_exact_call_and_isolates_sibling(
    decision: str,
) -> None:
    from poc.frameworks.approval import MockApprovalClient
    from poc.frameworks.common import RunBridge

    env = TaskEnvironment(TASK, 10)
    client = MockApprovalClient(timeout=0.03 if decision == "timeout" else 1)
    bridge = RunBridge(TASK, env, Budget(), RunUsage(), {}, approvals=client)
    root = bridge.start_task("root", "root")
    a = bridge.start_task("specialist", "A", root)
    b = bridge.start_task("specialist", "B", root)
    leaf = bridge.start_task("leaf", "A1", a)
    pending = asyncio.create_task(bridge.write_ledger(leaf, "entry"))
    try:
        await asyncio.sleep(0)
        request = next(iter(client.requests.values()))
        assert bridge.query(b, "SELECT 1")["rows"] == [[1]]
        assert not client.decide(request.approval_id, "wrong-call", request.argument_digest, True)
        assert not client.decide(request.approval_id, request.tool_call_id, "wrong-digest", True)
        if decision != "timeout":
            assert client.decide(
                request.approval_id,
                request.tool_call_id,
                request.argument_digest,
                decision == "grant",
            )
            assert not client.decide(
                request.approval_id, request.tool_call_id, request.argument_digest, True
            )
        result = await pending
        assert (
            result["status"]
            == {
                "grant": "executed",
                "deny": "denied",
                "timeout": "timed_out",
            }[decision]
        )
        assert len(bridge.effect_ledger) == int(decision == "grant")
    finally:
        bridge.close()
        env.close()


async def test_pydantic_nested_approval_does_not_block_parent_replanning() -> None:
    pytest.importorskip("pydantic_ai_harness")
    from poc.frameworks.approval import MockApprovalClient
    from poc.frameworks.common import RunBridge
    from poc.frameworks.pydantic_background import run

    env = TaskEnvironment(TASK, 10)
    events: list[dict[str, Any]] = []
    c_dispatched = asyncio.Event()
    a3_dispatched = asyncio.Event()
    steps: defaultdict[str, int] = defaultdict(int)
    client = MockApprovalClient(timeout=5)

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        detail = event.get("framework", {})
        if detail.get("event_type") == "task.dispatched":
            if detail.get("instruction") == "C":
                c_dispatched.set()
            if detail.get("instruction") == "A3":
                a3_dispatched.set()

    env.state.emit = emit

    def task_id(instruction: str) -> str:
        return next(
            e["framework"]["task_id"]
            for e in events
            if e.get("framework", {}).get("event_type") == "task.dispatched"
            and e["framework"].get("instruction") == instruction
        )

    def terminal_children(parent_id: str) -> int:
        return sum(
            e.get("framework", {}).get("event_type") == "task.succeeded"
            and e["framework"].get("parent_id") == parent_id
            for e in events
        )

    async def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        first = str(messages[0])
        role = (
            "root"
            if "You are a root" in (info.instructions or "")
            else next(
                (label for label in ("A1", "A2", "A3", "A", "B", "C") if f"Task: {label}" in first),
                "unknown",
            )
        )
        steps[messages[0].run_id] += 1
        n = steps[messages[0].run_id]
        history = str(messages)
        if role == "root":
            if n == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart("delegate", {"role_key": "specialist", "task": "A"}),
                        ToolCallPart("delegate", {"role_key": "specialist", "task": "B"}),
                    ]
                )
            if "N123" in history and not c_dispatched.is_set():
                return ModelResponse(
                    parts=[ToolCallPart("delegate", {"role_key": "specialist", "task": "C"})]
                )
            if terminal_children(task_id(TASK.prompt)) == 3:
                if "accepted" in history:
                    return ModelResponse(parts=[TextPart("done")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "finish_task", {"output": '{"values":{"x":7}}', "used_result_ids": []}
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("waiting")])
        if role == "A":
            if n == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart("delegate", {"role_key": "leaf", "task": "A1"}),
                        ToolCallPart("delegate", {"role_key": "leaf", "task": "A2"}),
                    ]
                )
            if "A2_READY" in history and not a3_dispatched.is_set():
                return ModelResponse(
                    parts=[ToolCallPart("delegate", {"role_key": "leaf", "task": "A3"})]
                )
            if terminal_children(task_id("A")) == 3:
                if "accepted" in history:
                    return ModelResponse(parts=[TextPart("done")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "finish_task", {"output": '{"values":{"a":1}}', "used_result_ids": []}
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("waiting")])
        if role == "A1" and n == 1:
            return ModelResponse(parts=[ToolCallPart("write_ledger", {"entry": "once"})])
        if role == "A2" and n == 1:
            await c_dispatched.wait()
        if "accepted" in history:
            return ModelResponse(parts=[TextPart("done")])
        output = {
            "B": '{"values":{"nonce":"N123"}}',
            "A2": '{"values":{"signal":"A2_READY"}}',
        }.get(role, '{"values":{"x":7}}')
        return ModelResponse(
            parts=[ToolCallPart("finish_task", {"output": output, "used_result_ids": []})]
        )

    async def approval_driver() -> None:
        await a3_dispatched.wait()
        while not client.requests:
            await asyncio.sleep(0)
        request = next(iter(client.requests.values()))
        assert client.decide(
            request.approval_id, request.tool_call_id, request.argument_digest, True
        )

    bridge = RunBridge(TASK, env, Budget(), RunUsage(), {}, approvals=client)
    driver = asyncio.create_task(approval_driver())
    try:
        answer = await asyncio.wait_for(run(bridge, FunctionModel(respond)), 5)
        await driver
        assert answer.values == {"x": 7}
        assert list(bridge.effect_ledger.values()) == ["once"]
        records = [e["framework"] for e in events if "framework" in e]
        a1_id = task_id("A1")
        a_id = task_id("A")
        a3_dispatch = next(i for i, e in enumerate(records) if e.get("instruction") == "A3")
        a1_terminal = next(
            i
            for i, e in enumerate(records)
            if e["event_type"] == "task.succeeded" and e["task_id"] == a1_id
        )
        assert a3_dispatch < a1_terminal
        assert any(
            e["event_type"] == "model.request" and e["task_id"] == a_id and "A2_READY" in e["input"]
            for e in records[:a3_dispatch]
        )
    finally:
        driver.cancel()
        env.close()


def test_unknown_framework_usage_is_reserved_but_not_reported_as_spent() -> None:
    from pydantic_ai.exceptions import UsageLimitExceeded

    from poc.frameworks.common import RunBridge

    env = TaskEnvironment(TASK, 10)
    usage = RunUsage()
    bridge = RunBridge(TASK, env, Budget(total_tokens=5000), usage, {})
    try:
        request_id = bridge.begin_model_request("first")
        bridge.finish_model_request(request_id, None, None, responded=True)
        assert env.state.request_attempts == env.state.request_responses == 1
        assert env.state.usage_complete is False
        assert env.state.unreported_token_reserve > 0
        assert usage.total_tokens == 0
        with pytest.raises(UsageLimitExceeded):
            bridge.begin_model_request("second")
    finally:
        env.close()


async def test_strands_reports_native_budget_failure_as_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("strands")
    from pydantic_ai.exceptions import UsageLimitExceeded

    from poc.frameworks.common import RunBridge
    from poc.frameworks.strands_background import run

    monkeypatch.setenv("POC_TEST_OPENAI_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("POC_TEST_OPENAI_KEY", "fixture")
    spec = ModelSpec(
        name="fixture",
        model_class="baseline",
        model="fixture",
        base_url_env="POC_TEST_OPENAI_URL",
        api_key_env="POC_TEST_OPENAI_KEY",
    )
    env = TaskEnvironment(TASK, 10)
    bridge = RunBridge(TASK, env, Budget(total_tokens=1000), RunUsage(), {}, spec)
    with pytest.raises(UsageLimitExceeded):
        await run(bridge)


async def test_llamaindex_reports_native_budget_failure_as_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("llama_index")
    from pydantic_ai.exceptions import UsageLimitExceeded

    from poc.frameworks.common import RunBridge
    from poc.frameworks.llamaindex_workflows import run

    monkeypatch.setenv("POC_TEST_OPENAI_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("POC_TEST_OPENAI_KEY", "fixture")
    spec = ModelSpec(
        name="fixture",
        model_class="baseline",
        model="fixture",
        base_url_env="POC_TEST_OPENAI_URL",
        api_key_env="POC_TEST_OPENAI_KEY",
    )
    env = TaskEnvironment(TASK, 10)
    bridge = RunBridge(TASK, env, Budget(total_tokens=1000), RunUsage(), {}, spec)
    with pytest.raises(UsageLimitExceeded):
        await run(bridge)


def test_framework_result_accepts_direct_json_object() -> None:
    from pydantic import ValidationError

    from poc.frameworks.common import result_answer

    assert result_answer(
        '{"affected_tenants":[],"changes":[],"evidence":[],"explanation":"x"}'
    ).values == {
        "affected_tenants": [],
        "changes": [],
        "evidence": [],
        "explanation": "x",
    }
    assert result_answer('{"values":{"x":7}}').values == {"x": 7}
    with pytest.raises(ValidationError):
        result_answer("[1,2,3]")


async def test_pydantic_cancelled_model_request_marks_usage_incomplete() -> None:
    pytest.importorskip("pydantic_ai_harness")
    from pydantic_ai import Agent

    from poc.frameworks.common import RunBridge
    from poc.frameworks.pydantic_background import TracedModel

    started = asyncio.Event()
    release = asyncio.Event()

    async def respond(messages: list[Any], info: AgentInfo) -> ModelResponse:
        started.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    env = TaskEnvironment(TASK, 10)
    bridge = RunBridge(TASK, env, Budget(), RunUsage(), {})
    scope = bridge.start_task("root", TASK.prompt)
    agent = Agent(TracedModel(FunctionModel(respond), bridge, scope))
    task = asyncio.create_task(agent.run("test"))
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert env.state.request_attempts == 1
        assert env.state.request_responses == 0
        assert env.state.usage_complete is False
    finally:
        release.set()
        env.close()
