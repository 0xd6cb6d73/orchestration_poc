"""Run the real benchmark CLI against a deterministic local model endpoint.

The endpoint supplies the task generator's witness. This checks adapter loading,
provider compatibility, report persistence and grading, not model quality.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from aiohttp import web

from poc.evaluation.suite.runner import read_report
from poc.evaluation.suite.tasks import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("backend", "strategy", "package", "openai_major"),
    [
        ("pydantic", "pydantic_background", "pydantic_ai_harness", 3),
        ("strands", "strands_background", "strands", None),
        ("llamaindex", "llamaindex_workflows", "llama_index", 2),
    ],
)
@pytest.mark.parametrize("family", ["scheduling", "routing", "authorization"])
async def test_framework_benchmark_cli(
    tmp_path: Path,
    backend: str,
    strategy: str,
    package: str,
    openai_major: int | None,
    family: str,
) -> None:
    pytest.importorskip(package)
    pytest.importorskip("openai")
    if openai_major is not None and int(version("openai").split(".", 1)[0]) != openai_major:
        pytest.skip(f"{backend} requires the OpenAI SDK from its isolated optional extra")

    case = generate(family, 0, "dev", "standard")
    expected = case.expected["submission"] if family == "authorization" else case.expected
    table = {"scheduling": "schedule_jobs", "routing": "stops", "authorization": "auth_tenants"}[
        family
    ]
    config_name = (
        f"evaluation-framework-authorization-{backend}.json"
        if family == "authorization"
        else f"evaluation-framework-{backend}.json"
    )

    model_calls = 0

    async def completion(request: web.Request) -> web.Response:
        nonlocal model_calls
        model_calls += 1
        body = await request.json()
        history = json.dumps(body["messages"])
        if '"accepted"' in history or '\\"accepted\\"' in history:
            message: dict[str, Any] = {"role": "assistant", "content": "done"}
            reason = "stop"
        else:
            queried = any(item["role"] == "tool" for item in body["messages"])
            tool_name = "finish_task" if queried else "query_sql"
            arguments: dict[str, Any] = (
                {
                    "output": json.dumps(
                        expected if family == "authorization" else {"values": expected}
                    ),
                    "used_result_ids": [],
                }
                if queried
                else {"sql": f"SELECT COUNT(*) AS n FROM {table}"}
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{uuid4().hex[:16]}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
            reason = "tool_calls"
        return web.json_response(
            {
                "id": f"cmpl-{uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "fixture",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completion)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = cast(tuple[str, int], runner.addresses[0])[1]
        output = tmp_path / "report.jsonl"
        env = {
            **os.environ,
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "fixture",
            "PYDANTIC_AI_NO_BANNER": "1",
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "poc.evaluation.suite",
            "--no-env-file",
            "--plugin",
            "poc.frameworks.evaluation",
            "run",
            "--config",
            str(ROOT / "configs" / config_name),
            "--output",
            str(output),
            "--families",
            family,
            "--seeds",
            "0",
            "--max-concurrency",
            "1",
            cwd=ROOT,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 60)
        assert process.returncode == 0, (stdout.decode()[-2000:], stderr.decode()[-2000:])
        report = read_report(output)
        repetitions = 1 if family == "authorization" else 2
        if backend in {"pydantic", "llamaindex"}:
            assert model_calls == 2 * repetitions
        assert report["coverage"] == {
            "expected": repetitions,
            "recorded": repetitions,
            "missing": 0,
            "complete": True,
        }
        for trial in report["trials"]:
            assert trial["strategy"] == strategy
            assert trial["family"] == family
            assert trial["status"] == "completed", trial["error"]
            assert trial["scores"]["exact"] == 1.0
            assert trial["total_tool_calls"] == 1
            assert trial["request_attempts"] == trial["request_responses"] == trial["requests"]
            assert trial["usage_complete"] is True
            assert trial["input_tokens"] > 0
            assert trial["output_tokens"] > 0
    finally:
        await runner.cleanup()
