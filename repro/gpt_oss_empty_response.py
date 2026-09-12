"""Standalone GPT-OSS tool/output reproducer; no orchestration or Phoenix required.

Run with the repository's .venv/bin/python. Default mode is an offline synthetic
replay of the observed response shape; --mode live uses OPENAI_* from .env.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
from importlib.metadata import version
from typing import Any

import httpx2 as httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: dict[str, Any]


async def run(mode: str, model_name: str) -> int:
    calls = 0
    empty_responses = 0

    async def replay(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        if calls > 1:
            assert any(m["role"] == "tool" for m in body["messages"])
        message: dict[str, Any] = {"role": "assistant", "content": None}
        if calls == 1:
            message["tool_calls"] = [
                {
                    "id": "query-1",
                    "type": "function",
                    "function": {"name": "query", "arguments": '{"sql":"SELECT * FROM jobs"}'},
                }
            ]
        elif mode == "control":
            message["tool_calls"] = [
                {
                    "id": "answer-1",
                    "type": "function",
                    "function": {"name": "final_result", "arguments": '{"values":{"task-0000":0}}'},
                }
            ]
        else:
            # Synthetic shape, not a verbatim provider response or reasoning transcript.
            message["reasoning"] = 'Inspect limits next: {"sql":"SELECT * FROM limits"}'
        return httpx.Response(
            200,
            json={
                "id": f"replay-{calls}",
                "object": "chat.completion",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    async def observe(response: httpx.Response) -> None:
        nonlocal empty_responses
        await response.aread()
        if response.status_code != 200:
            print(json.dumps({"http_status": response.status_code}))
            return
        body = response.json()
        for choice in body.get("choices", []):
            message = choice.get("message", {})
            if choice.get("finish_reason") == "stop" and not (
                message.get("content") or message.get("tool_calls")
            ):
                empty_responses += 1
            print(
                json.dumps(
                    {
                        "finish_reason": choice.get("finish_reason"),
                        "content_chars": len(message.get("content") or ""),
                        "reasoning_present": any(
                            message.get(k)
                            for k in ("reasoning", "reasoning_content", "reasoning_details")
                        ),
                        "tool_names": [
                            c.get("function", {}).get("name")
                            for c in message.get("tool_calls") or []
                        ],
                    }
                )
            )

    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE jobs(id TEXT, release INT);"
        "INSERT INTO jobs VALUES ('task-0000', 0);"
        "CREATE TABLE limits(horizon INT); INSERT INTO limits VALUES (10);"
    )

    async def query(sql: str) -> dict[str, Any]:
        """Read-only SQLite query."""
        print(json.dumps({"sql": sql}))
        try:
            cursor = db.execute(sql)
            return {"columns": [c[0] for c in cursor.description], "rows": cursor.fetchmany(200)}
        except sqlite3.Error as exc:
            return {"error": str(exc)}

    # Restrict live model SQL to reads, without importing the application environment.
    db.set_authorizer(
        lambda action, *_: (
            sqlite3.SQLITE_OK
            if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION)
            else sqlite3.SQLITE_DENY
        )
    )
    try:
        async with httpx.AsyncClient(
            transport=None if mode == "live" else httpx.MockTransport(replay),
            event_hooks={"response": [observe]},
            timeout=30,
            trust_env=mode == "live",
        ) as http:
            provider = OpenAIProvider(
                base_url=os.environ["OPENAI_BASE_URL"]
                if mode == "live"
                else "https://replay.invalid/v1",
                api_key=os.environ["OPENAI_API_KEY"] if mode == "live" else "offline",
                http_client=http,
            )
            agent = Agent(
                OpenAIChatModel(model_name, provider=provider),
                output_type=Answer,
                tools=[query],
                retries={"output": 1},
                model_settings={"temperature": 0},
                instructions="Use SQL to inspect the tables, then return the complete values mapping.",
            )
            try:
                async with asyncio.timeout(60):
                    result = await agent.run(
                        "Execute SELECT * FROM jobs and SELECT * FROM limits using query. "
                        "The schema is jobs(id, release), limits(horizon); do not inspect schema. "
                        "Return each job ID "
                        "mapped to its release time, provided it is below the horizon.",
                        usage_limits=UsageLimits(request_limit=6, tool_calls_limit=6),
                    )
            except UnexpectedModelBehavior as exc:
                if empty_responses < 2 or "Exceeded maximum output retries (1)" not in str(exc):
                    print("INCONCLUSIVE: different UnexpectedModelBehavior")
                    return 2
                print("REPRODUCED: UnexpectedModelBehavior (see response shapes above)")
                if mode == "replay":
                    assert calls == 3, f"Expected query plus two empty outputs, got {calls}"
                return 1 if mode == "control" else 0
            except Exception as exc:
                # Do not print provider exception bodies, headers, or credentials.
                print(f"INCONCLUSIVE: {type(exc).__name__}")
                return 2
            print("ANSWER:", result.output.model_dump_json())
            if mode == "replay":
                print("FAIL: expected output retry exhaustion")
                return 1
            if result.output.values != {"task-0000": 0}:
                print("FAIL: unexpected answer")
                return 1
            print("PASS: failure did not occur")
            return 0
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["replay", "control", "live"], default="replay")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    args = parser.parse_args()
    if args.mode == "live":
        load_dotenv()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "model": args.model,
                "pydantic-ai-slim": version("pydantic-ai-slim"),
                "openai": version("openai"),
            }
        )
    )
    raise SystemExit(asyncio.run(run(args.mode, args.model)))
