import asyncio
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from poc.api.app import create_app


@pytest.mark.asyncio
async def test_api_approval_status_page_and_artifact(tmp_path: Path) -> None:
    app = create_app(tmp_path / "api-data")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        created = await client.post("/runs", json={})
        assert created.status_code == 201
        body = cast(dict[str, Any], created.json())
        run_id = body["run"]["run_id"]
        approved = await client.post(body["approval_url"], json={"plan_version": 1})
        assert approved.status_code == 200
        status: dict[str, Any] = {}
        for _ in range(100):
            status = cast(dict[str, Any], (await client.get(body["status_url"])).json())
            if status["run"]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status["run"]["status"] == "completed"
        page = await client.get(f"/ui/runs/{run_id}")
        assert page.status_code == 200
        assert "Approved plan" in page.text
        assert "Agent hierarchy" in page.text
        assert "/static/run.js" in page.text
        events = cast(list[dict[str, Any]], status["events"])
        artifact_id = cast(
            str,
            next(
                event["data"]["artifact_id"]
                for event in events
                if event["event_type"] == "deliverable.accepted"
            ),
        )
        artifact = await client.get(f"/artifacts/{artifact_id}")
        assert artifact.status_code == 200
        assert artifact.headers["etag"]
        assert "Checkout latency regression" in artifact.text


@pytest.mark.asyncio
async def test_web_ui_drives_run_and_exposes_agent_relationships(tmp_path: Path) -> None:
    app = create_app(tmp_path / "ui-data")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        root = await client.get("/")
        assert root.status_code == 307
        assert root.headers["location"] == "/ui"

        dashboard = await client.get("/ui")
        assert dashboard.status_code == 200
        assert "Agent Operations" in dashboard.text
        assert 'action="/ui/runs"' in dashboard.text
        assert 'name="agent_backend"' in dashboard.text
        assert 'value="pydantic_ai"' in dashboard.text
        assert 'value="semantic_pydantic_ai"' in dashboard.text
        assert 'name="execution_mode"' in dashboard.text
        assert 'name="agent_model"' in dashboard.text

        created = await client.post(
            "/ui/runs",
            data={
                "objective": "Trace the checkout latency regression.",
                "agent_backend": "custom_python",
                "execution_mode": "managed_pool",
            },
        )
        assert created.status_code == 303
        run_url = created.headers["location"]
        run_id = run_url.rsplit("/", 1)[-1]

        proposed = await client.get(run_url)
        assert proposed.status_code == 200
        assert "Review and approve the plan" in proposed.text
        run_script = (await client.get("/static/run.js")).text
        assert "Agents appear after the plan is approved" in run_script
        assert "AbortController" in run_script
        assert "refreshInFlight" in run_script

        approved = await client.post(
            f"/ui/runs/{run_id}/approval",
            data={"plan_version": "1", "constraints": "Offline fixtures only"},
        )
        assert approved.status_code == 303
        assert approved.headers["location"] == run_url

        status: dict[str, Any] = {}
        for _ in range(100):
            status = cast(dict[str, Any], (await client.get(f"/runs/{run_id}")).json())
            if status["run"]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status["run"]["status"] == "completed"
        assert status["plan"]["version"] == 2
        assert status["plan"]["constraints"] == ["Offline fixtures only"]
        assert status["plan"]["agent_runtime"] == {
            "backend": "custom_python",
            "provider": None,
            "model": None,
            "options": {},
        }
        assert status["plan"]["execution_mode"] == "managed_pool"
        assert len(status["executions"]) == 3
        assert all(item["mode"] == "managed_pool" for item in status["executions"])
        assert all(
            item["policy"]["agent_backend"] == "custom_python" for item in status["executions"]
        )

        agents = cast(list[dict[str, Any]], status["agents"])
        main = next(agent for agent in agents if agent["tier"] == "main")
        supervisors = [agent for agent in agents if agent["tier"] == "sub_orchestrator"]
        workers = [agent for agent in agents if agent["tier"] == "worker"]
        assert supervisors
        assert workers
        assert all(agent["parent_agent_id"] == main["agent_instance_id"] for agent in supervisors)
        supervisor_ids = {agent["agent_instance_id"] for agent in supervisors}
        assert all(agent["parent_agent_id"] in supervisor_ids for agent in workers)

        finished_page = await client.get(run_url)
        assert "Live topology" in finished_page.text
        assert "main_orchestrator" in finished_page.text
        stylesheet = await client.get("/static/app.css")
        assert stylesheet.status_code == 200
        assert ".agent-tree" in stylesheet.text


@pytest.mark.asyncio
async def test_web_ui_persists_pydantic_ai_model_selection(tmp_path: Path) -> None:
    app = create_app(tmp_path / "ui-runtime-data")
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        created = await client.post(
            "/ui/runs",
            data={
                "objective": "Evaluate a selected agent runtime.",
                "agent_backend": "pydantic_ai",
                "execution_mode": "managed_pool",
                "agent_provider": "openai",
                "agent_model": "gpt-5",
            },
        )

        assert created.status_code == 303
        run_id = created.headers["location"].rsplit("/", 1)[-1]
        status = cast(dict[str, Any], (await client.get(f"/runs/{run_id}")).json())
        assert status["plan"]["agent_runtime"] == {
            "backend": "pydantic_ai",
            "provider": "openai",
            "model": "gpt-5",
            "options": {},
        }
        assert status["plan"]["execution_mode"] == "managed_pool"
        assert all(area["execution_mode"] == "managed_pool" for area in status["plan"]["areas"])

        page = await client.get(created.headers["location"])
        assert "pydantic_ai" in page.text
        assert "openai:gpt-5" in page.text

        invalid = await client.post(
            "/ui/runs",
            data={
                "objective": "Missing model configuration.",
                "agent_backend": "pydantic_ai",
                "execution_mode": "hierarchical_dag",
            },
        )
        assert invalid.status_code == 422
