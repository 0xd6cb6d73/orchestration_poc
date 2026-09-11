import asyncio

import httpx
import pytest

from poc.api.app import create_app


@pytest.mark.asyncio
async def test_api_approval_status_page_and_artifact(tmp_path):
    app = create_app(tmp_path / "api-data")
    async with app.router.lifespan_context(app):
      async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        created = await client.post("/runs", json={})
        assert created.status_code == 201
        body = created.json()
        run_id = body["run"]["run_id"]
        approved = await client.post(body["approval_url"], json={"plan_version": 1})
        assert approved.status_code == 200
        for _ in range(100):
            status = (await client.get(body["status_url"])).json()
            if status["run"]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.01)
        assert status["run"]["status"] == "completed"
        page = await client.get(f"/ui/runs/{run_id}")
        assert page.status_code == 200
        assert "Approved plan" in page.text
        artifact_id = next(e["data"]["artifact_id"] for e in status["events"] if e["event_type"] == "deliverable.accepted")
        artifact = await client.get(f"/artifacts/{artifact_id}")
        assert artifact.status_code == 200
        assert artifact.headers["etag"]
        assert "Checkout latency regression" in artifact.text
