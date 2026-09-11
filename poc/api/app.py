from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from poc.control.runtime import Runtime
from poc.models import ApprovalRequest, RunCreate
from poc.telemetry.bootstrap import configure_telemetry, shutdown_telemetry


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    data_path = Path(data_dir or os.getenv("POC_DATA_DIR", "var"))
    telemetry_provider = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal telemetry_provider
        telemetry_provider = configure_telemetry()
        app.state.runtime = Runtime(data_path)
        await app.state.runtime.recover()
        yield
        await app.state.runtime.close()
        shutdown_telemetry(telemetry_provider)

    app = FastAPI(title="Hierarchical OODA Incident PoC", version="0.1.0", lifespan=lifespan)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=201)
    async def create_run(request: RunCreate, http_request: Request) -> dict:
        run, plan = http_request.app.state.runtime.create_run(request)
        return {
            "run": run,
            "proposed_plan": plan.model_dump(mode="json"),
            "approval_url": f"/runs/{run['run_id']}/approval",
            "status_url": f"/runs/{run['run_id']}",
        }

    @app.post("/runs/{run_id}/approval")
    async def approve(run_id: str, request: ApprovalRequest, http_request: Request) -> dict:
        try:
            plan = await http_request.app.state.runtime.approve(run_id, request)
        except KeyError:
            raise HTTPException(404, "run not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {
            "run_id": run_id,
            "status": "running",
            "approved_plan": plan.model_dump(mode="json"),
        }

    @app.get("/runs/{run_id}")
    async def get_run(run_id: str, http_request: Request) -> dict:
        try:
            return http_request.app.state.runtime.status(run_id)
        except KeyError:
            raise HTTPException(404, "run not found") from None

    @app.delete("/runs/{run_id}", status_code=202)
    async def cancel(run_id: str, http_request: Request) -> dict:
        runtime: Runtime = http_request.app.state.runtime
        if not runtime.db.request_cancellation(run_id):
            raise HTTPException(404, "run not found")
        task = runtime.run_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        return {"run_id": run_id, "status": "cancelled", "authority_revoked": True}

    @app.get("/artifacts/{artifact_id}")
    async def artifact(artifact_id: str, http_request: Request) -> Response:
        try:
            record, content = http_request.app.state.runtime.artifacts.read(artifact_id)
        except KeyError:
            raise HTTPException(404, "artifact not found") from None
        return Response(
            content,
            media_type=record.media_type,
            headers={"ETag": f'"{record.sha256}"', "Cache-Control": "public, immutable"},
        )

    @app.get("/ui/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(run_id: str, request: Request):
        try:
            status = request.app.state.runtime.status(run_id)
        except KeyError:
            raise HTTPException(404, "run not found") from None
        return templates.TemplateResponse(request, "run.html", {"status": status, "run_id": run_id})

    return app


app = create_app()
