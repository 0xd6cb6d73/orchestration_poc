from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from poc.control.runtime import Runtime
from poc.execution.agent_executor import AgentExecutorNotRegistered
from poc.models import (
    AgentBackend,
    AgentRuntimeConfig,
    ApprovalRequest,
    ExecutionMode,
    PlanEdit,
    RunCreate,
)
from poc.telemetry.bootstrap import configure_telemetry, shutdown_telemetry


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    data_path = Path(data_dir or os.getenv("POC_DATA_DIR", "var"))
    telemetry_provider = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        nonlocal telemetry_provider
        telemetry_provider = configure_telemetry()
        app.state.runtime = Runtime(data_path)
        await app.state.runtime.recover()
        yield
        await app.state.runtime.close()
        shutdown_telemetry(telemetry_provider)

    app = FastAPI(title="Hierarchical OODA Incident PoC", version="0.1.0", lifespan=lifespan)
    api_dir = Path(__file__).parent
    templates = Jinja2Templates(directory=str(api_dir / "templates"))
    static_dir = api_dir / "static"
    static_assets = {
        "app.css": ("text/css", (static_dir / "app.css").read_text()),
        "index.js": ("text/javascript", (static_dir / "index.js").read_text()),
        "run.js": ("text/javascript", (static_dir / "run.js").read_text()),
    }

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/static/{asset_name}", include_in_schema=False)
    async def static_asset(asset_name: str) -> Response:
        try:
            media_type, content = static_assets[asset_name]
        except KeyError:
            raise HTTPException(404, "static asset not found") from None
        return Response(
            content,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui", status_code=307)

    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard(request: Request) -> Response:
        runtime = cast(Runtime, request.app.state.runtime)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "runs": runtime.db.list_runs(),
                "default_objective": RunCreate().objective,
                "agent_backends": sorted(runtime.agent_executors.backends),
                "execution_modes": sorted(mode.value for mode in runtime.strategies.modes),
                "default_backend": AgentBackend.CUSTOM_PYTHON,
                "default_execution_mode": ExecutionMode.HIERARCHICAL_DAG,
            },
        )

    @app.post("/ui/runs", include_in_schema=False)
    async def create_run_from_ui(
        request: Request,
        objective: Annotated[str, Form(min_length=1)],
        agent_backend: Annotated[str, Form(min_length=1)],
        execution_mode: Annotated[ExecutionMode, Form()],
        agent_provider: Annotated[str | None, Form()] = None,
        agent_model: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        runtime = cast(Runtime, request.app.state.runtime)
        normalized_objective = objective.strip()
        if not normalized_objective:
            raise HTTPException(422, "objective must not be blank")
        normalized_provider = agent_provider.strip() if agent_provider else None
        normalized_model = agent_model.strip() if agent_model else None
        try:
            agent_runtime = AgentRuntimeConfig(
                backend=agent_backend,
                provider=normalized_provider,
                model=normalized_model,
            )
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            runtime.agent_executors.get(agent_runtime.backend)
        except AgentExecutorNotRegistered as exc:
            raise HTTPException(422, str(exc)) from exc
        run, _ = runtime.create_run(
            RunCreate(
                objective=normalized_objective,
                agent_runtime=agent_runtime,
                execution_mode=execution_mode,
            )
        )
        return RedirectResponse(f"/ui/runs/{run['run_id']}", status_code=303)

    @app.post("/ui/runs/{run_id}/approval", include_in_schema=False)
    async def approve_from_ui(
        run_id: str,
        request: Request,
        plan_version: Annotated[int, Form(ge=1)],
        constraints: Annotated[str | None, Form()] = None,
    ) -> RedirectResponse:
        runtime = cast(Runtime, request.app.state.runtime)
        parsed_constraints = (
            [line.strip() for line in constraints.splitlines() if line.strip()]
            if constraints is not None
            else None
        )
        current_plan = runtime.db.get_plan(run_id)
        if current_plan is None:
            raise HTTPException(404, "run not found")
        constraints_changed = (
            parsed_constraints is not None and parsed_constraints != current_plan.constraints
        )
        approval = ApprovalRequest(
            plan_version=plan_version,
            edits=PlanEdit(constraints=parsed_constraints) if constraints_changed else None,
        )
        try:
            await runtime.approve(run_id, approval)
        except KeyError:
            raise HTTPException(404, "run not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return RedirectResponse(f"/ui/runs/{run_id}", status_code=303)

    @app.post("/ui/runs/{run_id}/cancel", include_in_schema=False)
    async def cancel_from_ui(run_id: str, request: Request) -> RedirectResponse:
        runtime = cast(Runtime, request.app.state.runtime)
        if not runtime.db.request_cancellation(run_id):
            raise HTTPException(404, "run not found")
        task = runtime.run_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        return RedirectResponse(f"/ui/runs/{run_id}", status_code=303)

    @app.post("/runs", status_code=201)
    async def create_run(request: RunCreate, http_request: Request) -> dict[str, Any]:
        runtime = cast(Runtime, http_request.app.state.runtime)
        run, plan = runtime.create_run(request)
        return {
            "run": run,
            "proposed_plan": plan.model_dump(mode="json"),
            "approval_url": f"/runs/{run['run_id']}/approval",
            "status_url": f"/runs/{run['run_id']}",
        }

    @app.post("/runs/{run_id}/approval")
    async def approve(
        run_id: str, request: ApprovalRequest, http_request: Request
    ) -> dict[str, Any]:
        runtime = cast(Runtime, http_request.app.state.runtime)
        try:
            plan = await runtime.approve(run_id, request)
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
    async def get_run(run_id: str, http_request: Request) -> dict[str, Any]:
        runtime = cast(Runtime, http_request.app.state.runtime)
        try:
            return runtime.status(run_id)
        except KeyError:
            raise HTTPException(404, "run not found") from None

    @app.delete("/runs/{run_id}", status_code=202)
    async def cancel(run_id: str, http_request: Request) -> dict[str, Any]:
        runtime = cast(Runtime, http_request.app.state.runtime)
        if not runtime.db.request_cancellation(run_id):
            raise HTTPException(404, "run not found")
        task = runtime.run_tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        return {"run_id": run_id, "status": "cancelled", "authority_revoked": True}

    @app.get("/artifacts/{artifact_id}")
    async def artifact(artifact_id: str, http_request: Request) -> Response:
        try:
            runtime = cast(Runtime, http_request.app.state.runtime)
            record, content = runtime.artifacts.read(artifact_id)
        except KeyError:
            raise HTTPException(404, "artifact not found") from None
        return Response(
            content,
            media_type=record.media_type,
            headers={"ETag": f'"{record.sha256}"', "Cache-Control": "public, immutable"},
        )

    @app.get("/ui/runs/{run_id}", response_class=HTMLResponse)
    async def run_page(run_id: str, request: Request) -> Response:
        runtime = cast(Runtime, request.app.state.runtime)
        try:
            status = runtime.status(run_id)
        except KeyError:
            raise HTTPException(404, "run not found") from None
        return templates.TemplateResponse(request, "run.html", {"status": status, "run_id": run_id})

    return app


app = create_app()
