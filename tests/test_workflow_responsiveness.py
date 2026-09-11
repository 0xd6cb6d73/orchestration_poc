from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from poc.execution.workflow_compiler import WorkflowCompiler, WorkflowState
from poc.execution.workflow_runner import WorkflowRunner
from poc.models import WorkflowSpec
from poc.persistence.database import Database


class _BlockingGraph:
    def __init__(self, started: threading.Event, release: threading.Event):
        self.started = started
        self.release = release

    def invoke(
        self,
        input: WorkflowState | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.started.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError("test graph was not released")
        return {"run_id": input.get("run_id", "test-run") if input else "test-run", "results": {}}


class _BlockingCompiler:
    def __init__(self, graph: _BlockingGraph):
        self.graph = graph

    def compile(self, spec: WorkflowSpec, checkpointer: Any) -> _BlockingGraph:
        return self.graph


@pytest.mark.asyncio
async def test_synchronous_graph_does_not_block_asyncio_loop(tmp_path: Path) -> None:
    db = Database(tmp_path / "application.sqlite")
    started = threading.Event()
    release = threading.Event()
    graph = _BlockingGraph(started, release)
    runner = WorkflowRunner(
        db,
        cast(WorkflowCompiler, _BlockingCompiler(graph)),
        tmp_path / "checkpoints.sqlite",
    )
    spec = WorkflowSpec(
        workflow_id="blocking-workflow",
        run_id="test-run",
        owner="test-owner",
        approved_plan_version=1,
        authorized_worker_roles=[],
        tasks=[],
    )
    timer = threading.Timer(0.5, release.set)
    timer.start()
    loop = asyncio.get_running_loop()
    before = loop.time()
    try:
        execution = asyncio.create_task(runner.start(spec))
        await asyncio.sleep(0.01)
        assert started.is_set()
        assert loop.time() - before < 0.25
        release.set()
        await asyncio.wait_for(execution, timeout=1)
    finally:
        timer.cancel()
        release.set()
        runner.close()
        db.close()
