from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from poc.execution.workflow_compiler import WorkflowCompiler, WorkflowGraph
from poc.models import WorkflowSpec
from poc.persistence.database import Database


class WorkflowRunner:
    def __init__(self, db: Database, compiler: WorkflowCompiler, checkpoint_path: str | Path):
        self.db = db
        self.compiler = compiler
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_connection = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.checkpoint_connection)
        self.checkpointer.setup()
        self._locks: dict[str, asyncio.Lock] = {}
        self._graphs: dict[tuple[str, int], WorkflowGraph] = {}

    def close(self) -> None:
        self.checkpoint_connection.close()

    async def start(self, spec: WorkflowSpec) -> dict[str, Any]:
        thread_id = f"{spec.run_id}:{spec.workflow_id}:r{spec.revision}"
        existing = self.db.get_workflow(spec.workflow_id, spec.revision)
        graph = self.compiler.compile(spec, self.checkpointer)
        self.db.put_workflow(spec, thread_id)
        self._graphs[(spec.workflow_id, spec.revision)] = graph
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
            "max_concurrency": spec.max_workers,
        }
        if existing and existing["status"] == "completed" and existing["result"]:
            return {"run_id": spec.run_id, "results": json.loads(existing["result"])}
        if existing:
            snapshot = graph.get_state(config)
            if snapshot.values:
                interrupts = tuple(
                    interrupt for task in snapshot.tasks for interrupt in task.interrupts
                )
                if interrupts:
                    return {**snapshot.values, "__interrupt__": interrupts}
                if not snapshot.next:
                    result = dict(snapshot.values)
                    self.db.update_workflow(
                        spec.workflow_id, spec.revision, "completed", result.get("results", {})
                    )
                    return result
                invocation: Any = None
            else:
                invocation = {"run_id": spec.run_id, "results": {}}
        else:
            invocation = {"run_id": spec.run_id, "results": {}}
        async with self._locks.setdefault(thread_id, asyncio.Lock()):
            result = graph.invoke(invocation, config)
        if result.get("__interrupt__"):
            self.db.update_workflow(spec.workflow_id, spec.revision, "paused")
        else:
            self.db.update_workflow(
                spec.workflow_id, spec.revision, "completed", result.get("results", {})
            )
        return result

    async def resume(self, spec: WorkflowSpec, resolution: dict[str, Any]) -> dict[str, Any]:
        thread_id = f"{spec.run_id}:{spec.workflow_id}:r{spec.revision}"
        graph = self._graphs.get((spec.workflow_id, spec.revision))
        if graph is None:
            graph = self.compiler.compile(spec, self.checkpointer)
            self._graphs[(spec.workflow_id, spec.revision)] = graph
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
            "max_concurrency": spec.max_workers,
        }
        self.db.update_workflow(spec.workflow_id, spec.revision, "resumed")
        async with self._locks.setdefault(thread_id, asyncio.Lock()):
            result = graph.invoke(Command(resume=resolution), config)
        if result.get("__interrupt__"):
            self.db.update_workflow(spec.workflow_id, spec.revision, "paused")
        else:
            self.db.update_workflow(
                spec.workflow_id, spec.revision, "completed", result.get("results", {})
            )
        return result
