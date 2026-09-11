from __future__ import annotations

import pytest

from poc.control.runtime import Runtime


@pytest.fixture
def runtime(tmp_path):
    instance = Runtime(tmp_path / "data")
    yield instance
    instance.runner.close()
    instance.db.close()


async def finish_run(runtime: Runtime, run_id: str) -> dict:
    await runtime.wait(run_id)
    return runtime.status(run_id)
