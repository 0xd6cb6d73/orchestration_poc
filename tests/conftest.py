from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from poc.control.runtime import Runtime


@pytest.fixture
def runtime(tmp_path: Path) -> Generator[Runtime, None, None]:
    instance = Runtime(tmp_path / "data")
    yield instance
    instance.runner.close()
    instance.db.close()


async def finish_run(runtime: Runtime, run_id: str) -> dict[str, Any]:
    await runtime.wait(run_id)
    return runtime.status(run_id)
