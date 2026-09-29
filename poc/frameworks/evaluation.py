"""Optional bindings from native framework strategies to the existing SQL evaluation suite.

Load with ``--plugin poc.frameworks.evaluation``. The evaluation task and grader stay
outside the framework backends; each backend receives only public task data/tools.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from poc.evaluation.suite.adapters import register_adapter
from poc.evaluation.suite.environment import TaskEnvironment
from poc.execution.sql_contracts import Answer, Budget, TaskInput
from poc.frameworks.common import RunBridge


async def pydantic_background(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    from poc.frameworks.pydantic_background import run

    return await run(
        RunBridge(
            task, env, budget, usage, settings, env.state.model_spec, backend="pydantic_background"
        ),
        model,
    )


async def strands_background(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    from poc.frameworks.strands_background import run

    return await run(
        RunBridge(
            task, env, budget, usage, settings, env.state.model_spec, backend="strands_background"
        )
    )


async def llamaindex_workflows(
    task: TaskInput,
    env: TaskEnvironment,
    model: Model | str,
    settings: ModelSettings,
    budget: Budget,
    usage: RunUsage,
) -> Answer:
    from poc.frameworks.llamaindex_workflows import run

    return await run(
        RunBridge(
            task, env, budget, usage, settings, env.state.model_spec, backend="llamaindex_workflows"
        )
    )


# The native clients construct their own provider from the same ModelSpec.
strands_background.native_model_spec = True  # type: ignore[attr-defined]
llamaindex_workflows.native_model_spec = True  # type: ignore[attr-defined]

register_adapter("pydantic_background", pydantic_background)
register_adapter("strands_background", strands_background)
register_adapter("llamaindex_workflows", llamaindex_workflows)
