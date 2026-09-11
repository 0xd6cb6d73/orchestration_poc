from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import ALWAYS_ON


def configure_telemetry() -> Any | None:
    """Configure Phoenix once when enabled; keep local development fully offline."""
    if os.getenv("PHOENIX_ENABLED", "0").lower() not in {"1", "true", "yes"}:
        return None
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError(
            "Install telemetry dependencies with `uv sync --extra telemetry`"
        ) from exc

    provider = register(
        project_name="hierarchical-ooda-poc",
        endpoint=os.getenv("PHOENIX_OTLP_HTTP_ENDPOINT", "http://localhost:6006/v1/traces"),
        protocol="http/protobuf",
        batch=True,
        auto_instrument=False,
        sampler=ALWAYS_ON,
        resource=Resource.create(
            {
                "service.name": "hierarchical-ooda-poc",
                "service.version": os.getenv("APP_REVISION", "development"),
            }
        ),
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    # Provider SDKs are optional. Instrument them only when an adapter actually installs them.
    if find_spec("openai") is not None:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=provider)
    if find_spec("anthropic") is not None:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        AnthropicInstrumentor().instrument(tracer_provider=provider)
    return provider


def shutdown_telemetry(provider: Any | None) -> None:
    if provider is not None:
        provider.force_flush()
        provider.shutdown()
