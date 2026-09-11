from __future__ import annotations

import logging
import os
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

logger = logging.getLogger(__name__)


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
    _instrument_optional_provider(
        package="openai",
        module="openinference.instrumentation.openai",
        instrumentor_name="OpenAIInstrumentor",
        provider=provider,
    )
    _instrument_optional_provider(
        package="anthropic",
        module="openinference.instrumentation.anthropic",
        instrumentor_name="AnthropicInstrumentor",
        provider=provider,
    )
    return provider


def _instrument_optional_provider(
    *, package: str, module: str, instrumentor_name: str, provider: Any
) -> bool:
    """Instrument an optional SDK without making telemetry an availability dependency."""
    if find_spec(package) is None:
        return False
    try:
        instrumentor = getattr(import_module(module), instrumentor_name)()
        instrumentor.instrument(tracer_provider=provider)
    except Exception as exc:
        logger.warning(
            "Skipping incompatible %s telemetry instrumentation: %s",
            package,
            exc,
        )
        return False
    return True


def shutdown_telemetry(provider: Any | None) -> None:
    if provider is not None:
        provider.force_flush()
        provider.shutdown()
