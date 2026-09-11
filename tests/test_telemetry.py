import logging
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from typing import Any

from poc.telemetry import bootstrap


class _BrokenInstrumentor:
    def instrument(self, *, tracer_provider: Any) -> None:
        raise ModuleNotFoundError("anthropic.resources.completions")


def _installed_package(package: str) -> ModuleSpec:
    return ModuleSpec(package, loader=None)


def _broken_instrumentation_module(module: str) -> Any:
    return SimpleNamespace(AnthropicInstrumentor=_BrokenInstrumentor)


def test_incompatible_optional_provider_does_not_disable_application(
    monkeypatch: Any, caplog: Any
) -> None:
    monkeypatch.setattr(bootstrap, "find_spec", _installed_package)
    monkeypatch.setattr(
        bootstrap,
        "import_module",
        _broken_instrumentation_module,
    )

    with caplog.at_level(logging.WARNING):
        instrumented = bootstrap._instrument_optional_provider(  # pyright: ignore[reportPrivateUsage]
            package="anthropic",
            module="openinference.instrumentation.anthropic",
            instrumentor_name="AnthropicInstrumentor",
            provider=object(),
        )

    assert not instrumented
    assert "Skipping incompatible anthropic telemetry instrumentation" in caplog.text
