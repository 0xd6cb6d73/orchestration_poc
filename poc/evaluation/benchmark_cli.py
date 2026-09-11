from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from poc.evaluation.benchmark import OrchestrationBenchmarkRunner, save_benchmark_report
from poc.evaluation.benchmark_models import (
    OrchestrationBenchmarkReport,
    OrchestrationBenchmarkVariant,
)
from poc.models import AgentBackend, ExecutionMode, SwarmStrategy

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GOOGLE_API_KEY",
    "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_SECRET_SETTING_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "secret",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hierarchical-ooda-benchmark",
        description="Compare orchestration modes on a complete incident-investigation workload.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in ExecutionMode],
        help="Mode to benchmark; repeat to select modes (default: all modes)",
    )
    parser.add_argument(
        "--strategy", choices=[item.value for item in SwarmStrategy], default="board"
    )
    parser.add_argument("--backend", default=AgentBackend.CUSTOM_PYTHON)
    parser.add_argument("--provider", help="Pydantic AI provider prefix")
    parser.add_argument("--model", help="Model name or provider:model identifier")
    parser.add_argument(
        "--model-setting",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="Pydantic AI model setting such as temperature=0; repeat as needed",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path, help="Write machine-readable JSON results")
    parser.add_argument(
        "--include-output",
        action="store_true",
        help="Include generated report text in JSON (hashes are always included)",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code zero even when a quality gate fails",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    modes = [ExecutionMode(value) for value in args.mode] if args.mode else list(ExecutionMode)
    strategy = SwarmStrategy(args.strategy)
    if strategy == SwarmStrategy.HYBRID_V1 and modes != [ExecutionMode.BOARD_CLAIM]:
        parser.error("--strategy hybrid_v1 requires exactly --mode board_claim")
    provider = args.provider
    if provider is None and args.model and ":" in args.model:
        provider = args.model.split(":", 1)[0]
    if args.backend == AgentBackend.PYDANTIC_AI:
        if args.model is None:
            parser.error("the pydantic_ai backend requires --model")
        if provider is None:
            parser.error(
                "the pydantic_ai backend requires --provider or a provider:model identifier"
            )
        _check_provider_environment(parser, provider)
    try:
        settings = _parse_settings(args.model_setting)
        variants = [
            OrchestrationBenchmarkVariant(
                name=_variant_name(mode, strategy, args.backend, provider, args.model),
                execution_mode=mode,
                swarm_strategy=strategy,
                agent_backend=args.backend,
                provider=provider,
                model=args.model,
                model_settings=settings,
            )
            for mode in modes
        ]
    except (ValueError, ValidationError) as exc:
        parser.error(str(exc))

    report = asyncio.run(
        OrchestrationBenchmarkRunner(
            fixture_root=args.fixture_root,
            work_dir=args.work_dir,
        ).evaluate(
            variants,
            repeat=args.repeat,
            max_concurrency=args.max_concurrency,
            include_output=args.include_output,
        )
    )
    _print_report(report)
    if args.report is not None:
        save_benchmark_report(report, args.report)
    passed = bool(report.trials) and all(trial.passed for trial in report.trials)
    return 0 if args.allow_failures or passed else 1


def main() -> None:
    raise SystemExit(run_cli())


def _parse_settings(values: Sequence[str]) -> dict[str, object]:
    settings: dict[str, object] = {}
    for value in values:
        name, separator, raw_value = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --model-setting {value!r}; expected NAME=JSON")
        if name in settings:
            raise ValueError(f"duplicate --model-setting name: {name!r}")
        normalized_name = name.casefold().replace("-", "_")
        if normalized_name in _SECRET_SETTING_NAMES or normalized_name.endswith("_api_key"):
            raise ValueError(
                f"secret-like model setting {name!r} is not accepted; use the provider's "
                "environment variable instead"
            )
        try:
            settings[name] = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in --model-setting {value!r}") from exc
    return settings


def _check_provider_environment(parser: argparse.ArgumentParser, provider: str) -> None:
    variable = _API_KEY_ENV.get(provider)
    if variable is not None and not os.getenv(variable):
        parser.error(
            f"provider {provider!r} requires {variable}; export it before running the benchmark"
        )


def _variant_name(
    mode: ExecutionMode,
    strategy: SwarmStrategy,
    backend: str,
    provider: str | None,
    model: str | None,
) -> str:
    parts: list[str] = [mode.value]
    if strategy != SwarmStrategy.BOARD:
        parts.append(strategy.value)
    parts.append(backend)
    if provider:
        parts.append(provider)
    if model:
        parts.append(model.replace(":", "-"))
    return "-".join(parts)


def _print_report(report: OrchestrationBenchmarkReport) -> None:
    print("variant\tpass\tcomplete\tmean_s\tp95_s\ttools\tworkers\tinput_tokens\toutput_tokens")
    for summary in report.summaries:
        print(
            f"{summary.variant}\t{summary.pass_rate:.0%}\t{summary.completion_rate:.0%}\t"
            f"{summary.mean_elapsed_seconds:.3f}\t{summary.p95_elapsed_seconds:.3f}\t"
            f"{summary.mean_tool_calls:.1f}\t{summary.mean_workers:.1f}\t"
            f"{summary.total_input_tokens}\t{summary.total_output_tokens}"
        )


if __name__ == "__main__":
    main()
