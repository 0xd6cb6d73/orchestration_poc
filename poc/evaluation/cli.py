from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from poc.evaluation.datasets import default_dataset_registry
from poc.evaluation.models import AgentEvaluationVariant
from poc.evaluation.runner import AgentEvaluationRunner, load_report, report_passed, save_report
from poc.models import AgentBackend, is_pydantic_ai_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hierarchical-ooda-eval",
        description="Evaluate agent prompt, model, and tool variants.",
    )
    parser.add_argument("--dataset", default="incident-workers")
    parser.add_argument("--name", help="Experiment/variant name")
    parser.add_argument("--backend", default=AgentBackend.CUSTOM_PYTHON)
    parser.add_argument("--provider", help="Pydantic AI provider prefix")
    parser.add_argument("--model", help="Model name or provider:model identifier")
    parser.add_argument("--prompt-file", type=Path, help="System prompt override")
    parser.add_argument(
        "--tools",
        action="append",
        default=[],
        metavar="ROLE=TOOL,TOOL",
        help="Replace the allowed tools for one role; repeat for multiple roles",
    )
    parser.add_argument(
        "--execution-limit",
        action="append",
        default=[],
        metavar="NAME=INTEGER",
        help="Override a role execution limit, for example max_tool_calls=2",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path, help="Write the full JSON report")
    parser.add_argument("--baseline", type=Path, help="Compare with a prior JSON report")
    parser.add_argument("--include-output", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code zero even when quality gates fail",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = default_dataset_registry()
    try:
        dataset = registry.get(args.dataset)
        prompt = args.prompt_file.read_text() if args.prompt_file is not None else None
        tools = _parse_tools(args.tools)
        limits = _parse_limits(args.execution_limit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    provider = args.provider
    if provider is None and args.model and ":" in args.model:
        provider = args.model.split(":", 1)[0]
    if is_pydantic_ai_backend(args.backend) and args.model is None:
        parser.error(f"the {args.backend} backend requires --model")
    if is_pydantic_ai_backend(args.backend) and provider is None:
        parser.error(
            f"the {args.backend} backend requires --provider or a provider:model identifier"
        )

    name = args.name or _default_name(args.backend, provider, args.model, args.prompt_file)
    variant = AgentEvaluationVariant(
        name=name,
        agent_backend=args.backend,
        provider=provider,
        model=args.model,
        system_prompt=prompt,
        allowed_tools_by_role=tools,
        execution_limits=limits,
    )
    runner = AgentEvaluationRunner(
        dataset,
        fixture_root=args.fixture_root,
        work_dir=args.work_dir,
    )
    report = asyncio.run(
        runner.evaluate(
            variant,
            repeat=args.repeat,
            max_concurrency=args.max_concurrency,
            progress=not args.no_progress,
        )
    )
    baseline = load_report(args.baseline) if args.baseline is not None else None
    report.print(
        baseline=baseline,
        include_output=args.include_output,
        include_reasons=True,
    )
    if args.report is not None:
        save_report(report, args.report)
    return 0 if args.allow_failures or report_passed(report) else 1


def main() -> None:
    raise SystemExit(run_cli())


def _parse_tools(values: Sequence[str]) -> dict[str, frozenset[str]]:
    parsed: dict[str, frozenset[str]] = {}
    for value in values:
        role, separator, raw_tools = value.partition("=")
        if not separator or not role:
            raise ValueError(f"invalid --tools value {value!r}; expected ROLE=TOOL,TOOL")
        if role in parsed:
            raise ValueError(f"duplicate --tools role: {role!r}")
        parsed[role] = frozenset(tool for tool in raw_tools.split(",") if tool)
    return parsed


def _parse_limits(values: Sequence[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        name, separator, raw_limit = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid --execution-limit value {value!r}; expected NAME=INTEGER")
        if name in parsed:
            raise ValueError(f"duplicate --execution-limit name: {name!r}")
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise ValueError(f"invalid integer in --execution-limit {value!r}") from exc
        if limit < 0:
            raise ValueError("execution limits cannot be negative")
        parsed[name] = limit
    return parsed


def _default_name(
    backend: str,
    provider: str | None,
    model: str | None,
    prompt_file: Path | None,
) -> str:
    parts: list[Any] = [backend]
    if provider:
        parts.append(provider)
    if model:
        parts.append(model)
    if prompt_file:
        parts.append(prompt_file.stem)
    return "-".join(str(part).replace(":", "-") for part in parts)
