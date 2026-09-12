from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from dotenv import load_dotenv

from poc.evaluation.suite.models import Matrix
from poc.evaluation.suite.phoenix import publish
from poc.evaluation.suite.runner import read_report, run_matrix
from poc.telemetry.bootstrap import configure_telemetry, shutdown_telemetry


def load_environment(env_file: Path | None = None) -> None:
    """Load local credentials as literal values, preserving exported overrides."""
    if env_file is not None and not env_file.is_file():
        raise ValueError(f"environment file does not exist: {env_file}")
    load_dotenv(env_file or Path.cwd() / ".env", override=False, interpolate=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seeded orchestration benchmark matrix")
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="Trusted Python module registering tasks/adapters",
    )
    environment = parser.add_mutually_exclusive_group()
    environment.add_argument(
        "--env-file", type=Path, help="Environment file (default: .env in the working directory)"
    )
    environment.add_argument(
        "--no-env-file", action="store_true", help="Use only exported environment variables"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum concurrent trials (overrides config; default: 4). Use 1 for serial execution",
    )
    run.add_argument("--phoenix", action="store_true", help="Publish results after local save")
    upload = commands.add_parser("publish")
    upload.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.max_concurrency is not None and args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    if not args.no_env_file:
        try:
            load_environment(args.env_file)
        except ValueError as exc:
            parser.error(str(exc))
    for module in args.plugin:
        importlib.import_module(module)
    if args.command == "publish":
        receipt = publish(read_report(args.report))
        args.report.with_suffix(".phoenix.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
        return
    matrix = Matrix.model_validate_json(args.config.read_bytes())
    if args.max_concurrency is not None:
        matrix = Matrix.model_validate(
            {**matrix.model_dump(), "max_concurrency": args.max_concurrency}
        )
    provider = configure_telemetry()
    try:
        report = asyncio.run(run_matrix(matrix, args.output))
    finally:
        shutdown_telemetry(provider)
    print(json.dumps(report["summaries"], indent=2))
    if args.phoenix:
        receipt = publish(report)
        args.output.with_suffix(".phoenix.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt, indent=2))
    # Incorrect answers are expected benchmark outcomes; infrastructure errors are not.
    raise SystemExit(int(any(t["status"] != "completed" for t in report["trials"])))


if __name__ == "__main__":
    main()
