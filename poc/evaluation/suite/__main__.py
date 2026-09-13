from __future__ import annotations

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from dotenv import load_dotenv

from poc.evaluation.diagnostic.cli import add_arguments as diagnostic_arguments
from poc.evaluation.diagnostic.cli import run_cli as run_diagnostic_cli
from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.models import Matrix
from poc.evaluation.suite.phoenix import publish, reconcile
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
    commands.add_parser("list", help="List registered orchestration strategies and task families")
    diagnostic = commands.add_parser(
        "diagnose", help="Fast infrastructure probes, without performance scoring"
    )
    diagnostic_arguments(diagnostic)
    monitor_parser = commands.add_parser(
        "monitor", help="Count trials and expected batch verification files"
    )
    monitor_parser.add_argument("directory", type=Path)
    calibration = commands.add_parser(
        "calibrate", help="Derive role budgets from recorded usage and time"
    )
    calibration.add_argument("reports", type=Path, nargs="+")
    calibration.add_argument("--source", required=True)
    calibration.add_argument("--headroom", type=float, default=1.25)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    for axis in ("models", "strategies", "families", "seeds"):
        run.add_argument(
            f"--{axis}",
            nargs="+",
            action="extend",
            type=int if axis == "seeds" else str,
            help=(
                f"Select configured {axis} (space-separated; repeatable; default: all)"
                + (". Use model names, not provider IDs" if axis == "models" else "")
            ),
        )
    run.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum concurrent trials (overrides config; default: 4). Use 1 for serial execution",
    )
    run.add_argument("--phoenix", action="store_true", help="Publish results after local save")
    run.add_argument(
        "--resume",
        action="store_true",
        help="Run only missing trials in an identical saved manifest",
    )
    upload = commands.add_parser("publish")
    upload.add_argument("report", type=Path)
    inspect = commands.add_parser(
        "inspect", help="Show coverage and matched comparisons without model calls"
    )
    inspect.add_argument("report", type=Path)
    inspect.add_argument(
        "--phoenix",
        action="store_true",
        help="Reconcile against the saved Phoenix receipt (read-only)",
    )
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
    if args.command == "diagnose":
        try:
            status = run_diagnostic_cli(args)
        except (ValueError, FileExistsError) as exc:
            parser.error(str(exc))
        raise SystemExit(status)
    if args.command == "monitor":
        from poc.evaluation.suite.batches import monitor

        print(json.dumps(monitor(args.directory), indent=2))
        return
    if args.command == "calibrate":
        from poc.evaluation.suite.budget_profiles import measured_profiles

        trials = [trial for path in args.reports for trial in read_report(path)["trials"]]
        profiles = measured_profiles(trials, args.source, headroom=args.headroom)
        print(
            json.dumps(
                {
                    model: {role: p.model_dump() for role, p in roles.items()}
                    for model, roles in profiles.items()
                },
                indent=2,
            )
        )
        return
    if args.command == "list":
        from poc.evaluation.suite.tasks import FACTORIES

        print(json.dumps({"strategies": sorted(ADAPTERS), "families": sorted(FACTORIES)}, indent=2))
        return
    if args.command == "inspect":
        report = read_report(args.report)
        result = {k: report[k] for k in ("coverage", "summaries", "comparisons")}
        if args.phoenix:
            from phoenix.client import Client

            receipt = json.loads(args.report.with_suffix(".phoenix.json").read_text())
            result["publication"] = reconcile(report, receipt, Client())
        print(json.dumps(result, indent=2))
        return
    if args.command == "publish":
        receipt = publish(
            read_report(args.report), receipt_path=args.report.with_suffix(".phoenix.json")
        )
        print(json.dumps(receipt, indent=2))
        return
    matrix = Matrix.model_validate_json(args.config.read_bytes())
    selected = matrix.model_dump()
    for axis in ("models", "strategies", "families", "seeds"):
        requested = getattr(args, axis)
        if requested is None:
            continue
        available = [m.name for m in matrix.models] if axis == "models" else selected[axis]
        unknown = set(requested) - set(available)
        if unknown:
            parser.error(
                f"--{axis}: values not in config: {', '.join(map(str, sorted(unknown)))}; "
                f"available: {', '.join(map(str, available))}"
            )
        selected[axis] = [
            value
            for value in selected[axis]
            if (value["name"] if axis == "models" else value) in requested
        ]
    selected["architecture_options"] = {
        key: value
        for key, value in selected["architecture_options"].items()
        if key in selected["strategies"]
    }
    if args.max_concurrency is not None:
        selected["max_concurrency"] = args.max_concurrency
    matrix = Matrix.model_validate(selected)
    provider = configure_telemetry(instrument_providers=False)
    try:
        report = asyncio.run(run_matrix(matrix, args.output, resume=args.resume))
    finally:
        shutdown_telemetry(provider)
    print(json.dumps({k: report[k] for k in ("coverage", "summaries", "comparisons")}, indent=2))
    if args.phoenix:
        receipt = publish(report, receipt_path=args.output.with_suffix(".phoenix.json"))
        print(json.dumps(receipt, indent=2))
    # Incorrect answers are expected benchmark outcomes; infrastructure errors are not.
    raise SystemExit(int(any(t["status"] != "completed" for t in report["trials"])))


if __name__ == "__main__":
    main()
