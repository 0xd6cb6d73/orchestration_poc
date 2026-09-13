"""CLI shared by the dedicated diagnostic entry point and `suite diagnose`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from poc.evaluation.diagnostic.runner import (
    POLICIES,
    PROTOCOLS,
    STRATEGIES,
    plan_probes,
    run_diagnostics,
    validate_limits,
)
from poc.execution.sql_contracts import ModelSpec


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output", type=Path, help="New JSONL evidence file; existing files are never overwritten"
    )
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument(
        "--live-config",
        type=Path,
        help="Opt into live canaries using one model binding from this config",
    )
    parser.add_argument(
        "--model", help="Required when --live-config contains multiple model bindings"
    )
    parser.add_argument("--case-seconds", type=float, help="Case watchdog (offline: 3; live: 20)")
    parser.add_argument(
        "--suite-seconds", type=float, help="Whole-run deadline (offline: 30; live: 180)"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--list-probes", action="store_true", help="Show planned cases without running them"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress, retaining the final diagnostic summary",
    )


def select_model(config: Path | None, name: str | None) -> ModelSpec | None:
    if config is None:
        if name is not None:
            raise ValueError("--model requires --live-config")
        return None
    data = json.loads(config.read_bytes())
    models = [ModelSpec.model_validate(model) for model in data["models"]]
    if name is None and len(models) != 1:
        raise ValueError(
            "select exactly one model using --model; available: "
            + ", ".join(m.name for m in models)
        )
    selected = [m for m in models if name is None or m.name == name]
    if len(selected) != 1:
        raise ValueError("--model must name exactly one configured model binding")
    return selected[0]


def run_cli(args: argparse.Namespace) -> int:
    model = select_model(args.live_config, args.model)
    planned = plan_probes(
        strategies=args.strategies,
        protocols=args.protocols,
        policies=args.policies,
        live=model is not None,
    )
    limits = {
        "case_seconds": args.case_seconds
        if args.case_seconds is not None
        else (20 if model else 3),
        "suite_seconds": args.suite_seconds
        if args.suite_seconds is not None
        else (180 if model else 30),
        "concurrency": args.concurrency,
    }
    validate_limits(**limits)
    manifest = {
        "schema_version": "infrastructure-diagnostic-v1",
        "purpose": "infrastructure_acceptance",
        "mode": "live_canary" if model else "offline_fault_injection",
        "model_binding": model.model_dump() if model else None,
        "limits": limits,
        "planned_probes": [p.id for p in planned],
    }
    if args.list_probes:
        print(json.dumps(manifest, indent=2))
        return 0
    stream: TextIO | None = None
    written: set[str] = set()

    def append(value: dict[str, Any]) -> None:
        if stream is not None:
            stream.write(json.dumps(value) + "\n")
            stream.flush()

    def progress(row: dict[str, Any]) -> None:
        append({"probe": row})
        written.add(row["id"])
        if not args.quiet and (row["status"] != "pass" or len(written) % 10 == 0):
            print(
                f"Diagnostics {len(written)}/{len(planned)}: {row['id']} {row['status']}",
                file=sys.stderr,
                flush=True,
            )

    try:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            stream = args.output.open("x")
        append({"manifest": manifest})
        report = asyncio.run(
            run_diagnostics(
                strategies=args.strategies,
                protocols=args.protocols,
                policies=args.policies,
                live_model=model,
                progress=progress,
                **limits,
            )
        )
        for row in report["probes"]:
            if row["id"] not in written:
                append({"probe": row})
        summary = {key: value for key, value in report.items() if key != "probes"}
        summary["findings"] = [
            {
                key: row.get(key)
                for key in (
                    "id",
                    "status",
                    "findings",
                    "error_type",
                    "provider_status_code",
                    "failure",
                )
            }
            for row in report["probes"]
            if row["status"] != "pass"
        ]
        append({"summary": summary})
        print(json.dumps(summary, indent=2))
        return int(report["status"] != "pass")
    finally:
        if stream is not None:
            stream.close()
