"""Batch manifest helpers; annotation verification files are never batch completions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def verified_batches(root: Path, groups: list[dict[str, Any]]) -> int:
    return sum((root / f"{group['name']}-verification.json").is_file() for group in groups)


def preflight_batches(root: Path, python: Path, groups: list[dict[str, Any]]) -> None:
    """Validate every config with its target implementation before launching any batch."""
    for group in groups:
        subprocess.run(
            [
                str(python),
                "-c",
                "import sys; from pathlib import Path; from poc.evaluation.suite.models import Matrix; "
                "Matrix.model_validate_json(Path(sys.argv[1]).read_bytes())",
                str(root / f"{group['name']}-config.json"),
            ],
            cwd=group["cwd"],
            check=True,
            capture_output=True,
        )


def monitor(root: Path) -> dict[str, int]:
    groups = json.loads((root / "index.json").read_text())
    trials: list[dict[str, Any]] = []
    for group in groups:
        report = root / f"{group['name']}.jsonl"
        if not report.exists():
            continue
        # Snapshot once. Only a torn final line may be ignored during an active append.
        lines = report.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith(b"\n"):
                    break
                raise
            if "trial" in entry:
                trials.append(entry["trial"])
    return {
        "trials": len(trials),
        "expected_trials": sum(g["trials"] for g in groups),
        "feasible": sum(t["scores"]["exact"] == 1 for t in trials),
        "verified_batches": verified_batches(root, groups),
        "expected_batches": len(groups),
    }
