from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from poc.evaluation.suite.models import ArchitectureOptions, Budget, Matrix, ModelSpec
from poc.evaluation.suite.reports import atomic_json, coverage, digest, locked, trial_key
from poc.evaluation.suite.runner import make_cases


def _runs(client: Any, experiment_id: str) -> list[dict[str, Any]]:
    """Read bounded pages of runs without fetching unrelated evaluation or dataset records."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        response = client._client.get(  # SDK transport; no public paginated runs method.
            f"v1/experiments/{experiment_id}/runs",
            params={"limit": 1000, **({"cursor": cursor} if cursor else {})},
        )
        response.raise_for_status()
        page = response.json()
        rows.extend(page["data"])
        cursor = page.get("next_cursor")
        if not cursor:
            return rows


def _trial_output(run: dict[str, Any]) -> Any:
    output = run["output"]
    if isinstance(output, dict) and set(cast(dict[str, Any], output)) == {"task_output"}:
        return cast(dict[str, Any], output)["task_output"]
    return cast(Any, output)


def _identity(report: dict[str, Any]) -> dict[str, Any]:
    return {
        k: report[k]
        for k in ("config", "cases", "suite_sha256", "implementation_sha256", "report_id")
        if k in report
    }


def reconcile(report: dict[str, Any], receipt: dict[str, Any], client: Any) -> dict[str, Any]:
    """Read-only comparison of local and remote terminal records, not a model rerun."""
    local = {trial_key(t): t for t in report["trials"]}
    remote: dict[str, Any] = {}
    for exp_id in receipt["experiments"].values():
        for run in _runs(client, exp_id):
            output = _trial_output(run)
            remote[trial_key(output)] = output
    return {
        **coverage(report),
        "remote_recorded": len(remote),
        "unpublished": sorted(local.keys() - remote.keys()),
        "remote_only": sorted(remote.keys() - local.keys()),
        "mismatched": sorted(
            k for k in local.keys() & remote.keys() if digest(local[k]) != digest(remote[k])
        ),
    }


def publish(
    report: dict[str, Any], client: Any = None, *, receipt_path: Path | None = None
) -> dict[str, Any]:
    """Publish saved trials. A durable receipt makes retries resume the same publication."""
    if client is None:
        from phoenix.client import Client

        client = Client()
    if receipt_path is not None:
        with locked(receipt_path):
            return _publish(report, client, receipt_path)
    return _publish(report, client, None)


def _publish(report: dict[str, Any], client: Any, receipt_path: Path | None) -> dict[str, Any]:
    matrix = Matrix.model_validate(report["config"])
    cases = make_cases(matrix)
    if {c.id: c.digest for c in cases} != {c["id"]: c["sha256"] for c in report["cases"]}:
        raise ValueError("generator changed since this report; refusing mismatched dataset")
    fingerprint = digest(_identity(report))
    receipt: dict[str, Any] = {}
    if receipt_path and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("report_fingerprint", fingerprint) != fingerprint:
            raise ValueError("publication receipt belongs to a different report")
    receipt.setdefault("report_fingerprint", fingerprint)
    receipt.setdefault("publication_id", uuid4().hex[:12])
    receipt.setdefault("experiments", {})
    receipt.setdefault("runs", {})
    receipt.setdefault("examples", {})
    suffix = receipt["publication_id"]
    receipt.setdefault("dataset_name", f"orchestration-{report['suite_sha256'][:12]}-{suffix}")

    def save() -> None:
        if receipt_path:
            atomic_json(receipt_path, receipt)

    save()  # Persist deterministic names BEFORE the first remote write.
    if not receipt.get("dataset_id"):
        existing = (
            [d for d in client.datasets.list() if d["name"] == receipt["dataset_name"]]
            if receipt_path
            else []
        )
        if existing:
            dataset = client.datasets.get_dataset(dataset=existing[0]["id"])
        else:
            dataset = client.datasets.create_dataset(
                name=receipt["dataset_name"],
                inputs=[c.input.model_dump() for c in cases],
                outputs=[c.expected for c in cases],
                metadata=[
                    {
                        "case_id": c.id,
                        "sha256": c.digest,
                        "family": c.family,
                        "split": c.split,
                        "difficulty": c.difficulty,
                    }
                    for c in cases
                ],
                dataset_description="Seeded orchestration suite v1; gold is evaluator-only, not model input.",
            )
        receipt.update(dataset_id=dataset.id, dataset_version_id=dataset.version_id)
        receipt["examples"] = {e["metadata"]["case_id"]: e["id"] for e in dataset.examples}
        save()
    if not receipt["examples"]:
        dataset = client.datasets.get_dataset(
            dataset=receipt["dataset_id"], version_id=receipt["dataset_version_id"]
        )
        actual = {e["metadata"]["case_id"]: e["metadata"]["sha256"] for e in dataset.examples}
        if actual != {c.id: c.digest for c in cases}:
            raise ValueError("receipt dataset does not match local cases")
        receipt["examples"] = {e["metadata"]["case_id"]: e["id"] for e in dataset.examples}
        save()
    remote_experiments: list[dict[str, Any]] = (
        client.experiments.list(dataset_id=receipt["dataset_id"]) if receipt_path else []
    )
    fresh: set[str] = set()
    for spec in matrix.models:
        for strategy in matrix.strategies:
            key = f"{spec.name}/{strategy}"
            if key in receipt["experiments"]:
                continue
            name = f"{key}/{suffix}"
            found = next((e for e in remote_experiments if e["name"] == name), None)
            if found:
                exp = found
            else:
                exp = client.experiments.create(
                    dataset_id=receipt["dataset_id"],
                    dataset_version_id=receipt["dataset_version_id"],
                    experiment_name=name,
                    repetitions=matrix.repetitions,
                    experiment_metadata={
                        "model": spec.model_dump(),
                        "strategy": strategy,
                        "budget": matrix.budget.model_dump(),
                        "max_concurrency": matrix.max_concurrency,
                        "suite_sha256": report["suite_sha256"],
                        "implementation_sha256": report["implementation_sha256"],
                        "report_fingerprint": fingerprint,
                        "architecture_options": matrix.architecture_options.get(
                            strategy, ArchitectureOptions()
                        ).model_dump(),
                    },
                )
                fresh.add(exp["id"])
            receipt["experiments"][key] = exp["id"]
            save()
    # Reconcile even when a receipt has run IDs: this detects lost responses and foreign writes.
    remote: dict[tuple[str, str, int], dict[str, Any]] = {}
    if receipt_path:
        expected_meta = {f"{m.name}/{s}": (m, s) for m in matrix.models for s in matrix.strategies}
        if set(receipt["experiments"]) != set(expected_meta):
            raise ValueError("receipt experiments do not match configured matrix")
        for axis, exp_id in receipt["experiments"].items():
            if exp_id not in fresh:
                meta = client.experiments.get(experiment_id=exp_id)
                if meta["dataset_id"] != receipt["dataset_id"]:
                    raise ValueError("receipt experiment belongs to another dataset")
                spec, strategy = expected_meta[axis]
                metadata = meta["metadata"]
                if (
                    ModelSpec.model_validate(metadata["model"]) != spec
                    or metadata["strategy"] != strategy
                    or Budget.model_validate(metadata["budget"]) != matrix.budget
                    or metadata["suite_sha256"] != report["suite_sha256"]
                    or ArchitectureOptions.model_validate(metadata.get("architecture_options", {}))
                    != matrix.architecture_options.get(strategy, ArchitectureOptions())
                ):
                    raise ValueError("receipt experiment configuration differs from report")
                for run in _runs(client, exp_id):
                    remote[exp_id, run["dataset_example_id"], run["repetition_number"]] = run
    for trial in report["trials"]:
        key = trial_key(trial)
        exp_id = receipt["experiments"][f"{trial['model_name']}/{trial['strategy']}"]
        example_id = receipt["examples"][trial["case_id"]]
        existing_run = remote.get((exp_id, example_id, trial["repetition"]))
        if existing_run is not None:
            if digest(_trial_output(existing_run)) != digest(trial):
                raise ValueError(f"remote trial differs from local report: {key}")
            run = existing_run
        else:
            run = client.experiments.log_run(
                experiment_id=exp_id,
                dataset_example_id=example_id,
                output=trial,
                repetition_number=trial["repetition"],
                start_time=datetime.fromisoformat(trial["start_time"]),
                end_time=datetime.fromisoformat(trial["end_time"]),
                trace_id=trial["trace_id"],
                error=trial["error"],
            )
        receipt["runs"][key] = run["id"]
        save()
        scores = dict(trial["scores"])
        diag = trial.get("diagnostics", {})
        if diag.get("answer_valid") is not None:
            scores["answer_valid"] = float(diag["answer_valid"])
        for name, counts in diag.get("dimensions", {}).items():
            scores[f"constraint_{name}"] = counts["passed"] / counts["total"]
        for name, score in scores.items():
            # Phoenix upserts evaluations, including after a lost response.
            client.experiments.log_evaluation(
                experiment_run_id=run["id"], name=name, score=score, annotator_kind="CODE"
            )
    receipt["coverage"] = coverage(report)
    receipt["published_runs"] = len(report["trials"])
    save()
    return receipt
