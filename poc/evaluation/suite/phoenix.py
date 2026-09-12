from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from poc.evaluation.suite.models import Matrix
from poc.evaluation.suite.runner import make_cases


def publish(report: dict[str, Any], client: Any = None) -> dict[str, Any]:
    """Publish saved trials without re-running models. Failure never loses local results."""
    if client is None:
        from phoenix.client import Client

        client = Client()  # PHOENIX_BASE_URL and PHOENIX_API_KEY, never stored in reports.
    matrix = Matrix.model_validate(report["config"])
    cases = make_cases(matrix)
    if {c.id: c.digest for c in cases} != {c["id"]: c["sha256"] for c in report["cases"]}:
        raise ValueError("generator changed since this report; refusing mismatched dataset")
    suffix = uuid4().hex[:12]
    dataset = client.datasets.create_dataset(
        name=f"orchestration-{report['suite_sha256'][:12]}-{suffix}",
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
    examples = {e["metadata"]["case_id"]: e["id"] for e in dataset.examples}
    experiments: dict[tuple[str, str], str] = {}
    for spec in matrix.models:
        for strategy in matrix.strategies:
            exp = client.experiments.create(
                dataset_id=dataset.id,
                dataset_version_id=dataset.version_id,
                experiment_name=f"{spec.name}/{strategy}/{suffix}",
                experiment_metadata={
                    "model": spec.model_dump(),
                    "strategy": strategy,
                    "budget": matrix.budget.model_dump(),
                    "max_concurrency": report["config"].get("max_concurrency", 1),
                    "suite_sha256": report["suite_sha256"],
                },
                repetitions=matrix.repetitions,
            )
            experiments[spec.name, strategy] = exp["id"]
    for trial in report["trials"]:
        run = client.experiments.log_run(
            experiment_id=experiments[trial["model_name"], trial["strategy"]],
            dataset_example_id=examples[trial["case_id"]],
            output=trial,
            repetition_number=trial["repetition"],
            start_time=datetime.fromisoformat(trial["start_time"]),
            end_time=datetime.fromisoformat(trial["end_time"]),
            trace_id=trial["trace_id"],
            error=trial["error"],
        )
        for name, score in trial["scores"].items():
            client.experiments.log_evaluation(
                experiment_run_id=run["id"],
                name=name,
                score=score,
                annotator_kind="CODE",
            )
    return {
        "dataset_id": dataset.id,
        "dataset_version_id": dataset.version_id,
        "experiments": {f"{m}/{s}": e for (m, s), e in experiments.items()},
    }
