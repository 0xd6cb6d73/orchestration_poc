"""Variant-driven agent evaluation infrastructure built on Pydantic Evals."""

from poc.evaluation.benchmark import (
    OrchestrationBenchmarkRunner,
    incident_investigation_workload,
    save_benchmark_report,
)
from poc.evaluation.benchmark_models import (
    OrchestrationBenchmarkReport,
    OrchestrationBenchmarkSummary,
    OrchestrationBenchmarkTrial,
    OrchestrationBenchmarkVariant,
    OrchestrationWorkload,
    TokenUsage,
)
from poc.evaluation.datasets import (
    AgentEvaluationCase,
    AgentEvaluationDataset,
    AgentEvaluationDatasetRegistry,
    create_agent_dataset,
    default_dataset_registry,
    incident_worker_dataset,
)
from poc.evaluation.evaluators import (
    ForbiddenToolsAvoided,
    RequiredToolsUsed,
    ResultContainsExpected,
    SuccessfulOutcome,
    ToolArgumentsContainExpected,
    ToolCallBudgetRespected,
    ToolCallsSucceeded,
    default_evaluators,
)
from poc.evaluation.models import (
    AgentEvaluationExpected,
    AgentEvaluationInput,
    AgentEvaluationOutput,
    AgentEvaluationVariant,
    AgentToolCall,
)
from poc.evaluation.runner import (
    AgentEvaluationReport,
    AgentEvaluationRunner,
    load_report,
    report_passed,
    save_report,
)

__all__ = [
    "AgentEvaluationCase",
    "AgentEvaluationDataset",
    "AgentEvaluationDatasetRegistry",
    "AgentEvaluationExpected",
    "AgentEvaluationInput",
    "AgentEvaluationOutput",
    "AgentEvaluationReport",
    "AgentEvaluationRunner",
    "AgentEvaluationVariant",
    "AgentToolCall",
    "ForbiddenToolsAvoided",
    "OrchestrationBenchmarkReport",
    "OrchestrationBenchmarkRunner",
    "OrchestrationBenchmarkSummary",
    "OrchestrationBenchmarkTrial",
    "OrchestrationBenchmarkVariant",
    "OrchestrationWorkload",
    "RequiredToolsUsed",
    "ResultContainsExpected",
    "SuccessfulOutcome",
    "TokenUsage",
    "ToolArgumentsContainExpected",
    "ToolCallBudgetRespected",
    "ToolCallsSucceeded",
    "create_agent_dataset",
    "default_dataset_registry",
    "default_evaluators",
    "incident_investigation_workload",
    "incident_worker_dataset",
    "load_report",
    "report_passed",
    "save_benchmark_report",
    "save_report",
]
