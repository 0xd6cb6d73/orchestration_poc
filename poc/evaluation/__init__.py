"""Variant-driven agent evaluation infrastructure built on Pydantic Evals."""

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
    "RequiredToolsUsed",
    "ResultContainsExpected",
    "SuccessfulOutcome",
    "ToolArgumentsContainExpected",
    "ToolCallBudgetRespected",
    "ToolCallsSucceeded",
    "create_agent_dataset",
    "default_dataset_registry",
    "default_evaluators",
    "incident_worker_dataset",
    "load_report",
    "report_passed",
    "save_report",
]
