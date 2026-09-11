"""Execution runtimes and pluggable scheduling strategies."""

from poc.execution.agent_executor import (
    AgentExecutionRequest,
    AgentExecutor,
    AgentExecutorNotRegistered,
    AgentExecutorRegistry,
    CustomPythonAgentExecutor,
)
from poc.execution.board_claim import BoardClaimStrategy, Claim, StaleClaim
from poc.execution.capacity import (
    CapacityScheduler,
    InProcessWorkerMaterializer,
    WorkerMaterializer,
)
from poc.execution.hierarchical_strategy import HierarchicalDAGStrategy
from poc.execution.managed_pool import Assignment, ManagedPoolStrategy, StaleAssignment
from poc.execution.pydantic_ai_executor import (
    PydanticAgentConfigurationError,
    PydanticAIAgentExecutor,
    PydanticModelFactory,
    PydanticWorkerOutput,
)
from poc.execution.speculative import CandidateGrant, ReconciliationDecision, SpeculativeStrategy
from poc.execution.strategy import (
    ExecutionCoordinator,
    ExecutionStrategy,
    PersistentExecutionStrategy,
    StrategyError,
    StrategyNotRegistered,
    StrategyRegistry,
)

__all__ = [
    "AgentExecutionRequest",
    "AgentExecutor",
    "AgentExecutorNotRegistered",
    "AgentExecutorRegistry",
    "Assignment",
    "BoardClaimStrategy",
    "CandidateGrant",
    "CapacityScheduler",
    "Claim",
    "CustomPythonAgentExecutor",
    "ExecutionCoordinator",
    "ExecutionStrategy",
    "HierarchicalDAGStrategy",
    "InProcessWorkerMaterializer",
    "ManagedPoolStrategy",
    "PersistentExecutionStrategy",
    "PydanticAIAgentExecutor",
    "PydanticAgentConfigurationError",
    "PydanticModelFactory",
    "PydanticWorkerOutput",
    "ReconciliationDecision",
    "SpeculativeStrategy",
    "StaleAssignment",
    "StaleClaim",
    "StrategyError",
    "StrategyNotRegistered",
    "StrategyRegistry",
    "WorkerMaterializer",
]
