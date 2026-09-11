"""Execution runtimes and pluggable scheduling strategies."""

from poc.execution.board_claim import BoardClaimStrategy, Claim, StaleClaim
from poc.execution.capacity import (
    CapacityScheduler,
    InProcessWorkerMaterializer,
    WorkerMaterializer,
)
from poc.execution.hierarchical_strategy import HierarchicalDAGStrategy
from poc.execution.managed_pool import Assignment, ManagedPoolStrategy, StaleAssignment
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
    "Assignment",
    "BoardClaimStrategy",
    "CandidateGrant",
    "CapacityScheduler",
    "Claim",
    "ExecutionCoordinator",
    "ExecutionStrategy",
    "HierarchicalDAGStrategy",
    "InProcessWorkerMaterializer",
    "ManagedPoolStrategy",
    "PersistentExecutionStrategy",
    "ReconciliationDecision",
    "SpeculativeStrategy",
    "StaleAssignment",
    "StaleClaim",
    "StrategyError",
    "StrategyNotRegistered",
    "StrategyRegistry",
    "WorkerMaterializer",
]
