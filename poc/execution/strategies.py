"""Stable public import surface for execution-strategy extensions."""

from poc.execution.board_claim import BoardClaimStrategy, Claim, StaleClaim
from poc.execution.capacity import CapacityScheduler, InProcessWorkerMaterializer, WorkerMaterializer
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

# Names that describe each component's responsibility, while preserving the more
# explicit strategy names used by the registry.
TaskBoard = BoardClaimStrategy
PoolCoordinator = ManagedPoolStrategy
SpeculationCoordinator = SpeculativeStrategy

__all__ = [
    "Assignment",
    "BoardClaimStrategy",
    "CapacityScheduler",
    "CandidateGrant",
    "Claim",
    "ExecutionCoordinator",
    "ExecutionStrategy",
    "HierarchicalDAGStrategy",
    "InProcessWorkerMaterializer",
    "ManagedPoolStrategy",
    "PersistentExecutionStrategy",
    "PoolCoordinator",
    "ReconciliationDecision",
    "SpeculationCoordinator",
    "SpeculativeStrategy",
    "StaleAssignment",
    "StaleClaim",
    "StrategyError",
    "StrategyNotRegistered",
    "StrategyRegistry",
    "TaskBoard",
    "WorkerMaterializer",
]
