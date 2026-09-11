from poc.execution.strategy import PersistentExecutionStrategy
from poc.models import ExecutionMode


class HierarchicalDAGStrategy(PersistentExecutionStrategy):
    """Authority record for the existing deterministic LangGraph DAG runner."""

    mode = ExecutionMode.HIERARCHICAL_DAG
