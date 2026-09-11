# pyright: reportTypedDictNotRequiredAccess=false
"""LangGraph decision nodes intentionally receive partial state updates."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from poc.models import Tier


class DecisionState(TypedDict, total=False):
    supervisor: dict[str, Any]
    event: dict[str, Any]
    proposed_commands: list[dict[str, Any]]
    interpretation: str
    accepted_commands: list[dict[str, Any]]
    rejected_commands: list[dict[str, Any]]


def build_decision_graph():
    """One bounded supervisor decision turn; it never waits for child work."""

    def interpret(state: DecisionState) -> dict[str, Any]:
        return {
            "interpretation": f"Handle {state['event']['type']} and return immediately after dispatch."
        }

    def validate(state: DecisionState) -> dict[str, Any]:
        tier = state["supervisor"]["tier"]
        accepted, rejected = [], []
        for command in state.get("proposed_commands", []):
            kind = command.get("kind")
            valid = (tier == Tier.MAIN and kind in {"assign_goal", "deliver_artifact"}) or (
                tier == Tier.SUB
                and kind in {"submit_workflow", "resolve_validation", "deliver_artifact"}
            )
            (accepted if valid else rejected).append(command)
        return {"accepted_commands": accepted, "rejected_commands": rejected}

    graph = StateGraph(DecisionState)
    graph.add_node("interpret_event", interpret)
    graph.add_node("validate_authority", validate)
    graph.add_edge(START, "interpret_event")
    graph.add_edge("interpret_event", "validate_authority")
    graph.add_edge("validate_authority", END)
    return graph.compile()
