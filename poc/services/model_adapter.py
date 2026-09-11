from __future__ import annotations

from typing import Any


class DeterministicModelAdapter:
    """Offline stand-in that emits inspectable facts and decisions, not hidden reasoning."""

    provider = "deterministic"
    model = "fixture-reasoner-v1"

    def orient(self, role: str, goal: str, observations: dict[str, Any]) -> dict[str, Any]:
        return {
            "facts": observations,
            "uncertainties": ["timestamp timezone"]
            if observations.get("timezone_ambiguous")
            else [],
            "constraints": [
                "offline fixtures only",
                "no customer identifiers",
                "one bounded output",
            ],
            "summary": f"Interpreted available evidence for {role}: {goal}",
        }

    def decide(
        self,
        role: str,
        oriented: dict[str, Any],
        tool_calls: int,
        validation_response: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if oriented.get("uncertainties") and not validation_response:
            return {
                "kind": "request_validation",
                "question": "May the timezone-less incident timestamps be treated as UTC?",
            }
        if tool_calls == 0:
            return {
                "kind": "act",
                "justification": "Use the role's narrow deterministic evidence tool.",
            }
        return {"kind": "finish", "justification": "The bounded result and evidence are available."}
