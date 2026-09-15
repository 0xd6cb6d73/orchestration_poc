from __future__ import annotations

from collections.abc import Callable

from poc.models import SwarmPolicySet, SwarmStrategy

PolicyFactory = Callable[[], SwarmPolicySet]


class SwarmPolicyResolver:
    """Resolves a strategy name into pinned runtime-enforced policy versions."""

    def __init__(self) -> None:
        self._factories: dict[SwarmStrategy, PolicyFactory] = {
            SwarmStrategy.BOARD: _board_policies,
            SwarmStrategy.HYBRID_V1: _hybrid_v1_policies,
            SwarmStrategy.HYBRID_V2: _hybrid_v2_policies,
        }

    def register(
        self,
        strategy: SwarmStrategy | str,
        factory: PolicyFactory,
        *,
        replace: bool = False,
    ) -> None:
        normalized = SwarmStrategy(strategy)
        if normalized in self._factories and not replace:
            raise ValueError(f"policies already registered for {normalized.value!r}")
        self._factories[normalized] = factory

    def resolve(self, strategy: SwarmStrategy | str) -> SwarmPolicySet:
        normalized = SwarmStrategy(strategy)
        try:
            resolved = self._factories[normalized]()
        except KeyError as exc:
            raise LookupError(f"no policies registered for {normalized.value!r}") from exc
        if resolved.strategy != normalized:
            raise ValueError("resolved policy set has the wrong strategy")
        return resolved

    @property
    def strategies(self) -> frozenset[SwarmStrategy]:
        return frozenset(self._factories)


def _board_policies() -> SwarmPolicySet:
    return SwarmPolicySet()


def _hybrid_v1_policies() -> SwarmPolicySet:
    return SwarmPolicySet(
        strategy=SwarmStrategy.HYBRID_V1,
        allocation_policy="capability_router_v1",
        context_policy="artifact_manifest_v1",
        communication_policy="scoped_team_v1",
        collaboration_policy="diverge_test_select_v1",
        acceptance_policy="independent_evidence_v1",
        completion_policy="attested_delivery_v1",
    )


def _hybrid_v2_policies() -> SwarmPolicySet:
    return SwarmPolicySet(
        strategy=SwarmStrategy.HYBRID_V2,
        allocation_policy="board_scoped_wave_v1",
        context_policy="artifact_manifest_v1",
        communication_policy="scoped_team_v1",
        collaboration_policy="orchestrator_planned_v2",
        acceptance_policy="independent_evidence_v1",
        completion_policy="attested_delivery_v1",
    )
