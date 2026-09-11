from __future__ import annotations

from poc.hybrid.contracts import CapabilityProfile, RouteDecision
from poc.models import AgentRuntimeConfig, RoleSpec
from poc.persistence.database import Database


class NoCompatibleCapability(LookupError):
    pass


class CapabilityRouter:
    """Deterministic, versioned routing over provisionable execution profiles."""

    policy_version = "capability_router_v1"

    def __init__(self, db: Database):
        self.db = db

    def profile_for_role(
        self,
        role: RoleSpec,
        runtime: AgentRuntimeConfig,
        *,
        task_classes: frozenset[str],
        preference: int = 0,
    ) -> CapabilityProfile:
        profile = CapabilityProfile(
            profile_id=(
                f"profile:{role.role_id}:v{role.version}:{runtime.backend}:"
                f"{runtime.provider or 'default'}:{runtime.model or 'default'}"
            ),
            role_id=role.role_id,
            role_version=role.version,
            agent_backend=runtime.backend,
            provider=runtime.provider,
            model=runtime.model,
            task_classes=task_classes,
            tools=frozenset(role.allowed_tools),
            preference=preference,
        )
        self.db.put_capability_profile(profile)
        return profile

    def route(
        self,
        *,
        run_id: str,
        task_id: str,
        task_class: str,
        required_tools: frozenset[str],
        profiles: list[CapabilityProfile],
        offer_generation: int = 1,
    ) -> RouteDecision:
        compatible = [
            profile
            for profile in profiles
            if task_class in profile.task_classes and required_tools <= profile.tools
        ]
        if not compatible:
            raise NoCompatibleCapability(
                f"no profile supports task class {task_class!r} and tools {sorted(required_tools)}"
            )
        compatible.sort(key=lambda item: (-item.preference, item.profile_id))
        selected = compatible[0]
        decision = RouteDecision(
            route_decision_id=f"route:{run_id}:{task_id}:g{offer_generation}",
            run_id=run_id,
            task_id=task_id,
            offer_generation=offer_generation,
            policy_version=self.policy_version,
            selected_profile_id=selected.profile_id,
            considered_profile_ids=tuple(profile.profile_id for profile in profiles),
            reasons=(
                f"supports task class {task_class}",
                f"contains required tools {sorted(required_tools)}",
                f"highest deterministic preference {selected.preference}",
            ),
        )
        self.db.put_route_decision(decision)
        return decision
