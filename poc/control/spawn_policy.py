from __future__ import annotations

from poc.models import AgentInstance, EventRecord, Tier, new_id
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry


class SpawnDenied(PermissionError):
    pass


class SpawnPolicy:
    def __init__(self, db: Database, roles: RoleRegistry):
        self.db = db
        self.roles = roles

    def spawn(
        self,
        *,
        run_id: str,
        parent: AgentInstance,
        child_role: str,
        plan_version: int,
        stable_key: str | None = None,
        agent_backend: str | None = None,
        agent_provider: str | None = None,
        agent_model: str | None = None,
    ) -> AgentInstance:
        parent_spec = self.roles.get(parent.role_id)
        child_spec = self.roles.get(child_role)
        if not self.db.authority_active(run_id, plan_version):
            self._deny(
                run_id,
                parent.agent_instance_id,
                child_role,
                "plan authority is inactive or superseded",
            )
        allowed_tier = (parent.tier == Tier.MAIN and child_spec.tier == Tier.SUB) or (
            parent.tier == Tier.SUB and child_spec.tier == Tier.WORKER
        )
        if not allowed_tier or child_role not in parent_spec.allowed_child_roles:
            self._deny(
                run_id,
                parent.agent_instance_id,
                child_role,
                "hierarchy or role policy forbids spawn",
            )
        suffix = stable_key or new_id("instance")
        agent_id = f"{child_role}-{suffix}" if stable_key else suffix
        selected_backend = agent_backend or child_spec.agent_backend
        selected_provider = agent_provider or child_spec.provider
        selected_model = agent_model or child_spec.model
        existing = self.db.get_agent(agent_id)
        if existing is not None:
            if (
                existing.run_id != run_id
                or existing.parent_agent_id != parent.agent_instance_id
                or existing.role_id != child_role
                or existing.plan_version != plan_version
            ):
                self._deny(
                    run_id,
                    parent.agent_instance_id,
                    child_role,
                    "a stable agent identity is already bound to different authority",
                )
            if existing.agent_backend != selected_backend:
                self._deny(
                    run_id,
                    parent.agent_instance_id,
                    child_role,
                    "a stable worker identity cannot change agent backend",
                )
            if (
                existing.agent_provider != selected_provider
                or existing.agent_model != selected_model
            ):
                self._deny(
                    run_id,
                    parent.agent_instance_id,
                    child_role,
                    "a stable worker identity cannot change provider or model",
                )
            agent = existing
        else:
            agent = AgentInstance(
                agent_instance_id=agent_id,
                run_id=run_id,
                parent_agent_id=parent.agent_instance_id,
                tier=child_spec.tier,
                role_id=child_role,
                role_version=child_spec.version,
                plan_version=plan_version,
                agent_backend=selected_backend,
                agent_provider=selected_provider,
                agent_model=selected_model,
            )
            self.db.put_agent(agent)
        self.db.record_event(
            EventRecord(
                run_id=run_id,
                event_type="agent.spawn_authorized",
                actor_id=parent.agent_instance_id,
                data={
                    "agent_instance_id": agent_id,
                    "child_role": child_role,
                    "agent_backend": selected_backend,
                    "agent_provider": selected_provider,
                    "agent_model": selected_model,
                },
            )
        )
        return agent

    def _deny(self, run_id: str, actor_id: str, child_role: str, reason: str) -> None:
        self.db.record_event(
            EventRecord(
                run_id=run_id,
                event_type="agent.spawn_denied",
                actor_id=actor_id,
                data={"child_role": child_role, "reason": reason},
            )
        )
        raise SpawnDenied(reason)
