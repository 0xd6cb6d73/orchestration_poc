from __future__ import annotations

from dataclasses import dataclass

from poc.models import AgentInstance, Tier
from poc.persistence.database import Database
from poc.roles.registry import RoleRegistry


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str


class RBACAdapter:
    def __init__(self, db: Database, roles: RoleRegistry):
        self.db = db
        self.roles = roles

    def authorize(self, actor: AgentInstance, tool_name: str) -> AuthorizationDecision:
        if actor.tier != Tier.WORKER:
            return AuthorizationDecision(False, "domain tools are restricted to workers")
        if not self.db.authority_active(actor.run_id, actor.plan_version):
            return AuthorizationDecision(
                False, "execution authority is inactive or plan was superseded"
            )
        role = self.roles.get(actor.role_id)
        if tool_name not in role.allowed_tools:
            return AuthorizationDecision(
                False, f"tool {tool_name!r} is not allowed for role {actor.role_id!r}"
            )
        return AuthorizationDecision(True, "role, plan revision, and run are active")
