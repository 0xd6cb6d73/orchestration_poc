from __future__ import annotations

from poc.blackboard.models import EpistemicStatus
from poc.hybrid.contracts import DeliveryDecision
from poc.models import MissionPlan
from poc.persistence.database import Database


class DeliveryBlocked(RuntimeError):
    pass


class CompletionGate:
    """Deterministic final gate; it enforces attestations, not semantic truth."""

    def __init__(self, db: Database):
        self.db = db

    def evaluate(
        self,
        *,
        plan: MissionPlan,
        deliverable_artifact_id: str,
        required_accepted_artifacts: tuple[str, ...],
    ) -> DeliveryDecision:
        run = self.db.get_run(plan.run_id)
        current_plan = self.db.get_plan(plan.run_id)
        unresolved = [
            record
            for record in self.db.list_blackboard_records(plan.run_id)
            if record.epistemic_status
            in {
                EpistemicStatus.PROPOSED,
                EpistemicStatus.DISPUTED,
                EpistemicStatus.UNRESOLVED,
            }
        ]
        checks = {
            "approved_plan_is_current": bool(
                run
                and current_plan
                and current_plan.status == "approved"
                and current_plan.version == plan.version
            ),
            "deliverable_has_provenance": self.db.get_provenance(deliverable_artifact_id)
            is not None,
            "required_artifacts_accepted": all(
                self.db.accepted_artifact(plan.run_id, artifact_id)
                for artifact_id in required_accepted_artifacts
            ),
            "unresolved_claims_treated": not unresolved,
        }
        reasons = tuple(name for name, passed in checks.items() if not passed)
        decision = DeliveryDecision(
            delivery_decision_id=(
                f"delivery:{plan.run_id}:v{plan.version}:{deliverable_artifact_id}"
            ),
            run_id=plan.run_id,
            artifact_id=deliverable_artifact_id,
            plan_version=plan.version,
            completion_policy_version=plan.policy_set.completion_policy,
            permitted=all(checks.values()),
            checks=checks,
            reasons=reasons,
        )
        self.db.put_delivery_decision(decision)
        return decision

    def require(self, decision: DeliveryDecision) -> None:
        if not decision.permitted:
            raise DeliveryBlocked(f"delivery blocked by checks: {', '.join(decision.reasons)}")
