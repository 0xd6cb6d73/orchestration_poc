from __future__ import annotations

from poc.blackboard.models import BlackboardRecord, EpistemicStatus, RecordType
from poc.hybrid.contracts import VisibilityPolicy
from poc.models import AgentInstance, EventRecord, Tier, new_id
from poc.persistence.database import Database


class BlackboardDenied(PermissionError):
    pass


class BlackboardService:
    def __init__(self, db: Database):
        self.db = db

    def publish(
        self,
        *,
        actor: AgentInstance,
        domain_id: str,
        record_type: RecordType,
        statement: str,
        supporting_artifacts: tuple[str, ...] = (),
        challenging_artifacts: tuple[str, ...] = (),
        status: EpistemicStatus = EpistemicStatus.PROPOSED,
        visibility: VisibilityPolicy = VisibilityPolicy.DOMAIN,
        visibility_ref: str | None = None,
        record_id: str | None = None,
    ) -> BlackboardRecord:
        self._require_active(actor)
        record = BlackboardRecord(
            record_id=record_id or new_id("blackboard"),
            run_id=actor.run_id,
            domain_id=domain_id,
            record_type=record_type,
            concise_statement=statement,
            supporting_artifact_refs=supporting_artifacts,
            challenging_artifact_refs=challenging_artifacts,
            author_agent_id=actor.agent_instance_id,
            epistemic_status=status,
            visibility_policy=visibility,
            visibility_ref=visibility_ref,
        )
        self.db.put_blackboard_record(record, event_type="blackboard.published")
        return record

    def transition(
        self,
        *,
        actor: AgentInstance,
        record_id: str,
        status: EpistemicStatus,
        supporting_artifacts: tuple[str, ...] = (),
        challenging_artifacts: tuple[str, ...] = (),
        visibility: VisibilityPolicy | None = None,
        visibility_ref: str | None = None,
    ) -> BlackboardRecord:
        self._require_active(actor)
        current = self.db.latest_blackboard_record(record_id)
        if current is None or current.run_id != actor.run_id:
            raise KeyError(record_id)
        author = self.db.get_agent(current.author_agent_id)
        authorized = actor.tier == Tier.MAIN or (
            actor.tier == Tier.SUB
            and author is not None
            and (
                author.parent_agent_id == actor.agent_instance_id
                # A SUB steward may govern records it published itself, e.g. candidate
                # revisions it records inside its own collaboration round.
                or author.agent_instance_id == actor.agent_instance_id
            )
        )
        if not authorized:
            self._deny(actor, "only the governing orchestrator can change epistemic status")
        updated = current.model_copy(
            update={
                "record_version": current.record_version + 1,
                "epistemic_status": status,
                "supporting_artifact_refs": tuple(
                    dict.fromkeys((*current.supporting_artifact_refs, *supporting_artifacts))
                ),
                "challenging_artifact_refs": tuple(
                    dict.fromkeys((*current.challenging_artifact_refs, *challenging_artifacts))
                ),
                "supersedes_record_version": current.record_version,
                **({"visibility_policy": visibility} if visibility is not None else {}),
                **({"visibility_ref": visibility_ref} if visibility is not None else {}),
            }
        )
        event_type = (
            "blackboard.challenged" if status == EpistemicStatus.DISPUTED else "blackboard.updated"
        )
        if visibility is not None and visibility != current.visibility_policy:
            event_type = "blackboard.released"
        self.db.put_blackboard_record(updated, event_type=event_type)
        return updated

    def visible_records(
        self,
        *,
        actor: AgentInstance,
        domain_id: str | None = None,
        team_id: str | None = None,
        round_id: str | None = None,
    ) -> list[BlackboardRecord]:
        records = self.db.list_blackboard_records(actor.run_id)
        return [
            record
            for record in records
            if _record_visible(self.db, record, actor, domain_id, team_id, round_id)
        ]

    def _require_active(self, actor: AgentInstance) -> None:
        if not self.db.authority_active(actor.run_id, actor.plan_version):
            self._deny(actor, "actor has no active plan authority")

    def _deny(self, actor: AgentInstance, reason: str) -> None:
        self.db.record_event(
            EventRecord(
                run_id=actor.run_id,
                event_type="blackboard.access_denied",
                actor_id=actor.agent_instance_id,
                data={"reason": reason},
            )
        )
        raise BlackboardDenied(reason)


def _record_visible(
    db: Database,
    record: BlackboardRecord,
    actor: AgentInstance,
    domain_id: str | None,
    team_id: str | None,
    round_id: str | None,
) -> bool:
    visibility = record.visibility_policy
    if visibility == VisibilityPolicy.RUN_WIDE:
        return True
    if visibility == VisibilityPolicy.DOMAIN:
        return domain_id == record.domain_id
    if visibility == VisibilityPolicy.RELEASED_TO_TEAM:
        if team_id is None or team_id != record.visibility_ref:
            return False
        team = db.get_team(team_id)
        return team is not None and actor.agent_instance_id in team.member_agent_ids
    if visibility == VisibilityPolicy.SEALED_TO_ROUND:
        return actor.agent_instance_id == record.author_agent_id
    if visibility == VisibilityPolicy.PRIVATE_TO_ATTEMPT:
        return actor.agent_instance_id == record.author_agent_id
    return round_id is not None and round_id == record.visibility_ref
