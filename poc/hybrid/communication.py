from __future__ import annotations

import json

from poc.hybrid.contracts import TeamSpec, VisibilityPolicy
from poc.models import AgentInstance, EventRecord, MessageEnvelope, utc_now
from poc.persistence.database import Database


class CommunicationDenied(PermissionError):
    pass


class CommunicationService:
    def __init__(self, db: Database):
        self.db = db

    def register_team(self, team: TeamSpec) -> None:
        steward = self.db.get_agent(team.steward_agent_id)
        if steward is None or steward.run_id != team.run_id:
            raise CommunicationDenied("team steward is not part of the run")
        members = [self.db.get_agent(member) for member in team.member_agent_ids]
        if any(member is None or member.run_id != team.run_id for member in members):
            raise CommunicationDenied("all team members must belong to the run")
        self.db.put_team(team)

    def send(
        self,
        *,
        team_id: str,
        sender: AgentInstance,
        recipient: AgentInstance,
        message_type: str,
        payload: dict[str, object],
        payload_ref: str | None = None,
    ) -> MessageEnvelope:
        team = self.db.get_team(team_id)
        reason = self._denial_reason(team, sender, recipient, message_type)
        if reason is None and payload_ref is not None:
            artifact = self.db.get_artifact(payload_ref)
            if artifact is None:
                reason = "message references an unknown artifact"
            elif artifact.run_id != sender.run_id:
                reason = "message artifact belongs to another run"
            elif (
                artifact.visibility
                in {
                    VisibilityPolicy.PRIVATE_TO_ATTEMPT,
                    VisibilityPolicy.SEALED_TO_ROUND,
                }
                and f"agent:{recipient.agent_instance_id}" not in artifact.access_labels
            ):
                reason = "message cannot expose a private or sealed artifact"
        if reason:
            self.db.record_event(
                EventRecord(
                    run_id=sender.run_id,
                    event_type="communication.denied",
                    actor_id=sender.agent_instance_id,
                    data={
                        "team_id": team_id,
                        "recipient_id": recipient.agent_instance_id,
                        "message_type": message_type,
                        "reason": reason,
                        "communication_policy_version": (
                            team.communication_policy_version if team else "unknown"
                        ),
                    },
                )
            )
            raise CommunicationDenied(reason)
        assert team is not None
        sent = sum(
            json.loads(str(row[0])).get("team_id") == team_id
            for row in self.db.conn.execute(
                "SELECT payload FROM messages WHERE run_id=?", (sender.run_id,)
            )
        )
        if sent >= team.message_budget:
            reason = "team message budget exhausted"
            self.db.record_event(
                EventRecord(
                    run_id=sender.run_id,
                    event_type="communication.denied",
                    actor_id=sender.agent_instance_id,
                    data={
                        "team_id": team_id,
                        "recipient_id": recipient.agent_instance_id,
                        "message_type": message_type,
                        "reason": reason,
                        "communication_policy_version": team.communication_policy_version,
                    },
                )
            )
            raise CommunicationDenied(reason)
        message = MessageEnvelope(
            run_id=sender.run_id,
            sender_id=sender.agent_instance_id,
            recipient_id=recipient.agent_instance_id,
            message_type=message_type,
            plan_version=sender.plan_version,
            payload={"team_id": team_id, **payload},
            payload_ref=payload_ref,
            delivered_at=utc_now(),
        )
        self.db.put_message(message)
        return message

    @staticmethod
    def _denial_reason(
        team: TeamSpec | None,
        sender: AgentInstance,
        recipient: AgentInstance,
        message_type: str,
    ) -> str | None:
        if team is None:
            return "unknown team"
        members = set(team.member_agent_ids)
        if sender.agent_instance_id not in members or recipient.agent_instance_id not in members:
            return "sender and recipient must be team members"
        if message_type not in team.allowed_message_types:
            return "message type is not allowed by the team policy"
        edge = f"{sender.agent_instance_id}->{recipient.agent_instance_id}"
        if team.permitted_edges and edge not in team.permitted_edges:
            return "communication edge is not allowed by the team policy"
        return None
