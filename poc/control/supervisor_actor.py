from __future__ import annotations

import asyncio
from typing import Any

from poc.control.decision_graph import build_decision_graph
from poc.models import AgentInstance, EventRecord, MessageEnvelope, utc_now
from poc.persistence.database import Database


class SupervisorActor:
    """A serialized mailbox facade around short, non-blocking LangGraph turns."""

    def __init__(self, agent: AgentInstance, db: Database):
        self.agent = agent
        self.db = db
        self._lock = asyncio.Lock()
        self.graph = build_decision_graph()
        self.state_version = db.supervisor_version(agent.agent_instance_id)

    async def turn(
        self, event: dict[str, Any], proposed_commands: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        async with self._lock:
            input_version = self.state_version
            message = MessageEnvelope(
                run_id=self.agent.run_id,
                sender_id=event.get("sender_id", "runtime"),
                recipient_id=self.agent.agent_instance_id,
                message_type=event["type"],
                plan_version=self.agent.plan_version,
                workflow_revision=event.get("workflow_revision"),
                payload=event,
                delivered_at=utc_now(),
                consumed_at=utc_now(),
            )
            self.db.put_message(message)
            result = self.graph.invoke(
                {
                    "supervisor": self.agent.model_dump(mode="json"),
                    "event": event,
                    "proposed_commands": proposed_commands,
                }
            )
            turn_event = EventRecord(
                run_id=self.agent.run_id,
                event_type="supervisor.turn",
                actor_id=self.agent.agent_instance_id,
                causation_id=message.message_id,
                data={
                    "input_state_version": input_version,
                    "trigger": event["type"],
                    "interpretation": result["interpretation"],
                    "accepted_command_ids": [
                        c.get("command_id") for c in result["accepted_commands"]
                    ],
                    "rejected_commands": result["rejected_commands"],
                },
            )
            self.state_version = self.db.commit_supervisor_turn(
                agent_id=self.agent.agent_instance_id,
                run_id=self.agent.run_id,
                input_version=input_version,
                snapshot={
                    "last_message_id": message.message_id,
                    "waiting_for": event.get("waiting_for", []),
                    "last_interpretation": result["interpretation"],
                },
                accepted_commands=result["accepted_commands"],
                event=turn_event,
            )
            return result["accepted_commands"]
