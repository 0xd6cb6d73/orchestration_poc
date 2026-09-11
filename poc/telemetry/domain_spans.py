from __future__ import annotations

import json

from opentelemetry import trace

from poc.models import EventRecord


def emit_event_span(event: EventRecord, *, event_seq: int | None = None) -> None:
    """Mirror coordination evidence into short searchable spans.

    The SQLite event/outbox commit remains authoritative if OTLP delivery fails.
    """
    tracer = trace.get_tracer("poc.coordination")
    with tracer.start_as_current_span(event.event_type) as span:
        span.set_attribute("session.id", event.run_id)
        span.set_attribute("swarm.run.id", event.run_id)
        span.set_attribute("swarm.event.id", event.event_id)
        span.set_attribute("metadata.run_id", event.run_id)
        span.set_attribute("metadata.event_id", event.event_id)
        span.set_attribute("metadata.event_type", event.event_type)
        if event.actor_id:
            span.set_attribute("metadata.agent_instance_id", event.actor_id)
        if event.correlation_id:
            span.set_attribute("metadata.correlation_id", event.correlation_id)
            span.set_attribute("swarm.correlation_id", event.correlation_id)
        if event.causation_id:
            span.set_attribute("swarm.causation_id", event.causation_id)
        if event_seq is not None:
            span.set_attribute("swarm.event.seq", event_seq)
        attribute_names = {
            "mode": "swarm.mode",
            "execution_id": "swarm.execution.id",
            "task_id": "swarm.task.id",
            "task_revision": "swarm.task.revision",
            "attempt_id": "swarm.attempt.id",
            "claim_generation": "swarm.claim.generation",
            "pool_id": "swarm.pool.id",
            "slot_id": "swarm.slot.id",
            "offer_id": "swarm.offer.id",
            "assignment_id": "swarm.assignment.id",
            "assignment_generation": "swarm.assignment.generation",
            "group_id": "swarm.speculation.group_id",
            "candidate_id": "swarm.candidate.id",
            "decision_id": "swarm.reconciliation.id",
            "result_ref": "swarm.result.ref",
            "role": "swarm.agent.role",
            "worker_id": "swarm.agent.id",
            "lease_expires_at": "swarm.lease.expires_at",
            "token_fingerprint": "swarm.ownership.token_fingerprint",
        }
        for source, target in attribute_names.items():
            value = event.data.get(source)
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(target, value)
        # Full artifacts never go in spans; the journal keeps the complete redacted record.
        compact = json.dumps(event.data, default=str)
        span.set_attribute("metadata.event_summary", compact[:2048])
