from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace

from poc.models import EventRecord


def emit_event_span(event: EventRecord) -> None:
    """Mirror coordination evidence into short searchable spans.

    The SQLite event/outbox commit remains authoritative if OTLP delivery fails.
    """
    tracer = trace.get_tracer("poc.coordination")
    with tracer.start_as_current_span(event.event_type) as span:
        span.set_attribute("session.id", event.run_id)
        span.set_attribute("metadata.run_id", event.run_id)
        span.set_attribute("metadata.event_id", event.event_id)
        span.set_attribute("metadata.event_type", event.event_type)
        if event.actor_id:
            span.set_attribute("metadata.agent_instance_id", event.actor_id)
        if event.correlation_id:
            span.set_attribute("metadata.correlation_id", event.correlation_id)
        # Full artifacts never go in spans; the journal keeps the complete redacted record.
        compact = json.dumps(event.data, default=str)
        span.set_attribute("metadata.event_summary", compact[:2048])

