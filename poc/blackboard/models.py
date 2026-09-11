from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from poc.hybrid.contracts import VisibilityPolicy
from poc.models import new_id, utc_now


class RecordType(StrEnum):
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    CONTRADICTION = "contradiction"
    DECISION = "decision"


class EpistemicStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    DISPUTED = "disputed"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class BlackboardRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(default_factory=lambda: new_id("blackboard"))
    record_version: int = 1
    run_id: str
    domain_id: str
    record_type: RecordType
    concise_statement: str = Field(min_length=1)
    supporting_artifact_refs: tuple[str, ...] = ()
    challenging_artifact_refs: tuple[str, ...] = ()
    author_agent_id: str
    epistemic_status: EpistemicStatus = EpistemicStatus.PROPOSED
    trust_labels: frozenset[str] = frozenset({"untrusted_content"})
    visibility_policy: VisibilityPolicy = VisibilityPolicy.DOMAIN
    visibility_ref: str | None = None
    supersedes_record_version: int | None = None
    created_at: str = Field(default_factory=utc_now)
