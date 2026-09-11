from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from poc.hybrid.contracts import VisibilityPolicy
from poc.models import AgentInstance, ArtifactRecord
from poc.persistence.database import Database


class ArtifactStore:
    def __init__(self, root: str | Path, db: Database):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = db

    def write(
        self,
        run_id: str,
        value: Any,
        *,
        producer_task_id: str | None = None,
        producer_attempt_id: str | None = None,
        media_type: str = "application/json",
        visibility: VisibilityPolicy | str = VisibilityPolicy.RUN_WIDE,
        visibility_ref: str | None = None,
        access_labels: frozenset[str] = frozenset(),
    ) -> ArtifactRecord:
        if isinstance(value, bytes):
            content = value
        elif isinstance(value, str) and media_type.startswith("text/"):
            content = value.encode()
        else:
            content = json.dumps(value, indent=2, sort_keys=True).encode()
        digest = hashlib.sha256(content).hexdigest()
        suffix = ".md" if media_type == "text/markdown" else ".json"
        path = self.root / digest[:2] / f"{digest}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
        artifact_key = hashlib.sha256(f"{run_id}:{digest}".encode()).hexdigest()
        record = ArtifactRecord(
            # Payload bytes remain globally content-addressed, while the public ID
            # is run-scoped so identical fixture output cannot alias another run's
            # visibility or provenance record.
            artifact_id=f"artifact-{artifact_key[:16]}",
            run_id=run_id,
            media_type=media_type,
            sha256=digest,
            path=str(path),
            producer_task_id=producer_task_id,
            producer_attempt_id=producer_attempt_id,
            visibility=str(visibility),
            visibility_ref=visibility_ref,
            access_labels=access_labels,
        )
        self.db.put_artifact(record)
        return record

    def read(self, artifact_id: str) -> tuple[ArtifactRecord, bytes]:
        record = self.db.get_artifact(artifact_id)
        if not record:
            raise KeyError(artifact_id)
        return record, Path(record.path).read_bytes()

    def read_public(self, artifact_id: str) -> tuple[ArtifactRecord, bytes]:
        """Read an artifact through a principal-less inspection endpoint."""
        record, content = self.read(artifact_id)
        if record.visibility in {
            VisibilityPolicy.PRIVATE_TO_ATTEMPT,
            VisibilityPolicy.SEALED_TO_ROUND,
        }:
            raise PermissionError(f"artifact {artifact_id!r} is not publicly inspectable")
        return record, content

    def read_for(
        self,
        actor: AgentInstance,
        artifact_id: str,
        *,
        attempt_id: str | None = None,
        round_id: str | None = None,
        team_id: str | None = None,
        domain_id: str | None = None,
    ) -> tuple[ArtifactRecord, bytes]:
        record, content = self.read(artifact_id)
        if record.visibility == VisibilityPolicy.RELEASED_TO_TEAM:
            team = self.db.get_team(record.visibility_ref or "")
            if team is None or actor.agent_instance_id not in team.member_agent_ids:
                raise PermissionError(
                    f"artifact {artifact_id!r} is outside the actor's visible context"
                )
        if record.run_id != actor.run_id or not _artifact_visible(
            record,
            actor,
            attempt_id=attempt_id,
            round_id=round_id,
            team_id=team_id,
            domain_id=domain_id,
        ):
            raise PermissionError(
                f"artifact {artifact_id!r} is outside the actor's visible context"
            )
        return record, content

    def set_visibility(
        self,
        artifact_id: str,
        visibility: VisibilityPolicy,
        *,
        visibility_ref: str | None = None,
        access_labels: frozenset[str] = frozenset(),
    ) -> ArtifactRecord:
        return self.db.update_artifact_access(
            artifact_id,
            visibility=visibility,
            visibility_ref=visibility_ref,
            access_labels=access_labels,
        )


def _artifact_visible(
    record: ArtifactRecord,
    actor: AgentInstance,
    *,
    attempt_id: str | None,
    round_id: str | None,
    team_id: str | None,
    domain_id: str | None,
) -> bool:
    visibility = VisibilityPolicy(record.visibility)
    labels = record.access_labels
    if visibility == VisibilityPolicy.RUN_WIDE:
        return True
    if visibility == VisibilityPolicy.PRIVATE_TO_ATTEMPT:
        return attempt_id is not None and f"attempt:{attempt_id}" in labels
    if visibility == VisibilityPolicy.SEALED_TO_ROUND:
        return f"agent:{actor.agent_instance_id}" in labels or (
            attempt_id is not None and f"attempt:{attempt_id}" in labels
        )
    if visibility == VisibilityPolicy.RELEASED_TO_TEAM:
        return team_id is not None and team_id == record.visibility_ref
    if visibility == VisibilityPolicy.DOMAIN:
        return domain_id is not None and domain_id == record.visibility_ref
    return round_id is not None and round_id == record.visibility_ref
