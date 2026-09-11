from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from poc.models import ArtifactRecord
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
        media_type: str = "application/json",
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
        record = ArtifactRecord(
            artifact_id=f"artifact-{digest[:16]}",
            run_id=run_id,
            media_type=media_type,
            sha256=digest,
            path=str(path),
            producer_task_id=producer_task_id,
        )
        self.db.put_artifact(record)
        return record

    def read(self, artifact_id: str) -> tuple[ArtifactRecord, bytes]:
        record = self.db.get_artifact(artifact_id)
        if not record:
            raise KeyError(artifact_id)
        return record, Path(record.path).read_bytes()
