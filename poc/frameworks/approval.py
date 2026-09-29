"""Deterministic per-call approval service for native backend conformance runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    run_id: str
    task_id: str
    tool_call_id: str
    tool_name: str
    argument_digest: str
    expires_at: float


class MockApprovalClient:
    """Correlates decisions to one immutable call and rejects stale/duplicate votes."""

    def __init__(self, timeout: float = 30.0, pending_limit: int = 4) -> None:
        self.timeout = timeout
        self.pending_limit = pending_limit
        self.requests: dict[str, ApprovalRequest] = {}
        self._pending: dict[str, asyncio.Future[bool]] = {}

    def request(
        self,
        run_id: str,
        task_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ApprovalRequest:
        if sum(not future.done() for future in self._pending.values()) >= self.pending_limit:
            raise RuntimeError("pending approval limit exhausted")
        digest = hashlib.sha256(
            json.dumps(arguments, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        request = ApprovalRequest(
            approval_id=uuid4().hex,
            run_id=run_id,
            task_id=task_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            argument_digest=digest,
            expires_at=monotonic() + self.timeout,
        )
        self.requests[request.approval_id] = request
        self._pending[request.approval_id] = asyncio.get_running_loop().create_future()
        return request

    def decide(
        self, approval_id: str, tool_call_id: str, argument_digest: str, approved: bool
    ) -> bool:
        request = self.requests.get(approval_id)
        future = self._pending.get(approval_id)
        if (
            request is None
            or future is None
            or future.done()
            or monotonic() >= request.expires_at
            or request.tool_call_id != tool_call_id
            or request.argument_digest != argument_digest
        ):
            return False
        future.set_result(approved)
        return True

    async def wait(self, request: ApprovalRequest) -> bool:
        remaining = request.expires_at - monotonic()
        if remaining <= 0:
            raise TimeoutError("approval expired")
        try:
            return await asyncio.wait_for(self._pending[request.approval_id], remaining)
        except TimeoutError as exc:
            raise TimeoutError("approval expired") from exc
