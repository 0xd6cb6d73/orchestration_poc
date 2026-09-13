"""Lossless transport normalization and explicit, bounded protocol errors."""

from __future__ import annotations

import json
from typing import Any, cast


class ProtocolError(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate_json_key")
        result[key] = value
    return result


def parse_action(text: str, policy: str = "strict-v1") -> dict[str, Any]:
    raw = text.strip()
    if policy == "json-normalize-v1" and raw.startswith("```"):
        lines = raw.splitlines()
        if lines[0] not in ("```", "```json") or lines[-1] != "```":
            raise ProtocolError("invalid_json_fence")
        raw = "\n".join(lines[1:-1])
    try:
        value = json.loads(raw, object_pairs_hook=_object)
        if policy == "json-normalize-v1" and isinstance(value, str):
            value = json.loads(value, object_pairs_hook=_object)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            "multiple_actions" if exc.msg == "Extra data" else "invalid_json"
        ) from None
    if not isinstance(value, dict):
        raise ProtocolError("expected_action_object")
    return cast(dict[str, Any], value)
