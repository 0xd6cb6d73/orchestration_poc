"""Calibrate declared role allowances from usage and timing, never grader scores."""

from __future__ import annotations

import math
from typing import Any

from poc.execution.sql_contracts import RoleBudgetProfile


def measured_profiles(
    trials: list[dict[str, Any]], source: str, *, headroom: float = 1.25
) -> dict[str, dict[str, RoleBudgetProfile]]:
    """Use observed phase totals as conservative request bounds, with explicit headroom.

    A phase may contain several requests. These are initial ceilings, not token cost
    predictions or evidence of optimal allocation. Include failed phases to avoid
    calibrating only on survivors; missing usage cannot provide a token measurement.
    """
    if not math.isfinite(headroom) or headroom < 1:
        raise ValueError("headroom must be finite and at least one")
    measurements: dict[tuple[str, str], tuple[int, float]] = {}
    for trial in trials:
        for phase in trial.get("phases", []):
            binding: dict[str, Any] = phase.get("model_binding") or {}
            model = binding.get("model") or phase.get("model") or trial["model_name"]
            role = phase["name"]
            key = model, role
            tokens, seconds = measurements.get(key, (0, 0.0))
            observed_tokens = phase.get("output_tokens", 0) if phase.get("usage_complete") else 0
            measurements[key] = (
                max(tokens, observed_tokens),
                max(seconds, phase.get("elapsed_seconds", 0)),
            )
    profiles: dict[str, dict[str, RoleBudgetProfile]] = {}
    for (model, role), (tokens, seconds) in sorted(measurements.items()):
        if tokens <= 0 or seconds <= 0:
            continue
        profiles.setdefault(model, {})[role] = RoleBudgetProfile(
            measurement_source=source,
            max_output_tokens=math.ceil(tokens * headroom),
            request_timeout_seconds=seconds * headroom,
        )
    return profiles
