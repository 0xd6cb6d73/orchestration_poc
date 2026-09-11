from __future__ import annotations

from poc.models import RoleSpec, Tier


class RoleRegistry:
    def __init__(self) -> None:
        worker_limits = {"max_ooda_cycles": 3, "max_tool_calls": 2, "max_validations": 1}
        self._roles = {
            "main_orchestrator": RoleSpec(
                role_id="main_orchestrator",
                tier=Tier.MAIN,
                system_prompt="Coordinate approved mission areas; never use domain tools.",
                allowed_child_roles=[
                    "metrics_supervisor",
                    "evidence_supervisor",
                    "reporting_supervisor",
                ],
            ),
            "metrics_supervisor": RoleSpec(
                role_id="metrics_supervisor",
                tier=Tier.SUB,
                system_prompt="Decompose and validate metrics work; never inspect raw metrics.",
                allowed_child_roles=[
                    "window_selector",
                    "percentile_calculator",
                    "metric_comparator",
                    "manifest_reader",
                ],
            ),
            "evidence_supervisor": RoleSpec(
                role_id="evidence_supervisor",
                tier=Tier.SUB,
                system_prompt="Decompose logs and changes work; never inspect raw evidence.",
                allowed_child_roles=[
                    "log_slice_selector",
                    "log_pattern_counter",
                    "deployment_matcher",
                ],
            ),
            "reporting_supervisor": RoleSpec(
                role_id="reporting_supervisor",
                tier=Tier.SUB,
                system_prompt="Coordinate evidence-backed reporting; never write report content directly.",
                allowed_child_roles=[
                    "claim_drafter",
                    "claim_checker",
                    "section_renderer",
                    "report_assembler",
                ],
            ),
            "window_selector": _worker(
                "window_selector", ["read_metric_slice"], "WindowSelection", worker_limits
            ),
            "percentile_calculator": _worker(
                "percentile_calculator",
                ["read_metric_slice", "calculate_percentile"],
                "PercentileResult",
                worker_limits,
            ),
            "metric_comparator": _worker(
                "metric_comparator", ["write_artifact"], "MetricComparison", worker_limits
            ),
            "manifest_reader": _worker(
                "manifest_reader", ["read_manifest"], "ManifestFact", worker_limits
            ),
            "log_slice_selector": _worker(
                "log_slice_selector", ["read_log_slice"], "LogSlice", worker_limits
            ),
            "log_pattern_counter": _worker(
                "log_pattern_counter", ["count_log_pattern"], "PatternCount", worker_limits
            ),
            "deployment_matcher": _worker(
                "deployment_matcher", ["read_deployment_record"], "DeploymentMatch", worker_limits
            ),
            "claim_drafter": _worker(
                "claim_drafter", ["write_artifact"], "DraftClaim", worker_limits
            ),
            "claim_checker": _worker(
                "claim_checker", ["write_artifact"], "CheckedClaim", worker_limits
            ),
            "section_renderer": _worker(
                "section_renderer", ["write_artifact"], "ReportSection", worker_limits
            ),
            "report_assembler": _worker(
                "report_assembler", ["write_artifact"], "FinalReport", worker_limits
            ),
        }

    def get(self, role_id: str) -> RoleSpec:
        try:
            return self._roles[role_id]
        except KeyError as exc:
            raise ValueError(f"unknown role: {role_id}") from exc

    def all(self) -> list[RoleSpec]:
        return list(self._roles.values())

    def register(self, role: RoleSpec, *, replace: bool = False) -> None:
        if role.role_id in self._roles and not replace:
            raise ValueError(f"role already registered: {role.role_id}")
        self._roles[role.role_id] = role


def _worker(role_id: str, tools: list[str], schema: str, limits: dict[str, int]) -> RoleSpec:
    return RoleSpec(
        role_id=role_id,
        tier=Tier.WORKER,
        system_prompt=f"Complete exactly one bounded {role_id} task and return typed evidence.",
        allowed_tools=tools,
        output_schemas=[schema],
        execution_limits=limits.copy(),
    )
