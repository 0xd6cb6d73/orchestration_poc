from __future__ import annotations

from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field


class OutputContractError(ValueError):
    """Raised when a workflow refers to an unknown or incompatible output contract."""


class StrictOutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeWindow(StrictOutputModel):
    start: str
    end: str
    service: str


class WindowSelection(StrictOutputModel):
    baseline: TimeWindow
    incident: TimeWindow
    sample_count: int = Field(ge=0)
    timezone_note: str


class PercentileValue(StrictOutputModel):
    percentile: float = Field(ge=0, le=100)
    value: float
    units: str
    sample_count: int = Field(ge=1)
    method: str


class PercentileResult(StrictOutputModel):
    result: PercentileValue
    window: TimeWindow


class ArtifactPublication(StrictOutputModel):
    artifact_id: str = Field(min_length=1)
    sha256: str = Field(min_length=1)
    media_type: str = Field(min_length=1)


class MetricComparisonContent(StrictOutputModel):
    baseline_p95_ms: float
    incident_p95_ms: float
    absolute_increase_ms: float
    ratio: float
    claim: str


class MetricComparison(StrictOutputModel):
    content: MetricComparisonContent
    published: ArtifactPublication


class ManifestData(StrictOutputModel):
    fixture: str
    timestamp_timezone: str
    note: str


class ManifestFact(StrictOutputModel):
    fact: str
    manifest: ManifestData


class LogRow(StrictOutputModel):
    timestamp: str
    service: str
    level: str
    pattern: str
    message: str


class LogSliceData(StrictOutputModel):
    rows: list[LogRow]
    count: int = Field(ge=0)
    start: str
    end: str


class LogSlice(StrictOutputModel):
    log_slice: LogSliceData


class PatternCountData(StrictOutputModel):
    counts: dict[str, int]
    total: int = Field(ge=0)
    interpretation: str


class PatternCount(StrictOutputModel):
    pattern_counts: PatternCountData


class DeploymentRecord(StrictOutputModel):
    deployment_id: str
    service: str
    version: str
    timestamp: str
    change: str
    minutes_before_incident: int = Field(ge=0)


class DeploymentMatchData(StrictOutputModel):
    matches: list[DeploymentRecord]
    count: int = Field(ge=0)


class DeploymentMatch(StrictOutputModel):
    deployment_match: DeploymentMatchData
    pattern_counts: PatternCountData


class DraftClaimContent(StrictOutputModel):
    hypothesis_key: str
    claim: str
    confidence: str
    alternatives: list[str]


class DraftClaim(StrictOutputModel):
    content: DraftClaimContent
    published: ArtifactPublication


class CheckedClaimContent(StrictOutputModel):
    hypothesis_key: str
    claim: str
    supported: bool
    checks: list[str]
    caveat: str


class CheckedClaim(StrictOutputModel):
    content: CheckedClaimContent
    published: ArtifactPublication


class VerificationContent(StrictOutputModel):
    subject_artifact_id: str
    supported: bool
    checks: dict[str, bool]
    findings: list[str]


class VerificationResult(StrictOutputModel):
    content: VerificationContent
    published: ArtifactPublication


class ReportSection(StrictOutputModel):
    content: str
    published: ArtifactPublication


class FinalReport(StrictOutputModel):
    content: str
    published: ArtifactPublication


OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "WindowSelection": WindowSelection,
    "PercentileResult": PercentileResult,
    "MetricComparison": MetricComparison,
    "ManifestFact": ManifestFact,
    "LogSlice": LogSlice,
    "PatternCount": PatternCount,
    "DeploymentMatch": DeploymentMatch,
    "DraftClaim": DraftClaim,
    "CheckedClaim": CheckedClaim,
    "VerificationResult": VerificationResult,
    "ReportSection": ReportSection,
    "FinalReport": FinalReport,
}


def output_model_for(schema: str) -> type[BaseModel]:
    try:
        return OUTPUT_MODELS[schema]
    except KeyError as exc:
        raise OutputContractError(f"no output contract is registered for {schema!r}") from exc


def validate_output(schema: str, value: Any) -> dict[str, Any]:
    validated = output_model_for(schema).model_validate(value)
    return validated.model_dump(mode="json")


def validate_output_field(schema: str, field_path: str) -> None:
    """Check that a dotted workflow binding path exists in a registered output model."""
    current: type[BaseModel] = output_model_for(schema)
    components = field_path.split(".")
    if not field_path or any(not component for component in components):
        raise OutputContractError(f"invalid empty field path for {schema!r}")
    for index, component in enumerate(components):
        field = current.model_fields.get(component)
        if field is None:
            raise OutputContractError(
                f"output contract {schema!r} has no field {'.'.join(components[: index + 1])!r}"
            )
        if index == len(components) - 1:
            return
        nested = _nested_model(field.annotation)
        if nested is None:
            raise OutputContractError(
                f"output contract field {'.'.join(components[: index + 1])!r} "
                "does not expose nested fields"
            )
        current = nested


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in {Union, UnionType}:
        for candidate in get_args(annotation):
            nested = _nested_model(candidate)
            if nested is not None:
                return nested
    return None
