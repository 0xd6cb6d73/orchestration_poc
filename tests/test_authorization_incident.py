from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

from poc.evaluation.authorization.__main__ import export_case
from poc.evaluation.authorization.generator import SCALES, Scale, generate_incident
from poc.evaluation.authorization.grading import grade_authorization
from poc.evaluation.authorization.reference import Entitlements
from poc.evaluation.authorization.simulator import AuthorizationSandbox, parse_changes
from poc.evaluation.suite.adapters import ADAPTERS, single, single_json
from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Answer, Budget, Matrix, ModelSpec, TaskCase
from poc.evaluation.suite.runner import read_report, run_matrix
from poc.evaluation.suite.tasks import generate
from poc.execution.sql_team_worker import TeamWorker

SMALL = Scale(4, 12, 8, 3500)


def small_case(seed: int = 0) -> TaskCase:
    public, expected = generate_incident(random.Random(seed), SMALL)
    return TaskCase(
        id=f"authorization-v1-dev-standard-{seed}",
        family="authorization",
        seed=seed,
        split="dev",
        difficulty="standard",
        input=public,
        expected=expected,
    )


@pytest.fixture(scope="module")
def case() -> TaskCase:
    return small_case()


@pytest.mark.parametrize("seed", range(8))
def test_generated_incident_has_observable_faults_and_feasible_repairs(seed: int) -> None:
    case = small_case(seed)
    witness = Answer(values=case.expected["submission"])
    assert grade_authorization(case, witness)["exact"] == 1
    deployed = witness.model_copy(deep=True)
    deployed.values["changes"] = []
    scores = grade_authorization(case, deployed)
    assert scores["exact"] == 0 and scores["restored_access"] < 1
    assert scores["tenant_impact"] == scores["evidence_support"] == 1
    engine = AuthorizationSandbox(case.input.tables)
    audit = {r["trace_id"]: r for r in case.input.tables["auth_audit"]}
    for req in case.input.tables["auth_requests"][:100]:
        assert (
            engine.evaluate(req, engine.configurations)["decision"]
            == audit[req["trace_id"]]["decision"]
        )


def test_reproducible_registration_and_distinct_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(SCALES, 80, SMALL)
    first = generate("authorization", 5, "dev", "standard")
    assert first.digest == generate("authorization", 5, "dev", "standard").digest
    assert first.digest != generate("authorization", 5, "test", "standard").digest
    assert first.digest != generate("authorization", 6, "dev", "standard").digest


def test_grader_accepts_alternative_mitigation_and_rejects_unsafe_fallback(case: TaskCase) -> None:
    bypass = Answer(
        values={
            **case.expected["submission"],
            "changes": [
                {**c, "setting": "cache_enabled", "value": False}
                for c in case.expected["submission"]["changes"]
            ],
        }
    )
    assert grade_authorization(case, bypass)["exact"] == 1
    unsafe = bypass.model_copy(deep=True)
    unsafe.values["changes"].extend(
        {
            "tenant_id": r["tenant_id"],
            "path": r["path"],
            "setting": "fallback",
            "value": "authenticated",
        }
        for r in case.input.tables["auth_runtime"]
    )
    scores = grade_authorization(case, unsafe)
    assert scores["exact"] == 0 and scores["preserved_denials"] < 1
    assert scores["restored_access"] == 1
    assert grade_authorization(case, Answer(values={}))["fraction_correct"] == 0


def test_grader_checks_impact_and_evidence_relations_not_just_existing_ids(case: TaskCase) -> None:
    wrong = Answer(values=case.expected["submission"]).model_copy(deep=True)
    wrong.values["affected_tenants"].append("nonexistent-tenant")
    assert grade_authorization(case, wrong)["tenant_impact"] < 1
    wrong = Answer(values=case.expected["submission"]).model_copy(deep=True)
    renderer = next(
        r for r in case.input.tables["auth_changes"] if r["component"] == "report-renderer"
    )
    wrong.values["evidence"][0]["change_id"] = renderer["change_id"]
    scores = grade_authorization(case, wrong)
    assert scores["evidence_support"] < 1 and scores["exact"] == 0
    wrong = Answer(values=case.expected["submission"]).model_copy(deep=True)
    wrong.values["evidence"].append(dict(wrong.values["evidence"][0]))
    assert grade_authorization(case, wrong)["evidence_support"] < 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"value": 1},
        {"tenant_id": "unknown"},
        {"setting": "grant_everything"},
        {"path": []},
    ],
)
def test_malformed_repairs_score_zero(case: TaskCase, mutation: dict[str, Any]) -> None:
    wrong = Answer(values=case.expected["submission"]).model_copy(deep=True)
    wrong.values["changes"][0] = {
        **wrong.values["changes"][0],
        "setting": "cache_enabled",
        "value": False,
    }
    wrong.values["changes"][0].update(mutation)
    assert grade_authorization(case, wrong)["fraction_correct"] == 0


def test_public_replay_large_queries_isolation_and_tool_budget(case: TaskCase) -> None:
    env = TaskEnvironment(case.input, 12)
    child = env.fork(2)
    try:
        rows = env.query("SELECT * FROM auth_audit LIMIT 100")
        assert "error" not in rows and len(json.dumps(rows).encode()) > 64000
        assert len(env.query("SELECT request_id FROM auth_requests")["rows"]) == 2000
        assert "error" in env.query("SELECT * FROM auth_audit LIMIT 2000")
        req = case.expected["submission"]["evidence"][0]["request_id"]
        patch = json.dumps(case.expected["submission"]["changes"])
        replay = env.query(f"SELECT authz_replay('{patch}', '{req}')")
        assert json.loads(replay["rows"][0][0])["decision"] == "allow"
        original = child.query(f"SELECT authz_replay('[]', '{req}')")
        assert json.loads(original["rows"][0][0])["decision"] == "deny"
        assert "error" in child.query("DROP TABLE auth_policies")
        with pytest.raises(RuntimeError, match="budget"):
            child.query("SELECT 1")
        env.absorb(child)
        assert env.tool_calls == 6
        unknown = env.query("SELECT authz_replay('[]', 'hidden-request')")
        assert "error" in json.loads(unknown["rows"][0][0])
        malformed = env.query(f"SELECT authz_replay('{{}}', '{req}')")
        assert "error" in json.loads(malformed["rows"][0][0])
        assert "error" in env.query("SELECT * FROM expected")
        assert "error" in env.query("ATTACH DATABASE '/tmp/host.db' AS host")
        excessive = env.query(
            "SELECT SUM(length(authz_replay('[]', request_id))) FROM auth_requests"
        )
        assert "2000 invocations" in excessive["error"]
        assert "rows" in env.query(
            "SELECT authz_replay('[]', request_id) FROM auth_requests LIMIT 1"
        )
    finally:
        child.close()
        env.close()


def test_oracle_and_runtime_obey_hand_constructed_entitlements(case: TaskCase) -> None:
    tables = case.input.model_copy(deep=True).tables
    tenant = tables["auth_tenants"][0]["tenant_id"]
    other = tables["auth_tenants"][1]["tenant_id"]
    subject = "user-00000"
    # Override one subject's assignments: inherited payroll allow PLUS direct payroll deny.
    tables["auth_memberships"] = [
        r
        for r in tables["auth_memberships"]
        if (r["tenant_id"], r["subject_id"]) != (tenant, subject)
    ]
    tables["auth_memberships"].extend(
        {"tenant_id": tenant, "subject_id": subject, "group_id": group, "revision": 2}
        for group in ("role-003", "role-002")
    )
    engine = AuthorizationSandbox(tables)
    oracle = Entitlements(tables)
    req = {
        "tenant_id": tenant,
        "token_id": f"token-{tenant}-0-1",
        "path": "interactive",
        "resource_id": f"report-{tenant}-payroll",
        "cache_state": "cold",
        "prior_tenant_id": other,
    }
    assert oracle.decision(req) == engine.evaluate(req, engine.configurations)["decision"] == "deny"
    req["resource_id"] = f"report-{tenant}-summary"
    assert (
        oracle.decision(req) == engine.evaluate(req, engine.configurations)["decision"] == "allow"
    )
    for override in (
        {"token_id": f"token-{tenant}-0-0"},
        {"token_id": f"token-{other}-0-1"},
        {"resource_id": f"report-{other}-summary"},
    ):
        assert oracle.decision({**req, **override}) == "deny"
        assert engine.evaluate({**req, **override}, engine.configurations)["decision"] == "deny"
    # Remove the explicit deny; transitive role-003 -> role-001 now allows payroll.
    tables["auth_memberships"] = [
        r
        for r in tables["auth_memberships"]
        if (r["tenant_id"], r["subject_id"], r["group_id"]) != (tenant, subject, "role-002")
    ]
    req["resource_id"] = f"report-{tenant}-payroll"
    assert Entitlements(tables).decision(req) == "allow"
    revised = AuthorizationSandbox(tables)
    assert revised.evaluate(req, revised.configurations)["decision"] == "allow"


def test_duplicate_setting_changes_rejected(case: TaskCase) -> None:
    engine = AuthorizationSandbox(case.input.tables)
    change = case.expected["submission"]["changes"][0]
    with pytest.raises(ValueError, match="duplicate"):
        parse_changes([change, change], engine.configurations)


def test_export_is_reproducible_public_only_and_never_overwrites(
    case: TaskCase, tmp_path: Path
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    first = export_case(case, a)
    assert export_case(case, b) == first
    assert not (a / "private").exists()
    for metadata in first["tables"].values():
        data = (a / metadata["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
        assert len(data) == metadata["bytes"]
    assert "submission" not in (a / "manifest.json").read_text()
    with pytest.raises(FileExistsError):
        export_case(case, a)
    export_case(case, tmp_path / "operator", include_reference=True)
    assert json.loads((tmp_path / "operator/private/reference.json").read_text()) == case.expected


@pytest.mark.parametrize("strategy", ["single", "single-json", "team", "team-json"])
async def test_existing_model_transports_can_run_repair_experiments(
    case: TaskCase, strategy: str
) -> None:
    count = 0
    req = case.expected["submission"]["evidence"][0]["request_id"]
    sql = f"SELECT authz_replay('[]', '{req}')"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal count
        count += 1
        payload = {"sql": sql} if count == 1 else {"values": case.expected["submission"]}
        if not strategy.endswith("-json"):
            return ModelResponse(
                parts=[ToolCallPart("query" if count == 1 else info.output_tools[0].name, payload)]
            )
        return ModelResponse(parts=[TextPart(json.dumps(payload))])

    env = TaskEnvironment(case.input, 4)
    try:
        if strategy.startswith("team"):
            worker = TeamWorker(
                "hierarchical_dag",
                case.input,
                env,
                FunctionModel(model),
                {},
                Budget(),
                RunUsage(),
                json_protocol=strategy.endswith("-json"),
            )
            output = await worker("Investigate and repair the authorization incident.")
        else:
            output = await (single if strategy == "single" else single_json)(
                case.input, env, FunctionModel(model), {}, Budget(), RunUsage()
            )
        assert grade_authorization(case, output)["exact"] == 1
        assert env.calls[0]["sql"] == sql
        assert all(c["diagnostics"]["feasible"] is None for c in env.state.diagnostic_candidates)
    finally:
        env.close()


async def test_registered_task_runs_through_matrix_and_records_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(SCALES, 80, SMALL)
    case = generate("authorization", 0, "dev", "standard")

    async def adapter(*args: Any) -> Answer:
        return Answer(values=case.expected["submission"])

    monkeypatch.setitem(ADAPTERS, "test-authz-reference", adapter)
    matrix = Matrix(
        models=[ModelSpec(name="offline", model_class="baseline", model="test")],
        strategies=["test-authz-reference"],
        families=["authorization"],
        seeds=[0],
        repetitions=1,
    )
    output = tmp_path / "report.jsonl"
    report = await run_matrix(matrix, output)
    assert report["trials"][0]["scores"]["exact"] == 1
    assert report["trials"][0]["scores"]["preserved_denials"] == 1
    assert read_report(output)["trials"] == report["trials"]
