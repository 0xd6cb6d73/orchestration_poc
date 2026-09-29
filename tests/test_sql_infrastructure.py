from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage, RunUsage

from poc.evaluation.suite import runner
from poc.evaluation.suite.adapters import ADAPTERS
from poc.evaluation.suite.batches import monitor
from poc.evaluation.suite.environment import TaskEnvironment
from poc.evaluation.suite.models import Matrix, ModelSpec
from poc.evaluation.suite.tasks import generate
from poc.execution.sql_artifacts import (
    ArtifactContractError,
    CommittedCandidates,
    validate_artifact,
)
from poc.execution.sql_contracts import (
    Answer,
    ArchitectureOptions,
    Budget,
    RoleBudgetProfile,
    TaskInput,
)
from poc.execution.sql_protocol import ProtocolError, parse_action
from poc.execution.sql_strategy import ContextBoundModel, single_json
from poc.execution.sql_team_worker import TeamWorker

TASK = TaskInput(prompt="Return x from data", tables={"data": [{"x": 7}]})


def fixture_model(*, reject: bool = False, fail_proposal: bool = False) -> FunctionModel:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = str(messages)
        if fail_proposal and "sealed proposer 1" in prompt:
            raise UnexpectedModelBehavior("fixture proposer failed")
        refs = re.findall(r"query:(?:critique|verify):\d+:\d+", prompt)
        if "Your role is critic" in prompt or "Your role is verifier" in prompt:
            if not refs:
                if info.function_tools:
                    return ModelResponse(
                        parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                    )
                return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
            if "Your role is critic" in prompt:
                values: dict[str, Any] = {
                    "structural_validity": True,
                    "feasibility": "unsupported" if reject else "supported",
                    "evidence_refs": [refs[-1]],
                    "reason": "Observed public x",
                    "evaluation": dict.fromkeys(
                        (
                            "validity",
                            "evidence",
                            "usefulness",
                            "novelty",
                            "constraint_satisfaction",
                        ),
                        4,
                    ),
                }
            else:
                # The verifier must cite its own query, not the critic's refs in its prompt.
                own = [ref for ref in refs if ref.startswith("query:verify:")]
                if not own:
                    if info.function_tools:
                        return ModelResponse(
                            parts=[ToolCallPart("query", {"sql": "SELECT x FROM data"})]
                        )
                    return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
                values = {
                    "answer_supported": not reject,
                    "evidence_refs": [own[-1]],
                    "reason": "Checked x",
                }
        elif "Your role is planner" in prompt:
            values = {"plan": "Read public x"}
        else:
            values = {"x": 7}
        await asyncio.sleep(0.01)
        payload = {"values": values}
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, payload)]
            if info.output_tools
            else [TextPart(json.dumps(payload))]
        )

    return FunctionModel(respond)


@pytest.mark.parametrize("native", [True, False])
async def test_endpoint_cap_applies_after_stage_override(native: bool) -> None:
    events: list[dict[str, Any]] = []
    seen: list[int] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_settings
        seen.append(info.model_settings.get("max_tokens") or 0)
        payload = {"values": {"x": 7}}
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, payload)]
            if native
            else [TextPart(json.dumps(payload))]
        )

    env = TaskEnvironment(TASK, 10)
    env.state.emit = events.append
    try:
        model = ContextBoundModel(FunctionModel(respond), 128000, 8192)
        await ADAPTERS["single" if native else "single-json"](
            TASK, env, model, {"max_tokens": 131072}, Budget(total_tokens=1000000), RunUsage()
        )
        assert seen == [8192]
        assert (
            next(e["request_settings"] for e in events if "request_settings" in e).get("max_tokens")
            == 8192
        )
        assert TASK.tables == {"data": [{"x": 7}]}
        assert model.admit([], None, ModelRequestParameters()).get("max_tokens") == 8192
        assert (
            ContextBoundModel(FunctionModel(respond), 128000)
            .admit([], None, ModelRequestParameters())
            .get("max_tokens")
            == 16384
        )
    finally:
        env.close()


@pytest.mark.parametrize("family", ["scheduling", "routing", "ledger", "dependencies", "access"])
def test_public_contract_rejects_id_drift_empty_and_types(family: str) -> None:
    case = generate(family, 7, "test", "standard")
    valid = Answer(values=case.expected)
    assert validate_artifact(case.input, valid) is valid
    for values in ({}, {"0": 0}, {**case.expected, next(iter(case.expected)): True}):
        with pytest.raises(ArtifactContractError):
            validate_artifact(case.input, Answer(values=values))
    subset = Answer(values=dict(list(case.expected.items())[:1]))
    assert validate_artifact(case.input, subset, allow_partial=True) is subset
    with pytest.raises(ArtifactContractError):
        validate_artifact(
            case.input, Answer(values={**subset.values, "unlisted-id": 0}), allow_partial=True
        )
    with pytest.raises(ArtifactContractError):
        validate_artifact(
            case.input, Answer(values={next(iter(subset.values)): True}), allow_partial=True
        )
    if family == "scheduling":
        # Shape-valid but known infeasible answers still reach grading.
        wrong = Answer(values=dict.fromkeys(case.expected, 0))
        assert validate_artifact(case.input, wrong) is wrong


@pytest.mark.parametrize("native", [True, False])
async def test_empty_revision_cannot_replace_committed_draft(native: bool) -> None:
    case = generate("scheduling", 0, "test", "standard")
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        payload: dict[str, Any] = (
            {"values": case.expected}
            if calls == 1
            else {"action": "revise", "reason": "bad replacement", "replacement": {"values": {}}}
        )
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, payload)]
            if native
            else [TextPart(json.dumps(payload))]
        )

    env = TaskEnvironment(case.input, 20)
    env.state.options = ArchitectureOptions(
        review_protocol="decision-v1", review_failure_policy="return_submitted_draft"
    )
    try:
        answer = await ADAPTERS["review" if native else "review-json"](
            case.input, env, FunctionModel(respond), {}, Budget(), RunUsage()
        )
        assert answer.values == case.expected
        assert calls == 3 and env.state.answer_source == "draft_fallback"
        assert env.state.review_decision is None
        assert all(c["answer"]["values"] for c in env.state.candidates)
    finally:
        env.close()


def test_committed_candidates_are_immutable() -> None:
    committed = CommittedCandidates()
    answer = Answer(values={"x": {"nested": 1}})
    committed.append(answer)
    answer.values["x"]["nested"] = 9
    committed[0].values.clear()
    assert committed[0].values == {"x": {"nested": 1}}


@pytest.mark.parametrize("policy", ["reliable-v2", "concurrent-v1"])
@pytest.mark.parametrize("native", [True, False])
async def test_hybrid_requires_supported_critique_and_independent_verification(
    policy: str, native: bool
) -> None:
    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions.model_validate({"team_policy": policy})
    events: list[dict[str, Any]] = []
    env.state.emit = events.append
    try:
        result = await ADAPTERS["hybrid_v1" + ("" if native else "-json")](
            TASK, env, fixture_model(), {}, Budget(requests=100), RunUsage()
        )
        assert result.values == {"x": 7}
        assert env.tool_calls == 3
        assert sum("evidence" in e for e in events) == 3
        assert len({e["evidence"]["ref"] for e in events if "evidence" in e}) == 3
        assert {p["name"] for p in env.state.phases} == {"proposal", "critique", "verify"}
    finally:
        env.close()


@pytest.mark.parametrize("quorum", [1, 2])
async def test_hybrid_branch_failure_respects_explicit_quorum(quorum: int) -> None:
    from poc.hybrid.collaboration_controller import CollaborationError

    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(
        team_policy="concurrent-v1", hybrid_proposal_quorum=quorum
    )
    try:
        run = ADAPTERS["hybrid_v1-json"](
            TASK, env, fixture_model(fail_proposal=True), {}, Budget(requests=100), RunUsage()
        )
        if quorum == 1:
            assert (await run).values == {"x": 7}
            assert any(p["name"] == "verify" for p in env.state.phases)
        else:
            with pytest.raises(CollaborationError, match="submission condition"):
                await run
            assert not any(p["name"] == "verify" for p in env.state.phases)
    finally:
        env.close()


async def test_rejecting_all_hybrid_candidates_does_not_bypass_gate() -> None:
    from poc.hybrid.collaboration_controller import CollaborationError

    env = TaskEnvironment(TASK, 100)
    try:
        with pytest.raises(CollaborationError):
            await ADAPTERS["hybrid_v1-json"](
                TASK, env, fixture_model(reject=True), {}, Budget(requests=100), RunUsage()
            )
        assert not any(p["name"] == "verify" for p in env.state.phases)
    finally:
        env.close()


@pytest.mark.parametrize(
    "method", ["hierarchical_dag", "board_claim", "managed_pool", "speculative"]
)
async def test_concurrent_branches_overlap_and_account_once(method: str) -> None:
    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(team_policy="concurrent-v1")
    events: list[dict[str, Any]] = []
    env.state.emit = events.append
    usage = RunUsage()
    try:
        answer = await ADAPTERS[method + "-json"](
            TASK, env, fixture_model(), {}, Budget(requests=100), usage
        )
        assert answer.values == {"x": 7}
        phases = sorted(env.state.phases, key=lambda p: p["branch_index"])
        assert (
            phases[1]["start_seconds"] < phases[0]["start_seconds"] + phases[0]["elapsed_seconds"]
        )
        assert usage.requests == len(phases)
        reservations = [e["budget_reservation"] for e in events if "budget_reservation" in e]
        assert sum(r["requests"] for r in reservations) <= 100
        assert sum(r["total_tokens"] for r in reservations) <= 100000
        if method == "board_claim":
            assert any(
                e.get("orchestration", {}).get("event_type") == "sql.claim_contended"
                for e in events
            )
    finally:
        env.close()


async def test_pool_reassigns_failed_plan_with_a_new_generation() -> None:
    calls = 0
    base = fixture_model()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise UnexpectedModelBehavior("fixture worker failure")
        assert base.function is not None
        return await cast(Awaitable[ModelResponse], base.function(messages, info))

    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(team_policy="concurrent-v1")
    events: list[dict[str, Any]] = []
    env.state.emit = events.append
    try:
        assert (
            await ADAPTERS["managed_pool-json"](
                TASK, env, FunctionModel(respond), {}, Budget(requests=100), RunUsage()
            )
        ).values == {"x": 7}
        created = [
            e["orchestration"]["data"]
            for e in events
            if e.get("orchestration", {}).get("event_type") == "assignment.created"
        ]
        retried = [a for a in created if a["task_id"] == "plan-0"]
        assert len(retried) == 2
        assert retried[0]["worker_id"] != retried[1]["worker_id"]
        assert retried[1]["assignment_generation"] == 2
    finally:
        env.close()


@pytest.mark.parametrize(
    "raw,kind",
    [
        ("{}{}", "multiple_actions"),
        ('{"values":{"x":1,"x":2}}', "duplicate_json_key"),
        ("[]", "expected_action_object"),
    ],
)
def test_protocol_rejects_ambiguous_actions(raw: str, kind: str) -> None:
    with pytest.raises(ProtocolError, match=kind):
        parse_action(raw, "json-normalize-v1")


def test_transport_normalization_is_opt_in_and_lossless() -> None:
    for text in ('```json\n{"values":{"x":7}}\n```', json.dumps('{"values":{"x":7}}')):
        with pytest.raises(ProtocolError):
            parse_action(text)
        assert parse_action(text, "json-normalize-v1") == {"values": {"x": 7}}


async def test_reasoning_only_response_never_executes_text() -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart('{"sql":"SELECT x FROM data"}')],
            usage=RequestUsage(input_tokens=10, output_tokens=20),
        )

    env = TaskEnvironment(TASK, 10)
    usage = RunUsage()
    try:
        with pytest.raises(UnexpectedModelBehavior, match="reasoning_only"):
            await single_json(TASK, env, FunctionModel(respond), {}, Budget(), usage)
        assert env.tool_calls == 0 and not env.state.candidates
        assert usage.requests == 1 and usage.total_tokens == 30
    finally:
        env.close()


@pytest.mark.parametrize("requests,success", [(2, True), (1, False)])
async def test_transient_retries_consume_declared_request_budget(
    requests: int, success: bool
) -> None:
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelHTTPError(429, "fixture", {"secret": "never log this"})
        return ModelResponse(parts=[TextPart('{"values":{"x":7}}')])

    env = TaskEnvironment(TASK, 10)
    env.state.options = ArchitectureOptions(provider_retries=1)
    usage = RunUsage()
    try:
        run = single_json(TASK, env, FunctionModel(respond), {}, Budget(requests=requests), usage)
        if success:
            assert (await run).values == {"x": 7}
        else:
            with pytest.raises(ModelHTTPError):
                await run
        assert calls == requests and usage.requests == requests
        assert not env.state.usage_complete
    finally:
        env.close()


async def test_role_profile_and_downstream_reserve_are_effective() -> None:
    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(
        stage_weights=[0.6, 0.4],
        role_profiles={
            "plan": RoleBudgetProfile(
                measurement_source="fixture:491s-plan",
                max_output_tokens=24000,
                request_timeout_seconds=510,
            )
        },
    )
    try:
        worker = TeamWorker(
            "hierarchical_dag",
            TASK,
            env,
            fixture_model(),
            {},
            Budget(seconds=900, total_tokens=1000000),
            RunUsage(),
            json_protocol=True,
        )
        await worker("Plan", "plan")
        phase = env.state.phases[0]
        assert phase["max_output_tokens"] == 24000
        assert 491 < phase["deadline_seconds"] < 540
        assert phase["token_limit"] == 600000
    finally:
        env.close()


async def test_phase_failure_keeps_scope_and_submission_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(10)
        raise AssertionError

    def resolve(spec: ModelSpec) -> FunctionModel:
        return FunctionModel(respond)

    monkeypatch.setattr(runner, "resolve_model", resolve)
    cfg = Matrix(
        models=[ModelSpec(name="fixture", model="fixture", model_class="baseline")],
        strategies=["review-json"],
        budget=Budget(seconds=0.1, draft_fraction=0.56),
    )
    result = await runner.run_trial(
        generate("scheduling", 0, "test", "standard"), cfg.models[0], "review-json", 1, cfg
    )
    assert result["status"] == "phase_timeout"
    assert result["failure"]["scope"] == "phase" and result["failure"]["stage"] == "draft"
    assert result["failure"]["threshold"]["seconds"] < 0.06
    summary = runner.summarize([result])[0]
    assert summary["no_submissions"] == 1 and summary["invalid_answers"] == 0


def test_monitor_excludes_annotation_verification(tmp_path: Path) -> None:
    (tmp_path / "index.json").write_text(json.dumps([{"name": "batch", "trials": 1}]))
    (tmp_path / "batch-verification.json").write_text("{}")
    (tmp_path / "annotations-verification.json").write_text("{}")
    assert monitor(tmp_path)["verified_batches"] == 1


@pytest.mark.parametrize("method,duration", [("hierarchical_dag", 491), ("review-json", 507)])
async def test_measured_long_stage_retains_downstream_time(
    monkeypatch: pytest.MonkeyPatch, method: str, duration: int
) -> None:
    from poc.execution import sql_strategy, sql_team_worker

    clock = [0.0]
    monkeypatch.setattr(sql_strategy, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(sql_team_worker, "perf_counter", lambda: clock[0])
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        clock[0] = float(duration if calls == 1 else duration + 16)
        values = {"plan": "Read x"} if calls == 1 and method == "hierarchical_dag" else {"x": 7}
        return ModelResponse(parts=[TextPart(json.dumps({"values": values}))])

    env = TaskEnvironment(TASK, 20)
    env.state.started = 0
    try:
        if method == "hierarchical_dag":
            env.state.options = ArchitectureOptions(stage_weights=[0.65, 0.35])
            worker = TeamWorker(
                method,
                TASK,
                env,
                FunctionModel(respond),
                {},
                Budget(seconds=900),
                RunUsage(),
                json_protocol=True,
            )
            await worker("Plan", "plan")
            result = await worker("Solve", "solve")
        else:
            result = await ADAPTERS[method](
                TASK,
                env,
                FunctionModel(respond),
                {},
                Budget(seconds=900, draft_fraction=0.65),
                RunUsage(),
            )
        assert result.values == {"x": 7}
        assert env.state.phases[0]["elapsed_seconds"] == duration
        assert env.state.phases[-1]["status"] == "completed"
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["zero_constraints", "foreign_evidence", "abstain"])
async def test_hybrid_cannot_approve_inconsistent_or_unsupported_critique(kind: str) -> None:
    from poc.hybrid.collaboration_controller import CollaborationError

    base = fixture_model()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert base.function is not None
        response = await cast(Awaitable[ModelResponse], base.function(messages, info))
        part = response.parts[0]
        if "Your role is critic" in str(messages) and isinstance(part, TextPart):
            payload = json.loads(part.content)
            if "values" in payload:
                if kind == "zero_constraints":
                    payload["values"]["evaluation"].update(validity=1, constraint_satisfaction=0)
                elif kind == "foreign_evidence":
                    payload["values"]["evidence_refs"] = ["query:another-worker:0:1"]
                else:
                    payload["values"]["feasibility"] = "abstain"
                return ModelResponse(parts=[TextPart(json.dumps(payload))])
        return response

    env = TaskEnvironment(TASK, 100)
    try:
        with pytest.raises(CollaborationError):
            await ADAPTERS["hybrid_v1-json"](
                TASK, env, FunctionModel(respond), {}, Budget(requests=100), RunUsage()
            )
        assert not any(p["name"] == "verify" for p in env.state.phases)
    finally:
        env.close()


@pytest.mark.parametrize("native", [True, False])
async def test_malformed_reconciliation_returns_first_committed_artifact(native: bool) -> None:
    calls = 0

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        payload: dict[str, Any] = (
            {"values": {"x": -9 if calls == 1 else 7}} if calls <= 2 else {"values": {}}
        )
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, payload)]
            if native
            else [TextPart(json.dumps(payload))]
        )

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(speculative_failure_policy="return_first_submitted")
    try:
        result = await ADAPTERS["speculative" + ("" if native else "-json")](
            TASK, env, FunctionModel(respond), {}, Budget(), RunUsage()
        )
        assert result.values == {"x": -9}
        assert calls == 4 and env.state.answer_source == "first_submitted_fallback"
    finally:
        env.close()


@pytest.mark.parametrize("shape", ["error", "duplicate", "multiple"])
async def test_invalid_provider_shapes_cannot_execute_tools(shape: str) -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if shape == "error":
            return ModelResponse(parts=[TextPart('{"values":{"x":7}}')], finish_reason="error")
        if shape == "duplicate":
            return ModelResponse(
                parts=[ToolCallPart("query", '{"sql":"SELECT 1","sql":"SELECT 2"}')]
            )
        return ModelResponse(
            parts=[
                ToolCallPart("query", {"sql": "SELECT 1"}),
                ToolCallPart("query", {"sql": "SELECT 2"}),
            ]
        )

    env = TaskEnvironment(TASK, 20)
    usage = RunUsage()
    try:
        with pytest.raises(UnexpectedModelBehavior):
            await single_json(TASK, env, FunctionModel(respond), {}, Budget(), usage)
        assert not env.calls and not env.state.candidates
        assert usage.requests == 1
    finally:
        env.close()


def test_new_coordination_families_preserve_public_contracts() -> None:
    for family in ("partitioned_ledger", "dependency_join"):
        for seed in (101, 102):
            case = generate(family, seed, "test", "standard")
            validate_artifact(case.input, Answer(values=case.expected))
            assert case.digest == generate(family, seed, "test", "standard").digest
    for path in Path("configs").glob("evaluation-*-teams.json"):
        Matrix.model_validate_json(path.read_text())


def test_profile_calibration_never_uses_grader_success() -> None:
    from poc.evaluation.suite.budget_profiles import measured_profiles

    rows = [
        {
            "model_name": "fixture",
            "scores": {"exact": 0},
            "phases": [
                {
                    "name": "plan",
                    "output_tokens": 10000,
                    "elapsed_seconds": 491,
                    "usage_complete": True,
                    "status": "timeout",
                }
            ],
        }
    ]
    first = measured_profiles(rows, "fixture")
    rows[0]["scores"] = {"exact": 1}
    assert measured_profiles(rows, "fixture") == first
    assert first["fixture"]["plan"].max_output_tokens == 12500


@pytest.mark.parametrize("policy", ["reliable-v2", "concurrent-v1"])
async def test_reconciliation_timeout_returns_before_harness_deadline(policy: str) -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if "Reconcile these independent" in str(messages):
            await asyncio.sleep(10)
        return ModelResponse(parts=[TextPart('{"values":{"x":7}}')])

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions.model_validate(
        {
            "team_policy": policy,
            "speculative_failure_policy": "return_first_submitted",
            "return_reserve_seconds": 0.05,
        }
    )
    try:
        async with asyncio.timeout(0.4):
            result = await ADAPTERS["speculative-json"](
                TASK, env, FunctionModel(respond), {}, Budget(seconds=0.4), RunUsage()
            )
        assert result.values == {"x": 7}
        assert env.state.recovery and env.state.recovery["error"] == "PhaseTimeout"
    finally:
        env.close()


async def test_concurrent_cancellation_drains_workers_and_closes_connections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    entered = 0
    drained = 0
    started = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal entered, drained
        entered += 1
        if entered == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained += 1
        raise AssertionError

    env = TaskEnvironment(TASK, 20)
    env.state.options = ArchitectureOptions(team_policy="concurrent-v1")
    running = asyncio.ensure_future(
        ADAPTERS["speculative-json"](TASK, env, FunctionModel(respond), {}, Budget(), RunUsage())
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert drained == 2 and list(tmp_path.iterdir()) == []
    finally:
        env.close()


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("role", ["orchestrate", "critique", "task"])
async def test_judge_roles_carry_provider_reasoning_limits(role: str, native: bool) -> None:
    seen: list[dict[str, Any]] = []
    base = fixture_model()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        assert base.function is not None
        return await cast(Awaitable[ModelResponse], base.function(messages, info))

    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(team_policy="reliable-v2")
    try:
        await TeamWorker(
            "hierarchical_dag",
            TASK,
            env,
            FunctionModel(respond),
            {},
            Budget(requests=100),
            RunUsage(),
            json_protocol=not native,
        )("Judge", cast(Any, role))
        assert seen
        for settings in seen:
            if role in {"orchestrate", "critique"}:
                assert settings["openrouter_reasoning"] == {"effort": "low"}
            else:
                assert "openrouter_reasoning" not in settings
    finally:
        env.close()


@pytest.mark.parametrize("role,expected", [("orchestrate", 180), ("critique", 180), ("task", 600)])
async def test_judge_stage_budget_clamps_request_timeout(role: str, expected: float) -> None:
    events: list[dict[str, Any]] = []
    env = TaskEnvironment(TASK, 100)
    env.state.options = ArchitectureOptions(team_policy="reliable-v2")
    env.state.emit = events.append
    try:
        await TeamWorker(
            "hierarchical_dag",
            TASK,
            env,
            fixture_model(),
            {},
            Budget(requests=100, seconds=900, request_timeout_seconds=600),
            RunUsage(),
            json_protocol=True,
        )("Judge", cast(Any, role))
        observed = {
            e["request_settings"]["stage"]: e["request_settings"]["request_timeout_seconds"]
            for e in events
            if "request_settings" in e
        }
        assert observed[role] == expected
    finally:
        env.close()


async def test_invalid_turn_retry_drops_invalid_response_keeps_prompts() -> None:
    calls: list[list[ModelMessage]] = []
    events: list[dict[str, Any]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls.append(list(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart("not an action")])
        return ModelResponse(parts=[TextPart('{"values":{"x":7}}')])

    env = TaskEnvironment(TASK, 10)
    env.state.emit = events.append
    try:
        answer = await TeamWorker(
            "hierarchical_dag",
            TASK,
            env,
            FunctionModel(respond),
            {},
            Budget(requests=10),
            RunUsage(),
            json_protocol=True,
        )("Solve", "solve")
        assert answer.values == {"x": 7}
        assert any("protocol_error" in e for e in events)
        assert len(calls) == 2
        assert len(calls[0]) == 1
        # The retry keeps legitimate prompts and drops the invalid model response:
        # the original task prompt is preserved and the retry instruction is appended.
        assert len(calls[1]) == 1
        assert not isinstance(calls[1][0], ModelResponse)
        assert "Return x from data" in str(calls[1][0])
        assert "Invalid JSON" in str(calls[1][0])
    finally:
        env.close()


async def test_invalid_turn_retry_keeps_query_evidence() -> None:
    calls: list[list[ModelMessage]] = []
    events: list[dict[str, Any]] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls.append(list(messages))
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart('{"sql":"SELECT x FROM data"}')])
        if len(calls) == 2:
            return ModelResponse(parts=[TextPart("not an action")])
        return ModelResponse(parts=[TextPart('{"values":{"x":7}}')])

    env = TaskEnvironment(TASK, 10)
    env.state.emit = events.append
    try:
        answer = await TeamWorker(
            "hierarchical_dag",
            TASK,
            env,
            FunctionModel(respond),
            {},
            Budget(requests=100),
            RunUsage(),
            json_protocol=True,
        )("Solve", "solve")
        assert answer.values == {"x": 7}
        # The retry after an invalid output still carries the SQL evidence, so the
        # model can answer instead of re-querying from a blank conversation.
        assert len(calls) == 3
        assert "SQL result:" in str(calls[2][0])
        assert all(not isinstance(m, ModelResponse) for m in calls[2])
    finally:
        env.close()
