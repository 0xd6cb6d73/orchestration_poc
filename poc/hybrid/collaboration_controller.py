from __future__ import annotations

from poc.blackboard.models import EpistemicStatus, RecordType
from poc.blackboard.service import BlackboardService
from poc.hybrid.communication import CommunicationService
from poc.hybrid.contracts import (
    Candidate,
    CandidateCluster,
    Challenge,
    CollaborationPhase,
    CollaborationRound,
    EvaluationVector,
    TeamSpec,
    TestResult,
    VisibilityPolicy,
)
from poc.models import AgentInstance, HybridConfig, new_id, utc_now
from poc.persistence.database import Database
from poc.services.artifact_store import ArtifactStore


class CollaborationError(RuntimeError):
    pass


class CollaborationController:
    """Durable application controller for bounded diverge-test-select rounds."""

    policy_version = "diverge_test_select_v1"

    def __init__(
        self,
        db: Database,
        artifacts: ArtifactStore,
        blackboard: BlackboardService,
        communications: CommunicationService,
    ):
        self.db = db
        self.artifacts = artifacts
        self.blackboard = blackboard
        self.communications = communications

    def frame(
        self,
        *,
        run_id: str,
        domain_id: str,
        steward: AgentInstance,
        member_agent_ids: tuple[str, ...],
        config: HybridConfig,
        team_id: str | None = None,
        round_id: str | None = None,
    ) -> CollaborationRound:
        if round_id is not None:
            existing = self.db.get_collaboration_round(round_id)
            if existing is not None:
                return existing
        if config.proposals_per_round > len(member_agent_ids):
            raise CollaborationError("team is smaller than the pinned proposal population")
        participants = tuple(dict.fromkeys((steward.agent_instance_id, *member_agent_ids)))
        edges = frozenset(
            f"{sender}->{recipient}"
            for sender in participants
            for recipient in participants
            if sender != recipient
        )
        team = TeamSpec(
            team_id=team_id or new_id("team"),
            run_id=run_id,
            domain_id=domain_id,
            steward_agent_id=steward.agent_instance_id,
            member_agent_ids=participants,
            permitted_edges=edges,
            message_budget=config.proposals_per_round * config.max_critics_per_candidate * 2,
        )
        self.communications.register_team(team)
        round_ = CollaborationRound(
            round_id=round_id or new_id("round"),
            run_id=run_id,
            domain_id=domain_id,
            steward_agent_id=steward.agent_instance_id,
            team_id=team.team_id,
            phase=CollaborationPhase.COLLECTING_SEALED_PROPOSALS,
            expected_proposals=config.proposals_per_round,
            policy_version=self.policy_version,
        )
        self.db.put_collaboration_round(round_)
        return round_

    def submit_candidate(
        self,
        *,
        round_id: str,
        author: AgentInstance,
        hypothesis_key: str,
        artifact_id: str,
        evidence_refs: tuple[str, ...],
    ) -> Candidate:
        existing = next(
            (
                item
                for item in self.db.list_candidates(author.run_id)
                if item.round_id == round_id and item.author_agent_id == author.agent_instance_id
            ),
            None,
        )
        if existing is not None:
            return existing
        round_ = self._round(round_id, CollaborationPhase.COLLECTING_SEALED_PROPOSALS)
        team = self.db.get_team(round_.team_id)
        if team is None or author.agent_instance_id not in team.member_agent_ids:
            raise CollaborationError("candidate author is not a member of this round")
        candidates = self.db.list_candidates(round_.run_id)
        if sum(item.round_id == round_id for item in candidates) >= round_.expected_proposals:
            raise CollaborationError("the round already has its configured proposal population")
        self.artifacts.set_visibility(
            artifact_id,
            VisibilityPolicy.SEALED_TO_ROUND,
            visibility_ref=round_id,
            access_labels=frozenset({f"agent:{author.agent_instance_id}", f"round:{round_id}"}),
        )
        candidate_id = f"candidate:{round_id}:{author.agent_instance_id}"
        blackboard = self.blackboard.publish(
            actor=author,
            domain_id=round_.domain_id,
            record_type=RecordType.HYPOTHESIS,
            statement=f"Sealed hypothesis {candidate_id}",
            supporting_artifacts=(artifact_id,),
            visibility=VisibilityPolicy.SEALED_TO_ROUND,
            visibility_ref=round_id,
            record_id=f"blackboard:{candidate_id}",
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            round_id=round_id,
            author_agent_id=author.agent_instance_id,
            hypothesis_key=hypothesis_key,
            artifact_id=artifact_id,
            blackboard_record_id=blackboard.record_id,
            evidence_refs=evidence_refs,
        )
        self.db.put_candidate(candidate, event_type="candidate.submitted_sealed")
        updated = round_.model_copy(
            update={
                "submitted_proposals": round_.submitted_proposals + 1,
                "updated_at": utc_now(),
            }
        )
        self.db.put_collaboration_round(updated)
        return candidate

    def release(self, round_id: str, *, minimum_proposals: int | None = None) -> list[Candidate]:
        round_ = self.db.get_collaboration_round(round_id)
        if round_ is None:
            raise KeyError(round_id)
        candidates = [
            item for item in self.db.list_candidates(round_.run_id) if item.round_id == round_id
        ]
        if round_.phase != CollaborationPhase.COLLECTING_SEALED_PROPOSALS:
            if round_.phase in {
                CollaborationPhase.RELEASED,
                CollaborationPhase.CLUSTERED,
                CollaborationPhase.CRITIQUED,
                CollaborationPhase.TESTED,
                CollaborationPhase.RECOMBINED,
                CollaborationPhase.VERIFIED_AND_SCORED,
                CollaborationPhase.COMPLETE,
            }:
                return candidates
            raise CollaborationError(f"round cannot release from phase {round_.phase.value!r}")
        minimum = round_.expected_proposals if minimum_proposals is None else minimum_proposals
        if not 1 <= minimum <= round_.expected_proposals:
            raise ValueError("proposal quorum must be between one and the expected population")
        if len(candidates) < minimum:
            raise CollaborationError(
                "sealed proposals cannot release before the submission condition"
            )
        released: list[Candidate] = []
        steward = self.db.get_agent(round_.steward_agent_id)
        if steward is None:
            raise CollaborationError("collaboration round has no active steward identity")
        for candidate in candidates:
            updated = candidate.model_copy(
                update={
                    "version": candidate.version + 1,
                    "visibility": VisibilityPolicy.RELEASED_TO_TEAM,
                }
            )
            self.artifacts.set_visibility(
                candidate.artifact_id,
                VisibilityPolicy.RELEASED_TO_TEAM,
                visibility_ref=round_.team_id,
            )
            self.blackboard.transition(
                actor=steward,
                record_id=candidate.blackboard_record_id,
                status=EpistemicStatus.PROPOSED,
                visibility=VisibilityPolicy.RELEASED_TO_TEAM,
                visibility_ref=round_.team_id,
            )
            self.db.put_candidate(updated, event_type="candidate.released")
            released.append(updated)
        self._transition(round_, CollaborationPhase.RELEASED)
        return released

    def record_evaluation(
        self,
        *,
        candidate_id: str,
        critic: AgentInstance,
        critique_artifact_id: str,
        passed: bool,
        findings: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        evaluation: EvaluationVector,
    ) -> Candidate:
        candidate = self.db.latest_candidate(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        if candidate.state in {"viable", "refuted", "selected"}:
            return candidate
        round_ = self.db.get_collaboration_round(candidate.round_id)
        if round_ is None or round_.phase not in {
            CollaborationPhase.RELEASED,
            CollaborationPhase.CLUSTERED,
            CollaborationPhase.CRITIQUED,
            CollaborationPhase.TESTED,
            CollaborationPhase.RECOMBINED,
        }:
            raise CollaborationError("candidate is not available for critique and testing")
        if critic.agent_instance_id == candidate.author_agent_id:
            raise CollaborationError("a producer cannot critique its own candidate")
        challenge = Challenge(
            challenge_id=f"challenge:{candidate.round_id}:{candidate_id}:{critic.agent_instance_id}",
            round_id=candidate.round_id,
            candidate_id=candidate_id,
            critic_agent_id=critic.agent_instance_id,
            assumption=findings[0] if findings else "No explicit assumption recorded",
            artifact_id=critique_artifact_id,
        )
        self.db.put_challenge(challenge)
        self.db.put_test_result(
            TestResult(
                test_result_id=f"test:{candidate.round_id}:{candidate_id}:{critic.agent_instance_id}",
                round_id=candidate.round_id,
                candidate_id=candidate_id,
                tester_agent_id=critic.agent_instance_id,
                passed=passed,
                findings=findings,
                evidence_refs=evidence_refs,
            )
        )
        updated = candidate.model_copy(
            update={
                "version": candidate.version + 1,
                "state": "viable" if passed else "refuted",
                "evaluation": evaluation,
            }
        )
        steward = self.db.get_agent(round_.steward_agent_id)
        if steward is None:
            raise CollaborationError("collaboration round has no active steward identity")
        self.blackboard.transition(
            actor=steward,
            record_id=candidate.blackboard_record_id,
            status=EpistemicStatus.SUPPORTED if passed else EpistemicStatus.REFUTED,
            supporting_artifacts=evidence_refs if passed else (),
            challenging_artifacts=() if passed else evidence_refs,
        )
        self.db.put_candidate(
            updated, event_type="candidate.retained" if passed else "candidate.refuted"
        )
        self.blackboard.publish(
            actor=critic,
            domain_id=round_.domain_id,
            record_type=RecordType.DECISION,
            statement=(
                f"Candidate {candidate_id} survived evidence testing"
                if passed
                else f"Candidate {candidate_id} was refuted by evidence testing"
            ),
            supporting_artifacts=evidence_refs if passed else (),
            challenging_artifacts=() if passed else evidence_refs,
            status=EpistemicStatus.SUPPORTED if passed else EpistemicStatus.REFUTED,
            visibility=VisibilityPolicy.DOMAIN,
            visibility_ref=round_.domain_id,
        )
        return updated

    def cluster_candidates(self, round_id: str) -> list[CandidateCluster]:
        round_ = self.db.get_collaboration_round(round_id)
        if round_ is None:
            raise KeyError(round_id)
        if round_.phase != CollaborationPhase.RELEASED:
            if round_.phase in {
                CollaborationPhase.CLUSTERED,
                CollaborationPhase.CRITIQUED,
                CollaborationPhase.TESTED,
                CollaborationPhase.RECOMBINED,
                CollaborationPhase.VERIFIED_AND_SCORED,
                CollaborationPhase.COMPLETE,
            }:
                return self.db.list_candidate_clusters(round_.run_id)
            raise CollaborationError(f"round cannot cluster from phase {round_.phase.value!r}")
        candidates = [
            item for item in self.db.list_candidates(round_.run_id) if item.round_id == round_id
        ]
        by_hypothesis: dict[str, list[str]] = {}
        for candidate in candidates:
            by_hypothesis.setdefault(candidate.hypothesis_key, []).append(candidate.candidate_id)
        clusters: list[CandidateCluster] = []
        for hypothesis_key, candidate_ids in sorted(by_hypothesis.items()):
            cluster = CandidateCluster(
                cluster_id=f"cluster:{round_id}:{hypothesis_key}",
                round_id=round_id,
                label=hypothesis_key,
                candidate_ids=tuple(sorted(candidate_ids)),
                rationale="Candidates share the same explicit hypothesis key.",
            )
            self.db.put_candidate_cluster(cluster)
            clusters.append(cluster)
        self._transition(round_, CollaborationPhase.CLUSTERED)
        return clusters

    def record_recombination(
        self,
        *,
        round_id: str,
        author: AgentInstance,
        hypothesis_key: str,
        artifact_id: str,
        evidence_refs: tuple[str, ...],
        parent_candidate_ids: tuple[str, ...],
    ) -> Candidate:
        round_ = self._round(round_id, CollaborationPhase.TESTED)
        if len(set(parent_candidate_ids)) < 2:
            raise CollaborationError("recombination requires at least two distinct parents")
        normalized_parents = tuple(dict.fromkeys(parent_candidate_ids))
        existing = next(
            (
                candidate
                for candidate in self.db.list_candidates(round_.run_id)
                if candidate.round_id == round_id
                and candidate.hypothesis_key == hypothesis_key
                and candidate.parent_candidate_ids == normalized_parents
            ),
            None,
        )
        if existing is not None:
            return existing
        known = {
            candidate.candidate_id
            for candidate in self.db.list_candidates(round_.run_id)
            if candidate.round_id == round_id
        }
        if not set(parent_candidate_ids) <= known:
            raise CollaborationError("recombination references an unknown candidate")
        team = self.db.get_team(round_.team_id)
        if team is None or author.agent_instance_id not in team.member_agent_ids:
            raise CollaborationError("recombination author is not a member of this round")
        self.artifacts.set_visibility(
            artifact_id,
            VisibilityPolicy.RELEASED_TO_TEAM,
            visibility_ref=round_.team_id,
        )
        candidate_id = f"candidate:{round_id}:recombined:{hypothesis_key}"
        blackboard = self.blackboard.publish(
            actor=author,
            domain_id=round_.domain_id,
            record_type=RecordType.HYPOTHESIS,
            statement=f"Recombined hypothesis {candidate_id}",
            supporting_artifacts=(artifact_id, *evidence_refs),
            visibility=VisibilityPolicy.RELEASED_TO_TEAM,
            visibility_ref=round_.team_id,
            record_id=f"blackboard:{candidate_id}",
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            round_id=round_id,
            author_agent_id=author.agent_instance_id,
            hypothesis_key=hypothesis_key,
            artifact_id=artifact_id,
            blackboard_record_id=blackboard.record_id,
            evidence_refs=evidence_refs,
            visibility=VisibilityPolicy.RELEASED_TO_TEAM,
            parent_candidate_ids=normalized_parents,
        )
        self.db.put_candidate(candidate, event_type="candidate.recombined")
        return candidate

    def revise_candidate(
        self,
        *,
        round_id: str,
        author: AgentInstance,
        hypothesis_key: str,
        artifact_id: str,
        evidence_refs: tuple[str, ...] = (),
        parent_candidate_ids: tuple[str, ...] = (),
    ) -> Candidate:
        """Publish a revision (or an added task with no parents) during the decision loop."""
        round_ = self._round(round_id, CollaborationPhase.TESTED)
        normalized_parents = tuple(dict.fromkeys(parent_candidate_ids))
        known = {
            candidate.candidate_id
            for candidate in self.db.list_candidates(round_.run_id)
            if candidate.round_id == round_id
        }
        if not set(normalized_parents) <= known:
            raise CollaborationError("revision references an unknown candidate")
        team = self.db.get_team(round_.team_id)
        if team is None or author.agent_instance_id not in team.member_agent_ids:
            raise CollaborationError("revision author is not a member of this round")
        self.artifacts.set_visibility(
            artifact_id,
            VisibilityPolicy.RELEASED_TO_TEAM,
            visibility_ref=round_.team_id,
        )
        revision_index = sum(
            1
            for candidate in self.db.list_candidates(round_.run_id)
            if candidate.round_id == round_id
            and candidate.hypothesis_key == hypothesis_key
            and candidate.candidate_id.startswith(f"candidate:{round_id}:revised:{hypothesis_key}")
        )
        candidate_id = f"candidate:{round_id}:revised:{hypothesis_key}:{revision_index + 1}"
        blackboard = self.blackboard.publish(
            actor=author,
            domain_id=round_.domain_id,
            record_type=RecordType.HYPOTHESIS,
            statement=f"Revised hypothesis {candidate_id}",
            supporting_artifacts=(artifact_id, *evidence_refs),
            visibility=VisibilityPolicy.RELEASED_TO_TEAM,
            visibility_ref=round_.team_id,
            record_id=f"blackboard:{candidate_id}",
        )
        candidate = Candidate(
            candidate_id=candidate_id,
            round_id=round_id,
            author_agent_id=author.agent_instance_id,
            hypothesis_key=hypothesis_key,
            artifact_id=artifact_id,
            blackboard_record_id=blackboard.record_id,
            evidence_refs=evidence_refs,
            visibility=VisibilityPolicy.RELEASED_TO_TEAM,
            parent_candidate_ids=normalized_parents,
        )
        self.db.put_candidate(candidate, event_type="candidate.revised")
        return candidate

    def select(self, round_id: str) -> Candidate:
        round_ = self.db.get_collaboration_round(round_id)
        if round_ is None:
            raise KeyError(round_id)
        already_selected = next(
            (
                item
                for item in self.db.list_candidates(round_.run_id)
                if item.round_id == round_id and item.state == "selected"
            ),
            None,
        )
        if already_selected is not None:
            return already_selected
        if round_.phase != CollaborationPhase.RECOMBINED:
            raise CollaborationError(
                f"round phase is {round_.phase.value!r}, expected "
                f"{CollaborationPhase.RECOMBINED.value!r}"
            )
        candidates = [
            item
            for item in self.db.list_candidates(round_.run_id)
            if item.round_id == round_id and item.state == "viable" and item.evaluation is not None
        ]
        if not candidates:
            self._transition(round_, CollaborationPhase.REFRAME)
            raise CollaborationError("no candidate survived evidence-based testing")
        selected = max(
            candidates,
            key=_candidate_score,
        )
        selected = selected.model_copy(
            update={"version": selected.version + 1, "state": "selected"}
        )
        self.db.put_candidate(selected, event_type="candidate.selected")
        return selected

    def advance(self, round_id: str, phase: CollaborationPhase) -> CollaborationRound:
        current = self.db.get_collaboration_round(round_id)
        if current is None:
            raise KeyError(round_id)
        allowed = {
            CollaborationPhase.RELEASED: CollaborationPhase.CLUSTERED,
            CollaborationPhase.CLUSTERED: CollaborationPhase.CRITIQUED,
            CollaborationPhase.CRITIQUED: CollaborationPhase.TESTED,
            CollaborationPhase.TESTED: CollaborationPhase.RECOMBINED,
            CollaborationPhase.RECOMBINED: CollaborationPhase.VERIFIED_AND_SCORED,
        }
        if allowed.get(current.phase) != phase:
            if current.phase == phase:
                return current
            raise CollaborationError(
                f"invalid collaboration transition {current.phase.value!r} -> {phase.value!r}"
            )
        return self._transition(current, phase)

    def complete(self, round_id: str) -> CollaborationRound:
        round_ = self.db.get_collaboration_round(round_id)
        if round_ is None:
            raise KeyError(round_id)
        if round_.phase == CollaborationPhase.COMPLETE:
            return round_
        if round_.phase != CollaborationPhase.VERIFIED_AND_SCORED:
            raise CollaborationError(
                f"round phase is {round_.phase.value!r}, expected "
                f"{CollaborationPhase.VERIFIED_AND_SCORED.value!r}"
            )
        return self._transition(round_, CollaborationPhase.COMPLETE)

    def _round(self, round_id: str, expected_phase: CollaborationPhase) -> CollaborationRound:
        round_ = self.db.get_collaboration_round(round_id)
        if round_ is None:
            raise KeyError(round_id)
        if round_.phase != expected_phase:
            raise CollaborationError(
                f"round phase is {round_.phase.value!r}, expected {expected_phase.value!r}"
            )
        return round_

    def _transition(
        self, round_: CollaborationRound, phase: CollaborationPhase
    ) -> CollaborationRound:
        updated = round_.model_copy(update={"phase": phase, "updated_at": utc_now()})
        self.db.put_collaboration_round(updated)
        return updated


def _evaluation_total(evaluation: EvaluationVector) -> int:
    return (
        evaluation.validity
        + evaluation.evidence
        + evaluation.usefulness
        + evaluation.novelty
        + evaluation.constraint_satisfaction
    )


def _candidate_score(candidate: Candidate) -> tuple[int, int, str]:
    evaluation = candidate.evaluation
    if evaluation is None:
        return (-1, -1, candidate.candidate_id)
    return (_evaluation_total(evaluation), evaluation.novelty, candidate.candidate_id)
