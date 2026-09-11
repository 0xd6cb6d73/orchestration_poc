from __future__ import annotations

from poc.hybrid.contracts import (
    AcceptanceDecision,
    VerificationRequest,
    VerificationVerdict,
)
from poc.models import AgentInstance, Tier
from poc.persistence.database import Database


class VerificationDenied(PermissionError):
    pass


class VerificationService:
    def __init__(self, db: Database):
        self.db = db

    def request(
        self,
        *,
        run_id: str,
        subject_artifact_id: str,
        producer_agent_id: str,
        schema: str,
        required_checks: tuple[str, ...],
        evidence_refs: tuple[str, ...],
    ) -> VerificationRequest:
        request = VerificationRequest(
            verification_id=f"verification:{run_id}:{subject_artifact_id}",
            run_id=run_id,
            subject_artifact_id=subject_artifact_id,
            producer_agent_id=producer_agent_id,
            claim_or_output_schema=schema,
            required_checks=required_checks,
            evidence_refs=evidence_refs,
        )
        self.db.put_verification_request(request)
        return self.db.get_verification_request(request.verification_id) or request

    def complete(
        self,
        *,
        verification_id: str,
        verifier: AgentInstance,
        checks: dict[str, bool],
        findings: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        limitations: tuple[str, ...] = (),
    ) -> VerificationVerdict:
        request = self.db.get_verification_request(verification_id)
        if request is None:
            raise KeyError(verification_id)
        if verifier.run_id != request.run_id or verifier.tier != Tier.WORKER:
            raise VerificationDenied("verification requires a worker in the same run")
        if verifier.agent_instance_id == request.producer_agent_id:
            raise VerificationDenied("a producer cannot verify its own artifact")
        if not self.db.authority_active(verifier.run_id, verifier.plan_version):
            raise VerificationDenied("verifier authority is inactive")
        if set(checks) != set(request.required_checks):
            raise VerificationDenied("verdict must address every required check exactly once")
        missing_evidence = set(request.evidence_refs) - set(evidence_refs)
        provenance = self.db.get_provenance(request.subject_artifact_id)
        if missing_evidence or provenance is None:
            verdict_value = "inconclusive"
        else:
            verdict_value = "pass" if all(checks.values()) else "fail"
        verdict = VerificationVerdict(
            verdict_id=f"verdict:{request.verification_id}:{verifier.agent_instance_id}",
            verification_id=request.verification_id,
            verdict=verdict_value,
            checks_performed=tuple(checks),
            findings=findings,
            evidence_refs=evidence_refs,
            limitations=limitations,
            verifier_agent_id=verifier.agent_instance_id,
        )
        self.db.put_verification_verdict(request.run_id, verdict)
        return self.db.get_verification_verdict(verdict.verdict_id) or verdict

    def accept(
        self,
        *,
        request: VerificationRequest,
        verdict: VerificationVerdict,
        policy_version: str,
        downstream_uses: tuple[str, ...],
    ) -> AcceptanceDecision:
        if verdict.verification_id != request.verification_id:
            raise VerificationDenied("verdict does not belong to the verification request")
        decision = AcceptanceDecision(
            acceptance_decision_id=f"acceptance:{request.verification_id}",
            run_id=request.run_id,
            subject_artifact_id=request.subject_artifact_id,
            verdict_refs=(verdict.verdict_id,),
            acceptance_policy_version=policy_version,
            permitted_downstream_uses=downstream_uses if verdict.verdict == "pass" else (),
            accepted=verdict.verdict == "pass",
        )
        self.db.put_acceptance_decision(decision)
        return self.db.get_acceptance_decision(decision.acceptance_decision_id) or decision
