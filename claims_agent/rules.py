"""Deterministic claim decision. The LLM never sets the status - this module does.

Precedence: needs_review > incomplete > complete.
Rationale: a conflict or an invalid value needs a human regardless of what the customer sends,
and we still message the customer in parallel for anything they can fix (next_action carries both).
"""
from __future__ import annotations

from . import consistency
from .config import settings
from .models import ClaimState, Evaluation, Issue
from .reconcile import LABEL, reconcile_all

FIELD_LABEL = {"vin": "VIN", "date_of_loss": "date of loss", "insurance_payout": "insurance payout",
               "loan_balance": "outstanding loan balance"}
FIELD_HOME = {"vin": "police_report", "date_of_loss": "police_report",
              "insurance_payout": "settlement_breakdown", "loan_balance": "finance_agreement"}
DOC_RESOLVER = {"police_report": "customer_or_internal", "finance_agreement": "customer_or_internal",
                "settlement_breakdown": "internal"}


def evaluate(state: ClaimState) -> Evaluation:
    fields, analyses = reconcile_all(state)
    issues: list[Issue] = []
    present = {d.doc_type.value for d in state.active_docs()}
    missing_docs = [d for d in settings.required_documents if d not in present]

    for d in missing_docs:
        said = [c for c in state.commitments if c.doc_type == d]
        details = None
        if said:
            c = said[-1]
            if c.eta == "unavailable":
                details = f"Customer says they cannot provide it: \"{c.quote}\""
            else:
                details = f"Customer promised to send it{(' (' + c.eta + ')') if c.eta else ''}: \"{c.quote}\""
        issues.append(Issue(type="missing", severity="blocking", code="missing_document",
                            description=f"Required document missing: {LABEL[d]}", details=details,
                            resolvable_by=DOC_RESOLVER[d]))

    for f, res in fields.items():
        a = analyses[f]
        label = FIELD_LABEL[f]
        if a.get("missing"):
            home_missing = FIELD_HOME[f] in missing_docs
            issues.append(Issue(type="missing", severity="blocking", code="missing_field", field=f,
                                description=f"{label} not available" + (f" (its source, the {LABEL[FIELD_HOME[f]]}, is missing)" if home_missing else ""),
                                resolvable_by="customer_or_internal" if home_missing else "internal"))
        elif a.get("invalid_only"):
            pending = any("pending" in (e.validation_note or "") for e in a["invalid"])
            if pending:
                issues.append(Issue(type="missing", severity="blocking", code="field_pending", field=f,
                                    description=f"{label} not yet determined",
                                    details=f"{res.source}: \"{a['invalid'][0].raw}\"", resolvable_by="internal"))
            else:
                issues.append(Issue(type="invalid", severity="blocking", code="invalid_value", field=f,
                                    description=f"{label} fails validation: {res.value}",
                                    details=res.reason, resolvable_by="customer_or_internal"))
        elif a.get("conflict"):
            groups = "; ".join(f"{', '.join(g['sources'])}: {g['value'].lstrip('~')}" for g in a["groups"])
            issues.append(Issue(type="inconsistency", severity="blocking", code="cross_document_conflict", field=f,
                                description=f"{label} disagrees across sources",
                                details=groups + (" | " + " ".join(a.get("hints", [])) if a.get("hints") else ""),
                                suggested_value=a.get("suggested_value"),
                                resolvable_by="internal" if f != "vin" else "customer_or_internal"))
        if not a.get("conflict") and a.get("invalid") and not a.get("invalid_only"):
            # some sources invalid while others agree on a valid value
            for e in a["invalid"]:
                issues.append(Issue(type="inconsistency", severity="blocking", code="invalid_in_one_source", field=f,
                                    description=f"{label} in {e.source_file} is invalid while other documents agree on {res.value}",
                                    details=f"{e.source_file}: {e.raw} ({e.validation_note})",
                                    suggested_value=res.value, resolvable_by="internal"))
        for amb in a.get("ocr_ambiguity", []):
            issues.append(Issue(
                type="low_confidence", severity="blocking", code="ocr_lookalike_mismatch", field=f,
                description=f"{label} readings look alike but are not identical - needs human confirmation",
                details=(f"{', '.join(amb['suggested_sources'])}: {amb['suggested']} vs "
                         f"{', '.join(amb['alternative_sources'])}: {amb['alternative']} ({amb['chars']}). "
                         "These characters are commonly confused by OCR and both are legal in a VIN, so the agent "
                         "will not pick one automatically. Check the original documents or the vehicle registration."),
                suggested_value=amb["suggested"], resolvable_by="internal"))
        if res.confidence == "low" and not (a.get("missing") or a.get("invalid_only") or a.get("conflict")
                                            or a.get("ocr_ambiguity")):
            issues.append(Issue(type="low_confidence", severity="blocking", code="low_confidence", field=f,
                                description=f"{label} could not be established with confidence",
                                details=res.reason, resolvable_by="customer_or_internal"))

    inactive = {d.file for d in state.documents.values() if d.status != "active"}
    for c in state.concerns:
        if c.source_file in inactive:  # e.g. math on a superseded settlement version
            continue
        blocking = (settings.strict_settlement_math and c.code == "settlement_math") or \
                   (settings.warnings_block_finalize and c.severity == "warning")
        issues.append(Issue(type="warning", severity="blocking" if blocking else c.severity, code=c.code,
                            description=c.description,
                            details=(f"{c.source_file}: \"{c.quote}\"" if c.quote else c.source_file),
                            resolvable_by="internal"))

    # Cross-document consistency of everything else the documents share (vehicle, lender, owner name, policy
    # number, police report number). Contradictions block; OCR variants and customer-only mismatches don't.
    xissues, cross_checks = consistency.check(state)
    issues.extend(xissues)

    blocking = [i for i in issues if i.severity == "blocking"]
    review_types = {"inconsistency", "invalid", "low_confidence", "warning"}
    if any(i.type in review_types for i in blocking):
        status = "needs_review"
        why = "; ".join(sorted({i.description for i in blocking if i.type in review_types}))
    elif blocking:
        status = "incomplete"
        why = "; ".join(i.description for i in blocking)
    else:
        status = "complete"
        why = "all required documents present; all four fields valid and consistent"

    customer_fixable = any(i.severity == "blocking" and i.resolvable_by.startswith("customer") for i in issues)
    if status == "complete":
        action = "finalize"
    elif status == "incomplete":
        action = "message_customer" if customer_fixable else "escalate"
    else:
        action = "escalate"
    return Evaluation(status=status, status_reason=why, extracted_fields=fields, missing_documents=missing_docs,
                      issues=issues, recommended_action=action, cross_checks=cross_checks)
