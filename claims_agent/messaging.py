"""Template-based customer/internal messages: used by the fallback planner and as the safety net
when the LLM cannot produce an acceptable message."""
from __future__ import annotations

from typing import Optional

from .models import ClaimState

_DOC_ASK = {
    "police_report": ("a copy of the police report for the accident",
                      "it officially confirms the accident and the vehicle involved. If you can't get a copy, "
                      "tell us the police department and report number and we can request it"),
    "finance_agreement": ("your auto finance (loan) agreement",
                          "we need it to confirm the loan details and pay your lender correctly. Your lender can "
                          "usually send a copy from your online account or by request"),
}
_FIELD_ASK = {
    "vin": "a clear photo of your vehicle registration or title showing the full 17-character vehicle "
           "identification number (VIN) - the VIN on the documents we have appears incomplete or unclear",
    "date_of_loss": "confirmation of the exact date of the accident",
    "loan_balance": "a recent loan statement or payoff letter showing your outstanding balance",
}


def _policy(state: ClaimState) -> Optional[str]:
    for d in state.documents.values():
        if d.meta.get("policy_number"):
            return d.meta["policy_number"]
    return None


def template_message(state: ClaimState) -> tuple[Optional[str], str]:
    ev = state.evaluation
    asks = []
    for d in ev.missing_documents:
        if d in _DOC_ASK:
            what, why = _DOC_ASK[d]
            asks.append(f"- {what[0].upper() + what[1:]}: {why}.")
    for i in ev.issues:
        if i.severity == "blocking" and i.field in _FIELD_ASK and i.resolvable_by.startswith("customer") \
                and i.code in ("invalid_value", "low_confidence", "cross_document_conflict") and i.field == "vin":
            asks.append(f"- {_FIELD_ASK[i.field][0].upper() + _FIELD_ASK[i.field][1:]}.")
    questions = [q for q in state.questions if q.turn == state.current_turn]
    internal = _internal_note(state)
    if not asks and not questions:
        return None, internal
    ref = _policy(state)
    lines = ["Hello,", "", f"Thank you for your patience with your claim{f' (policy {ref})' if ref else ''}."]
    if asks:
        lines += ["To keep things moving, we still need:", *asks, "",
                  "You can reply to this message with photos or PDFs attached."]
    for q in questions:
        lines += ["", f"About your question ({q.question.split('] ', 1)[-1]}): a claims specialist will follow up "
                      "with you directly."]
    lines += ["", "Thank you,", "Claims Team"]
    return "\n".join(lines), internal


def _internal_note(state: ClaimState) -> str:
    ev = state.evaluation
    parts = [f"Status {ev.status}: {ev.status_reason}."]
    for i in ev.issues:
        if i.severity == "blocking":
            s = f" Suggested: {i.suggested_value}." if i.suggested_value else ""
            parts.append(f"- {i.description}.{s}")
    warns = [i.description for i in ev.issues if i.severity == "warning"]
    if warns:
        parts.append("Warnings: " + " | ".join(warns))
    return "\n".join(parts)
