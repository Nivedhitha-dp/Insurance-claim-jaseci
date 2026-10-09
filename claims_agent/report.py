"""Structured output per claim + cross-claim prioritization."""
from __future__ import annotations

import json
from datetime import date
from typing import Any

from .models import ClaimState

_TYPE_LABEL = {"adjuster_note": "adjuster_note (supplementary - handwritten field notes)",
               "tow_receipt": "tow_receipt (supplementary)",
               "customer_correspondence": "customer_correspondence"}


def _short(result: Any) -> Any:
    """Human-scale summary of a tool result for tools_used."""
    if not isinstance(result, dict):
        return result
    if "error" in result:
        return f"error: {result['error']}"
    for key in ("relation", "status", "valid", "recorded", "accepted"):
        if key in result:
            break
    s = json.dumps({k: v for k, v in result.items() if k not in ("text", "meta", "type_suggestion", "documents",
                                                                  "fields", "checks", "groups", "issues")}, default=str)
    if "fields" in result and isinstance(result["fields"], list):
        s = json.dumps({"doc_type": result.get("doc_type"),
                        "fields": {f["field"]: f["value"] or f["raw"] for f in result["fields"]},
                        **({"not_found": result["not_found"]} if result.get("not_found") else {})})
    if "checks" in result:
        s = json.dumps({"ok": result.get("ok"), "checks": [f"{c['check']}: {'ok' if c['ok'] else 'MISMATCH'}"
                                                          for c in result["checks"]]})
    if "documents" in result:
        s = f"{len(result['documents'])} files; exact duplicate groups: {result.get('exact_duplicate_groups')}"
    if "groups" in result and result.get("conflict"):
        s = json.dumps({"conflict": True, "best_guess": result.get("value"), "groups": result["groups"]})
    elif "groups" in result:
        s = json.dumps({"value": result.get("value"), "confidence": result.get("confidence")})
    if "method" in result and "chars" in result:
        s = json.dumps({"method": result["method"], "ocr_conf": result.get("ocr_mean_confidence"),
                        "skew_corrected": result.get("skew_corrected_deg"),
                        "suggested_type": result.get("type_suggestion", {}).get("doc_type")})
    if "attributes" in result:  # cross_check_documents
        s = json.dumps({a: r["status"] for a, r in result["attributes"].items()})
    elif "issues" in result:  # evaluate_claim
        s = json.dumps({"status": result["status"], "blocking": sum(1 for i in result["issues"] if i["severity"] == "blocking")})
    return json.loads(s) if s.startswith("{") else s


def claim_output(state: ClaimState) -> dict:
    ev = state.evaluation
    last = state.turns[-1] if state.turns else None
    docs = state.documents.values()
    out = {
        "claim_id": state.claim_id,
        "status": ev.status,
        "status_reason": ev.status_reason,
        "extracted_fields": {k: {"value": v.value, "confidence": v.confidence, "source": v.source,
                                 "reason": v.reason, "corroborated_by": [s for s in v.sources if s != v.source]}
                             for k, v in ev.extracted_fields.items()},
        "documents": {
            "identified": [{"file": d.file, "type": _TYPE_LABEL.get(d.doc_type.value, d.doc_type.value),
                            "read_via": d.text_method, **({"ocr_confidence": d.ocr_confidence} if d.ocr_confidence else {}),
                            **({"received_in_turn": d.added_in_turn} if d.added_in_turn else {}),
                            "status": d.status} for d in docs],
            "missing": ev.missing_documents,
            "duplicates": [{"file": d.file, "duplicate_of": d.related_to, "reason": d.status_reason}
                           for d in docs if d.status == "duplicate"],
            "superseded": [{"file": d.file, "superseded_by": d.related_to, "reason": d.status_reason}
                           for d in docs if d.status == "superseded"],
        },
        "issues": [{"type": i.type, "severity": i.severity, "field": i.field, "description": i.description,
                    "details": i.details, **({"suggested_value": i.suggested_value} if i.suggested_value else {}),
                    "resolvable_by": i.resolvable_by} for i in ev.issues],
        "cross_document_checks": ev.cross_checks,
        "next_action": last.next_action.model_dump() if last and last.next_action else None,
        "conversation": [{"turn": t.index, "kind": t.kind,
                          **({"inbound": t.inbound} if t.inbound else {}),
                          **({"uploaded": t.inbound_files} if t.inbound_files else {}),
                          "status_after_turn": t.status,
                          "action": t.next_action.type if t.next_action else None,
                          "message_to_customer": t.next_action.message if t.next_action else None,
                          "agent_summary": t.agent_summary, "driven_by": t.driven_by} for t in state.turns],
        "tools_used": [{"turn": c.turn, "tool": c.tool, "input": c.input, "result": _short(c.result)}
                       for c in state.tool_log],
    }
    return out


# ------------------------------------------------------------------------------ prioritization
_FIELD_POINTS = {"high": 15, "medium": 9, "low": 3}


def decision_confidence(state: ClaimState) -> dict:
    """How much we trust the decision for this claim, 0-100 - separates claims with the same tier and the same
    number of blockers (e.g. several 'complete' claims with 0 blocking issues).

      field evidence  60 pts  per field: high 15 / medium 9 / low 3 / missing 0
      corroboration   20 pts  share of the four fields backed by 2+ independent documents
      cleanliness     20 pts  -5 per reviewer warning, -2 per info note (e.g. an OCR variant)
    """
    ev = state.evaluation
    customer_files = {d.file for d in state.documents.values() if d.doc_type.value == "customer_correspondence"}
    fields = ev.extracted_fields
    evidence = sum(_FIELD_POINTS.get(f.confidence, 0) if f.value else 0 for f in fields.values())
    corroborated = sum(1 for f in fields.values() if f.value and len(set(f.sources) - customer_files) >= 2)
    warnings = sum(1 for i in ev.issues if i.severity == "warning")
    infos = sum(1 for i in ev.issues if i.severity == "info")
    clean = max(0, 20 - 5 * warnings - 2 * infos)
    score = round(evidence + 20 * corroborated / 4 + clean)
    label = "high" if score >= 85 else "medium" if score >= 60 else "low"
    conf = [f.confidence for f in fields.values() if f.value]
    summary = (f"{conf.count('high')}/4 fields high-confidence, {corroborated}/4 corroborated by 2+ documents"
               + (f", {warnings} warning(s)" if warnings else "") + (f", {infos} note(s)" if infos else ""))
    return {"score": score, "label": label, "summary": summary,
            "breakdown": {"field_evidence": evidence, "corroboration": round(20 * corroborated / 4),
                          "cleanliness": clean}}


def prioritize(states: list[ClaimState], today: date | None = None) -> dict:
    """Order = how close each claim is to a payout we'd trust, then how long the customer has waited.

    tier 1  complete -> finalize now
    tier 2  all documents on file; only internal blockers that come with a best-guess fix
    tier 3  waiting on the customer for documents they *have* (promised, with an ETA)
    tier 4  review needed AND documents the customer cannot easily produce
    tier 5  blocked on our own unknowns (valuation pending) plus invalid data
    Within a tier: fewer blocking issues, then HIGHER decision confidence, then the oldest loss.
    """
    today = today or date.today()
    rows = []
    for s in states:
        ev = s.evaluation
        blocking = [i for i in ev.issues if i.severity == "blocking"]
        pending_internal = [i for i in blocking if i.code == "field_pending"]
        invalid = [i for i in blocking if i.type == "invalid"]
        unavailable = [c for c in s.commitments if c.eta == "unavailable" and c.doc_type in ev.missing_documents]
        promised = [c for c in s.commitments if c.eta != "unavailable" and c.doc_type in ev.missing_documents]
        if ev.status == "complete":
            tier = 1
        elif not ev.missing_documents and all((i.suggested_value and i.suggested_value != "No guess") or i.type == "warning" for i in blocking):
            tier = 2
        elif ev.status == "incomplete":
            tier = 3
        elif pending_internal or invalid:
            tier = 5
        else:
            tier = 4
        dol = ev.extracted_fields["date_of_loss"].value
        age = (today - date.fromisoformat(dol)).days if dol else None
        payout = ev.extracted_fields["insurance_payout"].value
        dc = decision_confidence(s)
        rows.append((tier, len(blocking), -dc["score"], -(age or 0), s, age, payout, promised, unavailable,
                     pending_internal, invalid, dc))
    rows.sort(key=lambda r: r[:4])

    order, parallel = [], []
    for rank, (tier, nblock, _, _, s, age, payout, promised, unavailable, pending_internal, invalid, dc) in enumerate(rows, 1):
        ev = s.evaluation
        blocking = [i for i in ev.issues if i.severity == "blocking"]
        warns = [i for i in ev.issues if i.severity == "warning"]
        if tier == 1:
            reason = "All required documents present; every field valid and consistent - ready to finalize."
            if warns:
                reason += f" Do a quick sanity check of {len(warns)} non-blocking warning(s) first: " + \
                          "; ".join(w.description.split(':')[0] for w in warns) + "."
        elif tier == 2:
            fixes = "; ".join(f"{i.description} (best guess {i.suggested_value})" for i in blocking if i.suggested_value and i.suggested_value != "No guess")
            reason = (f"Everything is on file; one internal decision away from final: {fixes}. "
                      "A single verification call (e.g. to the lender) unblocks it.")
        elif tier == 3:
            reason = ("Data present is clean and consistent; blocked only on " +
                      ", ".join(ev.missing_documents) + " which the customer has and promised" +
                      (f" ({promised[0].eta})" if promised and promised[0].eta else "") +
                      ". Can also be requested directly from the police department to save time.")
        elif tier == 4:
            reason = ("Needs a reviewer decision (" + "; ".join(i.description for i in blocking if i.type != "missing") +
                      ") and a document the customer could not provide (" + ", ".join(ev.missing_documents) + ").")
        else:
            reason = ("Furthest from finalization: " + "; ".join(i.description for i in pending_internal + invalid) +
                      ". Needs internal valuation plus corrected source data.")
        reason += f" Decision confidence {dc['score']}/100 ({dc['label']}): {dc['summary']}."
        if age is not None:
            reason += f" Loss was {age} days ago."
        order.append({"rank": rank, "claim_id": s.claim_id, "status": ev.status, "tier": tier,
                      "blocking_issues": nblock, "decision_confidence": dc, "reason": reason})
        last = s.turns[-1].next_action if s.turns else None
        if last and last.message:
            parallel.append({"claim_id": s.claim_id, "action": "send customer message now (they are the bottleneck)"})
        if age is not None and age > 90 and tier >= 4:
            parallel.append({"claim_id": s.claim_id, "action": f"escalate: loss is {age} days old and still blocked"})
    return {"processing_order": order,
            "do_in_parallel_today": parallel,
            "method": "Rank by distance-to-trustworthy-payout (tiers), then fewer blockers, then higher decision "
                      "confidence (field confidence, corroboration across documents, open warnings), then oldest loss. "
                      "Customer-facing requests go out immediately regardless of rank, since the customer is the "
                      "slowest dependency."}
