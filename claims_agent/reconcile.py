"""Cross-document reconciliation: turns a pile of Evidence into one FieldResult per field.

Principles
- Every document is one *independent source* (a revision chain counts once; duplicates don't vote).
- Sources carry *authority* per field: the police report owns date of loss, the settlement owns
  payout, the finance agreement owns loan balance. Customer statements are the weakest voice:
  they can corroborate or contradict, never satisfy a requirement on their own.
- Confidence is derived from evidence (source quality, corroboration, repairs, conflicts) -
  never from asking the model how sure it feels.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .config import settings
from .models import ClaimState, DocType, Evidence, FieldResult
from .validators import amounts_match, parse_date, vin_ocr_variant

AUTHORITY: dict[str, dict[DocType, float]] = {
    "vin": {DocType.police_report: 3, DocType.finance_agreement: 3, DocType.settlement_breakdown: 2,
            DocType.tow_receipt: 1, DocType.adjuster_note: 1, DocType.customer_correspondence: 0.5},
    "date_of_loss": {DocType.police_report: 4, DocType.settlement_breakdown: 2, DocType.adjuster_note: 1,
                     DocType.tow_receipt: 1, DocType.customer_correspondence: 1},
    "insurance_payout": {DocType.settlement_breakdown: 4, DocType.adjuster_note: 0.3,
                         DocType.customer_correspondence: 0.2},
    "loan_balance": {DocType.finance_agreement: 4, DocType.settlement_breakdown: 2, DocType.adjuster_note: 0.3,
                     DocType.customer_correspondence: 0.5},
}
PRIMARY = {"vin": DocType.police_report, "date_of_loss": DocType.police_report,
           "insurance_payout": DocType.settlement_breakdown, "loan_balance": DocType.finance_agreement}
LABEL = {"police_report": "police report", "finance_agreement": "finance agreement",
         "settlement_breakdown": "settlement breakdown", "adjuster_note": "adjuster note",
         "tow_receipt": "tow receipt", "customer_correspondence": "customer message"}


def _quality(ev: Evidence) -> float:
    if ev.method == "pdf_text":
        return 1.0
    if ev.method in ("ocr", "ocr_enhanced"):
        c = ev.line_confidence or 0
        return 0.9 if c >= settings.ocr_good else 0.7 if c >= settings.ocr_fair else 0.4
    if ev.method == "vision":
        return 0.7
    if ev.method == "llm_observation":
        return 0.6
    return 1.0  # customer_statement: authority weight already encodes how little it counts


def _strong(ev: Evidence) -> bool:
    return ev.method == "pdf_text" or (ev.method.startswith("ocr") and (ev.line_confidence or 0) >= settings.ocr_good)


def _describe(ev: Evidence) -> str:
    how = {"pdf_text": "PDF text", "ocr": "scan/OCR", "ocr_enhanced": "scan/enhanced OCR", "vision": "vision model",
           "llm_observation": "read from note", "customer_statement": "customer said"}.get(ev.method, ev.method)
    conf = f" {ev.line_confidence:.2f}" if ev.line_confidence is not None and ev.method.startswith("ocr") else ""
    return f"{ev.source_file} ({how}{conf})"


def usable_evidence(state: ClaimState, field: str) -> list[Evidence]:
    inactive = {d.file for d in state.documents.values() if d.status != "active"}
    return [e for e in state.evidence if e.field == field and e.source_file not in inactive]


def reconcile_field(state: ClaimState, field: str) -> tuple[FieldResult, dict]:
    """Returns (FieldResult, analysis). analysis carries conflict details for issue generation."""
    evs = usable_evidence(state, field)
    analysis: dict = {"field": field, "conflict": False, "invalid_only": False, "groups": [], "invalid": []}
    candidates = [{"value": e.value if e.valid else e.raw, "valid": e.valid, "source": e.source_file,
                   "method": e.method, "line_confidence": e.line_confidence,
                   "note": e.validation_note, "approximate": e.approximate} for e in evs]

    valid = [e for e in evs if e.valid and e.value is not None]
    invalid = [e for e in evs if not e.valid]
    analysis["invalid"] = invalid
    if not valid:
        if not invalid:
            return FieldResult(value=None, confidence="low", source=None, reason="not found in any document",
                               candidates=candidates), analysis | {"missing": True}
        analysis["invalid_only"] = True
        first = invalid[0]
        files = sorted({e.source_file for e in invalid})
        same = len({(e.value or e.raw) for e in invalid}) == 1
        reason = f"{first.validation_note or 'failed validation'}"
        pending = "pending" in (first.validation_note or "")
        if pending:
            reason = f"document shows \"{first.raw}\" - {reason}"
        if len(files) > 1 and same:
            reason += f" - identical in all {len(files)} documents, so the source documents themselves are wrong, not the OCR"
        return FieldResult(value=None if pending else (first.value or first.raw), confidence="low", source=first.source_file, sources=files,
                           reason=reason, candidates=candidates), analysis

    # --- group exact values; approximate (customer) amounts join a matching exact group ---------
    exact = [e for e in valid if not e.approximate]
    approx = [e for e in valid if e.approximate]
    groups: dict[str, list[Evidence]] = defaultdict(list)
    for e in exact:
        groups[e.value].append(e)
    for e in approx:
        match = next((v for v in groups if amounts_match(v, e.value, approximate=True)), None)
        groups[match or f"~{e.value}"].append(e)

    def score(group: list[Evidence]) -> float:
        best_per_file: dict[str, float] = {}
        for e in group:
            w = AUTHORITY[field].get(e.doc_type, 0.5) * _quality(e)
            best_per_file[e.source_file] = max(best_per_file.get(e.source_file, 0), w)
        return round(sum(best_per_file.values()), 2)

    # VINs that differ only by OCR look-alikes (S/5, B/8, Z/2, G/6, D/0) are *probably* one VIN read twice -
    # but both characters are legal in a VIN, so we cannot know which reading is right. We group them (so they
    # don't show up as an unrelated conflict), propose the stronger reading, force LOW confidence and let
    # rules.py raise a blocking "needs human confirmation" issue. Never silently "fixed".
    ocr_variants: list[str] = []
    ambiguity: list[dict] = []
    if field == "vin":
        for weak in sorted(groups, key=lambda v: score(groups[v])):
            strong = next((v for v in sorted(groups, key=lambda v: score(groups[v]), reverse=True)
                           if v != weak and score(groups[v]) >= score(groups[weak]) and vin_ocr_variant(v, weak)), None)
            if strong:
                pos = vin_ocr_variant(strong, weak)
                weak_files = sorted({e.source_file for e in groups[weak]})
                strong_files = sorted({e.source_file for e in groups[strong]})
                chars = ", ".join(f"pos {p}: '{strong[p - 1]}' vs '{weak[p - 1]}'" for p in pos)
                ocr_variants.append(f"{strong} ({', '.join(strong_files)}) and {weak} ({', '.join(weak_files)}) look "
                                    f"alike but are not identical ({chars}) - OCR look-alike characters, cannot tell "
                                    f"which is right; best guess {strong}, needs human confirmation")
                ambiguity.append({"suggested": strong, "alternative": weak, "positions": pos, "chars": chars,
                                  "suggested_sources": strong_files, "alternative_sources": weak_files})
                groups[strong].extend(groups.pop(weak))
    analysis["ocr_variants"] = ocr_variants
    analysis["ocr_ambiguity"] = ambiguity

    ranked = sorted(groups.items(), key=lambda kv: score(kv[1]), reverse=True)
    analysis["groups"] = [{"value": v, "score": score(g), "sources": sorted({e.source_file for e in g})}
                          for v, g in ranked]
    win_value, win = ranked[0]
    win_value = win_value.lstrip("~")
    win_files = sorted({e.source_file for e in win})
    doc_sources = {e.source_file for e in win if e.method != "customer_statement"}
    customer_only = not doc_sources
    primary = max(win, key=lambda e: (e.method != "customer_statement",
                                      AUTHORITY[field].get(e.doc_type, 0.5) * _quality(e)))

    # --- conflict -----------------------------------------------------------------------------
    if len(ranked) > 1:
        analysis["conflict"] = True
        runner_value, runner = ranked[1]
        s1, s2 = score(win), score(runner)
        hints = conflict_hints(state, field, win_value, runner_value.lstrip("~"), runner)
        analysis["hints"] = hints
        has_primary = any(e.doc_type == PRIMARY[field] for e in win)
        decisive = s1 >= 2 * s2 and has_primary
        others = "; ".join(f"{v.lstrip('~')} <- {', '.join(sorted({e.source_file for e in g}))}" for v, g in ranked[1:])
        # An educated guess needs a real edge: clearly more source weight, or a concrete explanation (hint)
        # for why the other value is wrong. Otherwise say so instead of pretending.
        can_guess = s1 >= 1.25 * s2 or bool(hints)
        if can_guess:
            reason = (f"CONFLICT. Best guess {win_value} (supported by {', '.join(win_files)}, weight {s1}) "
                      f"vs {others} (weight {s2})." + (f" {' '.join(hints)}" if hints else ""))
        else:
            reason = (f"CONFLICT. No guess - {win_value} ({', '.join(win_files)}, weight {s1}) vs {others} "
                      f"(weight {s2}): the sources are equally authoritative.")
        analysis["suggested_value"] = win_value if can_guess else "No guess"
        analysis["decisive"] = decisive and can_guess
        return FieldResult(value=win_value if can_guess else None, confidence="medium" if decisive and can_guess else "low",
                           source=primary.source_file if can_guess else None,
                           sources=win_files, reason=reason, candidates=candidates), analysis

    # --- agreement ------------------------------------------------------------------------------
    notes = []
    repaired = any((e.validation_note or "").lower().find("ocr") >= 0 or "repair" in (e.validation_note or "").lower()
                   for e in win)
    strong_docs = {e.source_file for e in win if _strong(e)}
    if customer_only:
        conf, notes = "low", ["only the customer's statement - no document supports it yet"]
        if any(e.approximate for e in win):
            notes.append("customer gave an approximate figure")
    elif len(doc_sources) >= 2 and strong_docs:
        conf = "high"
    elif len(doc_sources) == 1 and primary.method == "pdf_text":
        conf = "high"
    else:
        conf = "medium"
        if len(doc_sources) == 1:
            notes.append(f"single source: {_describe(primary)}; no second document to corroborate")
        else:
            notes.append("all sources are low-quality scans")
    if repaired and conf == "high":
        conf = "medium"
    if repaired:
        notes.append("value needed an automatic OCR correction")
    if conf == "high" and field != "vin" and primary.doc_type != PRIMARY[field] and not any(
            e.doc_type == PRIMARY[field] for e in win):
        conf = "medium"
        notes.append(f"the {LABEL[PRIMARY[field].value]} (authoritative source) is not on file")
    elif conf != "high" and not any(e.doc_type == PRIMARY[field] for e in win) and not customer_only:
        notes.append(f"the {LABEL[PRIMARY[field].value]} (authoritative source) is not on file")
    if ocr_variants:
        notes = ocr_variants + notes
        conf = "low"  # sources do not literally agree; a human must confirm the look-alike character(s)
    cust = [e for e in win if e.method == "customer_statement"]
    if cust and not customer_only:
        notes.append("customer's statement agrees" + (" (approximately)" if any(e.approximate for e in cust) else ""))
    return FieldResult(value=win_value, confidence=conf, source=primary.source_file, sources=win_files,
                       reason="; ".join(notes) or None, candidates=candidates), analysis


def conflict_hints(state: ClaimState, field: str, winner: str, loser: str, loser_evs: list[Evidence]) -> list[str]:
    """Explain *why* a value is probably wrong - the 'educated guess' rationale."""
    hints = []
    if field == "vin" and len(winner) == len(loser):
        diff = [i + 1 for i, (a, b) in enumerate(zip(winner, loser)) if a != b]
        if len(diff) <= 3:
            hints.append(f"Values differ only at position(s) {diff} - consistent with a transcription error.")
    if field == "date_of_loss":
        for e in loser_evs:
            doc = state.documents.get(e.source_file)
            prepared = (doc.meta.get("prepared_date") if doc else None)
            if prepared:
                p = parse_date(prepared)
                if p["valid"] and p["value"] == loser:
                    hints.append(f"{e.source_file} lists a date of loss identical to its own 'Date Prepared' "
                                 f"({loser}) - likely a clerical error copying the preparation date.")
    if any(e.method == "customer_statement" for e in loser_evs) and all(
            e.method == "customer_statement" for e in loser_evs):
        hints.append("The dissenting value comes only from the customer's statement.")
    return hints


def reconcile_all(state: ClaimState) -> tuple[dict[str, FieldResult], dict[str, dict]]:
    results, analyses = {}, {}
    for f in ("vin", "date_of_loss", "insurance_payout", "loan_balance"):
        results[f], analyses[f] = reconcile_field(state, f)
    return results, analyses


def reference_year(state: ClaimState) -> Optional[int]:
    for e in state.evidence:
        if e.field == "date_of_loss" and e.valid and e.value:
            return int(e.value[:4])
    return None
