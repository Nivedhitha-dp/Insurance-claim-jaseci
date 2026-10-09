"""Agent tools.

Each tool: a narrow capability with a clear boundary, a JSON schema, and a description that tells
the model *when* to use it. Tools read/write the ClaimState; they return compact JSON so the
model's context stays small (and so the tool log doubles as an audit trail).

Division of labour
- Tools own facts: reading files, validating values, comparing documents, computing status.
- The model owns judgement: which documents need a closer look, what an unstructured note or
  customer email *means*, whether two files are the same document, how to talk to the customer.
- Anything the model asserts about a document must quote it; quotes are verified (anti-hallucination).
"""
from __future__ import annotations

import difflib
import inspect
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Optional

from .config import settings
from .extraction import SETTLEMENT_LINES, classify, document_meta, extract_fields as layout_extract, vin_mentions
from .ingest import file_kind, lines_to_text, mean_conf, ocr_page, pdf_lines, sha256
from .models import (ClaimState, Concern, CustomerCommitment, CustomerQuestion, DocType, Document, Evidence,
                     TextLine, ToolCall)
from .reconcile import reconcile_field, reference_year
from .rules import evaluate
from .validators import normalize_field, parse_amount, validate_vin as _validate_vin

DOC_TYPES = [t.value for t in DocType]


class ToolError(Exception):
    pass


class ToolContext:
    def __init__(self, state: ClaimState, emit: Optional[Callable[[dict], None]] = None):
        self.state = state
        self.emit = emit or (lambda e: None)
        self.step = 0
        self.submitted: Optional[dict] = None
        self.vision_client = None  # injected by the agent when a vision model is configured

    @property
    def folder(self) -> Path:
        return Path(self.state.folder)

    def doc(self, file: str) -> Document:
        if file not in self.state.documents:
            raise ToolError(f"unknown file '{file}'. Known: {sorted(self.state.documents)}")
        return self.state.documents[file]

    def path(self, file: str) -> Path:
        p = self.folder / file
        if not p.exists():
            p = settings.state_dir / self.state.claim_id / file
        return p


TOOLS: dict[str, dict] = {}


def tool(schema_props: dict, required: list[str] | None = None):
    def deco(fn):
        TOOLS[fn.__name__] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": inspect.cleandoc(fn.__doc__ or ""),
                    "parameters": {"type": "object", "properties": schema_props, "required": required or []},
                },
            },
        }
        return fn

    return deco


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _quote_found(quote: str, text: str) -> tuple[bool, float]:
    """Is `quote` really in `text`? Exact after normalisation, else best fuzzy window (OCR noise)."""
    q, t = _squash(quote), _squash(text)
    if not q:
        return False, 0.0
    if q in t:
        return True, 1.0
    n, best = len(q), 0.0
    step = max(1, n // 6)
    for i in range(0, max(1, len(t) - n + 1), step):
        r = difflib.SequenceMatcher(None, q, t[i:i + n]).ratio()
        best = max(best, r)
        if best > 0.95:
            break
    return best >= 0.85, round(best, 2)


def _ensure_text(ctx: ToolContext, d: Document) -> str:
    if d.text is None:
        raise ToolError(f"{d.file} has not been read yet - call read_document first")
    return d.text


# =============================================================================== intake
@tool({})
def list_documents(ctx: ToolContext) -> dict:
    """Inventory of the claim folder: every file with its kind, hash and whether it was already read/typed.
    Exact byte-identical files are grouped (exact duplicates). Call this first in a new claim and again
    whenever the customer uploads something."""
    out, by_hash = [], {}
    for d in sorted(ctx.state.documents.values(), key=lambda d: d.file):
        by_hash.setdefault(d.sha256, []).append(d.file)
        out.append({"file": d.file, "kind": d.kind, "size_kb": round(d.size / 1024, 1), "sha256": d.sha256[:12],
                    "read": d.text is not None, "doc_type": d.doc_type.value if d.fields_extracted else None,
                    "status": d.status, "added_in_turn": d.added_in_turn})
    exact = [files for files in by_hash.values() if len(files) > 1]
    res = {"documents": out, "exact_duplicate_groups": exact}
    if ctx.state.pending_replies:
        res["note"] = "customer replies are held back until the initial review is submitted"
    return res


@tool({
    "file": {"type": "string"},
    "mode": {"type": "string", "enum": ["auto", "enhanced_ocr"],
             "description": "auto = PDF text layer / standard OCR. enhanced_ocr = deskew+denoise+upscale re-OCR; "
                            "use only when a scan's confidence is low or expected fields were not found."},
}, ["file"])
def read_document(ctx: ToolContext, file: str, mode: str = "auto") -> dict:
    """Read a file's text. PDFs use the embedded text layer (exact); scans are OCR'd (open-source RapidOCR)
    and report a confidence; text files are read as-is. Returns the text, OCR quality, low-confidence
    lines and a keyword-based document-type suggestion you should sanity-check against the content."""
    d = ctx.doc(file)
    p = ctx.path(file)
    skew = None
    if mode == "enhanced_ocr" and d.text_method == "pdf_text":
        # A born-digital PDF's text layer is exact; OCR'ing it can only add errors (seen in a real LLM run).
        return {"file": file, "result": "not applicable - this PDF has an exact text layer, OCR would only be worse",
                "method": "pdf_text", "hint": "use the text already returned; enhanced_ocr is for scans"}
    if d.kind == "text":
        d.text, d.text_method, d.lines = p.read_text(encoding="utf-8", errors="replace"), "plain_text", []
    elif d.kind == "pdf" and mode == "auto":
        lines = pdf_lines(p)
        if sum(len(l.text) for l in lines) < 40:  # scanned PDF without a text layer
            r = ocr_page(p, False, d.sha256)
            lines, d.text_method, skew = r["lines"], "ocr", r["skew_deg"]
        else:
            d.text_method = "pdf_text"
        d.lines = lines
    else:
        enhanced = mode == "enhanced_ocr"
        r = ocr_page(p, enhanced, d.sha256)
        if enhanced and d.lines and mean_conf(r["lines"]) is not None and mean_conf(d.lines) and \
                mean_conf(r["lines"]) < mean_conf(d.lines) - 0.02 and len(r["lines"]) < len(d.lines):
            return {"file": file, "result": "enhanced OCR was worse than the standard pass; keeping standard text",
                    "standard_conf": mean_conf(d.lines), "enhanced_conf": mean_conf(r["lines"])}
        d.lines, d.text_method, skew = r["lines"], "ocr_enhanced" if enhanced else "ocr", r["skew_deg"]
    if d.kind != "text":
        d.text = lines_to_text(d.lines)
    d.ocr_confidence = mean_conf(d.lines) if d.text_method and d.text_method.startswith("ocr") else None
    d.meta.update(document_meta(d.text))
    if d.lines:
        from .extraction import person_name
        name = person_name(d.lines)
        if name:
            d.meta["person_name"] = name
    suggestion = classify(d.text, file)
    low = [{"text": l.text, "conf": l.conf} for l in d.lines if l.conf < settings.ocr_fair][:8]
    text = d.text if len(d.text) < 3500 else d.text[:3500] + "\n...[truncated]"
    res = {"file": file, "method": d.text_method, "chars": len(d.text), "text": text,
           "type_suggestion": suggestion, "meta": d.meta}
    if d.ocr_confidence is not None:
        res["ocr_mean_confidence"] = d.ocr_confidence
        res["skew_corrected_deg"] = skew
        res["low_confidence_lines"] = low
        if d.ocr_confidence < settings.ocr_good:
            res["hint"] = "OCR quality is mediocre - consider enhanced_ocr or vision_read for key fields"
    return res


@tool({"file": {"type": "string"},
       "question": {"type": "string", "description": "what to read, e.g. 'the VIN written after door jamb'"}},
      ["file", "question"])
def vision_read(ctx: ToolContext, file: str, question: str) -> dict:
    """Ask a vision-language model to look at a scanned image directly. Expensive and can hallucinate - use
    ONLY when OCR failed on a field you need (handwriting, strike-throughs, smudges). Its transcript is
    stored so record_observation can verify quotes against it."""
    d = ctx.doc(file)
    if d.kind != "image":
        raise ToolError("vision_read only works on image files")
    if ctx.vision_client is None:
        raise ToolError("no vision model configured - rely on OCR text (and say so in your reasoning)")
    answer = ctx.vision_client(ctx.path(file), question)
    d.meta.setdefault("vision_transcripts", []).append(answer)
    return {"file": file, "answer": answer,
            "note": "treat as a single, medium-reliability reading; corroborate with other documents"}


# =============================================================================== extraction
@tool({
    "file": {"type": "string"},
    "doc_type": {"type": "string", "enum": DOC_TYPES,
                 "description": "your decision on what this document is (use the content, not the filename)"},
    "type_reason": {"type": "string"},
}, ["file", "doc_type"])
def extract_fields(ctx: ToolContext, file: str, doc_type: str, type_reason: str = "") -> dict:
    """Record the document type and run layout-aware extraction of the claim fields (VIN, date of loss,
    insurance payout, loan balance) for known form types. Values are validated/normalized and stored as
    evidence with provenance + OCR confidence. For types with no form template (adjuster_note,
    customer_correspondence, unknown) nothing is auto-extracted: read the text yourself and use
    record_observation for any field you find."""
    d = ctx.doc(file)
    _ensure_text(ctx, d)
    d.doc_type = DocType(doc_type)
    d.type_reason = type_reason or None
    d.type_confidence = classify(d.text, file)["confidence"] if classify(d.text, file)["doc_type"] == doc_type else 0.5
    # re-extraction (e.g. after enhanced OCR) replaces this file's previous layout evidence
    ctx.state.evidence = [e for e in ctx.state.evidence
                          if not (e.source_file == file and e.method in ("pdf_text", "ocr", "ocr_enhanced"))]
    raw = layout_extract(d.doc_type, d.lines)
    ref_year = reference_year(ctx.state)
    fields, seen = [], set()
    for r in raw:
        if (r["field"], r["raw"]) in seen:
            continue
        seen.add((r["field"], r["raw"]))
        n = normalize_field(r["field"], r["raw"], ref_year)
        note = n["note"]
        if n.get("repaired") and r["field"] == "vin":
            note = f"repaired from '{r['raw']}': " + n["detail"]["repairs"][0]["why"]
        ev = Evidence(field=r["field"], raw=r["raw"], value=n["value"], valid=n["valid"], validation_note=note,
                      source_file=file, doc_type=d.doc_type, method=d.text_method or "ocr",
                      line_confidence=r["line_confidence"], turn=ctx.state.current_turn,
                      approximate=bool(n.get("approximate")))
        ctx.state.evidence.append(ev)
        fields.append({"field": r["field"], "raw": r["raw"], "value": n["value"], "valid": n["valid"],
                       "note": note, "line_confidence": r["line_confidence"]})
    d.fields_extracted = True
    res: dict[str, Any] = {"file": file, "doc_type": doc_type, "fields": fields}
    # Intra-document cross-check: VIN mentioned in the narrative vs the labelled VIN box.
    labelled = {f["value"] for f in fields if f["field"] == "vin" and f["valid"]} | \
               {f["raw"] for f in fields if f["field"] == "vin"}
    mentions = set(vin_mentions(d.text)) - labelled
    if labelled and mentions:
        res["intra_document_vin_mismatch"] = {"labelled": sorted(labelled), "narrative": sorted(mentions)}
    from .extraction import SPECS
    expected = {s.field for s in SPECS.get(d.doc_type, [])}
    missing = sorted(expected - {f["field"] for f in fields})
    if missing:
        res["not_found"] = missing
        if d.text_method == "ocr":
            res["hint"] = "fields missing on a scan: try read_document(mode='enhanced_ocr') then extract again"
    if d.doc_type in (DocType.adjuster_note, DocType.customer_correspondence, DocType.unknown):
        res["hint"] = "no template for this type: read the text and call record_observation / flag_concern"
    if d.doc_type == DocType.settlement_breakdown:
        res["hint_settlement"] = "settlement has line items - check_settlement_math can verify them"
    return res


@tool({
    "file": {"type": "string", "description": "document (or customer message file) the value comes from"},
    "field": {"type": "string", "enum": ["vin", "date_of_loss", "insurance_payout", "loan_balance"]},
    "value": {"type": "string", "description": "the value as written"},
    "quote": {"type": "string", "description": "verbatim snippet from the document containing the value"},
    "approximate": {"type": "boolean", "description": "true if the source hedges ('about', 'around', '~')"},
}, ["file", "field", "value", "quote"])
def record_observation(ctx: ToolContext, file: str, field: str, value: str, quote: str,
                       approximate: bool = False) -> dict:
    """Record a field value YOU read from an unstructured source (handwritten adjuster note, customer email,
    chat message). The quote must appear in the document text (verified - fabricated quotes are rejected).
    Customer statements are stored as low-authority evidence: they can corroborate or contradict documents
    but never replace a required document. Do NOT use for values a form extraction already captured.
    Only record a value the text states AS that field: a document's own date (when a note/email was
    written, 'Date Prepared') is NOT the date of loss; an estimate or a range of vehicle value is NOT the
    insurance payout. Second-hand figures ('owner said loan is about $22k') -> approximate=true."""
    d = ctx.doc(file)
    text = _ensure_text(ctx, d)
    ok, score = _quote_found(quote, text)
    method = "customer_statement" if d.doc_type == DocType.customer_correspondence else "llm_observation"
    if not ok:
        for t in d.meta.get("vision_transcripts", []):
            ok, score = _quote_found(quote, t)
            if ok:
                method = "vision"
                break
    if not ok:
        raise ToolError(f"quote not found in {file} (best match {score}). Quote the document verbatim.")
    n = normalize_field(field, value, reference_year(ctx.state))
    # The quote must actually SAY this value: literally ("$22,340") or after normalisation
    # ("$22k" -> 22000.00, "January 28th" -> 2026-01-28). Stops a real quote carrying an invented value.
    if not _quote_found(value, quote)[0]:
        q = normalize_field(field, quote, reference_year(ctx.state))
        if not (q["value"] and n["value"] and q["value"] == n["value"]):
            raise ToolError(f"the quote does not state this value (quote reads as {q['value'] or 'nothing parseable'}, "
                            f"value reads as {n['value']}). Pass the value exactly as written in the quote.")
    ev = Evidence(field=field, raw=value, value=n["value"], valid=n["valid"], validation_note=n["note"],
                  source_file=file, doc_type=d.doc_type, method=method, quote=quote, turn=ctx.state.current_turn,
                  approximate=approximate or bool(n.get("approximate")))
    ctx.state.evidence.append(ev)
    return {"recorded": True, "method": method, "normalized": n["value"], "valid": n["valid"], "note": n["note"],
            "quote_match": score}


_SETTLEMENT_MATH_TOPIC = re.compile(r"arithmetic|calculat|math|deductible|net (insurance )?payout|gap amount|"
                                    r"(does not|doesn't|not) add up", re.I)
# A problem with one of the four fields (wrong length, mismatch, missing...) is already reported by
# reconcile / evaluate_claim; flagging it again would list the same finding twice.
_FIELD_TOPIC = {
    "vin": re.compile(r"\bvins?\b|vehicle identification", re.I),
    "date_of_loss": re.compile(r"date[ _]of[ _]loss|loss[ _]date|incident[ _]date|accident[ _]date", re.I),
    "insurance_payout": re.compile(r"insurance[ _]payout|payout amount|net payout", re.I),
    "loan_balance": re.compile(r"loan[ _]balance|outstanding[ _]balance|payoff amount", re.I),
}
_FIELD_PROBLEM = re.compile(r"mismatch|conflict|differ|disagree|(does not|doesn't|not) match|invalid|incorrect|"
                            r"wrong|too short|too long|length|characters|missing|not found|pending|tbd", re.I)


@tool({
    "code": {"type": "string", "description": "short snake_case id, e.g. possible_prior_damage"},
    "description": {"type": "string"},
    "file": {"type": "string"},
    "quote": {"type": "string"},
    "severity": {"type": "string", "enum": ["warning", "info"]},
}, ["code", "description"])
def flag_concern(ctx: ToolContext, code: str, description: str, file: str = "", quote: str = "",
                 severity: str = "warning") -> dict:
    """Surface something a human reviewer should know that is not one of the four fields - e.g. an adjuster
    doubting the valuation, identity details that don't match, a document that looks altered. Warnings are
    reported but do not by themselves block finalization (policy-configurable). Quote verified if given.
    Settlement arithmetic is not a concern to flag: it is routed to check_settlement_math. Neither is a
    problem with VIN / date of loss / payout / loan balance: those are routed to reconcile."""
    if file and quote:
        ok, score = _quote_found(quote, _ensure_text(ctx, ctx.doc(file)))
        if not ok:
            raise ToolError(f"quote not found in {file} (best match {score})")
    if file and ctx.doc(file).doc_type == DocType.settlement_breakdown and \
            _SETTLEMENT_MATH_TOPIC.search(f"{code} {description}"):
        # One finding, one owner: the deterministic check reports settlement math, so an LLM restatement of it
        # would show up twice in the output. Run the check instead (it records the warning at most once).
        return {"recorded": False, "reason": "settlement arithmetic is reported by check_settlement_math",
                "check_settlement_math": check_settlement_math(ctx, file)}
    text = f"{code} {description}"
    if _FIELD_PROBLEM.search(text):
        field = next((f for f, pat in _FIELD_TOPIC.items() if pat.search(text)), None)
        if field:
            return {"recorded": False, "reason": f"{field} problems are reported by reconcile / evaluate_claim",
                    "reconcile": reconcile(ctx, field)}
    if any(c.code == code and c.source_file == (file or None) for c in ctx.state.concerns):
        return {"recorded": False, "reason": "already recorded"}
    ctx.state.concerns.append(Concern(code=code, description=description, source_file=file or None,
                                      quote=quote or None, severity=severity, turn=ctx.state.current_turn))
    return {"recorded": True}


# =============================================================================== validation
@tool({"vin": {"type": "string"}}, ["vin"])
def validate_vin(ctx: ToolContext, vin: str) -> dict:
    """Check a VIN: assignment rule (17 alphanumeric), illegal letters I/O/Q (classic OCR confusions, with
    a suggested repair) and the ISO 3779 check digit (informational - synthetic VINs often fail it).
    Use for VINs that came from free text, customer messages, or looked suspicious."""
    return _validate_vin(vin)


@tool({"file_a": {"type": "string"}, "file_b": {"type": "string"}}, ["file_a", "file_b"])
def compare_documents(ctx: ToolContext, file_a: str, file_b: str) -> dict:
    """Decide how two documents relate: exact duplicate (same bytes), content duplicate (same document,
    e.g. a PDF and a scan of it), revision (one explicitly supersedes the other), or genuinely different.
    Call when two files have the same document type, or list_documents shows identical hashes. Duplicates
    and superseded versions are excluded from voting; the kept copy is the better-quality / newer one."""
    a, b = ctx.doc(file_a), ctx.doc(file_b)
    if a.sha256 == b.sha256:
        keep, drop = sorted([a, b], key=lambda d: (d.added_in_turn, d.file))
        drop.status, drop.related_to, drop.status_reason = "duplicate", keep.file, "byte-identical file"
        return {"relation": "exact_duplicate", "kept": keep.file, "excluded": drop.file}
    ta, tb = _ensure_text(ctx, a), _ensure_text(ctx, b)
    sim = round(difflib.SequenceMatcher(None, _squash(ta), _squash(tb)).ratio(), 3)
    fa = {(e.field, e.value) for e in ctx.state.evidence if e.source_file == file_a}
    fb = {(e.field, e.value) for e in ctx.state.evidence if e.source_file == file_b}
    differing = sorted({f for f, _ in fa ^ fb})
    # A document the agent hasn't typed yet still has a content-based type - don't let call order change the answer.
    for d, t in ((a, ta), (b, tb)):
        if not d.fields_extracted and d.doc_type == DocType.unknown:
            d.doc_type = DocType(classify(t, d.file)["doc_type"])
    same_type = a.doc_type == b.doc_type and a.doc_type != DocType.unknown
    rev_a, rev_b = a.meta.get("is_revision"), b.meta.get("is_revision")
    if same_type and (rev_a or rev_b) and rev_a != rev_b:
        new, old = (a, b) if rev_a else (b, a)
        old.status, old.related_to = "superseded", new.file
        old.status_reason = f"{new.file} carries a revision notice superseding it"
        changed = {f: {"old": next((e.value for e in ctx.state.evidence if e.source_file == old.file and e.field == f), None),
                       "new": next((e.value for e in ctx.state.evidence if e.source_file == new.file and e.field == f), None)}
                   for f in differing}
        return {"relation": "revision", "current": new.file, "superseded": old.file, "text_similarity": sim,
                "changed_fields": changed}
    if same_type and sim > 0.93 and not differing:
        rank = {"pdf_text": 0, "ocr_enhanced": 1, "ocr": 2}
        keep, drop = sorted([a, b], key=lambda d: (rank.get(d.text_method or "", 3), d.file))
        drop.status, drop.related_to = "duplicate", keep.file
        drop.status_reason = f"same content as {keep.file} (similarity {sim}); kept the higher-quality copy"
        return {"relation": "content_duplicate", "kept": keep.file, "excluded": drop.file, "text_similarity": sim}
    if same_type:
        return {"relation": "conflicting_versions", "text_similarity": sim, "differing_fields": differing,
                "note": "same document type, different values, no revision marker - both stay active; the "
                        "conflict will surface in reconciliation"}
    return {"relation": "different_documents", "text_similarity": sim}


@tool({"file": {"type": "string"}}, ["file"])
def check_settlement_math(ctx: ToolContext, file: str) -> dict:
    """Verify a settlement breakdown's arithmetic: Net payout = ACV - deductible - prior-damage adjustment,
    and Gap = loan balance - net payout. Records a warning when it does not add up."""
    d = ctx.doc(file)
    _ensure_text(ctx, d)
    vals: dict[str, Optional[Decimal]] = {}
    for r in layout_extract(d.doc_type, d.lines, SETTLEMENT_LINES):
        p = parse_amount(r["raw"])
        vals.setdefault(r["field"], Decimal(p["value"]) if p["valid"] else None)
    acv, ded, prior, net, loan, gap = (vals.get(k) for k in
                                       ("acv", "deductible", "prior_damage", "net_payout", "loan_balance", "gap"))
    checks = []
    if None in (acv, ded, net):
        return {"file": file, "parsed": {k: str(v) if v is not None else None for k, v in vals.items()},
                "result": "cannot verify - some amounts are missing or pending"}
    expected_net = acv - abs(ded) - abs(prior or 0)
    checks.append({"check": "net = ACV - deductible - prior", "expected": str(expected_net), "stated": str(net),
                   "ok": expected_net == net})
    if loan is not None and gap is not None:
        checks.append({"check": "gap = loan - net", "expected": str(loan - net), "stated": str(gap),
                       "ok": loan - net == gap})
    bad = [c for c in checks if not c["ok"]]
    if bad:
        desc = (f"Settlement arithmetic does not add up in {file}: " +
                "; ".join(f"{c['check']} -> expected {c['expected']}, stated {c['stated']}" for c in bad))
        if expected_net != net and acv == net:
            desc += " (the deductible appears not to have been subtracted)"
        if not any(c.code == "settlement_math" and c.source_file == file for c in ctx.state.concerns):
            ctx.state.concerns.append(Concern(code="settlement_math", description=desc, source_file=file,
                                              severity="warning", turn=ctx.state.current_turn))
    return {"file": file, "checks": checks, "ok": not bad,
            "policy": "warning only" if not settings.strict_settlement_math else "blocks finalization"}


@tool({"field": {"type": "string", "enum": ["vin", "date_of_loss", "insurance_payout", "loan_balance"]}}, ["field"])
def reconcile(ctx: ToolContext, field: str) -> dict:
    """Cross-check one field across every source on file (documents + customer statements). Returns the
    agreed value or, on conflict, which sources disagree, a weighted best guess and the reasoning
    (source authority, majority, transcription-error / clerical patterns). Call when a field has been seen
    in more than one source or when sources disagree."""
    res, a = reconcile_field(ctx.state, field)
    return {"field": field, "value": res.value, "confidence": res.confidence, "reason": res.reason,
            "conflict": a.get("conflict", False), "groups": a.get("groups"), "sources": res.sources}


@tool({})
def cross_check_documents(ctx: ToolContext) -> dict:
    """Compare everything else the documents share - vehicle (year/make/model), lender, owner/claimant/
    borrower name, policy number, police report number - across all active documents (duplicates and
    superseded versions don't vote). Returns per attribute: consistent / single_source / ocr_variant /
    contradiction, which documents disagree, and a best guess or 'No guess'. Call once the documents are
    read; evaluate_claim also applies these results."""
    from .consistency import check

    issues, report = check(ctx.state)
    return {"attributes": report,
            "issues": [{"severity": i.severity, "description": i.description, "details": i.details,
                        "suggested_value": i.suggested_value} for i in issues]}


@tool({})
def evaluate_claim(ctx: ToolContext) -> dict:
    """Run the deterministic decision rules over everything recorded so far: per-field results with
    confidence, missing documents, issues, status (complete / incomplete / needs_review) and the
    recommended action. You cannot override the status. Call before submit_turn, and again after new info."""
    ev = evaluate(ctx.state)
    ctx.state.evaluation = ev
    return {
        "status": ev.status,
        "status_reason": ev.status_reason,
        "recommended_action": ev.recommended_action,
        "missing_documents": ev.missing_documents,
        "fields": {k: {"value": v.value, "confidence": v.confidence, "reason": v.reason}
                   for k, v in ev.extracted_fields.items()},
        "issues": [{"type": i.type, "severity": i.severity, "field": i.field, "description": i.description,
                    "suggested_value": i.suggested_value, "resolvable_by": i.resolvable_by} for i in ev.issues],
        "open_customer_questions": [q.question for q in ctx.state.questions if q.turn == ctx.state.current_turn],
    }


# =============================================================================== conversation
@tool({
    "kind": {"type": "string", "enum": ["document_promise", "document_unavailable", "question", "authorization",
                                         "other_request"]},
    "text": {"type": "string", "description": "short paraphrase (e.g. the question being asked)"},
    "doc_type": {"type": "string", "enum": ["police_report", "finance_agreement", "settlement_breakdown", "other", ""]},
    "eta": {"type": "string"},
    "quote": {"type": "string"},
}, ["kind", "text"])
def note_customer_intent(ctx: ToolContext, kind: str, text: str, doc_type: str = "", eta: str = "",
                         quote: str = "") -> dict:
    """Record what the customer is telling us beyond field values: a document they will send later
    (document_promise), one they cannot provide (document_unavailable), a question they asked that your
    reply must answer, or an authorization they gave. A promise does NOT satisfy a missing document."""
    t = ctx.state.current_turn
    if kind in ("document_promise", "document_unavailable"):
        ctx.state.commitments.append(CustomerCommitment(
            doc_type=doc_type or "other", eta=(eta or None) if kind == "document_promise" else "unavailable",
            quote=quote or text, turn=t))
    else:
        ctx.state.questions.append(CustomerQuestion(question=f"[{kind}] {text}", turn=t))
    return {"recorded": True}


# Our own review findings are not the customer's business: math errors in OUR settlement, adjuster notes,
# confidence levels, OCR. They go in internal_note.
_INTERNAL_ONLY = [r"arithmetic", r"settlement math", r"settlement calculation", r"calculation (error|mismatch)", r"\badjuster", r"prior damage",
                  r"confidence", r"\bocr\b", r"check digit"]
# The agent can't change records or contact third parties - its messages must not claim it did / will.
_OVERCLAIM = [r"(updated|corrected|changed|amended) (our|your|the) (records?|file|claim|date)",
              r"we (will|'ll|are going to) (proceed by )?(contact|call|reach out to|request \w+ from) (your )?(lender|bank|police|department|dealer)",
              r"we have (contacted|requested|obtained|received the (police|finance))"]
# Only the rules decide what satisfies a requirement - the message must not invent substitutes or portals.
_POLICY_INVENTION = [r"substitute", r"in place of", r"instead of (a|the|your) (finance|police)", r"portal"]
_FORBIDDEN = [r"\bguarantee", r"\bwill be paid (on|by|within)\b", r"\bapproved\b", r"\bdeny|denied\b"]


@tool({
    "action": {"type": "string", "enum": ["finalize", "message_customer", "escalate"]},
    "customer_message": {"type": "string",
                         "description": "message to the customer (required when they must do something or asked a question)"},
    "internal_note": {"type": "string", "description": "concise note for the claims team / reviewer"},
    "follow_ups": {"type": "array", "items": {"type": "string"}, "description": "concrete next steps"},
    "summary": {"type": "string", "description": "one-paragraph summary of what you found this turn"},
}, ["action", "summary"])
def submit_turn(ctx: ToolContext, action: str, summary: str, customer_message: str = "", internal_note: str = "",
                follow_ups: Optional[list] = None) -> dict:
    """Finish this turn. Guardrails are checked: the action must be compatible with the evaluated status,
    the customer message must request every customer-fixable missing item and must not promise outcomes.
    If rejected, fix the problem and call again."""
    ev = evaluate(ctx.state)  # always fresh - the agent may have recorded more since evaluate_claim
    ctx.state.evaluation = ev
    problems = []
    # every submitted document must be analysed (or explicitly excluded as duplicate / superseded)
    unprocessed = [d.file for d in ctx.state.active_docs()
                   if d.doc_type != DocType.customer_correspondence and not d.fields_extracted]
    if unprocessed:
        problems.append(f"not every document was processed: {unprocessed} - read them and call extract_fields "
                        "(or compare_documents if they may be duplicates/revisions)")
    allowed = {"complete": {"finalize", "escalate"}, "incomplete": {"message_customer", "escalate"},
               "needs_review": {"escalate"}}[ev.status]
    if action not in allowed:
        problems.append(f"action '{action}' is not allowed when status is {ev.status} (allowed: {sorted(allowed)}); "
                        "for needs_review use escalate - you can still include a customer_message")
    if action == "message_customer" and not customer_message.strip():
        problems.append("message_customer requires customer_message")
    customer_fixable = [i for i in ev.issues if i.severity == "blocking" and i.resolvable_by.startswith("customer")]
    if customer_fixable and not customer_message.strip() and action != "finalize":
        problems.append("there are customer-fixable issues - include a customer_message asking for them")
    msg = customer_message.lower()
    for d in ev.missing_documents:
        words = {"police_report": ["police"], "finance_agreement": ["finance", "loan"],
                 "settlement_breakdown": ["settlement"]}[d]
        if customer_message and not any(w in msg for w in words) and d != "settlement_breakdown":
            problems.append(f"customer_message does not mention the missing {d.replace('_', ' ')}")
    open_q = [q for q in ctx.state.questions if q.turn == ctx.state.current_turn]
    if open_q and not customer_message.strip():
        problems.append("the customer asked questions this turn - your message must answer them")
    for pat in _FORBIDDEN:
        if re.search(pat, msg):
            problems.append(f"customer_message makes a promise/decision it should not ('{pat}')")
    for pat in _POLICY_INVENTION:
        if re.search(pat, msg):
            problems.append(f"customer_message invents policy ('{pat}'): only the required document satisfies the "
                            "requirement, and there is no portal - ask them to reply with the document attached")
    for pat in _OVERCLAIM:
        if re.search(pat, msg):
            problems.append("customer_message claims an action that was not taken (records are only changed by a "
                            "reviewer; we don't contact third parties). Say the team will review it instead.")
    for pat in _INTERNAL_ONLY:
        if re.search(pat, msg):
            problems.append(f"customer_message reveals internal review details ('{pat}') - keep those in internal_note")
    if problems:
        raise ToolError("submit rejected: " + " | ".join(problems))
    # PII: never echo full account numbers back over email - keep the last 4 digits only
    customer_message = re.sub(r"\b(\d{4})[- ](\d{4})[- ](\d{4})\b", r"****-****-\3", customer_message)
    ctx.submitted = {"action": action, "customer_message": customer_message.strip() or None,
                     "internal_note": internal_note.strip() or None, "follow_ups": follow_ups or [],
                     "summary": summary}
    return {"accepted": True, "status": ev.status}


# =============================================================================== dispatch
def schemas(names: Optional[list[str]] = None) -> list[dict]:
    return [t["schema"] for n, t in TOOLS.items() if names is None or n in names]


def _summarize(result: Any) -> Any:
    """Compact version of a tool result for the audit log / UI (drop bulky text)."""
    if isinstance(result, dict):
        return {k: ("<%d chars>" % len(v) if k == "text" and isinstance(v, str) else v) for k, v in result.items()
                if k not in ("low_confidence_lines",)}
    return result


def call(ctx: ToolContext, name: str, args: dict) -> tuple[bool, Any]:
    ctx.step += 1
    if name not in TOOLS:
        result, ok = {"error": f"unknown tool {name}"}, False
    else:
        try:
            result, ok = TOOLS[name]["fn"](ctx, **args), True
        except ToolError as e:
            result, ok = {"error": str(e)}, False
        except TypeError as e:
            result, ok = {"error": f"bad arguments: {e}"}, False
    rec = ToolCall(turn=ctx.state.current_turn, step=ctx.step, tool=name, input=args, result=_summarize(result), ok=ok)
    ctx.state.tool_log.append(rec)
    ctx.emit({"type": "tool", **json.loads(rec.model_dump_json())})
    return ok, result


def register_document(state: ClaimState, path: Path, turn: int = 0) -> Document:
    kind = file_kind(path)
    if kind == "unsupported":
        raise ToolError(f"unsupported file type: {path.name}")
    d = Document(file=path.name, kind=kind, sha256=sha256(path), size=path.stat().st_size, added_in_turn=turn)
    state.documents[path.name] = d
    return d


def add_text_document(state: ClaimState, name: str, text: str, turn: int) -> Document:
    folder = settings.state_dir / state.claim_id
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text(text, encoding="utf-8")
    d = register_document(state, p, turn)
    d.doc_type = DocType.customer_correspondence
    d.text, d.text_method = text, "plain_text"
    return d


def is_reply(text: str) -> bool:
    return bool(re.search(r"^Subject:\s*(RE|Re|re):", text, re.M))
