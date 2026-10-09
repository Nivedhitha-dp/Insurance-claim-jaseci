"""Cross-document consistency for EVERYTHING the documents share - not just the four required fields.

The brief: "when the same field shows up in multiple documents, the values should match; if not, flag it,
say which documents disagree and, if possible, give an educated guess". The four claim fields are handled in
reconcile.py (with confidence scoring). This module covers the other shared facts:

    vehicle (year / make / model)   lender   owner/claimant/borrower name   policy number   police report no.

Each mismatch is classified, because "different text" is not the same as "contradiction":
    compatible       same fact, different spelling ("Capital One" vs "Capital One Auto Finance",
                     "2019 Ford Focus" vs "2019 Ford Focus SE", "YPD-202502871" vs "YPD-2025-02871")
    ocr_variant      near-identical and at least one side was OCR'd / handwritten ("EX-C" vs "EX-L")
                     -> reported (info), never silently merged, not blocking
    contradiction    the documents genuinely disagree -> blocking issue (needs_review), which documents
                     disagree, and an educated guess - or explicitly "No guess" when the sources are
                     equally authoritative
Duplicates and superseded versions don't vote (they'd just echo another document).
A customer's statement that disagrees with the documents is reported as a warning (it's not a document).
"""
from __future__ import annotations

import difflib
import re
from typing import Optional

from .extraction import FieldSpec, extract_fields as layout_extract, person_name
from .models import ClaimState, DocType, Document, Issue

ATTRS = {
    "vehicle": "Vehicle (year / make / model)",
    "lender": "Lender",
    "person_name": "Owner / claimant / borrower name",
    "policy_number": "Policy number",
    "police_report_number": "Police report number",
}
AUTHORITY = {
    "vehicle": {DocType.police_report: 3, DocType.finance_agreement: 3, DocType.settlement_breakdown: 2,
                DocType.tow_receipt: 1, DocType.adjuster_note: 1, DocType.customer_correspondence: 0.5},
    "lender": {DocType.finance_agreement: 4, DocType.settlement_breakdown: 2, DocType.adjuster_note: 1,
               DocType.customer_correspondence: 0.5},
    "person_name": {DocType.police_report: 3, DocType.finance_agreement: 3, DocType.settlement_breakdown: 2,
                    DocType.tow_receipt: 1, DocType.adjuster_note: 1},
    "policy_number": {DocType.settlement_breakdown: 3, DocType.police_report: 3, DocType.adjuster_note: 1,
                      DocType.tow_receipt: 1, DocType.customer_correspondence: 0.5},
    "police_report_number": {DocType.police_report: 4, DocType.tow_receipt: 1},
}
_GENERIC_LENDER = r"(autofinance|autoloans?|financial|finance|auto|bankna|na|inc|llc|corp|corporation|company|co)$"


# ------------------------------------------------------------------------------------------- extraction
def _first(specs: list[FieldSpec], lines: list[TextLine]) -> Optional[tuple[str, float]]:
    """First labelled value found by the layout extractor (field name selects the value pattern)."""
    for spec in specs:
        hits = layout_extract(DocType.unknown, lines, [spec])
        if hits:
            return hits[0]["raw"], hits[0]["line_confidence"]
    return None


def extract_attributes(doc: Document) -> dict[str, dict]:
    """Shared, non-required facts in one document: {attr: {"raw", "conf"}}."""
    out: dict[str, dict] = {}
    text, lines = doc.text or "", doc.lines or []
    ocr = (doc.text_method or "").startswith("ocr") or doc.doc_type == DocType.adjuster_note

    def put(attr, raw, conf=1.0):
        if raw and attr not in out:
            out[attr] = {"raw": raw.strip(), "conf": conf, "ocr": ocr}

    if lines:
        v = _first([FieldSpec("vehicle", ["YEAR / MAKE / MODEL", "VEHICLE"], "below"),
                    FieldSpec("vehicle", ["VEHICLE"], "right")], lines)
        if v:
            put("vehicle", *v)
        r = _first([FieldSpec("report_number", ["POLICE REPORT #", "POLICE REPORT NO", "REPORT NO"], "right")], lines)
        if r:
            put("police_report_number", *r)
        name = person_name(lines)
        if name:
            put("person_name", name, 0.95 if ocr else 1.0)
        if doc.doc_type == DocType.settlement_breakdown:
            ld = _first([FieldSpec("lender", ["LENDER"], "below")], lines)
            if ld:
                put("lender", *ld)
        if doc.doc_type == DocType.finance_agreement:
            # the lender's own letterhead: left-most text in the top band of page 1 (the right side holds
            # the form title "RETAIL INSTALLMENT CONTRACT")
            page_top = min(l.y0 for l in lines if l.page == 0)
            band = [l for l in lines if l.page == 0 and l.y0 < page_top + 25
                    and re.fullmatch(r"[A-Za-z.&' ]{3,40}", l.text)
                    and not re.search(r"contract|agreement|installment", l.text, re.I)]
            if band:
                head = min(band, key=lambda l: l.x0)
                put("lender", head.text, head.conf)
            else:  # fallback: footer "Ally Financial | NMLS #382230 | ..."
                m = re.search(r"^([A-Z][A-Za-z.&' ]{2,40}?)\s*\|\s*NMLS", text, re.M)
                if m:
                    put("lender", m.group(1))
    # inline "Label: value" forms (handwritten notes, receipts, emails)
    for attr, pat in (("vehicle", r"Vehicle\s*:\s*((?:19|20)\d{2}\s*[A-Za-z][A-Za-z0-9 \-]*)"),
                      ("lender", r"Lender\s*:\s*([A-Z][A-Za-z.&' ]{2,40})"),
                      ("police_report_number", r"(?:Report No|Police Report #)\s*:?\s*([A-Z]{2,4}[\-=]?\d{4}[\-=]?\d{3,6})")):
        m = re.search(pat, text)
        if m:
            put(attr, m.group(1).split("  ")[0])
    if doc.doc_type == DocType.customer_correspondence:  # "My car (2019 Ford Focus, VIN ...)"
        m = re.search(r"\b((?:19|20)\d{2}\s+[A-Z][a-zA-Z]+\s+[A-Z0-9][A-Za-z0-9\-]*(?:\s+[A-Z0-9]{1,4}\b)?)", text)
        if m:
            put("vehicle", m.group(1))
    if doc.meta.get("policy_number"):
        put("policy_number", doc.meta["policy_number"])
    return out


# ------------------------------------------------------------------------------------------- comparison
def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def normalize(attr: str, raw: str) -> str:
    s = _squash(raw)
    if attr == "lender":
        for _ in range(3):  # "Capital One Auto Finance" ~ "Capital One"; "Ally Financial" ~ "Ally"
            s = re.sub(_GENERIC_LENDER, "", s) or s
        s = re.sub(r"^bankof", "bankof", s)
    if attr == "person_name":  # drop middle initials: "Sarah J Mitchell" ~ "Sarah Mitchell"
        parts = [p for p in re.split(r"\s+", raw.lower().strip()) if len(p.strip(".")) > 1]
        s = _squash(" ".join(parts))
    return s


def relation(attr: str, a: str, b: str) -> str:
    """'same' | 'compatible' | 'ocr_variant' | 'contradiction' for two normalised values."""
    if a == b:
        return "same"
    if attr == "vehicle":
        ya, yb = a[:4], b[:4]
        if ya != yb:
            return "contradiction"  # model year is never an OCR-ambiguous detail worth guessing about
        ra, rb = a[4:], b[4:]
        if ra.startswith(rb) or rb.startswith(ra):  # missing trim: "fordfocus" vs "fordfocusse"
            return "compatible"
        a, b = ra, rb
    if attr == "lender" and (a in b or b in a):
        return "compatible"
    sim = difflib.SequenceMatcher(None, a, b).ratio()
    if sim >= 0.85 or (len(a) == len(b) and sum(x != y for x, y in zip(a, b)) <= 1):
        return "ocr_variant"
    return "contradiction"


def check(state: ClaimState) -> tuple[list[Issue], dict]:
    """Compare every shared attribute across active documents (+ customer statements)."""
    report: dict[str, dict] = {}
    issues: list[Issue] = []
    observations: dict[str, list[dict]] = {a: [] for a in ATTRS}
    for doc in state.active_docs():
        if doc.text is None:
            continue
        attrs = extract_attributes(doc)
        doc.meta["attributes"] = {k: v["raw"] for k, v in attrs.items()}
        for attr, v in attrs.items():
            observations[attr].append({"file": doc.file, "type": doc.doc_type, "raw": v["raw"], "ocr": v["ocr"],
                                       "norm": normalize(attr, v["raw"])})

    for attr, obs in observations.items():
        if len(obs) < 2:
            if obs:
                report[attr] = {"status": "single_source", "value": obs[0]["raw"], "sources": [obs[0]["file"]]}
            continue
        # cluster: each observation joins the first cluster it is same/compatible with
        clusters: list[list[dict]] = []
        for o in obs:
            for c in clusters:
                if relation(attr, c[0]["norm"], o["norm"]) in ("same", "compatible"):
                    c.append(o)
                    break
            else:
                clusters.append([o])

        def weight(c):
            return sum(AUTHORITY[attr].get(o["type"], 0.5) for o in c)

        clusters.sort(key=weight, reverse=True)
        best = clusters[0]
        # display the most authoritative, most readable spelling (OCR can squash spaces: "CapitalOneAutoFinance")
        # (all members of a cluster are the same fact, so this only picks the cleanest spelling)
        best_value = max(best, key=lambda o: (not o["ocr"], o["raw"].count(" "),
                                              AUTHORITY[attr].get(o["type"], 0.5), len(o["raw"])))["raw"]
        if len(clusters) == 1:
            report[attr] = {"status": "consistent", "value": best_value, "sources": [o["file"] for o in best]}
            continue

        listing = "; ".join(f"{', '.join(o['file'] for o in c)}: {c[0]['raw']}" for c in clusters)
        for other in clusters[1:]:
            rel = relation(attr, best[0]["norm"], other[0]["norm"])
            customer_only = all(o["type"] == DocType.customer_correspondence for o in other) or \
                all(o["type"] == DocType.customer_correspondence for o in best)
            w1, w2 = weight(best), weight(other)
            guess = best_value if w1 >= 1.5 * w2 else None
            why = (f"best guess '{best_value}' (supported by {', '.join(o['file'] for o in best)})" if guess
                   else "No guess - the disagreeing sources are equally authoritative")
            if rel == "ocr_variant" and any(o["ocr"] for o in best + other):
                issues.append(Issue(type="warning", severity="info", code=f"{attr}_ocr_variant", field=attr,
                                    description=f"{ATTRS[attr]}: near-identical readings, likely an OCR/handwriting misread",
                                    details=f"{listing}. Not treated as a contradiction; {why}.",
                                    suggested_value=guess, resolvable_by="internal"))
                status = "ocr_variant"
            elif customer_only:
                issues.append(Issue(type="warning", severity="warning", code=f"{attr}_customer_mismatch", field=attr,
                                    description=f"{ATTRS[attr]}: customer's statement disagrees with the documents",
                                    details=f"{listing}. {why}.", suggested_value=guess, resolvable_by="internal"))
                status = "customer_mismatch"
            else:
                issues.append(Issue(type="inconsistency", severity="blocking", code=f"{attr}_mismatch", field=attr,
                                    description=f"{ATTRS[attr]} differs across documents",
                                    details=f"{listing}. {why}.",
                                    suggested_value=guess or "No guess", resolvable_by="internal"))
                status = "contradiction"
            report[attr] = {"status": status, "value": best_value if guess else None, "clusters": [
                {"value": c[0]["raw"], "sources": [o["file"] for o in c]} for c in clusters]}
    return issues, report
