"""Deterministic document classification + layout-aware field extraction.

Why deterministic for the structured forms? Every value we report must be traceable to a
line on the page with an OCR confidence. An LLM reading the page is used only where there
is no form structure (handwritten notes, emails) - and even then its output must quote
text that verifiably exists in the document (see tools.record_observation).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Optional

from .models import DocType, TextLine

# ----------------------------------------------------------------------------- classification
_KEYWORDS: dict[DocType, list[tuple[str, float]]] = {
    DocType.police_report: [("policedepartment", 3), ("incidentreport", 2), ("accidentreport", 2),
                            ("reportingofficer", 2), ("badgeno", 1), ("narrative", 1), ("dateofincident", 1)],
    DocType.finance_agreement: [("retailinstallmentcontract", 3), ("financeagreement", 3), ("borrower", 2),
                                ("currentoutstandingbalance", 2), ("interestrate", 1), ("lenderrepresentative", 1),
                                ("monthlypayment", 1)],
    DocType.settlement_breakdown: [("settlementbreakdown", 3), ("settlementstatement", 2), ("actualcashvalue", 2),
                                   ("netinsurancepayout", 2), ("deductible", 1), ("gapamount", 1)],
    DocType.adjuster_note: [("adjusternote", 3), ("fieldadjuster", 3), ("adjuster", 1), ("inspected", 1)],
    DocType.tow_receipt: [("towing", 3), ("towfee", 2), ("receipt", 1), ("pickup", 1), ("storage", 1)],
    DocType.customer_correspondence: [("from:", 2), ("subject:", 2), ("to:", 1), ("regards", 1), ("thanks", 1)],
}


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9:]", "", text.lower())


def classify(text: str, filename: str = "") -> dict:
    squashed = _squash(text)
    scores = {t: sum(w for k, w in kws if k in squashed) for t, kws in _KEYWORDS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (top, s1), (second, s2) = ranked[0], ranked[1]
    if s1 < 3:
        return {"doc_type": DocType.unknown.value, "confidence": 0.2, "scores": _nonzero(scores),
                "reason": "no strong content signals; needs judgement"}
    conf = round(min(1.0, s1 / 8) * (s1 / (s1 + s2)), 2)
    hint = _squash(filename.rsplit(".", 1)[0]).replace("_", "")
    name_agrees = top.value.replace("_", "") in hint
    return {
        "doc_type": top.value,
        "confidence": conf,
        "runner_up": second.value if s2 else None,
        "scores": _nonzero(scores),
        "filename_agrees": name_agrees,
        "reason": f"content keywords (score {s1} vs {s2})" + ("" if name_agrees or not hint else "; NOTE filename suggests otherwise"),
    }


def _nonzero(scores):
    return {k.value: v for k, v in scores.items() if v}


# ----------------------------------------------------------------------------- metadata
def document_meta(text: str) -> dict:
    meta: dict = {}
    squashed = _squash(text)
    if "revisionnotice" in squashed or "supersedes" in squashed or re.search(r"\bREVISED\b", text):
        meta["is_revision"] = True
        m = re.search(r"supersedes[^.]*?dated\s+([0-9/]+)", text, re.I)
        if m:
            meta["supersedes_dated"] = m.group(1)
    m = re.search(r"Date Prepared:?\s*([0-9OIl/]{8,10})", text, re.I)
    if m:
        meta["prepared_date"] = m.group(1)
    m = re.search(r"\b(AUT-\d{5,8})", text)
    if m:
        meta["policy_number"] = m.group(1)
    return meta


# ----------------------------------------------------------------------------- spatial extraction
@dataclass
class FieldSpec:
    field: str
    labels: list[str]
    layout: str  # below | right


SPECS: dict[DocType, list[FieldSpec]] = {
    DocType.police_report: [
        FieldSpec("vin", ["VIN"], "below"),
        FieldSpec("date_of_loss", ["DATE OF INCIDENT", "DATE OF LOSS", "DATE OF ACCIDENT"], "below"),
    ],
    DocType.finance_agreement: [
        FieldSpec("vin", ["VIN"], "below"),
        FieldSpec("loan_balance", ["CURRENT OUTSTANDING BALANCE", "OUTSTANDING BALANCE", "PAYOFF AMOUNT"], "below"),
    ],
    DocType.settlement_breakdown: [
        FieldSpec("vin", ["VIN"], "below"),
        FieldSpec("date_of_loss", ["DATE OF LOSS"], "below"),
        FieldSpec("insurance_payout", ["NET INSURANCE PAYOUT"], "right"),
        FieldSpec("loan_balance", ["OUTSTANDING LOAN BALANCE"], "right"),
    ],
    DocType.tow_receipt: [FieldSpec("vin", ["VIN"], "right")],
}

# CAPTURE patterns - deliberately looser than validation (validators.py). Their job is to pick the right
# text next to a label; whether it is a *valid* value is decided later, so that e.g. a truncated 10-char
# VIN is reported as "invalid" instead of silently "not found".
_VALUE_PATTERNS = {
    "vin": re.compile(r"^[A-Z0-9]{8,20}$", re.I),
    "date_of_loss": re.compile(r"[0-9OoIl]{1,4}\s*[/\-.]\s*[0-9OoIl]{1,2}\s*[/\-.]\s*[0-9OoIl]{2,4}|[A-Z][a-z]+\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}"),
    # Optional identity check only (see person_name): "Sarah Mitchell", "SARAH MITCHELL", "Kevin O'Brien",
    # "Mary-Jane de la Cruz". Not one of the four required fields.
    "name": re.compile(r"^[A-Za-z][A-Za-z'\-]+(?: [A-Za-z]\.?)?(?: [A-Za-z][A-Za-z'\-]+){1,3}$"),
    "money": re.compile(r"(?:US)?\$\s*[\d,.]+|\bUSD\s*[\d,.]+|\b\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\b(TBD|pending|N/A)\b", re.I),
    # shared, non-required facts used by the cross-document consistency check (consistency.py)
    "vehicle": re.compile(r"^(?:19|20)\d{2}\s*[A-Za-z][A-Za-z0-9 \-]*$"),
    "report_number": re.compile(r"^[A-Z]{2,4}[\-=]?\d{4}[\-=]?\d{3,6}$"),
    "lender": re.compile(r"^[A-Z][A-Za-z.&' ]{2,40}$"),
}
_WHOLE_MATCH = ("date_of_loss", "name", "vehicle", "report_number", "lender")


def _pattern(field: str):
    return _VALUE_PATTERNS.get(field, _VALUE_PATTERNS["money"])


def _label_match(line: str, label: str) -> Optional[str]:
    """Return the remainder of the line after the label if the line *is* (or starts with) the label.
    Tolerates OCR noise (LOCATI0N, VERICL&, missing spaces)."""
    sq_line, sq_label = _squash(line).replace(":", ""), _squash(label).replace(":", "")
    if sq_line.startswith(sq_label):
        # remainder in original text, after the label's last character
        idx = re.search(r"\s*".join(map(re.escape, label.replace(" ", ""))), line, re.I)
        return line[idx.end():].lstrip(" :#") if idx else ""
    head = sq_line[: len(sq_label)]
    if len(sq_label) >= 6 and difflib.SequenceMatcher(None, head, sq_label).ratio() >= 0.84:
        return line[len(label):].lstrip(" :#") if len(line) > len(label) + 2 else ""
    return None


def _clean_value(field: str, text: str) -> Optional[str]:
    text = text.strip()
    if field == "vin":
        tok = re.sub(r"[\s\-]", "", text.upper())
        tok = re.sub(r"^VIN:?", "", tok)
        # real VINs always carry digits; this rejects neighbouring labels/names ("ALLYFINANCIAL")
        return tok if _pattern("vin").match(tok) and sum(c.isdigit() for c in tok) >= 3 else None
    m = _pattern(field).search(text)
    if not m:
        return None
    if field in _WHOLE_MATCH:
        return m.group(0)
    return text[m.start():].split("  ")[0].strip()


NAME_SPECS = [FieldSpec("name", ["REGISTERED OWNER", "CLAIMANT", "BORROWER NAME", "OWNER"], "below")]


def person_name(lines: list[TextLine]) -> Optional[str]:
    """Policyholder / owner / borrower name - used for a cross-document identity check."""
    hits = extract_fields(DocType.unknown, lines, NAME_SPECS)
    if not hits:  # receipts: "Owner:   Marcus Thompson" on the same row
        hits = extract_fields(DocType.unknown, lines, [FieldSpec("name", ["OWNER"], "right")])
    return hits[0]["raw"] if hits else None


SETTLEMENT_LINES = [
    FieldSpec("acv", ["ACTUAL CASH VALUE"], "right"),
    FieldSpec("deductible", ["LESS: DEDUCTIBLE"], "right"),
    FieldSpec("prior_damage", ["LESS: PRIOR DAMAGE ADJUSTMENT"], "right"),
    FieldSpec("net_payout", ["NET INSURANCE PAYOUT"], "right"),
    FieldSpec("loan_balance", ["OUTSTANDING LOAN BALANCE"], "right"),
    FieldSpec("gap", ["GAP AMOUNT"], "right"),
]


def extract_fields(doc_type: DocType, lines: list[TextLine], specs: Optional[list[FieldSpec]] = None) -> list[dict]:
    """Returns [{field, raw, line_confidence, label, how}] - raw values, not yet validated."""
    found: list[dict] = []
    for spec in specs if specs is not None else SPECS.get(doc_type, []):
        hits = []
        for i, line in enumerate(lines):
            for label in spec.labels:
                rest = _label_match(line.text, label)
                if rest is None:
                    continue
                # 1) inline value ("VIN: 1N4...")
                val = _clean_value(spec.field, rest) if rest else None
                if val:
                    hits.append({"raw": val, "line_confidence": line.conf, "label": label, "how": "inline"})
                    break
                cand = _neighbour(spec, line, lines)
                if cand:
                    hits.append(cand | {"label": label})
                break
            if hits and spec.layout == "below":
                break  # first labelled occurrence is authoritative for form fields
        # de-duplicate identical raws (e.g. "Net Insurance Payout" appears in two tables)
        seen = set()
        for h in hits:
            if h["raw"] not in seen:
                seen.add(h["raw"])
                found.append({"field": spec.field, **h})
    return found


def _neighbour(spec: FieldSpec, label: TextLine, lines: list[TextLine]) -> Optional[dict]:
    lh = max(label.y1 - label.y0, 8)
    lc = (label.y0 + label.y1) / 2
    best, best_d = None, 1e9
    for other in lines:
        if other is label:
            continue
        val = _clean_value(spec.field, other.text)
        if not val:
            continue
        oc = (other.y0 + other.y1) / 2
        if spec.layout == "below":
            dy = oc - lc  # centre-to-centre: OCR boxes of label and value often overlap
            aligned = abs(other.x0 - label.x0) < 80 or (other.x0 < label.x1 and other.x1 > label.x0)
            if lh * 0.4 < dy < lh * 4 and aligned and dy < best_d:
                best, best_d = other, dy
        else:  # right: same row, any distance to the right (tables / receipts)
            if abs(oc - lc) < lh * 1.1 and other.x0 > label.x1 - 5:
                d = other.x0 - label.x1 + abs(oc - lc) * 10
                if d < best_d:
                    best, best_d = other, d
    if best is None:
        return None
    return {"raw": _clean_value(spec.field, best.text), "line_confidence": best.conf, "how": spec.layout}


# "VIN: X", "VIN X", "VIN from door jamb: X" - value must be on the same line.
VIN_IN_TEXT = re.compile(r"\bVIN(?:[ \t]*from[ \t]*[A-Za-z \t]{0,20}?)?[ \t]*[:#]?[ \t]*([A-Z0-9][A-Z0-9\-]{7,22})", re.I)


def vin_mentions(text: str) -> list[str]:
    """All 'VIN: XXXX' mentions in free text (narratives, emails, notes)."""
    out = []
    for m in VIN_IN_TEXT.finditer(text):
        tok = re.sub(r"[\s\-]", "", m.group(1).upper())
        tok = re.match(r"[A-Z0-9]+", tok).group(0) if tok else ""
        if 8 <= len(tok) <= 20 and sum(c.isdigit() for c in tok) >= 3:
            out.append(tok[:17] if len(tok) > 17 and tok[17:].isalpha() else tok)
    return out
