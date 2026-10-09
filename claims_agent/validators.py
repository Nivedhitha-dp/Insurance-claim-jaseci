"""Deterministic validators/normalizers. Pure functions, no LLM - these are the facts layer.

Design rule: be *liberal in what we read, strict in what we accept*. Extraction captures anything
that looks like a value (so a truncated VIN is reported as invalid instead of silently missed);
these functions decide whether it is valid, normalise it, and explain every assumption they made.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional

# ============================================================================= VIN
# The assignment's rule, verbatim: exactly 17 alphanumeric characters.
VIN_PATTERN = re.compile(r"^[A-Z0-9]{17}$", re.I)

_VIN_TRANSLIT = {
    **{str(i): i for i in range(10)},
    **dict(zip("ABCDEFGH", range(1, 9))),
    **dict(zip("JKLMN", range(1, 6))), "P": 7, "R": 9,
    **dict(zip("STUVWXYZ", range(2, 10))),
}
_VIN_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

# Unambiguous OCR repairs: ISO 3779 forbids I, O and Q in VINs, so they can only be misreads of 1/0/0.
# A lowercase "l" never appears on a printed VIN either (and uppercasing it would hide it as a legal "L").
_VIN_ILLEGAL = {"I": "1", "O": "0", "Q": "0"}

# Ambiguous OCR look-alikes: BOTH characters are legal in a VIN, so we never auto-correct them.
# They are used only to recognise that two documents probably show the *same* VIN (see reconcile.py).
VIN_CONFUSABLE = [{"S", "5"}, {"B", "8"}, {"Z", "2"}, {"G", "6"}, {"D", "0"}, {"I", "1"}, {"O", "0"}, {"Q", "0"}]


def vin_check_digit(vin: str) -> Optional[str]:
    if len(vin) != 17 or any(c not in _VIN_TRANSLIT for c in vin):
        return None
    total = sum(_VIN_TRANSLIT[c] * w for c, w in zip(vin, _VIN_WEIGHTS)) % 11
    return "X" if total == 10 else str(total)


def vin_ocr_variant(a: str, b: str) -> Optional[list[int]]:
    """If `a` and `b` differ ONLY at positions holding OCR look-alikes (S/5, B/8, Z/2, G/6, D/0...),
    return those 1-based positions; otherwise None. Equal strings -> None."""
    if len(a) != len(b) or a == b:
        return None
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if all({a[i], b[i]} in VIN_CONFUSABLE for i in diff):
        return [i + 1 for i in diff]
    return None


def validate_vin(raw: str) -> dict[str, Any]:
    """Assignment rule (17 alphanumeric) + unambiguous OCR repair + informational check digit."""
    text = (raw or "").strip()
    repairs = []
    if "l" in text:  # lowercase L is never printed in a VIN; OCR produces it for "1"
        text = text.replace("l", "1")
        repairs.append("lowercase 'l' read as '1'")
    normalized = re.sub(r"[\s\-.:]", "", text.upper())
    errors, warnings = [], []
    if not re.fullmatch(r"[A-Z0-9]*", normalized):
        errors.append("contains non-alphanumeric characters")
    if len(normalized) != 17:
        errors.append(f"length is {len(normalized)}, expected 17")

    illegal = sorted({c for c in normalized if c in _VIN_ILLEGAL})
    if illegal:
        normalized = "".join(_VIN_ILLEGAL.get(c, c) for c in normalized)
        repairs.append(f"{'/'.join(illegal)} can never appear in a VIN - read as "
                       f"{'/'.join(_VIN_ILLEGAL[c] for c in illegal)} (OCR misread)")

    valid = not errors and bool(VIN_PATTERN.match(normalized))
    cd = None
    if valid:
        expected = vin_check_digit(normalized)
        cd = {"expected": expected, "actual": normalized[8], "passes": expected == normalized[8]}
        if not cd["passes"]:
            warnings.append("check digit (position 9) does not match - informational only; non-North-American "
                            "and synthetic VINs often fail this")
    return {
        "input": raw,
        "normalized": normalized,
        "valid": valid,
        "repaired": bool(repairs) and valid,
        "errors": errors,
        "warnings": warnings,
        "repairs": repairs,
        "check_digit": cd,
    }


# ============================================================================= dates
_OCR_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8", "Z": "2"})
_ORDINAL = re.compile(r"(\d)(st|nd|rd|th)\b", re.I)
# numeric: 02/14/2026, 2-14-26, 14.02.2026, 2026-02-14 (OCR look-alikes allowed inside, fixed below)
_NUMERIC_DATE = re.compile(r"(?<![\w/])([0-9OoDlI|SBZ]{1,4})\s*([/\-.])\s*([0-9OoDlI|SBZ]{1,2})\s*\2\s*([0-9OoDlI|SBZ]{1,4})(?![\w/])")
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_TEXT_DATE = re.compile(rf"\b(?:(?P<d1>\d{{1,2}})(?:\s+of)?\s+(?P<m1>{_MONTH})\.?|(?P<m2>{_MONTH})\.?\s+(?P<d2>\d{{1,2}}))(?:,?\s+(?P<y>\d{{4}}))?\b", re.I)
_TEXT_MONTH_YEAR = re.compile(rf"\b{_MONTH}\.?,?\s+\d{{4}}\b", re.I)
EARLIEST_PLAUSIBLE = date(1980, 1, 1)


def _strptime(s: str, fmt: str) -> Optional[date]:
    try:
        return datetime.strptime(s, fmt).date()
    except ValueError:
        return None


def parse_date(raw: str, reference_year: Optional[int] = None, today: Optional[date] = None) -> dict[str, Any]:
    """Parse and validate a date with datetime.strptime (real calendar check: 99/09/2002, 02/30/2026,
    02/29/2025 all fail). US documents -> MM/DD/YYYY preferred; every assumption is reported in `notes`."""
    today = today or date.today()
    text = _ORDINAL.sub(r"\1", (raw or "").strip())
    notes: list[str] = []

    def result(value: Optional[date], errors: list[str]) -> dict[str, Any]:
        if value is not None and not errors:
            if value > today:
                errors.append(f"date {value.isoformat()} is in the future")
            elif value < EARLIEST_PLAUSIBLE:
                errors.append(f"date {value.isoformat()} is implausibly old for a vehicle loss")
        return {"input": raw, "valid": value is not None and not errors,
                "value": value.isoformat() if value and not errors else None, "errors": errors, "notes": notes}

    m = _NUMERIC_DATE.search(text)
    if m:
        a_raw, sep, b_raw, c_raw = m.groups()
        a, b, c = (p.translate(_OCR_DIGIT) for p in (a_raw, b_raw, c_raw))
        if (a, b, c) != (a_raw, b_raw, c_raw):
            notes.append("OCR look-alike characters corrected in date (e.g. O->0, l->1)")
        if not (a + b + c).isdigit():
            return result(None, [f"'{m.group(0)}' is not a date"])
        if len(a) == 4:  # ISO-style year first: 2026-02-14
            if len(c) > 2:
                return result(None, [f"'{m.group(0)}' is not a valid date format"])
            v = _strptime(f"{a}-{b}-{c}", "%Y-%m-%d")
            return result(v, [] if v else [f"'{m.group(0)}' is not a real calendar date"])
        if len(c) not in (2, 4) or len(a) > 2:
            return result(None, [f"'{m.group(0)}' has an invalid year/day/month length"])
        yfmt = "%Y" if len(c) == 4 else "%y"
        if yfmt == "%y":
            notes.append("two-digit year expanded")
        s = f"{a}/{b}/{c}"
        us = _strptime(s, f"%m/%d/{yfmt}")
        intl = _strptime(s, f"%d/%m/{yfmt}")
        if us and intl and us != intl:
            notes.append(f"day/month order ambiguous ({us.isoformat()} vs {intl.isoformat()}); assumed US MM/DD")
        if us:
            return result(us, [])
        if intl:
            notes.append("only valid as DD/MM/YYYY (first number > 12) - unusual on US documents, verify")
            return result(intl, [])
        return result(None, [f"'{m.group(0)}' is not a real calendar date"])

    m = _TEXT_DATE.search(text)
    if m:
        day = m.group("d1") or m.group("d2")
        month = (m.group("m1") or m.group("m2")).lower()[:3]
        year = m.group("y")
        if not year:
            if reference_year is None:
                return result(None, ["no year given and no claim context to infer it"])
            year = str(reference_year)
            notes.append(f"year not stated; assumed {year} from the claim's documents")
        v = _strptime(f"{day} {month} {year}", "%d %b %Y")
        return result(v, [] if v else [f"'{m.group(0)}' is not a real calendar date"])

    if _TEXT_MONTH_YEAR.search(text):
        return result(None, ["month and year only - the day is missing"])
    return result(None, ["no recognizable date"])


# ============================================================================= money
_PENDING = re.compile(r"\b(tbd|t\.b\.d|pending|n/?a|to be determined|unknown|not yet (?:known|determined)|none)\b|^[\s\-–—]*$", re.I)
_APPROX = re.compile(r"\b(about|around|approx\w*|roughly|maybe|est(?:imated)?\.?|nearly|almost|close to|or so|ish|"
                     r"a little (?:more|less|over|under)|ballpark)\b|~", re.I)
_SCALE = {"k": 1000, "thousand": 1000, "grand": 1000, "m": 1_000_000, "mm": 1_000_000, "mn": 1_000_000,
          "million": 1_000_000, "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000}
_FOREIGN = re.compile(r"€|£|¥|₹|\b(eur|gbp|inr|cad|aud|mxn)\b", re.I)
_NUMBER = re.compile(
    r"(?P<neg>\(|-|−)?\s*(?:US\$|USD|\$)?\s*(?P<neg2>-|−)?\s*"
    r"(?P<num>\.?[0-9OoIlSB](?:[0-9OoIlSB,.' ]*[0-9OoIlSB])?)"
    r"\s*(?P<scale>k|mm|mn|m|bn|b|thousand|grand|million|billion)?\b"
    r"(?:\s*(?:USD|dollars?|bucks))?\s*\)?",
    re.I,
)
_MONEYISH = r"(?:\$\s*[\d,.]+\s*[kKmM]?|\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d+)?\s*[kK]\b)"
# "$18,500-$19,000", "18,500 to 19,000", "$18.5k-19k": the left side must look like money (not "8841-5567")
_RANGE = re.compile(rf"({_MONEYISH})\s*(?:-|–|—|to)\s*({_MONEYISH}|\$?\s*[\d,.]+\s*[kK]?)")

_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                         "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_WORD_SCALES = {"thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}


def words_to_number(words: str) -> Optional[int]:
    """'twenty-two thousand three hundred forty' -> 22340. None if any token isn't a number word."""
    tokens = [t for t in re.split(r"[\s\-]+", words.lower().strip()) if t and t not in ("and", "a", "an")]
    if not tokens:
        return None
    total = current = 0
    seen = False
    for t in tokens:
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        elif t in _WORD_SCALES:
            total += (current or 1) * _WORD_SCALES[t]
            current = 0
        else:
            return None
        seen = True
    return total + current if seen else None


_NUM_WORDS = set(_UNITS) | set(_TENS) | {"hundred"} | set(_WORD_SCALES)


def _amount_from_words(text: str) -> Optional[Decimal]:
    """Find the first run of number words in a sentence; a following '... cents' run adds cents.
    'the balance is about twenty-two thousand dollars and fifty cents' -> 22000.50"""
    toks = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(toks):
        starts = toks[i] in _NUM_WORDS or (toks[i] == "a" and i + 1 < len(toks)
                                           and toks[i + 1] in ("hundred", "thousand", "million"))
        if not starts:
            i += 1
            continue
        j = i + 1
        while j < len(toks) and (toks[j] in _NUM_WORDS or (
                toks[j] == "and" and j + 1 < len(toks) and toks[j + 1] in _NUM_WORDS
                and not any(t.startswith("cent") for t in toks[j + 1:j + 4]))):
            j += 1
        runs.append((i, j))
        i = j
    if not runs:
        return None

    def followed_by_cents(end: int) -> bool:
        return end < len(toks) and toks[end].startswith("cent")

    s0, e0 = runs[0]
    first = words_to_number(" ".join(toks[s0:e0]))
    if first is None:
        return None
    if followed_by_cents(e0):  # "fifty cents"
        return Decimal(first) / 100 if first < 100 else None
    cents = 0
    if len(runs) > 1 and followed_by_cents(runs[1][1]):
        cents = words_to_number(" ".join(toks[runs[1][0]:runs[1][1]])) or 0
        if cents > 99:
            return None
    return Decimal(first) + Decimal(cents) / 100


def _parse_number_body(body: str, has_scale: bool) -> tuple[Optional[Decimal], list[str], Optional[str]]:
    """Interpret thousands/decimal separators. Returns (value, notes, error)."""
    notes: list[str] = []
    fixed = body.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"}))
    if fixed != body:
        notes.append("OCR look-alike characters corrected in amount")
    s = fixed.strip()
    if " " in s or "'" in s:  # 18 500 / 18'500 - only as 3-digit thousands groups
        if not re.fullmatch(r"\d{1,3}(?:[ '](?:\d{3}))+(?:[.,]\d{1,2})?", s):
            return None, notes, f"unrecognised number format '{body}'"
        s = re.sub(r"[ ']", "", s)
    has_c, has_d = "," in s, "." in s
    if has_c and has_d:
        dec = "," if s.rfind(",") > s.rfind(".") else "."
        thou = "." if dec == "," else ","
        int_part, frac = s.rsplit(dec, 1)
        if not re.fullmatch(rf"\d{{1,3}}(?:\{thou}\d{{3}})*", int_part) or not re.fullmatch(r"\d{1,2}", frac):
            return None, notes, f"inconsistent separators in '{body}'"
        if dec == ",":
            notes.append("European format (dot thousands, comma decimals)")
        s = int_part.replace(thou, "") + "." + frac
    elif has_c:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
            s = s.replace(",", "")
        elif re.fullmatch(r"\d{1,2}(?:,\d{2})+,\d{3}", s):
            notes.append("Indian digit grouping (lakh) interpreted")
            s = s.replace(",", "")
        elif re.fullmatch(r"\d+,\d{1,2}", s):
            return None, notes, f"ambiguous: '{body}' could be a decimal comma or a dropped digit"
        else:
            return None, notes, f"thousands separators in '{body}' are not in groups of 3"
    elif has_d:
        if s.count(".") > 1:
            if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
                notes.append("European format (dot thousands)")
                s = s.replace(".", "")
            else:
                return None, notes, f"unrecognised number format '{body}'"
        elif re.fullmatch(r"\d{1,3}\.\d{3}", s) and not has_scale:
            return None, notes, f"ambiguous: '{body}' could be {s} or {s.replace('.', '')}"
        elif not has_scale and not re.fullmatch(r"\d*\.\d{1,2}", s):
            return None, notes, f"more than 2 decimal places in '{body}'"
    try:
        return Decimal(s), notes, None
    except InvalidOperation:
        return None, notes, f"not numeric: '{body}'"


def parse_amount(raw: str) -> dict[str, Any]:
    """Accepts 18500 / 18,500 / 18500.00 / $18,500 / $18500.00 / USD 18,500 / 18,500 dollars /
    $18.5k / 22 grand / 1.2 million / (500.00) / -$500 / 18 500 / 18.500,00 / Indian 1,85,000 /
    'eighteen thousand five hundred dollars and fifty cents' / OCR '$l8,5OO'. Rejects ranges,
    ambiguous separators, >2 decimals, foreign currencies and 'TBD / pending / N/A'."""
    text = re.sub(r"\s+", " ", (raw or "").replace(" ", " ")).strip()
    notes: list[str] = []
    approximate = bool(_APPROX.search(text))

    def fail(msg: str, **extra) -> dict[str, Any]:
        return {"input": raw, "valid": False, "value": None, "errors": [msg], "pending": False,
                "approximate": approximate, "notes": notes, **extra}

    if _FOREIGN.search(text):
        return fail("not numeric in USD: foreign currency")
    r = _RANGE.search(text)
    if r:
        lo, hi = parse_amount(r.group(1)), parse_amount(r.group(2))
        if lo["valid"] and hi["valid"]:
            return fail(f"a range ({lo['value']} - {hi['value']}), not a single amount",
                        range=[lo["value"], hi["value"]])

    # 1) digits (+ optional scale word)
    # OCR look-alikes (O/l/S/B) count as digits only in tokens that are mostly digits, or that follow a
    # currency marker ("$l8,5OO") - this keeps the "o" in "around" from being read as a zero.
    candidates = [c for c in _NUMBER.finditer(text)
                  if any(ch.isdigit() for ch in c.group("num")) and (
                      sum(ch.isdigit() for ch in c.group("num")) > sum(ch.isalpha() for ch in c.group("num"))
                      or re.search(r"(\$|USD)\s*-?\s*$", text[: c.start("num")], re.I))]
    if candidates:
        c = candidates[0]
        if len(candidates) > 1:
            notes.append(f"{len(candidates)} numbers in text; used the first ('{c.group(0).strip()}')")
        value, n2, err = _parse_number_body(c.group("num"), bool(c.group("scale")))
        notes += n2
        if err:
            return fail(err)
        if c.group("scale"):
            value *= _SCALE[c.group("scale").lower()]
            approximate = True
            notes.append(f"shorthand '{c.group('scale')}' expanded - treated as approximate")
        if c.group("neg") or c.group("neg2"):
            value = -value
    else:
        # 2) number words: "eighteen thousand five hundred dollars and fifty cents"
        value = _amount_from_words(text)
        if value is None:
            pending = bool(_PENDING.search(text))
            reason = "value is pending / not yet determined" if pending else "no numeric amount found"
            out = fail(f"not numeric: {reason}")
            out["pending"] = pending
            return out
        notes.append("amount written in words")

    if approximate and not any("approximate" in n for n in notes):
        notes.append("hedged / approximate amount")
    value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {"input": raw, "valid": True, "value": f"{value:.2f}", "errors": [], "pending": False,
            "approximate": approximate, "notes": notes}


def amounts_match(a: str, b: str, approximate: bool = False) -> bool:
    da, db = Decimal(a), Decimal(b)
    if approximate:  # "around $35,000, maybe a little more"
        return abs(da - db) <= max(Decimal("0.05") * max(abs(da), abs(db)), Decimal("500"))
    return da == db


def normalize_field(field: str, raw: str, reference_year: Optional[int] = None) -> dict[str, Any]:
    if field == "vin":
        r = validate_vin(raw)
        return {"value": r["normalized"], "valid": r["valid"],
                "note": "; ".join(r["errors"] + [f"repaired: {x}" for x in r["repairs"]] + r["warnings"]) or None,
                "repaired": r["repaired"], "detail": r}
    if field == "date_of_loss":
        r = parse_date(raw, reference_year=reference_year)
        return {"value": r["value"], "valid": r["valid"], "note": "; ".join(r["errors"] + r["notes"]) or None,
                "repaired": any("OCR" in n for n in r["notes"]), "detail": r}
    r = parse_amount(raw)
    errors = list(r["errors"])
    if r["valid"] and Decimal(r["value"]) < 0:  # payout / loan balance can't be negative
        errors.append("negative amount is not valid for this field")
    return {"value": r["value"] if not errors else None, "valid": not errors,
            "note": "; ".join(errors + r["notes"]) or None,
            "repaired": any("OCR" in n for n in r["notes"]), "approximate": r["approximate"], "detail": r}


def today_iso() -> str:
    return datetime.now().date().isoformat()
