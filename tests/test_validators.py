from datetime import date

import pytest

from claims_agent.validators import (VIN_PATTERN, amounts_match, normalize_field, parse_amount, parse_date,
                                     validate_vin, vin_check_digit, vin_ocr_variant)

# ------------------------------------------------------------------------------------------- VIN
@pytest.mark.parametrize("raw,expected,valid,repaired", [
    ("1HGCM82633A004352", "1HGCM82633A004352", True, False),
    ("1hgcm82633a004352", "1HGCM82633A004352", True, False),     # case-insensitive
    ("1HGCM-82633-A004352", "1HGCM82633A004352", True, False),   # separators stripped
    ("1HGCM82633AO04352", "1HGCM82633A004352", True, True),      # O can't be in a VIN -> 0
    ("1HGCM82633A0Q4352", "1HGCM82633A004352", True, True),      # Q -> 0
    ("IHGCM82633A004352", "1HGCM82633A004352", True, True),      # I -> 1
    ("1HGCM8263lA004352", "1HGCM82631A004352", True, True),      # lowercase l -> 1
    ("5YJ3E1EA7K", "5YJ3E1EA7K", False, False),                  # truncated (CLM-004)
    ("1HGCM82633A0043521", "1HGCM82633A0043521", False, False),  # 18 chars
    ("1HGCM82633A00435$", "1HGCM82633A00435$", False, False),    # symbol
    ("", "", False, False),
])
def test_vin(raw, expected, valid, repaired):
    r = validate_vin(raw)
    assert (r["normalized"], r["valid"], r["repaired"]) == (expected, valid, repaired), r


def test_vin_pattern_is_the_assignment_rule():
    assert VIN_PATTERN.match("1hgcm82633a004352") and not VIN_PATTERN.match("1HGCM82633A00435")


def test_vin_check_digit_is_informational_only():
    assert vin_check_digit("1HGCM82633A004352") == "3"
    r = validate_vin("2T1BURHE5JC034127")  # synthetic sample VIN: fails the checksum but is valid per the rule
    assert r["valid"] and r["warnings"]


def test_vin_ambiguous_lookalikes_are_never_auto_corrected_but_are_recognised():
    # S and 5 are both legal VIN characters -> no auto-repair...
    assert validate_vin("1HGCM82633A00435S")["normalized"] == "1HGCM82633A00435S"
    # ...but two readings that differ only by look-alikes are flagged as the same VIN
    assert vin_ocr_variant("1HGCM82633A004352", "1HGCM82633A00435Z") == [17]
    assert vin_ocr_variant("2T1BURHE5JC034127", "2T1BURHE5JC034182") is None  # CLM-003: a real conflict


# ------------------------------------------------------------------------------------------- dates
TODAY = date(2026, 10, 7)


@pytest.mark.parametrize("raw,expected", [
    ("02/14/2026", "2026-02-14"),
    ("2/5/2026", "2026-02-05"),
    ("02-14-26", "2026-02-14"),
    ("2026-02-14", "2026-02-14"),
    ("02/29/2024", "2024-02-29"),          # leap year
    ("13/02/2026", "2026-02-13"),          # only valid as DD/MM (noted)
    ("14.02.2026", "2026-02-14"),
    ("o1/28/2026", "2026-01-28"),          # OCR o -> 0
    ("02/14/2026 14:32", "2026-02-14"),
    ("March 22nd, 2026", "2026-03-22"),
    ("22 March 2026", "2026-03-22"),
    ("the 22nd of March 2026", "2026-03-22"),
    ("Feb. 5 2026", "2026-02-05"),
    ("March 22nd", "2026-03-22"),          # year from claim context (noted)
])
def test_valid_dates(raw, expected):
    r = parse_date(raw, reference_year=2026, today=TODAY)
    assert r["valid"] and r["value"] == expected, r


@pytest.mark.parametrize("raw", [
    "99/09/2002", "13/13/2026", "00/10/2026", "02/30/2026", "02/29/2025", "2026-13-01", "Feb 30, 2026",
    "12/19/2099",   # future
    "01/01/1975",   # implausibly old
    "March 2026",   # no day
    "202/14/2026", "yesterday", "", "TBD",
])
def test_invalid_dates(raw):
    r = parse_date(raw, reference_year=2026, today=TODAY)
    assert not r["valid"] and r["value"] is None and r["errors"], r


def test_ambiguous_date_is_flagged():
    assert any("ambiguous" in n for n in parse_date("03/04/2026", today=TODAY)["notes"])


# ------------------------------------------------------------------------------------------- money
@pytest.mark.parametrize("raw,expected,approx", [
    ("18500", "18500.00", False),
    ("18,500", "18500.00", False),
    ("18500.00", "18500.00", False),
    ("$18,500", "18500.00", False),
    ("$18500.00", "18500.00", False),
    ("$ 18,500.00", "18500.00", False),
    ("$18,500.5", "18500.50", False),
    ("USD 18,500", "18500.00", False),
    ("US$18,500", "18500.00", False),
    ("18,500.00 USD", "18500.00", False),
    ("18500 dollars", "18500.00", False),
    ("1,000,000", "1000000.00", False),
    ("18 500", "18500.00", False),            # space thousands
    ("18'500", "18500.00", False),            # Swiss
    ("18.500,00", "18500.00", False),         # European
    ("1,85,000", "185000.00", False),         # Indian grouping
    ("$.50", "0.50", False),
    ("$0.00", "0.00", False),
    ("($500.00)", "-500.00", False),          # accounting negative
    ("-$500", "-500.00", False),
    ("$-500", "-500.00", False),
    ("$l8,5OO", "18500.00", False),           # OCR look-alikes after a $
    ("$18.5k", "18500.00", True),
    ("22 grand", "22000.00", True),
    ("1.2 million", "1200000.00", True),
    ("about $22k still", "22000.00", True),
    ("around $35,000, maybe a little more", "35000.00", True),
    ("eighteen thousand five hundred dollars", "18500.00", False),
    ("eighteen thousand five hundred dollars and fifty cents", "18500.50", False),
    ("twenty-two thousand three hundred forty dollars and fifty-five cents", "22340.55", False),
    ("three hundred and forty dollars", "340.00", False),
    ("a hundred dollars", "100.00", False),
    ("the balance is about twenty thousand dollars", "20000.00", True),
    ("fifty cents", "0.50", False),
])
def test_valid_amounts(raw, expected, approx):
    r = parse_amount(raw)
    assert r["valid"] and r["value"] == expected and r["approximate"] == approx, r


@pytest.mark.parametrize("raw,why", [
    ("TBD - pending", "pending"),
    ("N/A", "pending"),
    ("$18,500-$19,000", "range"),
    ("$18,500–19,000", "range"),            # en-dash
    ("18,500 to 19,000", "range"),
    ("18,50", "ambiguous"),                     # decimal comma or dropped digit?
    ("18.500", "ambiguous"),                    # 18.5 or 18500?
    ("18500.555", "decimal places"),
    ("18,50,0", "groups of 3"),
    ("€18,500", "foreign"),
    ("abc", "no numeric"),
    ("", "pending"),
])
def test_invalid_amounts(raw, why):
    r = parse_amount(raw)
    assert not r["valid"] and r["value"] is None and why in r["errors"][0], r


def test_account_number_is_not_a_range():
    assert "range" not in (parse_amount("8841-5567-2203")["errors"] or [""])[0]


def test_negative_payout_or_balance_rejected():
    assert not normalize_field("insurance_payout", "($500.00)")["valid"]


def test_amounts_match_with_tolerance_for_approximations():
    assert amounts_match("35120.75", "35000.00", approximate=True)
    assert not amounts_match("35120.75", "35000.00")
