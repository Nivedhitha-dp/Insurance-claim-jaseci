"""Cross-document consistency for shared non-required facts (vehicle, lender, names, policy/report numbers)."""
from claims_agent.consistency import check, relation, normalize
from claims_agent.models import ClaimState, DocType, Document, Evidence
from claims_agent.reconcile import reconcile_field
from claims_agent.rules import evaluate


def doc(file, doc_type, text, status="active", method="pdf_text"):
    return Document(file=file, kind="text", sha256=file, size=1, doc_type=doc_type, text=text,
                    text_method=method, status=status, fields_extracted=True)


def claim(*docs):
    st = ClaimState(claim_id="T", folder=".")
    for d in docs:
        st.documents[d.file] = d
    return st


def issue(issues, code):
    return next((i for i in issues if i.code == code), None)


def test_vehicle_year_contradiction_with_equal_authority_says_no_guess():
    st = claim(doc("police.pdf", DocType.police_report, "Vehicle: 2020 Honda Accord EX-L"),
               doc("finance.pdf", DocType.finance_agreement, "Vehicle: 2021 Honda Accord EX-L"))
    issues, report = check(st)
    i = issue(issues, "vehicle_mismatch")
    assert i and i.severity == "blocking" and i.suggested_value == "No guess"
    assert "police.pdf" in i.details and "finance.pdf" in i.details
    assert report["vehicle"]["status"] == "contradiction"


def test_contradiction_with_majority_gives_educated_guess_and_blocks_status():
    st = claim(doc("police.pdf", DocType.police_report, "Vehicle: 2020 Honda Accord EX-L"),
               doc("settle.pdf", DocType.settlement_breakdown, "Vehicle: 2020 Honda Accord EX-L"),
               doc("finance.pdf", DocType.finance_agreement, "Vehicle: 2021 Honda Accord EX-L"))
    issues, _ = check(st)
    assert issue(issues, "vehicle_mismatch").suggested_value == "2020 Honda Accord EX-L"
    assert evaluate(st).status == "needs_review"


def test_same_fact_spelled_differently_is_not_a_conflict():
    assert relation("lender", normalize("lender", "Capital One"), normalize("lender", "Capital One Auto Finance")) in ("same", "compatible")
    assert relation("lender", normalize("lender", "CapitalOneAutoFinance"), normalize("lender", "Capital One Auto Finance")) == "same"
    assert relation("vehicle", normalize("vehicle", "2019 Ford Focus"), normalize("vehicle", "2019 Ford Focus SE")) == "compatible"
    assert relation("police_report_number", normalize("police_report_number", "YPD-202502871"),
                    normalize("police_report_number", "YPD-2025-02871")) == "same"
    assert relation("lender", normalize("lender", "Chase Auto"), normalize("lender", "Ally Financial")) == "contradiction"


def test_handwriting_ocr_variant_is_reported_but_not_blocking():
    st = claim(doc("police.pdf", DocType.police_report, "Vehicle: 2020 Honda Accord EX-L"),
               doc("settle.pdf", DocType.settlement_breakdown, "Vehicle: 2020 Honda Accord EX-L"),
               doc("note.png", DocType.adjuster_note, "Vehicle: 2020 Honda Accord EX-C", method="ocr"))
    issues, report = check(st)
    i = issue(issues, "vehicle_ocr_variant")
    assert i and i.severity == "info" and report["vehicle"]["status"] == "ocr_variant"


def test_duplicates_and_superseded_versions_do_not_vote():
    st = claim(doc("police.pdf", DocType.police_report, "Vehicle: 2020 Honda Accord EX-L"),
               doc("old_settle.pdf", DocType.settlement_breakdown, "Vehicle: 2021 Honda Accord", status="superseded"),
               doc("copy.pdf", DocType.police_report, "Vehicle: 2019 Honda Accord", status="duplicate"))
    issues, _ = check(st)
    assert not issues


def test_customer_only_disagreement_is_a_warning():
    st = claim(doc("settle.pdf", DocType.settlement_breakdown, "Lender: Chase Auto"),
               doc("finance.pdf", DocType.finance_agreement, "Lender: Chase Auto"),
               doc("reply.txt", DocType.customer_correspondence, "Lender: Wells Fargo Auto"))
    issues, _ = check(st)
    i = issue(issues, "lender_customer_mismatch")
    assert i and i.severity == "warning"


def test_main_field_conflict_without_an_edge_says_no_guess():
    st = ClaimState(claim_id="T", folder=".")
    st.evidence = [Evidence(field="vin", raw=v, value=v, valid=True, source_file=f, doc_type=t, method="pdf_text")
                   for f, t, v in [("police.pdf", DocType.police_report, "1HGCM82633A004352"),
                                   ("finance.pdf", DocType.finance_agreement, "2T1BURHE5JC034127")]]
    res, analysis = reconcile_field(st, "vin")
    assert analysis["suggested_value"] == "No guess" and res.value is None and "No guess" in res.reason


def test_sample_claims_have_no_false_contradictions():
    from tests.test_end_to_end import CLAIMS, run

    for cid in CLAIMS:
        ev = run(cid).evaluation
        assert not [i for i in ev.issues if i.code.endswith("_mismatch") and i.severity == "blocking"], cid
        assert ev.cross_checks["vehicle"]["status"] in ("consistent", "ocr_variant")
