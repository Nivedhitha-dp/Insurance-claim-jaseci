from claims_agent.models import ClaimState, DocType, Evidence
from claims_agent.reconcile import reconcile_field


def _ev(file, doc_type, value, method="ocr", conf=0.95):
    return Evidence(field="vin", raw=value, value=value, valid=True, source_file=file, doc_type=doc_type,
                    method=method, line_confidence=conf)


def test_ocr_lookalike_vins_need_human_confirmation():
    """…4352 vs …435Z: probably the same VIN, but 2 and Z are both legal - never silently 'fixed'."""
    from claims_agent.models import Document
    from claims_agent.rules import evaluate

    st = ClaimState(claim_id="T", folder=".")
    st.evidence = [_ev("police.pdf", DocType.police_report, "1HGCM82633A004352", "pdf_text", 1.0),
                   _ev("finance.png", DocType.finance_agreement, "1HGCM82633A00435Z")]  # 2 read as Z
    res, analysis = reconcile_field(st, "vin")
    assert res.value == "1HGCM82633A004352"           # best guess = the stronger (PDF) reading
    assert res.confidence == "low" and "look alike" in res.reason
    for f, t in [("police.pdf", DocType.police_report), ("finance.png", DocType.finance_agreement)]:
        st.documents[f] = Document(file=f, kind="pdf", sha256=f, size=1, doc_type=t)
    ev = evaluate(st)
    issue = next(i for i in ev.issues if i.code == "ocr_lookalike_mismatch")
    assert issue.severity == "blocking" and issue.suggested_value == "1HGCM82633A004352"
    assert ev.status == "needs_review"


def test_real_vin_difference_is_still_a_conflict():
    st = ClaimState(claim_id="T", folder=".")
    st.evidence = [_ev("police.png", DocType.police_report, "2T1BURHE5JC034127"),
                   _ev("finance.pdf", DocType.finance_agreement, "2T1BURHE5JC034182", "pdf_text", 1.0)]
    _, analysis = reconcile_field(st, "vin")
    assert analysis["conflict"]
