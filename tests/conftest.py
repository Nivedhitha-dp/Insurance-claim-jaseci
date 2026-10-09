"""Tests run fully offline (deterministic planner) against a temp copy of the claims,
plus a synthetic claim containing a byte-identical duplicate document."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="claims_test_"))
shutil.copytree(ROOT / "claims", TMP / "claims")
dup = TMP / "claims" / "CLM-DUP"
shutil.copytree(ROOT / "claims" / "CLM-001", dup)
shutil.copy(dup / "police_report.pdf", dup / "police_report_resubmitted.pdf")

os.environ.update({
    "LLM_PROVIDER": "none",
    "CLAIMS_DIR": str(TMP / "claims"),
    "STATE_DIR": str(TMP / "state"),
    "OUTPUT_DIR": str(TMP / "out"),
    "CACHE_DIR": str(ROOT / ".cache"),  # reuse OCR cache - keeps the suite fast
})
sys.path.insert(0, str(ROOT))
