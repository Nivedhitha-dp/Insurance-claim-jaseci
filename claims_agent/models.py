"""Domain models. The ClaimState is the single source of truth the agent's tools read & write."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    police_report = "police_report"
    finance_agreement = "finance_agreement"
    settlement_breakdown = "settlement_breakdown"
    adjuster_note = "adjuster_note"
    tow_receipt = "tow_receipt"
    customer_correspondence = "customer_correspondence"
    unknown = "unknown"


FIELDS = ("vin", "date_of_loss", "insurance_payout", "loan_balance")
FieldName = Literal["vin", "date_of_loss", "insurance_payout", "loan_balance"]
Confidence = Literal["high", "medium", "low"]


class TextLine(BaseModel):
    """A line of text with its position on the page (PDF text layer or OCR box)."""

    text: str
    conf: float = 1.0
    x0: float = 0
    y0: float = 0
    x1: float = 0
    y1: float = 0
    page: int = 0


class Document(BaseModel):
    file: str
    kind: Literal["pdf", "image", "text"]
    sha256: str
    size: int
    added_in_turn: int = 0  # 0 = initial submission, n = uploaded during conversation n
    doc_type: DocType = DocType.unknown
    type_confidence: float = 0.0
    type_reason: Optional[str] = None
    text: Optional[str] = None  # persisted so sessions resume without re-OCR
    lines: list[TextLine] = Field(default_factory=list)
    text_method: Optional[str] = None  # pdf_text | ocr | ocr_enhanced | vision | plain_text
    ocr_confidence: Optional[float] = None
    status: Literal["active", "duplicate", "superseded"] = "active"
    related_to: Optional[str] = None  # duplicate_of / superseded_by
    status_reason: Optional[str] = None
    fields_extracted: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """One observed value for one field from one source. Reconciliation works on these."""

    field: FieldName
    raw: str
    value: Optional[str]  # normalized (VIN upper / ISO date / decimal string); None if unparseable
    valid: bool
    validation_note: Optional[str] = None
    source_file: str
    doc_type: DocType
    method: str  # pdf_text | ocr | ocr_enhanced | vision | llm_observation | customer_statement
    line_confidence: Optional[float] = None
    quote: Optional[str] = None
    turn: int = 0
    approximate: bool = False  # e.g. customer says "around $35,000"


class Concern(BaseModel):
    """A non-field observation worth surfacing (adjuster note, odd math, customer question...)."""

    code: str
    description: str
    source_file: Optional[str] = None
    quote: Optional[str] = None
    severity: Literal["warning", "info"] = "warning"
    turn: int = 0


class CustomerCommitment(BaseModel):
    doc_type: str
    eta: Optional[str] = None
    quote: Optional[str] = None
    turn: int


class CustomerQuestion(BaseModel):
    question: str
    turn: int


class FieldResult(BaseModel):
    value: Optional[str]
    confidence: Confidence
    source: Optional[str]
    sources: list[str] = Field(default_factory=list)
    reason: Optional[str] = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class Issue(BaseModel):
    type: Literal["inconsistency", "missing", "invalid", "low_confidence", "warning"]
    severity: Literal["blocking", "warning", "info"]
    code: str
    field: Optional[str] = None
    description: str
    details: Optional[str] = None
    suggested_value: Optional[str] = None
    resolvable_by: Literal["customer", "internal", "lender", "customer_or_internal"] = "internal"


class Evaluation(BaseModel):
    status: Literal["complete", "incomplete", "needs_review"]
    status_reason: str
    extracted_fields: dict[str, FieldResult]
    missing_documents: list[str]
    issues: list[Issue]
    recommended_action: Literal["finalize", "message_customer", "escalate"]
    cross_checks: dict[str, Any] = Field(default_factory=dict)  # shared non-required facts across documents


class ToolCall(BaseModel):
    turn: int
    step: int
    tool: str
    input: dict[str, Any]
    result: Any
    ok: bool = True


class NextAction(BaseModel):
    type: Literal["finalize", "message_customer", "escalate"]
    message: Optional[str] = None  # to the customer
    internal_note: Optional[str] = None  # to the claims team
    follow_ups: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    index: int
    kind: Literal["initial_review", "customer_message", "document_upload"]
    at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    inbound: Optional[str] = None
    inbound_files: list[str] = Field(default_factory=list)
    status: Optional[str] = None
    next_action: Optional[NextAction] = None
    agent_summary: Optional[str] = None
    driven_by: Optional[str] = None  # "llm:gemini/gemini-3.5-flash" or "deterministic-planner (backup: ...)"


class ClaimState(BaseModel):
    claim_id: str
    folder: str
    documents: dict[str, Document] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    concerns: list[Concern] = Field(default_factory=list)
    commitments: list[CustomerCommitment] = Field(default_factory=list)
    questions: list[CustomerQuestion] = Field(default_factory=list)
    tool_log: list[ToolCall] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)
    evaluation: Optional[Evaluation] = None
    pending_replies: list[str] = Field(default_factory=list)  # reply .txt files held back for multi-turn

    @property
    def current_turn(self) -> int:
        return len(self.turns) - 1 if self.turns else 0

    def active_docs(self) -> list[Document]:
        return [d for d in self.documents.values() if d.status == "active"]
