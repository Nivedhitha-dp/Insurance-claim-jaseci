export type Status = "complete" | "incomplete" | "needs_review";
export type Confidence = "high" | "medium" | "low";

export interface ClaimSummary {
  claim_id: string;
  files: string[];
  processed: boolean;
  status?: Status;
  turns?: number;
  pending_replies?: string[];
  blocking?: number;
  warnings?: number;
  claimant?: string | null;
  payout?: string | null;
}

export interface FieldOut {
  value: string | null;
  confidence: Confidence;
  source: string | null;
  reason: string | null;
  corroborated_by: string[];
}

export interface Candidate {
  value: string | null;
  valid: boolean;
  source: string;
  method: string;
  line_confidence: number | null;
  note: string | null;
  approximate: boolean;
}

export interface IssueOut {
  type: string;
  severity: "blocking" | "warning" | "info";
  field: string | null;
  description: string;
  details: string | null;
  suggested_value?: string;
  resolvable_by: string;
}

export interface NextAction {
  type: "finalize" | "message_customer" | "escalate";
  message: string | null;
  internal_note: string | null;
  follow_ups: string[];
}

export interface TurnOut {
  turn: number;
  kind: string;
  inbound?: string;
  uploaded?: string[];
  status_after_turn: Status | null;
  action: string | null;
  message_to_customer: string | null;
  agent_summary: string | null;
}

export interface DocDetail {
  file: string;
  kind: "pdf" | "image" | "text";
  doc_type: string;
  status: "active" | "duplicate" | "superseded";
  related_to: string | null;
  status_reason: string | null;
  method: string | null;
  ocr_confidence: number | null;
  added_in_turn: number;
  type_reason: string | null;
  meta: Record<string, unknown>;
}

export interface ToolLog {
  turn: number;
  step: number;
  tool: string;
  input: Record<string, unknown>;
  result: unknown;
  ok: boolean;
}

export interface CrossCheck {
  status: "consistent" | "single_source" | "ocr_variant" | "contradiction" | "customer_mismatch";
  value: string | null;
  sources?: string[];
  clusters?: { value: string; sources: string[] }[];
}

export interface ClaimDetail {
  claim_id: string;
  processed: boolean;
  status: Status;
  status_reason: string;
  claimant: string | null;
  extracted_fields: Record<string, FieldOut>;
  evidence: Record<string, Candidate[]>;
  documents: { missing: string[] };
  documents_detail: DocDetail[];
  issues: IssueOut[];
  next_action: NextAction | null;
  conversation: TurnOut[];
  tool_log: ToolLog[];
  cross_document_checks?: Record<string, CrossCheck>;
  pending_replies: string[];
  files?: string[];
}

export interface Priority {
  processing_order: {
    rank: number; claim_id: string; status: Status; tier: number; blocking_issues: number; reason: string;
    decision_confidence?: { score: number; label: "high" | "medium" | "low"; summary: string };
  }[];
  do_in_parallel_today: { claim_id: string; action: string }[];
  method?: string;
}

export interface AgentEvent {
  type: "turn_start" | "turn_end" | "tool" | "thought" | "error" | "done";
  [k: string]: unknown;
}

const j = async <T,>(r: Response): Promise<T> => {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};

export const api = {
  config: () => fetch("/api/config").then(j<Record<string, unknown>>),
  claims: () => fetch("/api/claims").then(j<ClaimSummary[]>),
  claim: (id: string) => fetch(`/api/claims/${id}`).then(j<ClaimDetail>),
  priority: () => fetch("/api/prioritization").then(j<Priority>),
  reset: (id: string) => fetch(`/api/claims/${id}/reset`, { method: "POST" }),
  preview: (id: string, file: string) => `/api/claims/${id}/preview/${encodeURIComponent(file)}`,
  file: (id: string, file: string) => `/api/claims/${id}/files/${encodeURIComponent(file)}`,
};

/** POST + Server-Sent-Events reader (EventSource can't POST). */
export async function stream(url: string, body: FormData | null, onEvent: (e: AgentEvent) => void) {
  const r = await fetch(url, { method: "POST", body });
  if (!r.ok || !r.body) {
    onEvent({ type: "error", message: `${r.status} ${await r.text()}` });
    onEvent({ type: "done" });
    return;
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, i);
      buf = buf.slice(i + 2);
      if (chunk.startsWith("data: ")) onEvent(JSON.parse(chunk.slice(6)));
    }
  }
}
