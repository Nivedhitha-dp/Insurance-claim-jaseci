import { useState } from "react";
import {
  ArrowRight, Ban, Check, ChevronDown, ChevronRight, Copy, FileImage, FileText, FileWarning, Gavel, Layers,
  ListChecks, ScanLine, Send, ShieldCheck, Sparkles, X,
} from "lucide-react";
import { api, type ClaimDetail, type DocDetail, type IssueOut } from "./api";
import { Card, ConfidencePill, DOC_LABEL, FIELD_LABEL, StatusBadge, cx, fmtValue } from "./ui";

// ------------------------------------------------------------------ status hero + next action
const ACTION: Record<string, { label: string; cls: string; icon: JSX.Element }> = {
  finalize: { label: "Finalize", cls: "bg-emerald-600", icon: <ShieldCheck className="h-4 w-4" /> },
  message_customer: { label: "Message customer", cls: "bg-indigo-600", icon: <Send className="h-4 w-4" /> },
  escalate: { label: "Escalate to reviewer", cls: "bg-rose-600", icon: <Gavel className="h-4 w-4" /> },
};

export function StatusHero({ d }: { d: ClaimDetail }) {
  const na = d.next_action;
  const a = na ? ACTION[na.type] : null;
  const turns = d.conversation.length;
  return (
    <Card className="overflow-hidden">
      <div className="flex flex-wrap items-start justify-between gap-4 p-4">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-3">
            <StatusBadge status={d.status} size="lg" />
            <span className="text-xs text-slate-500">after {turns} turn{turns === 1 ? "" : "s"}</span>
          </div>
          <p className="mt-2 text-sm text-slate-600">{d.status_reason}</p>
        </div>
        {a && (
          <div className={cx("flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-semibold text-white shadow-sm", a.cls)}>
            {a.icon}
            {a.label}
          </div>
        )}
      </div>
      {na?.internal_note && (
        <div className="border-t border-slate-100 bg-slate-50/70 px-4 py-3">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Note for the claims team</div>
          <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-slate-700">{na.internal_note}</p>
          {na.follow_ups.length > 0 && (
            <ul className="mt-2 space-y-1">
              {na.follow_ups.map((f, i) => (
                <li key={i} className="flex items-start gap-2 text-[13px] text-slate-700">
                  <ArrowRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-indigo-500" />
                  {f}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </Card>
  );
}

// ------------------------------------------------------------------ fields
const METHOD: Record<string, string> = {
  pdf_text: "PDF text",
  ocr: "OCR",
  ocr_enhanced: "Enhanced OCR",
  vision: "Vision model",
  llm_observation: "Read by agent",
  customer_statement: "Customer said",
};

export function FieldsPanel({ d }: { d: ClaimDetail }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <Card title="Extracted fields" icon={<ListChecks className="h-4 w-4 text-indigo-500" />}>
      <div className="divide-y divide-slate-100">
        {Object.entries(d.extracted_fields).map(([k, f]) => {
          const ev = d.evidence[k] || [];
          const isOpen = open === k;
          return (
            <div key={k}>
              <button onClick={() => setOpen(isOpen ? null : k)} className="grid w-full grid-cols-[140px_1fr_auto] items-center gap-3 px-4 py-3 text-left hover:bg-slate-50/70">
                <div className="flex items-center gap-1.5 text-[13px] font-medium text-slate-500">
                  {isOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                  {FIELD_LABEL[k]}
                </div>
                <div className="min-w-0">
                  <div className={cx("truncate text-[15px] font-semibold", f.value ? "text-slate-900" : "text-slate-400", k === "vin" && "font-mono text-[14px]")}>
                    {fmtValue(k, f.value)}
                  </div>
                  <div className="mt-0.5 truncate text-[11px] text-slate-400">
                    {f.source ? `from ${f.source}` : "no source"}
                    {f.corroborated_by.length > 0 && ` · corroborated by ${f.corroborated_by.length}`}
                  </div>
                </div>
                <ConfidencePill c={f.confidence} />
              </button>
              {f.reason && !isOpen && <p className="-mt-1 truncate px-4 pb-2.5 pl-[164px] text-[12px] text-slate-500">{f.reason}</p>}
              {isOpen && (
                <div className="slide-in space-y-2 bg-slate-50/60 px-4 pb-3 pt-1">
                  {f.reason && <p className="text-[12.5px] leading-relaxed text-slate-600">{f.reason}</p>}
                  <table className="w-full text-[12px]">
                    <thead>
                      <tr className="text-left text-[10.5px] uppercase tracking-wider text-slate-400">
                        <th className="py-1 font-semibold">Source</th>
                        <th className="font-semibold">Value</th>
                        <th className="font-semibold">Read via</th>
                        <th className="font-semibold">Valid</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ev.map((c, i) => (
                        <tr key={i} className="border-t border-slate-200/70 align-top">
                          <td className="py-1.5 pr-2 text-slate-700">{c.source}</td>
                          <td className="pr-2 font-mono text-slate-900">
                            {c.approximate && "~"}
                            {fmtValue(k, c.value)}
                          </td>
                          <td className="pr-2 text-slate-500">
                            {METHOD[c.method] || c.method}
                            {c.line_confidence != null && c.method.startsWith("ocr") && ` · ${(c.line_confidence * 100).toFixed(0)}%`}
                          </td>
                          <td>
                            {c.valid ? <Check className="h-4 w-4 text-emerald-600" /> : (
                              <span className="flex items-center gap-1 text-rose-600" title={c.note || ""}>
                                <X className="h-4 w-4" />
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ------------------------------------------------------------------ issues
const SEV: Record<string, string> = {
  blocking: "border-l-rose-500 bg-rose-50/40",
  warning: "border-l-amber-400 bg-amber-50/40",
  info: "border-l-slate-300 bg-slate-50",
};

export function IssuesPanel({ issues }: { issues: IssueOut[] }) {
  const order = { blocking: 0, warning: 1, info: 2 } as const;
  const sorted = [...issues].sort((a, b) => order[a.severity] - order[b.severity]);
  const nb = issues.filter((i) => i.severity === "blocking").length;
  return (
    <Card
      title="Issues"
      icon={<FileWarning className="h-4 w-4 text-rose-500" />}
      right={<span className="text-xs text-slate-400">{nb} blocking · {issues.length - nb} other</span>}
    >
      {sorted.length === 0 ? (
        <div className="flex items-center gap-2 px-4 py-5 text-sm text-emerald-700">
          <ShieldCheck className="h-4 w-4" /> No issues - everything checks out.
        </div>
      ) : (
        <ul className="space-y-2 p-3">
          {sorted.map((i, n) => (
            <li key={n} className={cx("rounded-lg border-l-[3px] px-3 py-2", SEV[i.severity])}>
              <div className="flex flex-wrap items-center gap-2">
                <span className={cx("text-[10px] font-bold uppercase tracking-wider", i.severity === "blocking" ? "text-rose-600" : i.severity === "warning" ? "text-amber-700" : "text-slate-500")}>
                  {i.severity === "blocking" ? i.type.replace("_", " ") : i.severity}
                </span>
                <span className="text-[13px] font-medium text-slate-800">{i.description}</span>
              </div>
              {i.details && <p className="mt-1 break-words text-[12px] leading-relaxed text-slate-500">{i.details}</p>}
              <div className="mt-1.5 flex flex-wrap gap-2">
                {i.suggested_value && (
                  <span className="inline-flex items-center gap-1 rounded-md bg-indigo-50 px-1.5 py-0.5 text-[11px] font-medium text-indigo-700">
                    <Sparkles className="h-3 w-3" /> best guess: <span className="font-mono">{fmtValue(i.field || "", i.suggested_value)}</span>
                  </span>
                )}
                {i.severity === "blocking" && (
                  <span className="rounded-md bg-white px-1.5 py-0.5 text-[11px] text-slate-500 ring-1 ring-slate-200">
                    fix: {i.resolvable_by.replaceAll("_", " ")}
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

// ------------------------------------------------------------------ documents
export function DocumentsPanel({ d, requiredDocs }: { d: ClaimDetail; requiredDocs: string[] }) {
  const [view, setView] = useState<DocDetail | null>(null);
  const docs = d.documents_detail.filter((x) => x.doc_type !== "customer_correspondence");
  return (
    <Card title="Documents" icon={<Layers className="h-4 w-4 text-indigo-500" />} right={<span className="text-xs text-slate-400">{requiredDocs.length - d.documents.missing.length}/{requiredDocs.length} required</span>}>
      <div className="grid grid-cols-2 gap-3 p-3 xl:grid-cols-3">
        {docs.map((doc) => (
          <button key={doc.file} onClick={() => setView(doc)} className={cx("group overflow-hidden rounded-lg border text-left transition hover:shadow-md", doc.status !== "active" ? "border-slate-200 opacity-60" : "border-slate-200")}>
            <div className="relative h-28 overflow-hidden bg-slate-100">
              <img src={api.preview(d.claim_id, doc.file)} className="h-full w-full object-cover object-top transition group-hover:scale-[1.03]" loading="lazy" />
              {doc.status !== "active" && (
                <span className="absolute left-2 top-2 inline-flex items-center gap-1 rounded bg-slate-900/80 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-white">
                  {doc.status === "duplicate" ? <Copy className="h-3 w-3" /> : <Ban className="h-3 w-3" />}
                  {doc.status}
                </span>
              )}
              {doc.added_in_turn > 0 && <span className="absolute right-2 top-2 rounded bg-indigo-600 px-1.5 py-0.5 text-[10px] font-semibold text-white">turn {doc.added_in_turn}</span>}
            </div>
            <div className="p-2">
              <div className="truncate text-[12.5px] font-semibold text-slate-800">{DOC_LABEL[doc.doc_type] || doc.doc_type}</div>
              <div className="truncate text-[11px] text-slate-400">{doc.file}</div>
              <div className="mt-1 flex items-center gap-1 text-[10.5px] text-slate-500">
                {doc.kind === "pdf" && doc.method === "pdf_text" ? <FileText className="h-3 w-3" /> : <ScanLine className="h-3 w-3" />}
                {doc.method === "pdf_text" ? "PDF text layer" : doc.ocr_confidence != null ? `OCR ${(doc.ocr_confidence * 100).toFixed(0)}%` : doc.method}
                {!requiredDocs.includes(doc.doc_type) && <span className="ml-auto rounded bg-slate-100 px-1 text-slate-500">extra</span>}
              </div>
            </div>
          </button>
        ))}
        {d.documents.missing.map((m) => (
          <div key={m} className="flex h-full min-h-[170px] flex-col items-center justify-center rounded-lg border-2 border-dashed border-rose-200 bg-rose-50/40 p-3 text-center">
            <FileImage className="h-6 w-6 text-rose-300" />
            <div className="mt-2 text-[12.5px] font-semibold text-rose-700">{DOC_LABEL[m]}</div>
            <div className="text-[11px] text-rose-500">missing - required</div>
          </div>
        ))}
      </div>
      {view && <DocModal claimId={d.claim_id} doc={view} onClose={() => setView(null)} />}
    </Card>
  );
}

function DocModal({ claimId, doc, onClose }: { claimId: string; doc: DocDetail; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-6 backdrop-blur-sm" onClick={onClose}>
      <div className="flex max-h-full w-full max-w-5xl overflow-hidden rounded-2xl bg-white shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="scroll-thin flex-1 overflow-auto bg-slate-100">
          <img src={api.preview(claimId, doc.file)} className="mx-auto" />
        </div>
        <aside className="w-72 shrink-0 space-y-3 border-l border-slate-200 p-4 text-[13px]">
          <div className="flex items-start justify-between">
            <div>
              <div className="font-semibold">{DOC_LABEL[doc.doc_type]}</div>
              <div className="text-xs text-slate-400">{doc.file}</div>
            </div>
            <button onClick={onClose} className="rounded p-1 hover:bg-slate-100">
              <X className="h-4 w-4" />
            </button>
          </div>
          <Row k="Status" v={doc.status + (doc.related_to ? ` â†’ ${doc.related_to}` : "")} />
          {doc.status_reason && <Row k="Why" v={doc.status_reason} />}
          <Row k="Read via" v={doc.method || "-"} />
          {doc.ocr_confidence != null && <Row k="OCR confidence" v={`${(doc.ocr_confidence * 100).toFixed(1)}%`} />}
          {doc.type_reason && <Row k="Typed because" v={doc.type_reason} />}
          {Object.entries(doc.meta).map(([k, v]) => (
            <Row key={k} k={k.replaceAll("_", " ")} v={String(v)} />
          ))}
          <a href={api.file(claimId, doc.file)} target="_blank" className="inline-flex items-center gap-1 text-indigo-600 hover:underline">
            Open original <ArrowRight className="h-3 w-3" />
          </a>
        </aside>
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <div className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-400">{k}</div>
      <div className="break-words text-slate-700">{v}</div>
    </div>
  );
}


// ------------------------------------------------------------------ cross-document checks
const XC_LABEL: Record<string, string> = {
  vehicle: "Vehicle",
  lender: "Lender",
  person_name: "Owner / borrower",
  policy_number: "Policy number",
  police_report_number: "Police report no.",
};
const XC_STATUS: Record<string, { label: string; cls: string }> = {
  consistent: { label: "Consistent", cls: "bg-emerald-50 text-emerald-700" },
  single_source: { label: "1 source", cls: "bg-slate-100 text-slate-500" },
  ocr_variant: { label: "OCR variant", cls: "bg-sky-50 text-sky-700" },
  customer_mismatch: { label: "Customer differs", cls: "bg-amber-50 text-amber-800" },
  contradiction: { label: "Contradiction", cls: "bg-rose-50 text-rose-700" },
};

export function CrossChecksPanel({ d }: { d: ClaimDetail }) {
  const checks = d.cross_document_checks || {};
  const entries = Object.entries(checks);
  if (!entries.length) return null;
  return (
    <Card title="Cross-document checks" icon={<Layers className="h-4 w-4 text-indigo-500" />}
      right={<span className="text-xs text-slate-400">everything the documents share must agree</span>}>
      <div className="divide-y divide-slate-100">
        {entries.map(([attr, c]) => (
          <div key={attr} className="grid grid-cols-[150px_1fr_auto] items-start gap-3 px-4 py-2.5">
            <div className="text-[13px] font-medium text-slate-500">{XC_LABEL[attr] || attr}</div>
            <div className="min-w-0 text-[13px]">
              {c.clusters ? (
                <ul className="space-y-0.5">
                  {c.clusters.map((g, i) => (
                    <li key={i}>
                      <span className={cx("font-semibold", i === 0 ? "text-slate-900" : "text-rose-700")}>{g.value}</span>
                      <span className="text-[11px] text-slate-400"> · {g.sources.join(", ")}</span>
                    </li>
                  ))}
                  <li className="text-[11px] text-slate-500">
                    {c.value ? <>best guess: <span className="font-semibold">{c.value}</span></> : "No guess"}
                  </li>
                </ul>
              ) : (
                <>
                  <span className="font-semibold text-slate-900">{c.value}</span>
                  <span className="text-[11px] text-slate-400"> · {(c.sources || []).join(", ")}</span>
                </>
              )}
            </div>
            <span className={cx("rounded-md px-1.5 py-0.5 text-[11px] font-semibold", XC_STATUS[c.status]?.cls)}>
              {XC_STATUS[c.status]?.label || c.status}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}
