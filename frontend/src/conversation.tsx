import { useEffect, useRef, useState } from "react";
import {
  Bot, Brain, CheckCircle2, FileSearch, GitCompare, Inbox, MessageSquare, Paperclip, Scale, ScanLine, SearchCheck,
  Send, ShieldQuestion, Sigma, Tag, User, Workflow, XCircle,
} from "lucide-react";
import type { AgentEvent, ClaimDetail, ToolLog } from "./api";
import { Dots, StatusBadge, cx } from "./ui";

const TOOL_ICON: Record<string, JSX.Element> = {
  list_documents: <Inbox className="h-3.5 w-3.5" />,
  read_document: <ScanLine className="h-3.5 w-3.5" />,
  vision_read: <FileSearch className="h-3.5 w-3.5" />,
  extract_fields: <Tag className="h-3.5 w-3.5" />,
  record_observation: <SearchCheck className="h-3.5 w-3.5" />,
  flag_concern: <ShieldQuestion className="h-3.5 w-3.5" />,
  validate_vin: <CheckCircle2 className="h-3.5 w-3.5" />,
  compare_documents: <GitCompare className="h-3.5 w-3.5" />,
  check_settlement_math: <Sigma className="h-3.5 w-3.5" />,
  reconcile: <Scale className="h-3.5 w-3.5" />,
  evaluate_claim: <Workflow className="h-3.5 w-3.5" />,
  note_customer_intent: <MessageSquare className="h-3.5 w-3.5" />,
  submit_turn: <Send className="h-3.5 w-3.5" />,
};

function argSummary(t: string, input: Record<string, unknown>) {
  const v = Object.entries(input)
    .filter(([k]) => !["customer_message", "internal_note", "summary", "follow_ups", "type_reason"].includes(k))
    .map(([k, x]) => (k === "file" || k === "field" || k === "vin" ? String(x) : `${k}=${typeof x === "string" ? x : JSON.stringify(x)}`))
    .join(" · ");
  return t === "submit_turn" ? String(input.action) : v;
}

function resultSummary(r: unknown): string {
  if (r == null) return "";
  if (typeof r !== "object") return String(r);
  const o = r as Record<string, unknown>;
  if (o.error) return `⚠ ${o.error}`;
  if (o.status && o.recommended_action) return `status=${o.status} → ${o.recommended_action}`;
  if (o.relation) return String(o.relation).replaceAll("_", " ");
  if (Array.isArray(o.fields)) return (o.fields as { field: string; value: string | null; raw: string }[]).map((f) => `${f.field}=${f.value ?? f.raw}`).join(", ") || "no template fields";
  if (o.method && o.chars) return `${o.method}${o.ocr_mean_confidence ? ` · conf ${(Number(o.ocr_mean_confidence) * 100).toFixed(0)}%` : ""}${o.skew_corrected_deg ? ` · deskewed ${o.skew_corrected_deg}°` : ""}`;
  if ("conflict" in o) return o.conflict ? `CONFLICT → best guess ${o.value}` : `agreed: ${o.value} (${o.confidence})`;
  if ("checks" in o) return o.ok ? "arithmetic OK" : "arithmetic MISMATCH";
  if ("valid" in o) return o.valid ? "valid" : `invalid${Array.isArray(o.errors) ? ": " + (o.errors as string[]).join(", ") : ""}`;
  if (o.recorded) return `recorded${o.method ? ` (${o.method})` : ""}`;
  if (o.accepted) return `accepted - ${o.status}`;
  if (Array.isArray(o.documents)) return `${(o.documents as unknown[]).length} files`;
  return JSON.stringify(o).slice(0, 90);
}

type TraceItem = { kind: "tool"; t: ToolLog } | { kind: "thought"; text: string; turn: number } | { kind: "turn"; turn: number; label: string };

export function Trace({ log, live, running }: { log: ToolLog[]; live: AgentEvent[]; running: boolean }) {
  const items: TraceItem[] = [];
  const src: ToolLog[] = running ? [] : log;
  let lastTurn = -1;
  for (const t of src) {
    if (t.turn !== lastTurn) {
      items.push({ kind: "turn", turn: t.turn, label: t.turn === 0 ? "Initial review" : `Turn ${t.turn} · customer message` });
      lastTurn = t.turn;
    }
    items.push({ kind: "tool", t });
  }
  if (running) {
    for (const e of live) {
      if (e.type === "turn_start") items.push({ kind: "turn", turn: Number(e.turn), label: `${Number(e.turn) === 0 ? "Initial review" : `Turn ${e.turn}`} · ${e.agent}` });
      if (e.type === "tool") items.push({ kind: "tool", t: e as unknown as ToolLog });
      if (e.type === "thought") items.push({ kind: "thought", text: String(e.text), turn: 0 });
      if (e.type === "error") items.push({ kind: "thought", text: "Error: " + String(e.message), turn: 0 });
    }
  }
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (running) end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [live.length, running]);
  const [open, setOpen] = useState<number | null>(null);

  return (
    <div className="space-y-1 p-3">
      {items.length === 0 && <p className="p-4 text-center text-sm text-slate-400">No agent activity yet.</p>}
      {items.map((it, i) => {
        if (it.kind === "turn")
          return (
            <div key={i} className="sticky top-0 z-10 -mx-3 mt-2 bg-white/95 px-3 py-1.5 text-[10.5px] font-bold uppercase tracking-wider text-indigo-600 backdrop-blur first:mt-0">
              {it.label}
            </div>
          );
        if (it.kind === "thought")
          return (
            <div key={i} className="slide-in flex gap-2 rounded-md bg-violet-50/70 px-2 py-1.5 text-[12px] italic text-violet-900">
              <Brain className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span className="whitespace-pre-wrap">{it.text}</span>
            </div>
          );
        const t = it.t;
        return (
          <div key={i} className="slide-in">
            <button onClick={() => setOpen(open === i ? null : i)} className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-slate-50">
              <span className={cx("mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded", t.ok ? "bg-indigo-50 text-indigo-600" : "bg-rose-50 text-rose-600")}>
                {t.ok ? TOOL_ICON[t.tool] || <Bot className="h-3.5 w-3.5" /> : <XCircle className="h-3.5 w-3.5" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="font-mono text-[12px] font-medium text-slate-800">{t.tool}</span>
                <span className="ml-1.5 text-[11.5px] text-slate-500">{argSummary(t.tool, t.input)}</span>
                <span className={cx("block truncate text-[11.5px]", t.ok ? "text-slate-500" : "text-rose-600")}>→ {resultSummary(t.result)}</span>
              </span>
            </button>
            {open === i && (
              <pre className="scroll-thin mx-2 mb-2 max-h-72 overflow-auto rounded-md bg-slate-900 p-2.5 font-mono text-[11px] leading-relaxed text-slate-100">
                {JSON.stringify({ input: t.input, result: t.result }, null, 2)}
              </pre>
            )}
          </div>
        );
      })}
      {running && (
        <div className="flex items-center gap-2 px-2 py-2 text-xs text-indigo-600">
          <Dots /> agent working
        </div>
      )}
      <div ref={end} />
    </div>
  );
}

export function Chat({
  d, running, onSend, onReplay,
}: {
  d: ClaimDetail;
  running: boolean;
  onSend: (text: string, files: File[]) => void;
  onReplay: () => void;
}) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [d.conversation.length, running]);

  const send = () => {
    if (!text.trim() && !files.length) return;
    onSend(text, files);
    setText("");
    setFiles([]);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="scroll-thin flex-1 space-y-4 overflow-y-auto p-4">
        {d.conversation.map((t) => (
          <div key={t.turn} className="space-y-3">
            {(t.inbound || t.uploaded) && (
              <div className="flex justify-end gap-2">
                <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-600 px-3.5 py-2.5 text-[13px] leading-relaxed text-white shadow-sm">
                  {t.inbound && <p className="whitespace-pre-wrap">{t.inbound}</p>}
                  {t.uploaded?.map((f) => (
                    <div key={f} className="mt-1.5 inline-flex items-center gap-1 rounded bg-white/15 px-1.5 py-0.5 text-[11px]">
                      <Paperclip className="h-3 w-3" /> {f}
                    </div>
                  ))}
                </div>
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-indigo-700">
                  <User className="h-4 w-4" />
                </div>
              </div>
            )}
            <div className="flex gap-2">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white">
                <Bot className="h-4 w-4" />
              </div>
              <div className="max-w-[88%] space-y-1.5">
                <div className="flex items-center gap-2 text-[11px] text-slate-400">
                  {t.turn === 0 ? "Initial review" : `Turn ${t.turn}`} · status <StatusBadge status={t.status_after_turn} />
                </div>
                {t.message_to_customer ? (
                  <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-3.5 py-2.5 text-[13px] leading-relaxed text-slate-800 shadow-sm">
                    <p className="whitespace-pre-wrap">{t.message_to_customer}</p>
                  </div>
                ) : (
                  <div className="rounded-xl border border-dashed border-slate-200 px-3 py-2 text-[12px] italic text-slate-500">
                    No message to the customer - action: {t.action?.replaceAll("_", " ")}.
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {running && (
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="flex h-7 w-7 items-center justify-center rounded-full bg-slate-900 text-white">
              <Bot className="h-4 w-4" />
            </div>
            <Dots /> reviewing…
          </div>
        )}
        <div ref={end} />
      </div>

      {d.pending_replies.length > 0 && !running && (
        <div className="mx-3 mb-2 flex items-center justify-between rounded-lg bg-amber-50 px-3 py-2 text-[12px] text-amber-900">
          <span>
            This claim has a recorded customer reply (<span className="font-mono">{d.pending_replies[0]}</span>).
          </span>
          <button onClick={onReplay} className="rounded-md bg-amber-600 px-2.5 py-1 font-semibold text-white hover:bg-amber-700">
            Deliver it
          </button>
        </div>
      )}

      <div className="border-t border-slate-200 bg-white p-3">
        {files.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1.5">
            {files.map((f) => (
              <span key={f.name} className="inline-flex items-center gap-1 rounded bg-slate-100 px-1.5 py-0.5 text-[11px] text-slate-600">
                <Paperclip className="h-3 w-3" />
                {f.name}
              </span>
            ))}
          </div>
        )}
        <div className="flex items-end gap-2">
          <input ref={fileRef} type="file" multiple hidden onChange={(e) => setFiles([...files, ...Array.from(e.target.files || [])])} />
          <button onClick={() => fileRef.current?.click()} title="Attach documents" className="rounded-lg p-2 text-slate-500 hover:bg-slate-100">
            <Paperclip className="h-4 w-4" />
          </button>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={2}
            disabled={running}
            placeholder="Reply as the customer… (Enter to send, attach photos/PDFs)"
            className="flex-1 resize-none rounded-lg border border-slate-200 px-3 py-2 text-[13px] outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 disabled:bg-slate-50"
          />
          <button onClick={send} disabled={running} className="rounded-lg bg-indigo-600 p-2.5 text-white shadow-sm hover:bg-indigo-700 disabled:opacity-40">
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
