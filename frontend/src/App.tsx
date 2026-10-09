import { useCallback, useEffect, useState } from "react";
import { Activity, FileStack, ListOrdered, MessagesSquare, Play, PlayCircle, RotateCcw, Zap } from "lucide-react";
import { api, stream, type AgentEvent, type ClaimDetail, type ClaimSummary, type Priority } from "./api";
import { Chat, Trace } from "./conversation";
import { CrossChecksPanel, DocumentsPanel, FieldsPanel, IssuesPanel, StatusHero } from "./panels";
import { Card, Dots, StatusBadge, cx, fmtValue } from "./ui";

export default function App() {
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [claims, setClaims] = useState<ClaimSummary[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [detail, setDetail] = useState<ClaimDetail | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [live, setLive] = useState<AgentEvent[]>([]);
  const [view, setView] = useState<"claim" | "priority">("claim");
  const [priority, setPriority] = useState<Priority | null>(null);
  const [tab, setTab] = useState<"chat" | "trace">("chat");
  const [batch, setBatch] = useState(false);

  const refresh = useCallback(async (id?: string | null) => {
    const cs = await api.claims();
    setClaims(cs);
    setPriority(await api.priority());
    const target = id ?? sel;
    if (target) setDetail(await api.claim(target));
  }, [sel]);

  useEffect(() => {
    api.config().then(setConfig);
    api.claims().then((cs) => {
      setClaims(cs);
      if (cs.length) setSel(cs[0].claim_id);
    });
    api.priority().then(setPriority);
  }, []);
  useEffect(() => {
    if (sel) api.claim(sel).then(setDetail);
  }, [sel]);

  const run = async (id: string, url: string, body: FormData | null = null) => {
    setRunning(id);
    setLive([]);
    if (id === sel) setTab(url.endsWith("process") ? "trace" : tab);
    await stream(url, body, (e) => setLive((l) => [...l, e]));
    setRunning(null);
    await refresh(id);
  };

  const processAll = async () => {
    setBatch(true);
    for (const c of claims) {
      setSel(c.claim_id);
      setTab("trace");
      await run(c.claim_id, `/api/claims/${c.claim_id}/process`);
      const d = await api.claim(c.claim_id);
      for (let i = 0; i < d.pending_replies.length; i++) {
        await run(c.claim_id, `/api/claims/${c.claim_id}/replay-reply`);
      }
    }
    setBatch(false);
    setView("priority");
  };

  const rank = (id: string) => priority?.processing_order.find((p) => p.claim_id === id)?.rank;

  return (
    <div className="flex h-full flex-col">
      {/* top bar */}
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-5">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow">
            <FileStack className="h-4.5 w-4.5" />
          </div>
          <div>
            <div className="text-[15px] font-bold leading-tight tracking-tight">Claims Desk</div>
            <div className="text-[11px] text-slate-400">Total-loss claims agent</div>
          </div>
          <nav className="ml-6 flex h-14 items-stretch gap-1">
            {([["claim", "Claims", FileStack], ["priority", "Priority ranking", ListOrdered]] as const).map(([v, label, Icon]) => (
              <button key={v} onClick={() => setView(v)}
                className={cx("flex items-center gap-2 border-b-2 px-3 text-[13.5px] font-semibold transition",
                  view === v ? "border-indigo-600 text-indigo-700" : "border-transparent text-slate-500 hover:text-slate-800")}>
                <Icon className="h-4 w-4" /> {label}
                {v === "priority" && priority?.processing_order.length ? (
                  <span className="rounded-full bg-indigo-600 px-1.5 text-[10.5px] font-bold text-white">{priority.processing_order.length}</span>
                ) : null}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={processAll} disabled={!!running} className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-[12.5px] font-semibold text-white hover:bg-slate-800 disabled:opacity-40">
            {batch ? <Dots /> : <Zap className="h-3.5 w-3.5" />} Process all claims
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        {/* queue */}
        <aside className="scroll-thin w-64 shrink-0 overflow-y-auto border-r border-slate-200 bg-white/60 p-3">
          <button onClick={() => setView("priority")}
            className={cx("mb-3 flex w-full items-center gap-2.5 rounded-lg border p-2.5 text-left transition",
              view === "priority" ? "border-indigo-300 bg-indigo-50" : "border-indigo-100 bg-gradient-to-br from-indigo-50 to-white hover:border-indigo-300")}>
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-indigo-600 text-white"><ListOrdered className="h-4 w-4" /></span>
            <span className="min-w-0">
              <span className="block text-[13px] font-semibold text-indigo-900">Recommended order</span>
              <span className="block truncate text-[11px] text-indigo-600/80">
                {priority?.processing_order.length ? `Start with ${priority.processing_order[0].claim_id}` : "Process claims to rank them"}
              </span>
            </span>
          </button>
          <div className="mb-2 px-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400">Claim queue</div>
          <div className="space-y-1.5">
            {claims.map((c) => (
              <button
                key={c.claim_id}
                onClick={() => {
                  setSel(c.claim_id);
                  setView("claim");
                }}
                className={cx("w-full rounded-lg border p-2.5 text-left transition", sel === c.claim_id && view === "claim" ? "border-indigo-300 bg-indigo-50/60 shadow-sm" : "border-transparent hover:bg-white hover:shadow-sm")}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[13px] font-semibold">{c.claim_id}</span>
                  {rank(c.claim_id) && <span className="rounded bg-slate-900 px-1.5 text-[10px] font-bold text-white">#{rank(c.claim_id)}</span>}
                </div>
                <div className="mt-0.5 truncate text-[12px] text-slate-500">{c.claimant || `${c.files.length} files`}</div>
                <div className="mt-1.5 flex items-center justify-between">
                  {running === c.claim_id ? (
                    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-indigo-600">
                      <Activity className="h-3 w-3" /> processing <Dots />
                    </span>
                  ) : (
                    <StatusBadge status={c.status} />
                  )}
                  {c.processed && c.payout && <span className="text-[11px] text-slate-400">{fmtValue("insurance_payout", c.payout)}</span>}
                </div>
              </button>
            ))}
          </div>
        </aside>

        {view === "priority" ? (
          <PriorityView p={priority} onOpen={(id) => { setSel(id); setView("claim"); }} />
        ) : !sel || !detail ? (
          <div className="flex flex-1 items-center justify-center text-slate-400">Select a claim</div>
        ) : !detail.processed && running !== sel ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-4">
            <div className="text-center">
              <div className="font-mono text-lg font-semibold">{detail.claim_id}</div>
              <div className="mt-1 text-sm text-slate-500">{(detail.files || []).join(" · ")}</div>
            </div>
            <button onClick={() => run(sel, `/api/claims/${sel}/process`)} className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-200 hover:bg-indigo-700">
              <PlayCircle className="h-5 w-5" /> Run agent on this claim
            </button>
          </div>
        ) : (
          <main className="flex min-h-0 flex-1">
            <div className="scroll-thin min-w-0 flex-1 space-y-4 overflow-y-auto p-5">
              <div className="flex items-center justify-between">
                <div>
                  <h1 className="font-mono text-xl font-bold">{detail.claim_id}</h1>
                  <p className="text-sm text-slate-500">{detail.claimant || ""}</p>
                </div>
                <div className="flex gap-2">
                  <button disabled={!!running} onClick={() => run(sel, `/api/claims/${sel}/process`)} className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-[12.5px] font-medium hover:bg-slate-50 disabled:opacity-40">
                    <RotateCcw className="h-3.5 w-3.5" /> Re-run from scratch
                  </button>
                </div>
              </div>
              {running === sel && !detail.processed ? (
                <Card className="p-8 text-center text-sm text-slate-500"><Dots /> Agent is reviewing the documents - follow along in the trace →</Card>
              ) : (
                <>
                  <StatusHero d={detail} />
                  <div className="grid gap-4 2xl:grid-cols-2">
                    <FieldsPanel d={detail} />
                    <IssuesPanel issues={detail.issues} />
                  </div>
                  <CrossChecksPanel d={detail} />
                  <DocumentsPanel d={detail} requiredDocs={(config.required_documents as string[]) || []} />
                </>
              )}
            </div>
            {/* right rail */}
            <aside className="flex w-[420px] shrink-0 flex-col border-l border-slate-200 bg-white">
              <div className="flex border-b border-slate-200 px-2">
                {(["chat", "trace"] as const).map((t) => (
                  <button key={t} onClick={() => setTab(t)} className={cx("flex items-center gap-1.5 border-b-2 px-3 py-2.5 text-[13px] font-medium", tab === t ? "border-indigo-600 text-indigo-700" : "border-transparent text-slate-500 hover:text-slate-700")}>
                    {t === "chat" ? <MessagesSquare className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    {t === "chat" ? "Customer conversation" : "Agent trace"}
                    {t === "trace" && running === sel && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-indigo-500" />}
                  </button>
                ))}
              </div>
              <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
                {tab === "chat" ? (
                  <Chat
                    d={detail}
                    running={running === sel}
                    onReplay={() => run(sel, `/api/claims/${sel}/replay-reply`)}
                    onSend={(text, files) => {
                      const fd = new FormData();
                      fd.append("text", text);
                      files.forEach((f) => fd.append("files", f));
                      run(sel, `/api/claims/${sel}/message`, fd);
                    }}
                  />
                ) : (
                  <Trace log={detail.tool_log || []} live={live} running={running === sel} />
                )}
              </div>
            </aside>
          </main>
        )}
      </div>
    </div>
  );
}

const TIER: Record<number, { label: string; cls: string }> = {
  1: { label: "Finalize now", cls: "bg-emerald-600" },
  2: { label: "One decision away", cls: "bg-sky-600" },
  3: { label: "Waiting on customer", cls: "bg-amber-500" },
  4: { label: "Review + missing docs", cls: "bg-orange-600" },
  5: { label: "Blocked", cls: "bg-rose-600" },
};

function PriorityView({ p, onOpen }: { p: Priority | null; onOpen: (id: string) => void }) {
  if (!p || !p.processing_order.length)
    return <div className="flex flex-1 items-center justify-center text-sm text-slate-400">Process claims to see the recommended order.</div>;
  return (
    <main className="scroll-thin flex-1 overflow-y-auto p-6">
      <div className="mx-auto max-w-4xl space-y-5">
        <div>
          <h1 className="text-xl font-bold">Recommended processing order</h1>
          <p className="mt-1 text-sm text-slate-500">{p.method}</p>
          <p className="mt-1 text-xs text-slate-400">Confidence = field evidence (60) + corroboration across documents (20) + no open warnings (20).</p>
        </div>
        <ol className="space-y-3">
          {p.processing_order.map((r) => (
            <li key={r.claim_id}>
              <button onClick={() => onOpen(r.claim_id)} className="flex w-full gap-4 rounded-xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:border-indigo-300 hover:shadow-md">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-900 text-lg font-bold text-white">{r.rank}</div>
                {r.decision_confidence && (
                  <div className="order-last w-40 shrink-0 self-center" title={r.decision_confidence.summary}>
                    <div className="flex items-baseline justify-between">
                      <span className="text-[10.5px] font-semibold uppercase tracking-wider text-slate-400">Confidence</span>
                      <span className="text-[15px] font-bold text-slate-800">{r.decision_confidence.score}<span className="text-[11px] font-medium text-slate-400">/100</span></span>
                    </div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-100">
                      <div className={cx("h-full rounded-full", r.decision_confidence.label === "high" ? "bg-emerald-500" : r.decision_confidence.label === "medium" ? "bg-amber-400" : "bg-rose-500")}
                        style={{ width: `${r.decision_confidence.score}%` }} />
                    </div>
                    <div className="mt-1 text-[10.5px] leading-snug text-slate-500">{r.decision_confidence.summary}</div>
                  </div>
                )}
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-[15px] font-bold">{r.claim_id}</span>
                    <StatusBadge status={r.status} />
                    <span className={cx("rounded px-1.5 py-0.5 text-[10.5px] font-semibold text-white", TIER[r.tier].cls)}>{TIER[r.tier].label}</span>
                    <span className="text-xs text-slate-400">{r.blocking_issues} blocking</span>
                  </div>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-slate-600">{r.reason.replace(/ Decision confidence \d+\/100 \([a-z]+\): [^.]*\./, "")}</p>
                </div>
              </button>
            </li>
          ))}
        </ol>
        {p.do_in_parallel_today.length > 0 && (
          <Card title="Do in parallel today" icon={<Zap className="h-4 w-4 text-amber-500" />}>
            <ul className="divide-y divide-slate-100">
              {p.do_in_parallel_today.map((a, i) => (
                <li key={i} className="flex gap-3 px-4 py-2.5 text-[13px]">
                  <span className="font-mono font-semibold">{a.claim_id}</span>
                  <span className="text-slate-600">{a.action}</span>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>
    </main>
  );
}
