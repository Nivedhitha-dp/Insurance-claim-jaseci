import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, CircleDashed, ShieldAlert } from "lucide-react";
import type { Confidence, Status } from "./api";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

export const STATUS: Record<Status, { label: string; cls: string; dot: string; icon: ReactNode }> = {
  complete: {
    label: "Complete",
    cls: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    dot: "bg-emerald-500",
    icon: <CheckCircle2 className="h-4 w-4" />,
  },
  incomplete: {
    label: "Incomplete",
    cls: "bg-amber-50 text-amber-800 ring-amber-200",
    dot: "bg-amber-500",
    icon: <CircleDashed className="h-4 w-4" />,
  },
  needs_review: {
    label: "Needs review",
    cls: "bg-rose-50 text-rose-700 ring-rose-200",
    dot: "bg-rose-500",
    icon: <ShieldAlert className="h-4 w-4" />,
  },
};

export function StatusBadge({ status, size = "sm" }: { status?: Status | null; size?: "sm" | "lg" }) {
  if (!status)
    return <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">Not processed</span>;
  const s = STATUS[status];
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full font-semibold ring-1 ring-inset",
        s.cls,
        size === "lg" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-xs",
      )}
    >
      {size === "lg" ? s.icon : <span className={cx("h-1.5 w-1.5 rounded-full", s.dot)} />}
      {s.label}
    </span>
  );
}

const CONF: Record<Confidence, string> = {
  high: "bg-emerald-100 text-emerald-800",
  medium: "bg-amber-100 text-amber-800",
  low: "bg-rose-100 text-rose-800",
};

export function ConfidencePill({ c }: { c: Confidence }) {
  const bars = c === "high" ? 3 : c === "medium" ? 2 : 1;
  return (
    <span className={cx("inline-flex items-center gap-1.5 rounded-md px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide", CONF[c])}>
      <span className="flex items-end gap-[2px]">
        {[1, 2, 3].map((i) => (
          <span key={i} className={cx("w-[3px] rounded-sm", i <= bars ? "bg-current" : "bg-current opacity-20")} style={{ height: 3 + i * 3 }} />
        ))}
      </span>
      {c}
    </span>
  );
}

export function Card({ title, icon, right, children, className }: { title?: ReactNode; icon?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cx("rounded-xl border border-slate-200/80 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.04)]", className)}>
      {title && (
        <header className="flex items-center justify-between border-b border-slate-100 px-4 py-2.5">
          <h3 className="flex items-center gap-2 text-[13px] font-semibold text-slate-700">
            {icon}
            {title}
          </h3>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export const DOC_LABEL: Record<string, string> = {
  police_report: "Police report",
  finance_agreement: "Finance agreement",
  settlement_breakdown: "Settlement breakdown",
  adjuster_note: "Adjuster note",
  tow_receipt: "Tow receipt",
  customer_correspondence: "Customer message",
  unknown: "Unknown",
};

export const FIELD_LABEL: Record<string, string> = {
  vin: "VIN",
  date_of_loss: "Date of loss",
  insurance_payout: "Insurance payout",
  loan_balance: "Loan balance",
};

export function fmtValue(field: string, v: string | null) {
  if (v == null) return "—";
  if (field === "insurance_payout" || field === "loan_balance") {
    const n = Number(v.replace(/^~/, ""));
    return isNaN(n) ? v : n.toLocaleString("en-US", { style: "currency", currency: "USD" });
  }
  if (field === "date_of_loss" && /^\d{4}-\d{2}-\d{2}$/.test(v)) {
    return new Date(v + "T00:00:00").toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  }
  return v;
}

export function Warn({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-start gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-900">
      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <div>{children}</div>
    </div>
  );
}

export function Dots() {
  return (
    <span className="inline-flex gap-0.5">
      <span className="dot h-1.5 w-1.5 rounded-full bg-current" />
      <span className="dot h-1.5 w-1.5 rounded-full bg-current" />
      <span className="dot h-1.5 w-1.5 rounded-full bg-current" />
    </span>
  );
}
