// BriefsPanel — read the per-desk daily briefs another workstream renders to
// data/briefs/dt=YYYY-MM-DD/<desk>.md (+ optional <desk>.json with movers /
// driver sections). We list them (GET /api/briefs), render the markdown with
// marked + DOMPurify (same pattern as ReportsPanel), and show movers as a
// compact red/green table when the sibling json exists. Everything from the
// json is optional — render defensively.

import { useCallback, useEffect, useMemo, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { BookOpenText, FlaskConical } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "@/components/AsyncStates";
import { getBrief, getBriefs, type Brief, type BriefMover, type BriefRef } from "@/lib/mc-api";

export function BriefsPanel() {
  const [briefs, setBriefs] = useState<BriefRef[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [selected, setSelected] = useState<BriefRef | null>(null);

  const load = useCallback(async () => {
    setListError(null);
    try {
      const list = await getBriefs();
      setBriefs(list);
      setSelected((cur) => cur ?? list[0] ?? null);
    } catch (e) {
      setBriefs(null);
      setListError(String(e instanceof Error ? e.message : e));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="mx-auto w-full max-w-[980px] px-6 py-6">
      <h1 className="flex items-center gap-2 text-[18px] font-semibold text-foreground">
        <BookOpenText className="size-4 text-primary" /> Desk briefs
      </h1>
      <p className="mt-1 text-[12.5px] text-muted-foreground">
        Point-in-time market-color notes per desk — movers, drivers with citations, and an honest
        “no clear driver” when the evidence is thin.
      </p>

      <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-[240px_1fr]">
        <div className="space-y-2">
          {briefs === null && !listError && <LoadingPanel label="Loading briefs…" />}
          {listError && (
            <ErrorPanel title="Could not list briefs" message={listError} onRetry={() => void load()} />
          )}
          {briefs?.length === 0 && (
            <EmptyPanel
              title="No briefs yet"
              message="Briefs appear here once the daily brief generator has run."
            />
          )}
          {briefs?.map((b) => {
            const isSelected = selected?.date === b.date && selected?.desk === b.desk;
            return (
              <button
                key={`${b.date}/${b.desk}`}
                onClick={() => setSelected(b)}
                className={cn(
                  "w-full rounded-md border px-3 py-2 text-left transition-colors",
                  isSelected
                    ? "border-primary/40 bg-primary/[0.07]"
                    : "border-outline-subtle bg-surface-panel hover:border-primary/30",
                )}
              >
                <div className="text-[12.5px] font-medium capitalize text-foreground">{b.desk}</div>
                <div className="mt-0.5 font-mono text-[10.5px] text-muted-foreground">{b.date}</div>
              </button>
            );
          })}
        </div>

        <div>
          {selected ? (
            <BriefDetail date={selected.date} desk={selected.desk} />
          ) : (
            !listError &&
            briefs !== null &&
            briefs.length > 0 && (
              <EmptyPanel title="Pick a brief" message="Select a desk brief on the left to read it." />
            )
          )}
        </div>
      </div>
    </div>
  );
}

function BriefDetail({ date, desk }: { date: string; desk: string }) {
  const [brief, setBrief] = useState<Brief | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setBrief(null);
    setError(null);
    try {
      setBrief(await getBrief(date, desk));
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, [date, desk]);

  useEffect(() => {
    void load();
  }, [load]);

  const html = useMemo(
    () => (brief ? DOMPurify.sanitize(marked.parse(brief.markdown || "", { async: false })) : ""),
    [brief],
  );

  if (error) {
    return <ErrorPanel title="Could not load the brief" message={error} onRetry={() => void load()} />;
  }
  if (!brief) return <LoadingPanel label="Loading brief…" />;

  const movers = Array.isArray(brief.data?.movers) ? brief.data.movers : [];
  const newFacts = brief.data?.new_fact_count;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[14px] font-semibold capitalize text-foreground">
          {brief.desk} · {brief.date}
        </span>
        {typeof newFacts === "number" && (
          <Badge variant="secondary" className="text-[10px]">
            {newFacts} new fact{newFacts === 1 ? "" : "s"}
          </Badge>
        )}
        {brief.data?.fixture === true && (
          <Badge variant="secondary" className="gap-1 text-[10px] text-amber-300">
            <FlaskConical className="size-3" /> dev fixture
          </Badge>
        )}
        {brief.data?.generated_utc && (
          <span className="font-mono text-[10.5px] text-muted-foreground">
            generated {brief.data.generated_utc}
          </span>
        )}
      </div>

      {movers.length > 0 && <MoversTable movers={movers} />}

      <div className="rounded-md border border-outline-subtle bg-surface-panel px-4 py-3">
        <div
          className={cn(
            "prose-chat text-[13px] leading-relaxed text-foreground/90",
            "[&_h1]:text-[15px] [&_h2]:mt-3 [&_h2]:text-[14px] [&_h3]:text-[13px]",
            "[&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold",
            "[&_p]:my-1.5 [&_ul]:my-1.5 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-1.5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5",
            "[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-outline-subtle [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground",
            "[&_a]:text-primary [&_a]:underline [&_code]:rounded [&_code]:bg-surface-input [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[11.5px]",
          )}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
    </div>
  );
}

function MoversTable({ movers }: { movers: BriefMover[] }) {
  const fmtRet = (r: number) => `${r > 0 ? "+" : ""}${(r * 100).toFixed(1)}%`;
  return (
    <div className="overflow-hidden rounded-md border border-outline-subtle bg-surface-panel">
      <table className="w-full text-[12px]">
        <thead>
          <tr className="border-b border-outline-subtle text-left text-[10.5px] uppercase tracking-wide text-muted-foreground">
            <th className="px-3 py-1.5 font-medium">Symbol</th>
            <th className="px-3 py-1.5 font-medium">Name</th>
            <th className="px-3 py-1.5 text-right font-medium">Close</th>
            <th className="px-3 py-1.5 text-right font-medium">1d</th>
            <th className="px-3 py-1.5 text-right font-medium">z(20d)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-outline-subtle">
          {movers.map((m, i) => (
            <tr key={m.symbol ?? i}>
              <td className="px-3 py-1.5 font-mono text-foreground">{m.symbol ?? "—"}</td>
              <td className="px-3 py-1.5 text-muted-foreground">{m.name ?? ""}</td>
              <td className="px-3 py-1.5 text-right font-mono text-foreground/90">
                {typeof m.close === "number" ? m.close.toFixed(2) : "—"}
              </td>
              <td
                className={cn(
                  "px-3 py-1.5 text-right font-mono",
                  typeof m.ret_1d === "number"
                    ? m.ret_1d < 0
                      ? "text-rose-300"
                      : m.ret_1d > 0
                        ? "text-emerald-300"
                        : "text-muted-foreground"
                    : "text-muted-foreground",
                )}
              >
                {typeof m.ret_1d === "number" ? fmtRet(m.ret_1d) : "—"}
              </td>
              <td className="px-3 py-1.5 text-right font-mono text-muted-foreground">
                {typeof m.zscore_20d === "number" ? m.zscore_20d.toFixed(1) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
