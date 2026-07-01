// ReportsPanel — generate + read grounded market-color reports.
//
// Reports are written from the indexed facts (Claude + search_market_facts MCP):
// automatically after each daily run (kind='daily'), or on demand from a prompt
// here (kind='adhoc'). Generation is async — we poll getReport until it settles.

import { useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { marked } from "marked";
import { ExternalLink, FileText, Loader2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  generateReport,
  getReport,
  getReports,
  type Report,
  type ReportFact,
} from "@/lib/mc-api";

export function ReportsPanel() {
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState("");
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollers = useRef<Set<number>>(new Set());

  const refresh = async () => {
    try {
      setReports(await getReports());
    } catch (e) {
      setError(String(e));
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  // Poll any pending report until it settles.
  const pollUntilDone = (id: number) => {
    if (pollers.current.has(id)) return;
    pollers.current.add(id);
    const tick = async () => {
      try {
        const r = await getReport(id);
        setReports((prev) => prev.map((x) => (x.id === id ? r : x)));
        if (r.status === "pending") {
          setTimeout(tick, 3000);
        } else {
          pollers.current.delete(id);
          void refresh();
        }
      } catch {
        pollers.current.delete(id);
      }
    };
    setTimeout(tick, 2000);
  };

  const generate = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const req: { prompt: string; title?: string } = { prompt: prompt.trim() };
      if (title.trim()) req.title = title.trim();
      const r = await generateReport(req);
      setPrompt("");
      setTitle("");
      setSelectedId(r.id);
      await refresh();
      pollUntilDone(r.id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const selected = reports.find((r) => r.id === selectedId) ?? null;

  return (
    <div className="mx-auto w-full max-w-[860px] px-6 py-6">
      <h1 className="text-[18px] font-semibold">Reports</h1>
      <p className="mt-1 text-[12.5px] text-muted-foreground">
        Grounded summaries written from the indexed facts — generated automatically after each
        daily run, or on demand here.
      </p>

      <section className="mt-5 rounded-md border border-outline-subtle bg-surface-panel p-4">
        <div className="text-[13px] font-semibold">Generate a report</div>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Title (optional)"
          className="mt-2 w-full rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[12.5px]"
        />
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. Summarize today's energy and macro color, with the key figures."
          rows={3}
          className="mt-2 w-full resize-none rounded border border-outline-subtle bg-surface-input px-2.5 py-2 text-[12.5px] leading-relaxed"
        />
        <div className="mt-2 flex items-center gap-2">
          <Button onClick={generate} disabled={!prompt.trim() || busy}>
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
            Generate report
          </Button>
          {error && <span className="text-[11.5px] text-rose-300">{error}</span>}
        </div>
      </section>

      <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-[280px_1fr]">
        <div className="space-y-2">
          {reports.length === 0 && (
            <div className="rounded-md border border-outline-subtle bg-surface-panel px-3 py-4 text-center text-[12px] text-muted-foreground">
              No reports yet.
            </div>
          )}
          {reports.map((r) => (
            <button
              key={r.id}
              onClick={() => setSelectedId(r.id)}
              className={cn(
                "w-full rounded-md border px-3 py-2 text-left transition-colors",
                selectedId === r.id
                  ? "border-primary/40 bg-primary/[0.07]"
                  : "border-outline-subtle bg-surface-panel hover:border-primary/30",
              )}
            >
              <div className="flex items-center gap-1.5">
                <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium text-foreground">
                  {r.title}
                </span>
                <Badge variant="secondary" className="shrink-0 text-[9.5px] uppercase">
                  {r.kind}
                </Badge>
              </div>
              <div className="mt-1 flex items-center gap-1.5 text-[10.5px] text-muted-foreground">
                {r.status === "pending" && <Loader2 className="size-3 animate-spin text-primary" />}
                <span>{r.status}</span>
                {r.run_date && <span>· {r.run_date}</span>}
              </div>
            </button>
          ))}
        </div>

        <div>{selected ? <ReportDetail report={selected} /> : <EmptyDetail />}</div>
      </div>
    </div>
  );
}

function EmptyDetail() {
  return (
    <div className="grid h-full min-h-[160px] place-items-center rounded-md border border-outline-subtle bg-surface-panel text-[12px] text-muted-foreground">
      Select a report to read it.
    </div>
  );
}

function ReportDetail({ report }: { report: Report }) {
  const html = useMemo(
    () => DOMPurify.sanitize(marked.parse(report.body || "", { async: false })),
    [report.body],
  );

  if (report.status === "pending") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-outline-subtle bg-surface-panel px-3 py-4 text-[12.5px] text-muted-foreground">
        <Loader2 className="size-4 animate-spin text-primary" /> Generating…
      </div>
    );
  }
  if (report.status === "error") {
    return (
      <div className="rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-3 text-[12px] text-rose-200">
        {report.error || "Report generation failed."}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-outline-subtle bg-surface-panel px-4 py-3">
        <div className="text-[14px] font-semibold text-foreground">{report.title}</div>
        <div
          className={cn(
            "prose-chat mt-2 text-[13px] leading-relaxed text-foreground/90",
            "[&_h1]:mt-3 [&_h1]:text-[15px] [&_h2]:mt-3 [&_h2]:text-[14px] [&_h3]:text-[13px]",
            "[&_h1]:font-semibold [&_h2]:font-semibold [&_h3]:font-semibold",
            "[&_p]:my-1.5 [&_ul]:my-1.5 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-1.5 [&_ol]:list-decimal [&_ol]:pl-5 [&_li]:my-0.5",
            "[&_a]:text-primary [&_a]:underline [&_code]:rounded [&_code]:bg-surface-input [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[11.5px]",
          )}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </div>
      {report.facts?.length > 0 && <ReportFacts facts={report.facts} />}
    </div>
  );
}

function ReportFacts({ facts }: { facts: ReportFact[] }) {
  return (
    <div className="rounded-md border border-outline-subtle bg-surface-panel">
      <div className="border-b border-outline-subtle px-3 py-2 text-[11px] font-semibold text-foreground">
        {facts.length} sourced fact{facts.length === 1 ? "" : "s"}
      </div>
      <ul className="divide-y divide-outline-subtle">
        {facts.map((f, i) => (
          <li key={f.fact_id ?? i} className="px-3 py-2">
            <p className="text-[12px] leading-relaxed text-foreground/90">{f.claim}</p>
            <div className="mt-1 flex items-center gap-2 text-[10.5px] text-muted-foreground">
              <span className="truncate">
                {f.source_name ?? "unknown"}
                {f.published_date ? ` · ${f.published_date}` : ""}
              </span>
              {f.url && (
                <a
                  href={f.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-auto inline-flex shrink-0 items-center gap-1 text-primary hover:underline"
                >
                  source <ExternalLink className="size-3" />
                </a>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
