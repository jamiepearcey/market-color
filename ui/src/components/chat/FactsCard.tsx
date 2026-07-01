// FactsCard — renders the structured facts an answer was grounded on.
//
// This is the market-color analogue of the quant-algos result cards: instead of
// engine outputs it shows the facts the LLM retrieved from Qdrant via MCP, each
// linking back to the source article. Same visual language (surface-panel card,
// outline-subtle borders, desk/direction chips).

import { ArrowDownRight, ArrowRight, ArrowUpRight, ExternalLink, FileSearch } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Fact } from "@/lib/chat/model";

function DirectionChip({ direction }: { direction?: string }) {
  const d = direction ?? "neutral";
  const cfg =
    d === "bullish"
      ? { Icon: ArrowUpRight, cls: "text-emerald-300 border-emerald-400/30 bg-emerald-400/[0.08]" }
      : d === "bearish"
        ? { Icon: ArrowDownRight, cls: "text-rose-300 border-rose-400/30 bg-rose-400/[0.08]" }
        : { Icon: ArrowRight, cls: "text-muted-foreground border-outline-subtle bg-surface-toolbar" };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium capitalize",
        cfg.cls,
      )}
    >
      <cfg.Icon className="size-3" />
      {d}
    </span>
  );
}

export function FactsCard({ facts }: { facts: Fact[] }) {
  if (!facts.length) return null;
  return (
    <div className="rounded-md border border-outline-subtle bg-surface-panel">
      <div className="flex items-center gap-1.5 border-b border-outline-subtle px-3 py-2 text-[11px] font-semibold text-foreground">
        <FileSearch className="size-3.5 text-primary" />
        {facts.length} retrieved fact{facts.length === 1 ? "" : "s"}
        <span className="ml-1 font-normal text-muted-foreground">
          — grounded in the corpus, click through to verify
        </span>
      </div>
      <ul className="divide-y divide-outline-subtle">
        {facts.map((f, i) => (
          <li key={f.fact_id ?? i} className="px-3 py-2.5">
            <div className="flex items-start gap-2">
              <div className="min-w-0 flex-1">
                <p className="text-[12.5px] leading-relaxed text-foreground/90">{f.claim}</p>
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  {f.desk && (
                    <span className="rounded border border-outline-subtle bg-surface-toolbar px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                      {f.desk}
                    </span>
                  )}
                  <DirectionChip direction={f.direction ?? "neutral"} />
                  {f.metric && (
                    <span className="rounded border border-primary/30 bg-primary/[0.08] px-1.5 py-0.5 font-mono text-[10.5px] text-foreground/90">
                      {f.metric}
                    </span>
                  )}
                  {typeof f.score === "number" && (
                    <span className="font-mono text-[10px] text-muted-foreground/70">
                      {f.score.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-1.5 flex items-center gap-2 text-[10.5px] text-muted-foreground">
              <span className="truncate">
                {f.source_name ?? "unknown source"}
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
