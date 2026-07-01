// RunsPanel — trigger and monitor daily news-scrape/decompose/index/report runs.

import { useCallback, useEffect, useRef, useState } from "react";
import { Calendar, Play, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getRuns, triggerRun, type Run, type RunStatus } from "@/lib/mc-api";

const ACTIVE_STATUSES: Set<RunStatus> = new Set([
  "pending",
  "scraping",
  "decomposing",
  "indexing",
  "reporting",
]);

function statusBadgeClass(status: RunStatus): string {
  if (status === "done") return "border-emerald-400/30 bg-emerald-400/[0.08] text-emerald-300";
  if (status === "error") return "border-rose-400/30 bg-rose-400/[0.08] text-rose-300";
  return "border-amber-400/30 bg-amber-400/[0.08] text-amber-300";
}

function statsLabel(stats: Record<string, unknown>): string {
  const picks = ["articles", "facts", "docs", "chunks"];
  const parts: string[] = [];
  for (const k of picks) {
    if (k in stats) parts.push(`${k}: ${stats[k]}`);
  }
  if (parts.length) return parts.join(", ");
  const keys = Object.keys(stats);
  if (!keys.length) return "—";
  return keys
    .slice(0, 4)
    .map((k) => `${k}: ${stats[k]}`)
    .join(", ");
}

function RunRow({ run }: { run: Run }) {
  return (
    <li className="px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[12px] text-foreground/90">{run.run_date}</span>
        <span className="rounded border border-outline-subtle bg-surface-toolbar px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          {run.kind}
        </span>
        <Badge
          className={cn(
            "border px-1.5 py-0.5 text-[11px] font-medium capitalize",
            statusBadgeClass(run.status),
          )}
        >
          {run.status}
        </Badge>
        {run.stage && (
          <span className="text-[11px] text-muted-foreground">{run.stage}</span>
        )}
        <span className="ml-auto font-mono text-[11px] text-muted-foreground">
          {statsLabel(run.stats)}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-muted-foreground/70">
        <span>started {run.started_at.slice(0, 19).replace("T", " ")}</span>
        {run.finished_at && (
          <span>finished {run.finished_at.slice(0, 19).replace("T", " ")}</span>
        )}
      </div>
      {run.error && (
        <div className="mt-2 rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
          {run.error}
        </div>
      )}
    </li>
  );
}

export function RunsPanel() {
  const today = new Date().toISOString().slice(0, 10);
  const [runs, setRuns] = useState<Run[]>([]);
  const [dateInput, setDateInput] = useState<string>(today);
  const [triggering, setTriggering] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchRuns = useCallback(async () => {
    try {
      const data = await getRuns();
      setRuns(data.sort((a, b) => b.id - a.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load runs");
    }
  }, []);

  // Polling: run while any active status exists
  useEffect(() => {
    void fetchRuns();
  }, [fetchRuns]);

  useEffect(() => {
    const hasActive = runs.some((r) => ACTIVE_STATUSES.has(r.status));
    if (hasActive) {
      if (!intervalRef.current) {
        intervalRef.current = setInterval(() => {
          void fetchRuns();
        }, 3000);
      }
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [runs, fetchRuns]);

  const handleTrigger = useCallback(
    async (date?: string) => {
      setTriggering(true);
      setError(null);
      try {
        await triggerRun(date);
        await fetchRuns();
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to trigger run");
      } finally {
        setTriggering(false);
      }
    },
    [fetchRuns],
  );

  return (
    <div className="mx-auto w-full max-w-[860px] px-6 py-6">
      <h1 className="text-[18px] font-semibold text-foreground">Daily Runs</h1>
      <p className="mt-1 text-[13px] text-muted-foreground">
        A daily run scrapes the configured RSS sources, decomposes articles into facts, indexes them
        into Qdrant, then writes a report — all under a dated partition.
      </p>

      {/* Control card */}
      <div className="mt-5 rounded-md border border-outline-subtle bg-surface-panel px-4 py-4">
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">Run for date</label>
            <div className="flex items-center gap-2">
              <input
                type="date"
                value={dateInput}
                onChange={(e) => setDateInput(e.target.value)}
                className="rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
              <Button
                size="sm"
                variant="secondary"
                disabled={triggering || !dateInput}
                onClick={() => void handleTrigger(dateInput)}
              >
                <Calendar className="size-3.5" />
                Run for date
              </Button>
            </div>
          </div>

          <Button
            disabled={triggering}
            onClick={() => void handleTrigger()}
            className="gap-1.5"
          >
            {triggering ? (
              <RefreshCw className="size-3.5 animate-spin" />
            ) : (
              <Play className="size-3.5" />
            )}
            Run today
          </Button>
        </div>

        {error && (
          <div className="mt-3 rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
            {error}
          </div>
        )}
      </div>

      {/* Run list */}
      <div className="mt-5 rounded-md border border-outline-subtle bg-surface-panel">
        <div className="flex items-center gap-2 border-b border-outline-subtle px-4 py-2.5 text-[11px] font-semibold text-foreground">
          {runs.length} run{runs.length === 1 ? "" : "s"}
        </div>
        {runs.length === 0 ? (
          <div className="px-4 py-6 text-center text-[13px] text-muted-foreground">
            No runs yet — trigger one above.
          </div>
        ) : (
          <ul className="divide-y divide-outline-subtle">
            {runs.map((r) => (
              <RunRow key={r.id} run={r} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
