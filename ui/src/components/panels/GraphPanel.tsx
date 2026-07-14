// GraphPanel — the "transmission map": entity co-occurrence + causal edges
// built from the indexed facts (GET /api/graph), laid out with a hand-rolled
// force simulation (src/lib/graph-layout.ts — no new deps). Node size ∝ fact
// count, edge width ∝ shared-fact weight; causal edges (from `cause_entities`,
// once the fact loads carry it) are directed, amber, and arrow-headed.
// Clicking a node opens a side panel with that entity's facts, each linking
// back to the source article.

import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { ExternalLink, Waypoints, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "@/components/AsyncStates";
import {
  getDesks,
  getEntityFacts,
  getGraph,
  type FactPayload,
  type Graph,
} from "@/lib/mc-api";
import { layoutGraph, type Layout } from "@/lib/graph-layout";

// Desk keys present in the corpus; extended with /api/desks when the DB is up.
const DEFAULT_DESKS = [
  "energy",
  "macro",
  "equities",
  "fx",
  "rates",
  "metals",
  "crypto",
  "geopolitics",
  "asia",
];

const LABELED_NODES = 30; // label only the biggest nodes to keep the map readable

export function GraphPanel() {
  const [desk, setDesk] = useState("energy");
  const [since, setSince] = useState("");
  const [until, setUntil] = useState("");
  const [deskOptions, setDeskOptions] = useState<string[]>(DEFAULT_DESKS);

  const [graph, setGraph] = useState<Graph | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    // Optional: merge DB-managed desks in. The graph works without Postgres.
    getDesks()
      .then((ds) =>
        setDeskOptions((prev) => [...new Set([...prev, ...ds.map((d) => d.key)])]),
      )
      .catch(() => {});
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setSelected(null);
    try {
      const params: Parameters<typeof getGraph>[0] = { maxNodes: 150 };
      if (desk) params.desk = desk;
      if (since) params.since = since;
      if (until) params.until = until;
      setGraph(await getGraph(params));
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
      setGraph(null);
    } finally {
      setLoading(false);
    }
  }, [desk, since, until]);

  useEffect(() => {
    void load();
  }, [load]);

  const layout = useMemo(() => (graph ? layoutGraph(graph) : null), [graph]);

  return (
    <div className="flex h-full min-h-0 w-full flex-col px-6 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-[18px] font-semibold text-foreground">
            <Waypoints className="size-4 text-primary" /> Transmission map
          </h1>
          <p className="mt-1 text-[12.5px] text-muted-foreground">
            Entities from the indexed facts; edges are shared-fact co-occurrence, amber arrows are
            causal links. Click a node to read its facts.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[12px]">
          <label className="flex items-center gap-1.5 text-muted-foreground">
            Desk
            <select
              value={desk}
              onChange={(e) => setDesk(e.target.value)}
              className="rounded border border-outline-subtle bg-surface-input px-2 py-1.5 text-[12px] text-foreground"
            >
              <option value="">all desks</option>
              {deskOptions.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-1.5 text-muted-foreground">
            From
            <input
              type="date"
              value={since}
              onChange={(e) => setSince(e.target.value)}
              className="rounded border border-outline-subtle bg-surface-input px-2 py-1 text-[12px] text-foreground"
            />
          </label>
          <label className="flex items-center gap-1.5 text-muted-foreground">
            To
            <input
              type="date"
              value={until}
              onChange={(e) => setUntil(e.target.value)}
              className="rounded border border-outline-subtle bg-surface-input px-2 py-1 text-[12px] text-foreground"
            />
          </label>
        </div>
      </div>

      <div className="mt-4 flex min-h-0 flex-1 gap-4">
        <div className="min-w-0 flex-1">
          {loading && <LoadingPanel label="Building the transmission map…" className="h-full" />}
          {!loading && error && (
            <ErrorPanel
              title="Could not build the graph"
              message={error}
              onRetry={() => void load()}
            />
          )}
          {!loading && !error && layout && layout.nodes.length === 0 && (
            <EmptyPanel
              title="No facts in range"
              message="No indexed facts match this desk / date range. Widen the range or pick another desk."
            />
          )}
          {!loading && !error && layout && layout.nodes.length > 0 && (
            <GraphSvg
              layout={layout}
              selected={selected}
              hovered={hovered}
              onHover={setHovered}
              onSelect={(id) => setSelected((cur) => (cur === id ? null : id))}
            />
          )}
        </div>

        {selected && (
          <EntityFactsPanel entity={selected} desk={desk} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  );
}

function GraphSvg({
  layout,
  selected,
  hovered,
  onHover,
  onSelect,
}: {
  layout: Layout;
  selected: string | null;
  hovered: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const labeled = useMemo(
    () =>
      new Set(
        [...layout.nodes]
          .sort((a, b) => b.count - a.count)
          .slice(0, LABELED_NODES)
          .map((n) => n.id),
      ),
    [layout],
  );
  const neighborhood = useMemo(() => {
    const active = selected ?? hovered;
    if (!active) return null;
    const set = new Set([active]);
    for (const e of layout.edges) {
      if (e.source.id === active) set.add(e.target.id);
      if (e.target.id === active) set.add(e.source.id);
    }
    return set;
  }, [layout, selected, hovered]);

  const active = selected ?? hovered;
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="h-full min-h-[420px] overflow-hidden rounded-md border border-outline-subtle bg-surface-panel"
    >
      <svg
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        className="h-full w-full"
        role="img"
        aria-label="Transmission map of fact entities"
        onClick={(e) => {
          if (e.target === e.currentTarget) onHover(null);
        }}
      >
        <defs>
          <marker
            id="graph-arrow"
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M0,0 L8,4 L0,8 z" className="fill-amber-400" />
          </marker>
        </defs>

        {layout.edges.map((e, i) => {
          const dimmed = neighborhood && !(neighborhood.has(e.source.id) && neighborhood.has(e.target.id));
          const onPath =
            active !== null && (e.source.id === active || e.target.id === active);
          // pull the causal arrowhead back to the target's rim
          const dx = e.target.x - e.source.x;
          const dy = e.target.y - e.source.y;
          const d = Math.sqrt(dx * dx + dy * dy) || 1;
          const tx = e.causal ? e.target.x - (dx / d) * (e.target.r + 3) : e.target.x;
          const ty = e.causal ? e.target.y - (dy / d) * (e.target.r + 3) : e.target.y;
          return (
            <line
              key={i}
              x1={e.source.x}
              y1={e.source.y}
              x2={tx}
              y2={ty}
              strokeWidth={Math.min(0.8 + Math.log2(e.weight + 1), 4.5)}
              strokeDasharray={e.causal ? "5 3" : undefined}
              markerEnd={e.causal ? "url(#graph-arrow)" : undefined}
              className={cn(
                e.causal ? "stroke-amber-400" : "stroke-muted-foreground",
                dimmed ? "opacity-[0.06]" : e.causal ? "opacity-80" : onPath ? "opacity-60" : "opacity-25",
              )}
            />
          );
        })}

        {layout.nodes.map((n) => {
          const dimmed = neighborhood && !neighborhood.has(n.id);
          const isActive = n.id === active;
          return (
            <g
              key={n.id}
              transform={`translate(${n.x},${n.y})`}
              className={cn("cursor-pointer", dimmed && "opacity-25")}
              onMouseEnter={() => onHover(n.id)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onSelect(n.id)}
            >
              <circle
                r={n.r}
                className={cn(
                  "transition-opacity",
                  n.id === selected
                    ? "fill-amber-400 stroke-amber-200"
                    : "fill-primary/70 stroke-primary",
                )}
                strokeWidth={isActive ? 2 : 1}
              />
              {(labeled.has(n.id) || isActive) && (
                <text
                  y={-n.r - 4}
                  textAnchor="middle"
                  className={cn(
                    "select-none fill-foreground",
                    isActive ? "font-semibold" : "opacity-80",
                  )}
                  fontSize={10}
                >
                  {n.id}
                </text>
              )}
              <title>{`${n.id} — ${n.count} fact${n.count === 1 ? "" : "s"}`}</title>
            </g>
          );
        })}
      </svg>
    </motion.div>
  );
}

function EntityFactsPanel({
  entity,
  desk,
  onClose,
}: {
  entity: string;
  desk: string;
  onClose: () => void;
}) {
  const [facts, setFacts] = useState<FactPayload[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setFacts(null);
    setError(null);
    try {
      const res = await getEntityFacts(entity, desk || undefined);
      setFacts(res.facts);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    }
  }, [entity, desk]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <motion.aside
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex w-[340px] shrink-0 flex-col overflow-hidden rounded-md border border-outline-subtle bg-surface-panel"
    >
      <div className="flex items-center gap-2 border-b border-outline-subtle px-3 py-2">
        <span className="min-w-0 flex-1 truncate text-[12.5px] font-semibold capitalize text-foreground">
          {entity}
        </span>
        {facts && (
          <span className="shrink-0 text-[10.5px] text-muted-foreground">
            {facts.length} fact{facts.length === 1 ? "" : "s"}
          </span>
        )}
        <button
          type="button"
          aria-label="Close entity panel"
          onClick={onClose}
          className="grid size-6 shrink-0 place-items-center rounded text-muted-foreground hover:bg-surface-hover hover:text-foreground"
        >
          <X className="size-3.5" />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {!facts && !error && <LoadingPanel label="Loading facts…" className="m-3" />}
        {error && (
          <ErrorPanel
            title="Could not load facts"
            message={error}
            onRetry={() => void load()}
            className="m-3"
          />
        )}
        {facts && facts.length === 0 && (
          <EmptyPanel
            title="No facts"
            message="No indexed facts mention this entity."
            className="m-3"
          />
        )}
        {facts && facts.length > 0 && (
          <ul className="divide-y divide-outline-subtle">
            {facts.map((f, i) => (
              <li key={f.fact_id ?? i} className="px-3 py-2.5">
                <p className="text-[12px] leading-relaxed text-foreground/90">{f.claim ?? "—"}</p>
                {f.cause && (
                  <p className="mt-1 text-[10.5px] text-amber-300/90">cause: {f.cause}</p>
                )}
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
        )}
      </div>
    </motion.aside>
  );
}
