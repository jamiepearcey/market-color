// SourcesPanel — manage RSS feeds and ingest ad-hoc documents.

import { useCallback, useEffect, useState } from "react";
import { Trash2, ToggleLeft, ToggleRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  getDesks,
  getSources,
  addSource,
  setSourceEnabled,
  deleteSource,
  uploadAdhoc,
  type Desk,
  type Source,
  type Run,
} from "@/lib/mc-api";

// ── RSS feed form ─────────────────────────────────────────────────────────────

interface FeedFormState {
  name: string;
  url: string;
  method: "rss" | "google_news_rss" | "scrape";
  scope: "global" | "focused";
  tier: number;
  access: "open" | "headline" | "paywall" | "licensed";
  desks: string[];
}

const FEED_FORM_DEFAULTS: FeedFormState = {
  name: "",
  url: "",
  method: "rss",
  scope: "global",
  tier: 2,
  access: "open",
  desks: [],
};

function DeskChips({
  all,
  selected,
  onToggle,
}: {
  all: Desk[];
  selected: string[];
  onToggle: (key: string) => void;
}) {
  if (!all.length)
    return <span className="text-[11px] text-muted-foreground/60">No desks configured.</span>;
  return (
    <div className="flex flex-wrap gap-1.5">
      {all.map((d) => {
        const active = selected.includes(d.key);
        return (
          <button
            key={d.key}
            type="button"
            onClick={() => onToggle(d.key)}
            className={cn(
              "rounded border px-2 py-0.5 text-[11px] font-medium transition-colors",
              active
                ? "border-primary/40 bg-primary/[0.12] text-primary"
                : "border-outline-subtle bg-surface-toolbar text-muted-foreground hover:border-primary/30 hover:text-foreground",
            )}
          >
            {d.key}
          </button>
        );
      })}
    </div>
  );
}

// ── Source row ────────────────────────────────────────────────────────────────

function SourceRow({
  source,
  desks,
  onToggle,
  onDelete,
  toggling,
  deleting,
}: {
  source: Source;
  desks: Desk[];
  onToggle: (id: number, enabled: boolean) => void;
  onDelete: (id: number) => void;
  toggling: boolean;
  deleting: boolean;
}) {
  const deskMap = new Map(desks.map((d) => [d.key, d]));
  return (
    <li className="flex items-start gap-3 px-4 py-3">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-[13px] font-medium text-foreground">{source.name}</span>
          <span className="font-mono text-[11px] text-muted-foreground/70 truncate max-w-[260px]">
            {source.url ?? "—"}
          </span>
        </div>
        {source.desks.length > 0 && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {source.desks.map((k) => (
              <Badge key={k} variant="secondary" className="text-[10px]">
                {deskMap.get(k)?.key ?? k}
              </Badge>
            ))}
          </div>
        )}
        <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-muted-foreground/60">
          <span>{source.method}</span>
          <span>{source.scope}</span>
          <span>tier {source.tier}</span>
          <span>{source.access}</span>
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-1.5 mt-0.5">
        <Button
          size="icon"
          variant="ghost"
          disabled={toggling}
          onClick={() => onToggle(source.id, source.enabled)}
          title={source.enabled ? "Disable" : "Enable"}
          className={cn(
            "text-muted-foreground",
            source.enabled ? "text-emerald-400 hover:text-emerald-300" : "hover:text-foreground",
          )}
        >
          {source.enabled ? (
            <ToggleRight className="size-4" />
          ) : (
            <ToggleLeft className="size-4" />
          )}
        </Button>
        <Button
          size="icon"
          variant="ghost"
          disabled={deleting}
          onClick={() => onDelete(source.id)}
          className="text-muted-foreground hover:text-rose-300"
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>
    </li>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function SourcesPanel() {
  const [desks, setDesks] = useState<Desk[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Feed form
  const [feed, setFeed] = useState<FeedFormState>(FEED_FORM_DEFAULTS);
  const [feedError, setFeedError] = useState<string | null>(null);
  const [addingFeed, setAddingFeed] = useState<boolean>(false);

  // Source row states
  const [togglingId, setTogglingId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  // Adhoc form
  const [adhocTitle, setAdhocTitle] = useState<string>("");
  const [adhocBody, setAdhocBody] = useState<string>("");
  const [adhocUrl, setAdhocUrl] = useState<string>("");
  const [adhocDate, setAdhocDate] = useState<string>("");
  const [adhocDesks, setAdhocDesks] = useState<string[]>([]);
  const [adhocError, setAdhocError] = useState<string | null>(null);
  const [adhocSuccess, setAdhocSuccess] = useState<string | null>(null);
  const [ingesting, setIngesting] = useState<boolean>(false);

  const fetchAll = useCallback(async () => {
    try {
      const [d, s] = await Promise.all([getDesks(), getSources()]);
      setDesks(d);
      setSources(s);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Failed to load data");
    }
  }, []);

  useEffect(() => {
    void fetchAll();
  }, [fetchAll]);

  const toggleDeskInFeed = (key: string) => {
    setFeed((prev) => ({
      ...prev,
      desks: prev.desks.includes(key)
        ? prev.desks.filter((k) => k !== key)
        : [...prev.desks, key],
    }));
  };

  const handleAddFeed = async () => {
    if (!feed.name.trim() || !feed.url.trim()) {
      setFeedError("Name and URL are required.");
      return;
    }
    setAddingFeed(true);
    setFeedError(null);
    try {
      await addSource({
        name: feed.name.trim(),
        url: feed.url.trim(),
        method: feed.method,
        scope: feed.scope,
        tier: feed.tier,
        access: feed.access,
        desks: feed.desks,
      });
      setFeed(FEED_FORM_DEFAULTS);
      await fetchAll();
    } catch (e) {
      setFeedError(e instanceof Error ? e.message : "Failed to add source");
    } finally {
      setAddingFeed(false);
    }
  };

  const handleToggle = async (id: number, currentEnabled: boolean) => {
    setTogglingId(id);
    try {
      await setSourceEnabled(id, !currentEnabled);
      await fetchAll();
    } catch {
      // ignore
    } finally {
      setTogglingId(null);
    }
  };

  const handleDelete = async (id: number) => {
    setDeletingId(id);
    try {
      await deleteSource(id);
      await fetchAll();
    } catch {
      // ignore
    } finally {
      setDeletingId(null);
    }
  };

  const toggleAdhocDesk = (key: string) => {
    setAdhocDesks((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  const handleIngest = async () => {
    if (!adhocTitle.trim() || !adhocBody.trim()) {
      setAdhocError("Title and body are required.");
      return;
    }
    setIngesting(true);
    setAdhocError(null);
    setAdhocSuccess(null);
    try {
      const payload: {
        title: string;
        body: string;
        url?: string;
        published_date?: string;
        desks?: string[];
      } = {
        title: adhocTitle.trim(),
        body: adhocBody.trim(),
      };
      if (adhocUrl.trim()) payload.url = adhocUrl.trim();
      if (adhocDate.trim()) payload.published_date = adhocDate.trim();
      if (adhocDesks.length) payload.desks = adhocDesks;

      const result = await uploadAdhoc(payload);
      const run = result.run as Run;
      setAdhocSuccess(`Ingest run #${run.id} started — check Daily Runs.`);
      setAdhocTitle("");
      setAdhocBody("");
      setAdhocUrl("");
      setAdhocDate("");
      setAdhocDesks([]);
    } catch (e) {
      setAdhocError(e instanceof Error ? e.message : "Failed to ingest document");
    } finally {
      setIngesting(false);
    }
  };

  const rssSources = sources.filter((s) => s.kind === "rss");

  return (
    <div className="mx-auto w-full max-w-[860px] px-6 py-6">
      <h1 className="text-[18px] font-semibold text-foreground">Sources</h1>
      <p className="mt-1 text-[13px] text-muted-foreground">
        Configure RSS feeds for the daily crawler, or ingest ad-hoc documents directly into the
        corpus.
      </p>

      {loadError && (
        <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
          {loadError}
        </div>
      )}

      {/* ── Section A: RSS feeds ── */}
      <h2 className="mt-7 text-[14px] font-semibold text-foreground">RSS feeds</h2>

      {/* Add feed form */}
      <div className="mt-3 rounded-md border border-outline-subtle bg-surface-panel px-4 py-4">
        <p className="mb-3 text-[12px] font-semibold text-foreground">Add feed</p>
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">Name</label>
              <input
                type="text"
                value={feed.name}
                onChange={(e) => setFeed((p) => ({ ...p, name: e.target.value }))}
                placeholder="Reuters Energy"
                className="w-44 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">URL</label>
              <input
                type="url"
                value={feed.url}
                onChange={(e) => setFeed((p) => ({ ...p, url: e.target.value }))}
                placeholder="https://feeds.reuters.com/…"
                className="w-64 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">Method</label>
              <select
                value={feed.method}
                onChange={(e) =>
                  setFeed((p) => ({
                    ...p,
                    method: e.target.value as FeedFormState["method"],
                  }))
                }
                className="rounded border border-outline-subtle bg-surface-input px-2 py-1.5 text-[13px] text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"
              >
                <option value="rss">rss</option>
                <option value="google_news_rss">google_news_rss</option>
                <option value="scrape">scrape</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">Scope</label>
              <select
                value={feed.scope}
                onChange={(e) =>
                  setFeed((p) => ({
                    ...p,
                    scope: e.target.value as FeedFormState["scope"],
                  }))
                }
                className="rounded border border-outline-subtle bg-surface-input px-2 py-1.5 text-[13px] text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"
              >
                <option value="global">global</option>
                <option value="focused">focused</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">Tier</label>
              <input
                type="number"
                value={feed.tier}
                onChange={(e) =>
                  setFeed((p) => ({ ...p, tier: parseInt(e.target.value, 10) || 1 }))
                }
                min={1}
                max={10}
                className="w-16 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">Access</label>
              <select
                value={feed.access}
                onChange={(e) =>
                  setFeed((p) => ({
                    ...p,
                    access: e.target.value as FeedFormState["access"],
                  }))
                }
                className="rounded border border-outline-subtle bg-surface-input px-2 py-1.5 text-[13px] text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"
              >
                <option value="open">open</option>
                <option value="headline">headline</option>
                <option value="paywall">paywall</option>
                <option value="licensed">licensed</option>
              </select>
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">Desks</label>
            <DeskChips all={desks} selected={feed.desks} onToggle={toggleDeskInFeed} />
          </div>
          {feedError && (
            <div className="rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
              {feedError}
            </div>
          )}
          <div>
            <Button size="sm" disabled={addingFeed} onClick={() => void handleAddFeed()}>
              {addingFeed ? "Adding…" : "Add feed"}
            </Button>
          </div>
        </div>
      </div>

      {/* RSS source list */}
      <div className="mt-4 rounded-md border border-outline-subtle bg-surface-panel">
        <div className="border-b border-outline-subtle px-4 py-2.5 text-[11px] font-semibold text-foreground">
          {rssSources.length} RSS feed{rssSources.length === 1 ? "" : "s"}
        </div>
        {rssSources.length === 0 ? (
          <div className="px-4 py-5 text-center text-[13px] text-muted-foreground">
            No RSS sources yet.
          </div>
        ) : (
          <ul className="divide-y divide-outline-subtle">
            {rssSources.map((s) => (
              <SourceRow
                key={s.id}
                source={s}
                desks={desks}
                onToggle={(id, enabled) => void handleToggle(id, enabled)}
                onDelete={(id) => void handleDelete(id)}
                toggling={togglingId === s.id}
                deleting={deletingId === s.id}
              />
            ))}
          </ul>
        )}
      </div>

      {/* ── Section B: Ad-hoc upload ── */}
      <h2 className="mt-8 text-[14px] font-semibold text-foreground">Ad-hoc upload</h2>
      <div className="mt-3 rounded-md border border-outline-subtle bg-surface-panel px-4 py-4">
        <p className="mb-3 text-[12px] font-semibold text-foreground">Ingest document</p>
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">Title</label>
            <input
              type="text"
              value={adhocTitle}
              onChange={(e) => setAdhocTitle(e.target.value)}
              placeholder="Goldman Sachs Oil Note — Jun 2026"
              className="w-full rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">Body</label>
            <textarea
              value={adhocBody}
              onChange={(e) => setAdhocBody(e.target.value)}
              rows={6}
              placeholder="Paste the full text of the document here…"
              className="w-full resize-y rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
            />
          </div>
          <div className="flex flex-wrap gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">
                URL <span className="text-muted-foreground/50">(optional)</span>
              </label>
              <input
                type="url"
                value={adhocUrl}
                onChange={(e) => setAdhocUrl(e.target.value)}
                placeholder="https://…"
                className="w-64 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">
                Published date <span className="text-muted-foreground/50">(optional)</span>
              </label>
              <input
                type="date"
                value={adhocDate}
                onChange={(e) => setAdhocDate(e.target.value)}
                className="rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">Desks</label>
            <DeskChips all={desks} selected={adhocDesks} onToggle={toggleAdhocDesk} />
          </div>
          {adhocError && (
            <div className="rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
              {adhocError}
            </div>
          )}
          {adhocSuccess && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/[0.08] px-3 py-2 text-[12px] text-emerald-300">
              {adhocSuccess}
            </div>
          )}
          <div>
            <Button size="sm" disabled={ingesting} onClick={() => void handleIngest()}>
              {ingesting ? "Ingesting…" : "Ingest"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
