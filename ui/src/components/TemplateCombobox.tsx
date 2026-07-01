import { useEffect, useRef } from "react";
import { proxy, useSnapshot } from "valtio";
import { ChevronsUpDown, Search, Check, Star, Clock, Library, Boxes } from "lucide-react";
import { cn } from "@/lib/utils";
import { actions, store } from "@/lib/store";

/// Minimal shape the picker needs — keeps it assignable from a (deeply readonly)
/// Valtio snapshot of `Template[]`.
type TemplateOption = { name: string; displayName?: string; shortName?: string; description?: string | null; category: string };

/// A searchable template picker for ETL steps, organised into primary groups —
/// ⭐ Favorites, 🕘 Recently used, and the full category-grouped catalog. Local
/// UI state is a per-instance Valtio proxy (project rule: Valtio over state
/// hooks); the panel is fixed-positioned so it never clips inside the scroll area.
export function TemplateCombobox({
  value,
  onChange,
}: {
  value: string;
  onChange: (name: string) => void;
}) {
  const local = useRef(proxy({ open: false, query: "", top: 0, left: 0, width: 0 })).current;
  const snap = useSnapshot(local);
  const g = useSnapshot(store); // store-derived groups + favorites
  const btnRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  // The grouped catalog / favorites / recents are derived once in the store. With
  // no query we read those stable refs directly (zero work per render); only an
  // active search does any filtering — the one genuinely dynamic, local input.
  const q = snap.query.trim().toLowerCase();
  const match = (t: TemplateOption) => {
    if (!q) return true;
    const text = [t.shortName, t.displayName, t.name, t.category, t.description].filter(Boolean).join(" ").toLowerCase();
    return q
      .split(/[\s,._/-]+/)
      .filter(Boolean)
      .every((term) => text.includes(term) || fuzzyHit(term, text));
  };
  const favItems: readonly TemplateOption[] = q ? g.favoriteTemplates.filter(match) : g.favoriteTemplates;
  const recentItems: readonly TemplateOption[] = (q ? g.recentTemplates.filter(match) : g.recentTemplates).slice(0, 8);
  const catalog: readonly { cat: string; items: readonly TemplateOption[] }[] = q
    ? g.catalogGroups.map((grp) => ({ cat: grp.cat, items: grp.items.filter(match) })).filter((grp) => grp.items.length > 0)
    : g.catalogGroups;
  const total = catalog.reduce((n, grp) => n + grp.items.length, 0);
  const packs: readonly { pack: string; items: readonly TemplateOption[] }[] = q
    ? g.packGroups.map((grp) => ({ pack: grp.pack, items: grp.items.filter(match) })).filter((grp) => grp.items.length > 0)
    : g.packGroups;
  const packTotal = packs.reduce((n, grp) => n + grp.items.length, 0);

  function open() {
    const r = btnRef.current?.getBoundingClientRect();
    if (r) {
      local.top = r.bottom + 4;
      local.left = r.left;
      local.width = Math.max(r.width, 280);
    }
    local.query = "";
    local.open = true;
  }
  const close = () => (local.open = false);

  useEffect(() => {
    if (!snap.open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!btnRef.current?.contains(t) && !panelRef.current?.contains(t)) close();
    };
    const dismiss = (e: Event) => {
      const t = e.target as Node | null;
      if (t && panelRef.current?.contains(t)) return;
      close();
    };
    document.addEventListener("mousedown", onDown);
    window.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("scroll", dismiss, true);
      window.removeEventListener("resize", dismiss);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap.open]);

  const Row = (t: TemplateOption) => {
    const fav = g.favorites.includes(t.name);
    return (
      <div
        key={t.name}
        className={cn(
          "group/row flex items-center gap-1 pr-1 hover:bg-primary/[0.08]",
          t.name === value && "bg-primary/[0.08]",
        )}
      >
        <button
          type="button"
          onClick={() => {
            onChange(t.name);
            close();
          }}
          title={t.description ?? undefined}
          className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1 text-left font-mono text-[12px]"
        >
          {t.name === value ? (
            <Check className="size-3 shrink-0 text-primary" />
          ) : (
            <span className="size-3 shrink-0" />
          )}
          <span className="truncate" title={t.name}>{t.shortName || t.displayName || t.name}</span>
        </button>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            actions.toggleFavorite(t.name);
          }}
          title={fav ? "Unfavorite" : "Favorite"}
          className="shrink-0 rounded p-1 text-muted-foreground hover:bg-primary/[0.12] hover:text-primary"
        >
          <Star className={cn("size-3", fav && "fill-primary/35 text-primary")} />
        </button>
      </div>
    );
  };

  const SectionHeader = ({ icon, label, count }: { icon: React.ReactNode; label: string; count: number }) => (
    <div className="sticky top-0 z-10 flex items-center gap-1.5 bg-surface-header px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
      {icon}
      {label} <span className="opacity-50">({count})</span>
    </div>
  );

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        onClick={() => (snap.open ? close() : open())}
        className="flex h-8 min-w-0 flex-1 items-center justify-between gap-2 rounded-md border border-input bg-surface-input px-2 font-mono text-[12px] outline-none hover:bg-surface-hover"
      >
        <span className={cn("truncate", !value && "text-muted-foreground")}>{value || "select template…"}</span>
        <ChevronsUpDown className="size-3.5 shrink-0 opacity-60" />
      </button>

      {snap.open && (
        <div
          ref={panelRef}
          style={{ position: "fixed", top: snap.top, left: snap.left, width: snap.width }}
          className="z-50 overflow-hidden rounded-md border border-outline-strong bg-surface-panel shadow-[0_18px_52px_rgba(0,0,0,0.34)]"
        >
          <div className="flex items-center gap-2 border-b border-outline-subtle bg-surface-header px-2">
            <Search className="size-3.5 shrink-0 opacity-60" />
            <input
              autoFocus
              value={snap.query}
              onChange={(e) => (local.query = e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && close()}
              placeholder="Search templates…"
              className="h-8 w-full bg-transparent text-[12px] outline-none placeholder:text-muted-foreground"
            />
          </div>
          <div className="max-h-80 overflow-y-auto py-1">
            {favItems.length === 0 && recentItems.length === 0 && packTotal === 0 && total === 0 && (
              <div className="px-3 py-2 text-[12px] text-muted-foreground">No templates match “{snap.query}”.</div>
            )}
            {favItems.length > 0 && (
              <div className="mb-1">
                <SectionHeader icon={<Star className="size-3 text-primary" />} label="Favorites" count={favItems.length} />
                {favItems.map(Row)}
              </div>
            )}
            {recentItems.length > 0 && (
              <div className="mb-1">
                <SectionHeader icon={<Clock className="size-3" />} label="Recently used" count={recentItems.length} />
                {recentItems.map(Row)}
              </div>
            )}
            {packs.length > 0 && (
              <div className="mb-1">
                <SectionHeader icon={<Boxes className="size-3" />} label="Packs" count={packTotal} />
                {packs.map((grp) => (
                  <div key={grp.pack}>
                    <div className="bg-surface-toolbar px-3 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground/70">
                      {grp.pack} <span className="opacity-50">({grp.items.length})</span>
                    </div>
                    {grp.items.map(Row)}
                  </div>
                ))}
              </div>
            )}
            {catalog.length > 0 && (
              <div>
                <SectionHeader icon={<Library className="size-3" />} label="Full catalog" count={total} />
                {catalog.map((grp) => (
                  <div key={grp.cat}>
                    <div className="bg-surface-toolbar px-3 py-0.5 text-[10px] uppercase tracking-wide text-muted-foreground/70">
                      {grp.cat} <span className="opacity-50">({grp.items.length})</span>
                    </div>
                    {grp.items.map(Row)}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function fuzzyHit(term: string, value: string) {
  if (term.length < 3) return false;
  return value.split(/[\s,._/-]+/).some((part) => {
    let i = 0;
    for (const ch of part) if (ch === term[i]) i += 1;
    return i === term.length;
  });
}
