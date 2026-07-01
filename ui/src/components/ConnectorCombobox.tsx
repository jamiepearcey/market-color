import { useEffect, useRef } from "react";
import { proxy, useSnapshot } from "valtio";
import { ChevronsUpDown, Search, Check, X, Plug } from "lucide-react";
import { cn } from "@/lib/utils";
import { store } from "@/lib/store";
import { Badge } from "@/components/ui/badge";
import { iconForCategory } from "@/lib/connector-icons";
import type { ConnectorSpec } from "@/lib/types";

/** Minimal shape the picker needs — assignable from a (deeply readonly) Valtio
 *  snapshot of `ConnectorInstance[]`. */
type Option = {
  id: string;
  name: string;
  spec: string;
  driver: string;
  params: Record<string, string>;
};

/** Non-secret, configured params surfaced as compact "config tags" so two
 *  instances of the same connector are distinguishable at a glance. */
function configTags(
  inst: Option,
  spec: ConnectorSpec | undefined,
): { label: string; value: string }[] {
  if (!spec) return [];
  return spec.params
    .filter((p) => p.kind !== "secret")
    .map((p) => ({
      label: p.label || p.name,
      value: (inst.params[p.name] ?? "").trim(),
    }))
    .filter((t) => t.value !== "");
}

/**
 * A searchable connector picker aligned to the template search popover: a
 * fixed-positioned panel (never clips in a scroll area), per-instance Valtio
 * proxy for local UI state (project rule: Valtio over state hooks), instances
 * grouped under the connector catalog's category headers + icons, each row
 * badged with its driver and any configured params.
 */
export function ConnectorCombobox({
  instances,
  selectedId,
  onSelect,
  placeholder,
}: {
  instances: readonly Option[];
  selectedId?: string | undefined;
  onSelect: (id: string) => void;
  placeholder: string;
}) {
  const local = useRef(
    proxy({ open: false, query: "", top: 0, left: 0, width: 0 }),
  ).current;
  const snap = useSnapshot(local);
  const g = useSnapshot(store);
  const btnRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  const specOf = (name: string): ConnectorSpec | undefined =>
    (g.connectorSpecs as ConnectorSpec[]).find((s) => s.name === name);

  const selected = instances.find((c) => c.id === selectedId);
  const selectedSpec = selected ? specOf(selected.spec) : undefined;
  const selectedTags = selected ? configTags(selected, selectedSpec) : [];

  const q = snap.query.trim().toLowerCase();
  const match = (c: Option) => {
    if (!q) return true;
    const spec = specOf(c.spec);
    const text = [c.name, c.driver, spec?.category, ...Object.values(c.params)]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return q
      .split(/[\s,._/-]+/)
      .filter(Boolean)
      .every((term) => text.includes(term));
  };

  // Group the (filtered) instances under their spec's category, matching the
  // connector catalog's grouping + ordering.
  const groups: { category: string; items: Option[] }[] = [];
  for (const c of instances) {
    if (!match(c)) continue;
    const category = specOf(c.spec)?.category ?? "Other";
    (
      groups.find((grp) => grp.category === category) ??
      groups[groups.push({ category, items: [] }) - 1]
    ).items.push(c);
  }
  groups.sort((a, b) => a.category.localeCompare(b.category));
  const total = groups.reduce((n, grp) => n + grp.items.length, 0);

  function open() {
    const r = btnRef.current?.getBoundingClientRect();
    if (r) {
      local.top = r.bottom + 4;
      local.left = r.left;
      local.width = Math.max(r.width, 320);
    }
    local.query = "";
    local.open = true;
  }
  const close = () => (local.open = false);
  const pick = (id: string) => {
    onSelect(id);
    close();
  };

  useEffect(() => {
    if (!snap.open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!btnRef.current?.contains(t) && !panelRef.current?.contains(t))
        close();
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

  const TriggerIcon = selectedSpec
    ? iconForCategory(selectedSpec.category)
    : Plug;

  const Row = (c: Option) => {
    const spec = specOf(c.spec);
    const CatIcon = iconForCategory(spec?.category ?? "");
    const tags = configTags(c, spec);
    const isSel = c.id === selectedId;
    return (
      <button
        key={c.id}
        type="button"
        onClick={() => pick(c.id)}
        className={cn(
          "flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-primary/[0.08]",
          isSel && "bg-primary/[0.08]",
        )}
      >
        {isSel ? (
          <Check className="size-3.5 shrink-0 text-primary" />
        ) : (
          <CatIcon className="size-3.5 shrink-0 text-muted-foreground" />
        )}
        <span
          className="min-w-0 flex-1 truncate text-[12px] tracking-[-0.01em]"
          title={c.name}
        >
          {c.name}
        </span>
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground/70">
          {c.driver}
        </span>
        {tags.slice(0, 2).map((t) => (
          <Badge
            key={t.label}
            variant="outline"
            className="shrink-0 max-w-[120px] px-1 py-0 text-[9px]"
            title={`${t.label}: ${t.value}`}
          >
            <span className="truncate">{t.value}</span>
          </Badge>
        ))}
        {tags.length > 2 && (
          <Badge variant="outline" className="shrink-0 px-1 py-0 text-[9px]">
            +{tags.length - 2}
          </Badge>
        )}
        {spec?.exampleUnresolved && (
          <Badge
            variant="outline"
            className="shrink-0 px-1 py-0 text-[9px] text-amber-300/90"
          >
            example
          </Badge>
        )}
        {g.connectorPack[c.spec] && (
          <Badge
            variant="secondary"
            className="shrink-0 px-1 py-0 text-[9px]"
            title={`From pack: ${g.connectorPack[c.spec]}`}
          >
            {g.connectorPack[c.spec]}
          </Badge>
        )}
      </button>
    );
  };

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        onClick={() => (snap.open ? close() : open())}
        className="flex h-8 min-w-0 flex-1 items-center justify-between gap-2 rounded-md border border-input bg-surface-input px-2 text-[12px] outline-none hover:bg-surface-hover"
      >
        {selected ? (
          <span className="flex min-w-0 items-center gap-2">
            <TriggerIcon className="size-3.5 shrink-0 text-muted-foreground" />
            <span className="truncate" title={selected.name}>
              {selected.name}
            </span>
            {selectedTags[0] && (
              <Badge
                variant="secondary"
                className="shrink-0 max-w-[140px] px-1 py-0 text-[9px]"
                title={`${selectedTags[0].label}: ${selectedTags[0].value}`}
              >
                <span className="truncate">{selectedTags[0].value}</span>
              </Badge>
            )}
          </span>
        ) : (
          <span className="truncate text-muted-foreground">{placeholder}</span>
        )}
        <ChevronsUpDown className="size-3.5 shrink-0 opacity-60" />
      </button>

      {snap.open && (
        <div
          ref={panelRef}
          style={{
            position: "fixed",
            top: snap.top,
            left: snap.left,
            width: snap.width,
          }}
          className="z-50 overflow-hidden rounded-md border border-outline-strong bg-surface-panel shadow-[0_18px_52px_rgba(0,0,0,0.34)]"
        >
          <div className="flex items-center gap-2 border-b border-outline-subtle bg-surface-header px-2">
            <Search className="size-3.5 shrink-0 opacity-60" />
            <input
              autoFocus
              value={snap.query}
              onChange={(e) => (local.query = e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && close()}
              placeholder="Search connectors…"
              className="h-8 w-full bg-transparent text-[12px] outline-none placeholder:text-muted-foreground"
            />
          </div>
          <div className="max-h-80 overflow-y-auto py-1">
            {/* Clear selection */}
            <button
              type="button"
              onClick={() => pick("")}
              className={cn(
                "flex w-full items-center gap-2 px-2 py-1.5 text-left text-[12px] hover:bg-primary/[0.08]",
                !selectedId && "bg-primary/[0.08]",
              )}
            >
              {!selectedId ? (
                <Check className="size-3.5 shrink-0 text-primary" />
              ) : (
                <X className="size-3.5 shrink-0 text-muted-foreground" />
              )}
              <span className="text-muted-foreground">None</span>
            </button>

            {total === 0 && (
              <div className="px-3 py-2 text-[12px] text-muted-foreground">
                {snap.query
                  ? `No connectors match “${snap.query}”.`
                  : "No enabled connectors."}
              </div>
            )}

            {groups.map((grp) => {
              const CatIcon = iconForCategory(grp.category);
              return (
                <div key={grp.category}>
                  <div className="sticky top-0 z-10 flex items-center gap-1.5 bg-surface-header px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    <CatIcon className="size-3" />
                    {grp.category}{" "}
                    <span className="opacity-50">({grp.items.length})</span>
                  </div>
                  {grp.items.map(Row)}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}
