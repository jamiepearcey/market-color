// DesksPanel — manage the desk taxonomy used to classify and steer retrieval.

import { useCallback, useEffect, useState } from "react";
import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getDesks, addDesk, deleteDesk, type Desk } from "@/lib/mc-api";

export function DesksPanel() {
  const [desks, setDesks] = useState<Desk[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [formKey, setFormKey] = useState<string>("");
  const [formLabel, setFormLabel] = useState<string>("");
  const [formAnchor, setFormAnchor] = useState<string>("");
  const [formError, setFormError] = useState<string | null>(null);
  const [saving, setSaving] = useState<boolean>(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const fetchDesks = useCallback(async () => {
    try {
      const data = await getDesks();
      setDesks(data);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Failed to load desks");
    }
  }, []);

  useEffect(() => {
    void fetchDesks();
  }, [fetchDesks]);

  const handleAdd = async () => {
    if (!formKey.trim()) {
      setFormError("Key (slug) is required.");
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      await addDesk({ key: formKey.trim(), label: formLabel.trim(), anchor: formAnchor.trim() });
      setFormKey("");
      setFormLabel("");
      setFormAnchor("");
      await fetchDesks();
    } catch (e) {
      setFormError(e instanceof Error ? e.message : "Failed to add desk");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: number) => {
    setDeletingId(id);
    try {
      await deleteDesk(id);
      await fetchDesks();
    } catch {
      // silently ignore — desk list will be unchanged
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="mx-auto w-full max-w-[860px] px-6 py-6">
      <h1 className="text-[18px] font-semibold text-foreground">Desks</h1>
      <p className="mt-1 text-[13px] text-muted-foreground">
        Desks are the taxonomy facts get tagged with; the anchor text is used to classify and steer
        retrieval.
      </p>

      {loadError && (
        <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
          {loadError}
        </div>
      )}

      {/* Add-desk form */}
      <div className="mt-5 rounded-md border border-outline-subtle bg-surface-panel px-4 py-4">
        <p className="mb-3 text-[12px] font-semibold text-foreground">Add desk</p>
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">
                Key <span className="text-muted-foreground/50">(slug)</span>
              </label>
              <input
                type="text"
                value={formKey}
                onChange={(e) => setFormKey(e.target.value)}
                placeholder="e.g. energy"
                className="w-36 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-[11px] font-medium text-muted-foreground">Label</label>
              <input
                type="text"
                value={formLabel}
                onChange={(e) => setFormLabel(e.target.value)}
                placeholder="e.g. Energy & Commodities"
                className="w-56 rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
              />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium text-muted-foreground">Anchor text</label>
            <textarea
              value={formAnchor}
              onChange={(e) => setFormAnchor(e.target.value)}
              rows={3}
              placeholder="Descriptive text used to classify and retrieve facts for this desk…"
              className="w-full resize-none rounded border border-outline-subtle bg-surface-input px-2.5 py-1.5 text-[13px] text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-primary/60"
            />
          </div>
          {formError && (
            <div className="rounded-md border border-rose-500/30 bg-rose-500/[0.08] px-3 py-2 text-[12px] text-rose-200">
              {formError}
            </div>
          )}
          <div>
            <Button size="sm" disabled={saving} onClick={() => void handleAdd()}>
              {saving ? "Adding…" : "Add desk"}
            </Button>
          </div>
        </div>
      </div>

      {/* Desk list */}
      <div className="mt-5 rounded-md border border-outline-subtle bg-surface-panel">
        <div className="border-b border-outline-subtle px-4 py-2.5 text-[11px] font-semibold text-foreground">
          {desks.length} desk{desks.length === 1 ? "" : "s"}
        </div>
        {desks.length === 0 ? (
          <div className="px-4 py-6 text-center text-[13px] text-muted-foreground">
            No desks yet — add one above.
          </div>
        ) : (
          <ul className="divide-y divide-outline-subtle">
            {desks.map((d) => (
              <li key={d.id} className="flex items-start gap-3 px-4 py-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary" className="font-mono text-[11px]">
                      {d.key}
                    </Badge>
                    <span className="text-[13px] font-medium text-foreground">{d.label}</span>
                  </div>
                  {d.anchor && (
                    <p className="mt-1 line-clamp-2 text-[12px] text-muted-foreground">
                      {d.anchor}
                    </p>
                  )}
                </div>
                <Button
                  size="icon"
                  variant="ghost"
                  disabled={deletingId === d.id}
                  onClick={() => void handleDelete(d.id)}
                  className="mt-0.5 shrink-0 text-muted-foreground hover:text-rose-300"
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
