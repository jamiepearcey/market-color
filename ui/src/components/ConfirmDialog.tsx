import * as Dialog from "@radix-ui/react-dialog";
import { useSnapshot } from "valtio";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { actions, store } from "@/lib/store";

/**
 * App-wide confirmation modal for risky/irreversible actions. Driven by the
 * store's `confirm` slice (project rule: Valtio over state hooks); the pending
 * callback is held in the store action layer, so any call site is a one-liner:
 * `actions.requestConfirm({ title, message, … }, () => actions.doThing())`.
 */
export function ConfirmDialog() {
  const snap = useSnapshot(store);
  const c = snap.confirm;
  const danger = c.tone === "danger";
  return (
    <Dialog.Root open={c.open} onOpenChange={(open) => { if (!open) void actions.resolveConfirm(false); }}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[100] bg-black/55 backdrop-blur-[1px] data-[state=open]:animate-in data-[state=open]:fade-in-0" />
        <Dialog.Content
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void actions.resolveConfirm(true);
            }
          }}
          className="fixed left-1/2 top-1/2 z-[101] w-[400px] max-w-[90vw] -translate-x-1/2 -translate-y-1/2 rounded-lg border border-outline-strong bg-surface-panel p-4 shadow-[0_24px_64px_rgba(0,0,0,0.5)] focus:outline-none"
        >
          <div className="flex items-start gap-3">
            {danger && (
              <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-md border border-destructive/40 bg-destructive/[0.12]">
                <AlertTriangle className="size-4 text-destructive" />
              </span>
            )}
            <div className="min-w-0">
              <Dialog.Title className="text-sm font-semibold tracking-[-0.01em]">{c.title}</Dialog.Title>
              <Dialog.Description className="mt-1 text-[12px] leading-relaxed text-muted-foreground">
                {c.message}
              </Dialog.Description>
            </div>
          </div>
          <div className="mt-4 flex justify-end gap-2">
            <Button size="sm" variant="outline" onClick={() => void actions.resolveConfirm(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              variant={danger ? "destructive" : "default"}
              autoFocus
              onClick={() => void actions.resolveConfirm(true)}
            >
              {c.confirmLabel}
            </Button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
