import { useSnapshot } from "valtio";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, CircleX, Info, X } from "lucide-react";
import { actions, store, type ToastTone } from "@/lib/store";
import { cn } from "@/lib/utils";

/**
 * App-wide toast stack for action feedback (project rule: Valtio over state
 * hooks). Driven by the store's `toasts` queue; any mutating action confirms
 * itself or surfaces an error via `actions.notify(...)` instead of failing
 * silently. Rendered once at app root next to `<ConfirmDialog/>`.
 */
const toneStyles: Record<
  ToastTone,
  { icon: typeof Info; ring: string; accent: string }
> = {
  ok: { icon: CheckCircle2, ring: "border-emerald-500/30", accent: "text-emerald-400" },
  error: { icon: CircleX, ring: "border-destructive/40", accent: "text-destructive" },
  info: { icon: Info, ring: "border-outline-strong", accent: "text-primary" },
};

export function Toaster() {
  const snap = useSnapshot(store);
  return (
    <div className="pointer-events-none fixed bottom-9 right-3 z-[120] flex w-[340px] max-w-[90vw] flex-col gap-2">
      <AnimatePresence initial={false}>
        {snap.toasts.map((t) => {
          const s = toneStyles[t.tone];
          const Icon = s.icon;
          return (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, y: 8, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 4, scale: 0.98 }}
              transition={{ duration: 0.16 }}
              role="status"
              aria-live={t.tone === "error" ? "assertive" : "polite"}
              className={cn(
                "pointer-events-auto flex items-start gap-2.5 rounded-lg border bg-surface-panel px-3 py-2.5 shadow-[0_16px_40px_rgba(0,0,0,0.45)]",
                s.ring,
              )}
            >
              <Icon className={cn("mt-0.5 size-4 shrink-0", s.accent)} />
              <div className="min-w-0 flex-1">
                {t.title && (
                  <div className="text-[12px] font-semibold leading-tight tracking-[-0.01em]">
                    {t.title}
                  </div>
                )}
                <div
                  className={cn(
                    "text-[12px] leading-snug text-muted-foreground",
                    t.title && "mt-0.5",
                  )}
                >
                  {t.message}
                </div>
              </div>
              <button
                type="button"
                aria-label="Dismiss"
                onClick={() => actions.dismissToast(t.id)}
                className="rounded p-0.5 text-muted-foreground/70 transition-colors hover:text-foreground"
              >
                <X className="size-3.5" />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
