import { FlaskConical } from "lucide-react";
import { useSnapshot } from "valtio";
import { store } from "@/lib/store";
import { cn } from "@/lib/utils";

/**
 * Honesty markers for the browser preview build, where runs/probes/topology are
 * synthesised by the `api.ts` mocks rather than produced by a real engine.
 * Render these next to any such result so a green outcome is never mistaken for
 * a real one (the #1 trust hazard called out in the systems review). Both
 * render `null` on desktop, so call sites can mount them unconditionally.
 */
export function SimulatedBadge({ className }: { className?: string }) {
  const snap = useSnapshot(store);
  if (snap.runtime.engineMode !== "preview") return null;
  return (
    <span
      title="Preview build: this result is synthesised, not produced by a real engine run."
      className={cn(
        "inline-flex items-center gap-1 rounded border border-amber-400/30 bg-amber-400/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-200",
        className,
      )}
    >
      <FlaskConical className="size-3" />
      Simulated
    </span>
  );
}

export function SimulatedNotice({
  children,
  className,
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  const snap = useSnapshot(store);
  if (snap.runtime.engineMode !== "preview") return null;
  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border border-amber-400/25 bg-amber-400/[0.07] px-3 py-2 text-[11px] leading-relaxed text-amber-200/90",
        className,
      )}
    >
      <FlaskConical className="mt-0.5 size-3.5 shrink-0" />
      <span>
        {children ??
          "Preview build — results are simulated, not produced by a real engine. Run the desktop app for live execution."}
      </span>
    </div>
  );
}
