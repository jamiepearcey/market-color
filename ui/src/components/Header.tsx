import type { ReactNode } from "react";
import { Newspaper, Settings, BookOpen } from "lucide-react";

// Market Color header — chat-first, decoupled from the ETL template's store.
// The brand block replaces Celeritas; the action group exposes Settings + Docs.
export function Header({
  onOpenSettings,
  onHome,
}: {
  onOpenSettings?: () => void;
  onHome?: () => void;
}) {
  return (
    <header className="z-20 flex h-12 items-stretch gap-2 border-b border-outline-subtle bg-surface-header px-2 py-0.5">
      <button
        type="button"
        onClick={onHome}
        className="ml-0.5 flex h-11 items-center gap-2.5 rounded-md px-1 text-left transition-colors hover:bg-surface-hover"
      >
        <span
          aria-hidden="true"
          className="grid size-9 shrink-0 place-items-center rounded-md bg-gradient-to-br from-amber-400 to-orange-600 shadow-[0_0_12px_-2px] shadow-amber-500/50"
        >
          <Newspaper className="size-5 text-white" />
        </span>
        <div className="leading-tight">
          <div className="text-[16px] font-semibold tracking-tight">
            Market<span className="text-primary"> Color</span>
          </div>
          <div className="-mt-0.5 text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
            news corpus assistant
          </div>
        </div>
      </button>

      <div className="min-w-0 flex-1" />

      <div className="ml-auto flex h-11 items-center gap-1 rounded-md border border-outline-subtle bg-surface-toolbar p-1">
        <HeaderAction
          title="Documentation"
          label="Documentation"
          onClick={() =>
            window.open(
              "https://github.com/", // project README is the source of truth; see research/market-color/README.md
              "_blank",
              "noopener,noreferrer",
            )
          }
        >
          <BookOpen className="size-3.5" />
          <span className="hidden lg:inline">Docs</span>
        </HeaderAction>
        <HeaderAction title="Settings" label="Settings" onClick={onOpenSettings}>
          <Settings className="size-3.5" />
          <span className="hidden lg:inline">Settings</span>
        </HeaderAction>
      </div>
    </header>
  );
}

function HeaderAction({
  title,
  label,
  onClick,
  children,
}: {
  title: string;
  label: string;
  onClick?: (() => void) | undefined;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={label}
      onClick={onClick}
      className="inline-flex h-8 items-center justify-center gap-1.5 rounded-[5px] px-2 text-[12px] font-medium tracking-[-0.01em] text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45"
    >
      {children}
    </button>
  );
}
