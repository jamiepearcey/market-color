import { useEffect } from "react";
import { useSnapshot } from "valtio";
import { proxy } from "valtio";
import {
  BookOpenText,
  Circle,
  Database,
  LayoutGrid,
  MessageSquare,
  Rocket,
  Rss,
  Settings as SettingsIcon,
  FileText,
  Waypoints,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Header } from "@/components/Header";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { SettingsPanel } from "@/components/SettingsPanel";
import { ReportsPanel } from "@/components/panels/ReportsPanel";
import { RunsPanel } from "@/components/panels/RunsPanel";
import { SourcesPanel } from "@/components/panels/SourcesPanel";
import { DesksPanel } from "@/components/panels/DesksPanel";
import { GraphPanel } from "@/components/panels/GraphPanel";
import { BriefsPanel } from "@/components/panels/BriefsPanel";
import { chatStore, probeReady } from "@/lib/chat/store";

// Market Color is a chat-first market-intelligence client. A slim icon rail
// switches between the LLM chat and the management surfaces (reports, briefs,
// the transmission-map graph, daily runs, sources, desks) — all served by the
// API the client points at.
type View =
  | "chat"
  | "reports"
  | "briefs"
  | "graph"
  | "runs"
  | "sources"
  | "desks"
  | "settings";

const nav = proxy<{ view: View }>({ view: "chat" });

const RAIL: { view: View; label: string; Icon: typeof MessageSquare }[] = [
  { view: "chat", label: "Chat", Icon: MessageSquare },
  { view: "reports", label: "Reports", Icon: FileText },
  { view: "briefs", label: "Desk Briefs", Icon: BookOpenText },
  { view: "graph", label: "Transmission Map", Icon: Waypoints },
  { view: "runs", label: "Daily Runs", Icon: Rocket },
  { view: "sources", label: "Sources", Icon: Rss },
  { view: "desks", label: "Desks", Icon: LayoutGrid },
];

export default function App() {
  const snap = useSnapshot(chatStore);
  const { view } = useSnapshot(nav);
  const setView = (v: View) => (nav.view = v);

  useEffect(() => {
    void probeReady();
  }, []);

  return (
    <div className="relative flex h-screen flex-col overflow-hidden bg-surface-app">
      <Header onOpenSettings={() => setView("settings")} onHome={() => setView("chat")} />
      <div className="flex min-h-0 flex-1">
        <nav className="flex w-[56px] shrink-0 flex-col items-center gap-1 border-r border-outline-subtle bg-surface-header py-2">
          {RAIL.map(({ view: v, label, Icon }) => (
            <RailButton
              key={v}
              label={label}
              active={view === v}
              onClick={() => setView(v)}
              Icon={Icon}
            />
          ))}
          <div className="flex-1" />
          <RailButton
            label="Settings"
            active={view === "settings"}
            onClick={() => setView("settings")}
            Icon={SettingsIcon}
          />
        </nav>

        <main id="main-content" className="relative flex min-h-0 flex-1 flex-col overflow-y-auto bg-surface-app">
          {view === "chat" && <ChatPanel />}
          {view === "reports" && <ReportsPanel />}
          {view === "briefs" && <BriefsPanel />}
          {view === "graph" && <GraphPanel />}
          {view === "runs" && <RunsPanel />}
          {view === "sources" && <SourcesPanel />}
          {view === "desks" && <DesksPanel />}
          {view === "settings" && <SettingsPanel />}
        </main>
      </div>
      <StatusBar view={view} ready={snap.ready} model={snap.model} sending={snap.sending} />
    </div>
  );
}

function RailButton({
  label,
  active,
  onClick,
  Icon,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  Icon: typeof MessageSquare;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={cn(
        "grid size-10 place-items-center rounded-md transition-colors",
        active
          ? "bg-primary/[0.14] text-icon-tile-foreground"
          : "text-muted-foreground hover:bg-surface-hover hover:text-foreground",
      )}
    >
      <Icon className="size-[18px]" />
    </button>
  );
}

function StatusBar({
  view,
  ready,
  model,
  sending,
}: {
  view: View;
  ready: boolean | null;
  model: string;
  sending: boolean;
}) {
  const label = view.charAt(0).toUpperCase() + view.slice(1);
  return (
    <footer className="flex h-7 shrink-0 items-center gap-2 border-t border-outline-subtle bg-surface-header px-2 text-[11px] text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <Circle className="size-2 fill-current text-primary" />
        <span className="font-medium text-foreground">{label}</span>
      </span>
      <div className="min-w-0 flex-1" />
      {sending && <span className="hidden md:inline">working…</span>}
      <span className="flex items-center gap-1.5">
        <Database className="size-3.5" />
        <span className="font-mono">{model}</span>
      </span>
      <span className="h-3.5 w-px bg-outline-subtle" />
      <span className="flex items-center gap-1.5">
        <Circle
          className={`size-2 fill-current ${
            ready === true ? "text-emerald-400" : ready === false ? "text-amber-300/80" : "text-muted-foreground"
          }`}
        />
        <span>{ready === true ? "backend ready" : ready === false ? "backend offline" : "checking…"}</span>
      </span>
    </footer>
  );
}
