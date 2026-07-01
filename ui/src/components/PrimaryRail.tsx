import {
  Activity,
  ArrowDownToLine,
  ArrowUpFromLine,
  CalendarClock,
  Settings,
  Store,
  type LucideIcon,
} from "lucide-react";
import { useSnapshot } from "valtio";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { actions, store } from "@/lib/store";

type NavItem = {
  id: string;
  label: string;
  icon: LucideIcon;
  active: boolean;
  badge?: number;
  onSelect: () => void;
};

export function PrimaryRail({
  narrow = false,
  open = false,
  onClose,
}: {
  narrow?: boolean;
  open?: boolean;
  onClose?: () => void;
}) {
  const snap = useSnapshot(store);
  const onConnectors = snap.view === "connectors";

  const items: NavItem[] = [
    {
      id: "sources",
      label: "Sources",
      icon: ArrowDownToLine,
      active: onConnectors && snap.connectorRole === "source",
      badge: snap.sourceInstances.length,
      onSelect: () => actions.setConnectorRole("source"),
    },
    {
      id: "targets",
      label: "Targets",
      icon: ArrowUpFromLine,
      active: onConnectors && snap.connectorRole === "target",
      badge: snap.targetInstances.length,
      onSelect: () => actions.setConnectorRole("target"),
    },
    {
      id: "observability",
      label: "Observability",
      icon: Activity,
      active: snap.view === "observability",
      onSelect: () => actions.setView("observability"),
    },
    {
      id: "store",
      label: "Store",
      icon: Store,
      active: snap.view === "store",
      onSelect: () => actions.openStore(),
    },
    {
      id: "jobs",
      label: "Scheduled Jobs",
      icon: CalendarClock,
      active: snap.view === "jobs",
      badge: snap.jobs.length,
      onSelect: () => actions.setView("jobs"),
    },
    {
      id: "settings",
      label: "Settings",
      icon: Settings,
      active: snap.view === "settings",
      onSelect: () => actions.setView("settings"),
    },
  ];

  const rail = (
    <nav
      aria-label="Primary"
      className="flex w-[214px] shrink-0 flex-col border-r border-outline-subtle bg-surface-header"
    >
      <div className="border-b border-outline-subtle px-3 py-2">
        <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Navigation
        </div>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-2">
        <ul className="flex flex-col gap-0.5">
          {items.map((item) => {
            const Icon = item.icon;
            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => {
                    item.onSelect();
                    onClose?.();
                  }}
                  aria-current={item.active ? "page" : undefined}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13px] font-medium tracking-[-0.01em] transition-colors",
                    item.active
                      ? "bg-surface-active text-foreground"
                      : "text-muted-foreground hover:bg-surface-hover hover:text-foreground",
                  )}
                >
                  <Icon
                    className={cn(
                      "size-4 shrink-0",
                      item.active && "text-icon-tile-foreground",
                    )}
                  />
                  <span className="min-w-0 flex-1 truncate">{item.label}</span>
                  {item.badge != null && item.badge > 0 && (
                    <Badge variant="secondary" className="h-4 px-1 text-[9px]">
                      {item.badge}
                    </Badge>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </nav>
  );

  if (!narrow) return rail;
  if (!open) return null;
  return (
    <>
      <button
        type="button"
        aria-label="Close navigation"
        className="fixed inset-0 z-30 bg-black/40"
        onClick={onClose}
      />
      <div
        className={cn(
          "fixed inset-y-0 left-0 z-40 transition-transform duration-200",
          "translate-x-0",
        )}
      >
        {rail}
      </div>
    </>
  );
}
