import { ChevronRight } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

export type MultiLevelMenuItem = {
  id: string;
  label: string;
  icon?: LucideIcon;
  badge?: ReactNode;
  active?: boolean;
  onSelect?: () => void;
  children?: MultiLevelMenuItem[];
};

export function MultiLevelCollapsibleMenu({
  items,
  defaultOpenIds = [],
  className,
}: {
  items: MultiLevelMenuItem[];
  defaultOpenIds?: string[];
  className?: string;
}) {
  const initialOpenIds = useMemo(() => new Set(defaultOpenIds), [defaultOpenIds]);
  const [openIds, setOpenIds] = useState(initialOpenIds);

  const toggle = (id: string) => {
    setOpenIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className={cn("flex flex-col gap-1", className)}>
      {items.map((item) => (
        <MenuNode
          key={item.id}
          item={item}
          level={0}
          openIds={openIds}
          onToggle={toggle}
        />
      ))}
    </div>
  );
}

function MenuNode({
  item,
  level,
  openIds,
  onToggle,
}: {
  item: MultiLevelMenuItem;
  level: number;
  openIds: Set<string>;
  onToggle: (id: string) => void;
}) {
  const hasChildren = Boolean(item.children?.length);
  const open = openIds.has(item.id);

  if (!hasChildren) {
    return (
      <button
        type="button"
        onClick={item.onSelect}
        className={cn(
          "group flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[12px] font-medium tracking-[-0.01em] transition-colors",
          item.active
            ? "bg-surface-active text-foreground"
            : "text-muted-foreground hover:bg-surface-hover hover:text-foreground",
        )}
        style={{ paddingLeft: level * 14 + 8 }}
      >
        {item.icon && (
          <item.icon
            className={cn(
              "size-3.5 shrink-0",
              item.active ? "text-primary" : "text-icon-muted",
            )}
          />
        )}
        <span className="min-w-0 flex-1 truncate">{item.label}</span>
        {item.badge && <span className="shrink-0">{item.badge}</span>}
      </button>
    );
  }

  return (
    <Collapsible open={open} onOpenChange={() => onToggle(item.id)}>
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className={cn(
            "group flex h-8 w-full items-center gap-2 rounded-md px-2 text-left text-[12px] font-semibold tracking-[-0.01em] transition-colors",
            item.active
              ? "bg-surface-active text-foreground"
              : "text-foreground/90 hover:bg-surface-hover",
          )}
          style={{ paddingLeft: level * 14 + 8 }}
        >
          <ChevronRight
            className={cn(
              "size-3.5 shrink-0 text-muted-foreground transition-transform",
              open && "rotate-90",
            )}
          />
          {item.icon && (
            <item.icon className="size-3.5 shrink-0 text-icon-muted" />
          )}
          <span className="min-w-0 flex-1 truncate">{item.label}</span>
          {item.badge && <span className="shrink-0">{item.badge}</span>}
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-0.5 flex flex-col gap-0.5 border-l border-outline-subtle/80 pl-1">
          {item.children?.map((child) => (
            <MenuNode
              key={child.id}
              item={child}
              level={level + 1}
              openIds={openIds}
              onToggle={onToggle}
            />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
