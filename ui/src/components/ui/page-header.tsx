import * as React from "react";
import { cn } from "@/lib/utils";

export function PageHeader({
  icon,
  title,
  meta,
  actions,
  className,
}: {
  icon?: React.ReactNode;
  title: React.ReactNode;
  meta?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex h-11 items-center gap-3 border-b border-outline-subtle px-4", className)}>
      {icon && <span className="shrink-0">{icon}</span>}
      <span className="shrink-0 text-sm font-semibold">{title}</span>
      {/* Meta absorbs/truncates the remaining width so long content can never
          push the actions off the right edge. */}
      {meta && <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">{meta}</div>}
      {actions && <div className={cn("flex shrink-0 items-center", !meta && "ml-auto")}>{actions}</div>}
    </div>
  );
}
