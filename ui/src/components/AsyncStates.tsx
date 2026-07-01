import type { ReactNode } from "react";
import { AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function LoadingPanel({
  label,
  className,
}: {
  label: string;
  className?: string;
}) {
  return (
    <div className={cn("grid place-items-center rounded-md border border-outline-subtle bg-surface-panel py-12 text-sm text-muted-foreground", className)}>
      <div className="flex items-center gap-2">
        <Loader2 className="size-4 animate-spin" /> {label}
      </div>
    </div>
  );
}

export function EmptyPanel({
  title,
  message,
  action,
  className,
}: {
  title: string;
  message: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("rounded-md border border-dashed border-outline-subtle bg-surface-panel p-8 text-center", className)}>
      <div className="text-sm font-semibold text-foreground">{title}</div>
      <p className="mt-1 text-sm text-muted-foreground">{message}</p>
      {action ? <div className="mt-4 flex justify-center">{action}</div> : null}
    </div>
  );
}

export function ErrorPanel({
  title,
  message,
  onRetry,
  className,
}: {
  title: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div className={cn("rounded-md border border-destructive/30 bg-destructive/[0.06] p-4", className)}>
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-foreground">{title}</div>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{message}</p>
          {onRetry ? (
            <Button size="sm" variant="outline" className="mt-3" onClick={onRetry}>
              <RefreshCw className="size-3.5" /> Retry
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export function SkeletonRows({
  count = 4,
  className,
}: {
  count?: number;
  className?: string;
}) {
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: count }, (_, idx) => (
        <div
          key={idx}
          className="h-12 animate-pulse rounded-md border border-outline-subtle bg-surface-toolbar"
        />
      ))}
    </div>
  );
}
