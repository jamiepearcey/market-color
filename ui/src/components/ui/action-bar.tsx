import * as React from "react";
import { cn } from "@/lib/utils";

export function ActionBar({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      role="toolbar"
      className={cn(
        "inline-flex h-9 items-center gap-0.5 rounded-md border border-outline-subtle/80 bg-surface-input p-0.5",
        className,
      )}
      {...props}
    />
  );
}

export function ActionBarSeparator({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("mx-0.5 h-5 w-px bg-outline-subtle/80", className)} {...props} />;
}

export function ActionBarButton({
  tone = "default",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: "default" | "primary" | "danger" }) {
  return (
    <button
      className={cn(
        "inline-flex h-8 items-center justify-center gap-1.5 rounded-[5px] border px-2.5 text-[12px] font-medium tracking-[-0.01em] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-3.5 [&_svg]:shrink-0",
        tone === "default" &&
          "border-transparent text-muted-foreground hover:bg-surface-toolbar hover:text-foreground",
        tone === "primary" &&
          "border-transparent bg-primary/[0.14] text-foreground hover:bg-primary/[0.2]",
        tone === "danger" &&
          "border-transparent text-muted-foreground hover:bg-destructive/10 hover:text-destructive",
        className,
      )}
      {...props}
    />
  );
}
