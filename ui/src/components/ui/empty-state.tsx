import type { LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface EmptyStateAction {
  label: string;
  onClick: () => void;
  icon?: LucideIcon;
  variant?: "default" | "outline";
}

/**
 * Shared first-run / empty-state panel. Every empty surface should answer
 * "what is this, and what do I do next?" — so this always pairs a one-line
 * concept with at least one actionable CTA (and an optional learn-more link),
 * replacing the dead-end "nothing here" boxes called out in the systems review.
 */
export function EmptyState({
  icon: Icon,
  title,
  description,
  primary,
  secondary,
  learnMore,
  className,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  primary?: EmptyStateAction;
  secondary?: EmptyStateAction;
  learnMore?: { label: string; onClick: () => void };
  className?: string;
}) {
  return (
    <div
      className={cn(
        "mx-auto flex max-w-md flex-col items-center rounded-lg border border-dashed border-outline-subtle bg-surface-app/40 px-6 py-10 text-center",
        className,
      )}
    >
      <span className="mb-3 grid size-10 place-items-center rounded-lg border border-transparent bg-primary/[0.14]">
        <Icon className="size-5 text-icon-tile-foreground" />
      </span>
      <div className="text-[14px] font-semibold tracking-[-0.01em] text-foreground">{title}</div>
      <p className="mt-1.5 text-[12px] leading-relaxed text-muted-foreground">{description}</p>
      {(primary || secondary) && (
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          {primary && (
            <Button size="sm" variant={primary.variant ?? "default"} onClick={primary.onClick}>
              {primary.icon && <primary.icon className="size-4" />}
              {primary.label}
            </Button>
          )}
          {secondary && (
            <Button size="sm" variant={secondary.variant ?? "outline"} onClick={secondary.onClick}>
              {secondary.icon && <secondary.icon className="size-4" />}
              {secondary.label}
            </Button>
          )}
        </div>
      )}
      {learnMore && (
        <button
          type="button"
          onClick={learnMore.onClick}
          className="mt-3 text-[11px] text-primary underline-offset-2 hover:underline"
        >
          {learnMore.label}
        </button>
      )}
    </div>
  );
}
