import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cn } from "@/lib/utils";

export const ButtonGroup = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    orientation?: "horizontal" | "vertical";
    selectionMode?: "single" | "none";
  }
>(({ className, orientation = "horizontal", selectionMode = "single", onKeyDown, ...props }, ref) => (
  <div
    ref={ref}
    role={selectionMode === "single" ? "radiogroup" : "group"}
    aria-orientation={orientation}
    data-orientation={orientation}
    className={cn(
      "inline-flex rounded-md border border-outline-subtle bg-surface-toolbar p-0.5",
      orientation === "vertical" ? "flex-col" : "items-center",
      className,
    )}
    onKeyDown={(event) => {
      onKeyDown?.(event);
      if (event.defaultPrevented || selectionMode !== "single") return;
      if (!["ArrowRight", "ArrowDown", "ArrowLeft", "ArrowUp"].includes(event.key)) return;

      const radios = Array.from(
        event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]:not(:disabled)'),
      );
      if (radios.length === 0) return;

      event.preventDefault();
      const current = document.activeElement instanceof HTMLButtonElement ? document.activeElement : null;
      const currentIndex = current ? radios.indexOf(current) : -1;
      const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
      const next = radios[(Math.max(currentIndex, 0) + direction + radios.length) % radios.length];
      next.focus();
      next.click();
    }}
    {...props}
  />
));
ButtonGroup.displayName = "ButtonGroup";

export const ButtonGroupItem = React.forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    active?: boolean;
  }
>(({ className, active, type = "button", ...props }, ref) => (
  <button
    ref={ref}
    type={type}
    role="radio"
    aria-checked={active}
    data-active={active ? "" : undefined}
    className={cn(
      "inline-flex h-7 items-center justify-center gap-1.5 rounded px-2.5 text-[12px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/45 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-3.5 [&_svg]:shrink-0",
      active
        ? "bg-surface-active text-foreground"
        : "text-muted-foreground hover:bg-surface-hover hover:text-foreground",
      className,
    )}
    {...props}
  />
));
ButtonGroupItem.displayName = "ButtonGroupItem";

export const ButtonGroupSeparator = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & {
    orientation?: "horizontal" | "vertical";
  }
>(({ className, orientation = "vertical", ...props }, ref) => (
  <div
    ref={ref}
    role="separator"
    aria-orientation={orientation}
    className={cn(
      "shrink-0 bg-outline-subtle",
      orientation === "vertical" ? "mx-0.5 h-5 w-px" : "my-0.5 h-px w-full",
      className,
    )}
    {...props}
  />
));
ButtonGroupSeparator.displayName = "ButtonGroupSeparator";

export const ButtonGroupText = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement> & {
    asChild?: boolean;
  }
>(({ className, asChild, ...props }, ref) => {
  const Comp = asChild ? Slot : "span";
  return (
    <Comp
      ref={ref}
      className={cn(
        "inline-flex h-7 items-center px-2 text-[11px] font-medium text-muted-foreground",
        className,
      )}
      {...props}
    />
  );
});
ButtonGroupText.displayName = "ButtonGroupText";
