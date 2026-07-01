import * as React from "react";
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area";
import { cn } from "@/lib/utils";

export const ScrollArea = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>
>(({ className, children, type = "auto", ...props }, ref) => (
  <ScrollAreaPrimitive.Root
    ref={ref}
    type={type}
    className={cn("relative min-w-0 overflow-hidden", className)}
    {...props}
  >
    <ScrollAreaPrimitive.Viewport className="h-full w-full min-w-0 rounded-[inherit] [&>div]:!block [&>div]:!w-full [&>div]:!min-w-0">
      {children}
    </ScrollAreaPrimitive.Viewport>
    <ScrollAreaPrimitive.Scrollbar
      orientation="vertical"
      className="flex w-2.5 touch-none select-none bg-[hsl(var(--scrollbar-track)/0.72)] p-[2px] transition-colors"
    >
      <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-md bg-[hsl(var(--scrollbar-thumb)/0.9)] hover:bg-[hsl(var(--scrollbar-thumb-hover)/0.95)]" />
    </ScrollAreaPrimitive.Scrollbar>
    <ScrollAreaPrimitive.Scrollbar
      orientation="horizontal"
      className="flex h-2.5 touch-none select-none bg-[hsl(var(--scrollbar-track)/0.72)] p-[2px] transition-colors"
    >
      <ScrollAreaPrimitive.Thumb className="relative flex-1 rounded-md bg-[hsl(var(--scrollbar-thumb)/0.9)] hover:bg-[hsl(var(--scrollbar-thumb-hover)/0.95)]" />
    </ScrollAreaPrimitive.Scrollbar>
    <ScrollAreaPrimitive.Corner />
  </ScrollAreaPrimitive.Root>
));
ScrollArea.displayName = "ScrollArea";
