import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded border px-1.5 py-0.5 text-[11px] font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "border-primary/35 bg-primary/12 text-primary",
        accent: "border-outline-subtle bg-surface-active text-icon-tile-foreground",
        secondary: "border-outline-subtle bg-surface-toolbar text-secondary-foreground",
        outline: "border-outline-subtle bg-transparent text-muted-foreground",
        success: "border-transparent bg-emerald-500/15 text-emerald-400",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}
