import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md border border-transparent text-[13px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-1 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-3.5 [&_svg]:shrink-0 active:bg-surface-active",
  {
    variants: {
      variant: {
        default:
          "border-primary/[0.24] bg-primary/[0.14] text-foreground hover:border-primary/[0.34] hover:bg-primary/[0.2]",
        secondary: "border-outline-subtle bg-surface-toolbar text-secondary-foreground hover:bg-surface-hover",
        outline: "border-outline-strong bg-surface-panel text-foreground hover:bg-surface-hover",
        ghost: "border-transparent text-muted-foreground hover:bg-surface-hover hover:text-foreground",
        accent: "border-outline-subtle bg-surface-active text-icon-tile-foreground hover:bg-surface-hover",
        destructive: "border-destructive/50 bg-destructive/88 text-destructive-foreground hover:bg-destructive",
      },
      size: {
        default: "h-8 px-3 py-1.5",
        sm: "h-7 rounded px-2.5 text-[12px]",
        lg: "h-9 rounded-md px-4",
        icon: "h-8 w-8",
      },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />;
  },
);
Button.displayName = "Button";
export { buttonVariants };
