import {
  ChevronDown,
  CircleHelp,
  LogOut,
  Settings,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

export function CollapsibleUserProfile({
  name,
  email,
  role,
  initials,
  onSettings,
}: {
  name: string;
  email: string;
  role: string;
  initials: string;
  onSettings?: (() => void) | undefined;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="relative">
      <div
        className={cn(
          "min-w-[210px] rounded-md border border-outline-subtle bg-surface-panel transition-colors",
          open && "border-outline-strong",
        )}
      >
        <div className="flex items-center gap-2 px-2 py-1.5">
          <div className="grid size-8 shrink-0 place-items-center rounded-md border border-primary/[0.22] bg-primary/[0.14] text-[11px] font-semibold text-icon-tile-foreground">
            {initials}
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-[12px] font-semibold tracking-[-0.01em] text-foreground">
              {name}
            </div>
            <div className="truncate text-[10.5px] text-muted-foreground">
              {email}
            </div>
          </div>
          <CollapsibleTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-7 shrink-0"
              aria-label={
                open ? "Collapse user profile" : "Expand user profile"
              }
            >
              <ChevronDown
                className={cn(
                  "size-3.5 transition-transform",
                  open && "rotate-180",
                )}
              />
            </Button>
          </CollapsibleTrigger>
        </div>
        <CollapsibleContent>
          <div className="absolute right-0 top-[calc(100%+6px)] z-50 w-full rounded-md border border-outline-strong bg-surface-panel p-1.5 shadow-none">
            <div className="mb-1 flex items-center gap-1.5 rounded px-2 py-1 text-[10.5px] text-muted-foreground">
              <ShieldCheck className="size-3.5 text-icon-muted" />
              <span className="truncate">{role}</span>
            </div>
            <ProfileAction icon={<UserRound className="size-3.5" />}>
              Profile
            </ProfileAction>
            <ProfileAction
              icon={<Settings className="size-3.5" />}
              onClick={onSettings}
            >
              Settings
            </ProfileAction>
            <ProfileAction icon={<CircleHelp className="size-3.5" />}>
              Help
            </ProfileAction>
            <ProfileAction icon={<LogOut className="size-3.5" />}>
              Sign out
            </ProfileAction>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function ProfileAction({
  icon,
  children,
  onClick,
}: {
  icon: ReactNode;
  children: ReactNode;
  onClick?: (() => void) | undefined;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex h-7 w-full items-center gap-2 rounded px-2 text-left text-[11.5px] font-medium text-muted-foreground transition-colors hover:bg-surface-hover hover:text-foreground"
    >
      <span className="text-icon-muted">{icon}</span>
      <span className="truncate">{children}</span>
    </button>
  );
}
