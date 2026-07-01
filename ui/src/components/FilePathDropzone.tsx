import { FileUp, FolderSearch, X } from "lucide-react";
import { useMemo, useState } from "react";
import type { DropEvent, FileRejection } from "react-dropzone";
import {
  Dropzone,
  DropzoneContent,
  DropzoneEmptyState,
} from "@/components/ui/dropzone";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type FileWithPath = File & {
  path?: string;
};

function pathFile(path: string): File[] | undefined {
  if (!path) return undefined;
  const name = path.split(/[\\/]/).filter(Boolean).at(-1) || path;
  return [new File([], name)];
}

function pathFromFile(file: File): string | null {
  const candidate = (file as FileWithPath).path;
  return typeof candidate === "string" && candidate.length > 0
    ? candidate
    : null;
}

export function FilePathDropzone({
  value,
  onChange,
  onBrowse,
  disabled,
  compact = false,
  placeholder = "/data/returns.csv",
  status,
  className,
}: {
  value: string;
  onChange: (path: string) => void | Promise<void>;
  onBrowse: () => void | Promise<void>;
  disabled?: boolean | undefined;
  compact?: boolean | undefined;
  placeholder?: string | undefined;
  status?: string | undefined;
  className?: string | undefined;
}) {
  const [dropError, setDropError] = useState<string | null>(null);
  const src = useMemo(() => pathFile(value), [value]);

  const handleDrop = (
    files: File[],
    _rejections: FileRejection[],
    _event: DropEvent,
  ) => {
    setDropError(null);
    const path = files.map(pathFromFile).find(Boolean);
    if (path) {
      void onChange(path);
      return;
    }
    setDropError("Dropped files must expose a local filesystem path.");
  };

  return (
    <div className={cn("min-w-0", className)}>
      <Dropzone
        {...(src ? { src } : {})}
        maxFiles={1}
        {...(disabled !== undefined ? { disabled } : {})}
        noClick
        noKeyboard
        onDrop={handleDrop}
        onError={(error) => setDropError(error.message)}
        className={cn(
          "items-stretch rounded-md border-outline-strong bg-surface-panel px-3 text-left hover:bg-surface-hover",
          compact ? "min-h-16 py-2" : "min-h-24 py-3",
        )}
      >
        <DropzoneContent>
          <div className="flex w-full min-w-0 items-center gap-2">
            <div className="grid size-8 shrink-0 place-items-center rounded-md bg-surface-toolbar text-primary">
              <FileUp className="size-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate font-mono text-[12px] text-foreground">
                {value}
              </div>
              <div className="mt-0.5 text-[10px] text-muted-foreground">
                Drop a local file here or browse to replace it.
              </div>
            </div>
            {status && (
              <span className="shrink-0 text-[10px] text-muted-foreground">
                {status}
              </span>
            )}
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className="size-7 shrink-0"
              title="Clear file"
              onClick={(event) => {
                event.stopPropagation();
                void onChange("");
              }}
            >
              <X className="size-3.5" />
            </Button>
            <Button
              type="button"
              size="icon"
              variant="outline"
              className="size-7 shrink-0"
              title="Browse for a file"
              onClick={(event) => {
                event.stopPropagation();
                void onBrowse();
              }}
            >
              <FolderSearch className="size-3.5" />
            </Button>
          </div>
        </DropzoneContent>
        <DropzoneEmptyState>
          <div className="flex w-full min-w-0 items-center gap-2">
            <div className="grid size-8 shrink-0 place-items-center rounded-md bg-surface-toolbar text-muted-foreground">
              <FileUp className="size-4" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-[12px] font-medium text-foreground">
                Drop source file
              </div>
              <div className="mt-0.5 truncate font-mono text-[11px] text-muted-foreground">
                {placeholder}
              </div>
            </div>
            {status && (
              <span className="shrink-0 text-[10px] text-muted-foreground">
                {status}
              </span>
            )}
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="shrink-0"
              title="Browse for a file"
              onClick={(event) => {
                event.stopPropagation();
                void onBrowse();
              }}
            >
              <FolderSearch className="size-3.5" /> Browse
            </Button>
          </div>
        </DropzoneEmptyState>
      </Dropzone>
      {dropError && (
        <div className="mt-1 text-[10px] text-amber-300/90">{dropError}</div>
      )}
    </div>
  );
}
