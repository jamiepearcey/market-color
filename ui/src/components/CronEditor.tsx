import { useMemo } from "react";
import { CalendarClock, AlertTriangle } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

// ===========================================================================
// Cron engine — a real 5-field cron parser + next-fire computation. Pure and
// clock-injectable (`nextRuns` takes `from`) so it is unit-testable without
// waiting on real time. Fields: minute hour day-of-month month day-of-week.
// Supports `*`, `a`, `a-b`, `a,b,c`, and `*/n` / `a-b/n` step forms. Times are
// evaluated in UTC to match the backend scheduler ("cron, UTC").
// ===========================================================================

export interface CronFields {
  minute: string;
  hour: string;
  dom: string;
  month: string;
  dow: string;
}

export interface CronFieldDef {
  key: keyof CronFields;
  label: string;
  min: number;
  max: number;
  hint: string;
}

/** The five standard fields, in cron order, with their allowed ranges. */
export const CRON_FIELDS: CronFieldDef[] = [
  { key: "minute", label: "Minute", min: 0, max: 59, hint: "0–59" },
  { key: "hour", label: "Hour", min: 0, max: 23, hint: "0–23" },
  { key: "dom", label: "Day of month", min: 1, max: 31, hint: "1–31" },
  { key: "month", label: "Month", min: 1, max: 12, hint: "1–12" },
  // dow accepts 0–7 (0 and 7 both = Sunday); displayed as 0–6.
  { key: "dow", label: "Day of week", min: 0, max: 7, hint: "0–6 (Sun–Sat)" },
];

export interface CronPreset {
  id: string;
  label: string;
  expr: string;
}

export const CRON_PRESETS: CronPreset[] = [
  { id: "minute", label: "Every minute", expr: "* * * * *" },
  { id: "every5", label: "Every 5 minutes", expr: "*/5 * * * *" },
  { id: "every15", label: "Every 15 minutes", expr: "*/15 * * * *" },
  { id: "hourly", label: "Hourly (on the hour)", expr: "0 * * * *" },
  { id: "daily", label: "Daily at 02:00", expr: "0 2 * * *" },
  { id: "weekdays", label: "Weekdays at 08:00", expr: "0 8 * * 1-5" },
  { id: "weekly", label: "Weekly (Mon 02:00)", expr: "0 2 * * 1" },
  { id: "monthly", label: "Monthly (1st, 02:00)", expr: "0 2 1 * *" },
];

export function normalizeCron(expr: string): string {
  return expr.trim().replace(/\s+/g, " ");
}

export function fieldsToCron(f: CronFields): string {
  return [f.minute, f.hour, f.dom, f.month, f.dow].map((s) => s.trim() || "*").join(" ");
}

export function cronToFields(expr: string): CronFields {
  const [minute = "*", hour = "*", dom = "*", month = "*", dow = "*"] = normalizeCron(expr).split(" ");
  return { minute, hour, dom, month, dow };
}

/** Expand one cron field into the set of values it permits, or `null` if the
 *  field is malformed / out of range. */
export function parseField(expr: string, min: number, max: number): number[] | null {
  const out = new Set<number>();
  for (const rawPart of expr.split(",")) {
    const part = rawPart.trim();
    if (part === "") return null;
    const [rangePart, stepPart, ...rest] = part.split("/");
    if (rest.length) return null;
    let step = 1;
    if (stepPart !== undefined) {
      step = Number(stepPart);
      if (!Number.isInteger(step) || step < 1) return null;
    }
    let lo: number;
    let hi: number;
    if (rangePart === "*") {
      lo = min;
      hi = max;
    } else if (rangePart.includes("-")) {
      const [a, b, ...more] = rangePart.split("-");
      if (more.length) return null;
      lo = Number(a);
      hi = Number(b);
    } else {
      lo = Number(rangePart);
      hi = lo;
    }
    if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < min || hi > max || lo > hi) return null;
    for (let v = lo; v <= hi; v += step) out.add(v);
  }
  return [...out].sort((a, b) => a - b);
}

interface ParsedCron {
  minute: number[];
  hour: number[];
  dom: number[];
  month: number[];
  dow: number[];
  domStar: boolean;
  dowStar: boolean;
}

export function parseCron(expr: string): ParsedCron | null {
  const f = cronToFields(expr);
  const minute = parseField(f.minute, 0, 59);
  const hour = parseField(f.hour, 0, 23);
  const dom = parseField(f.dom, 1, 31);
  const month = parseField(f.month, 1, 12);
  const dowRaw = parseField(f.dow, 0, 7);
  if (!minute || !hour || !dom || !month || !dowRaw) return null;
  // Normalize 7 → 0 (both mean Sunday).
  const dow = [...new Set(dowRaw.map((v) => (v === 7 ? 0 : v)))].sort((a, b) => a - b);
  return { minute, hour, dom, month, dow, domStar: f.dom === "*", dowStar: f.dow === "*" };
}

export function isValidCron(expr: string): boolean {
  return parseCron(expr) !== null;
}

function matches(p: ParsedCron, d: Date): boolean {
  if (!p.minute.includes(d.getUTCMinutes())) return false;
  if (!p.hour.includes(d.getUTCHours())) return false;
  if (!p.month.includes(d.getUTCMonth() + 1)) return false;
  const domHit = p.dom.includes(d.getUTCDate());
  const dowHit = p.dow.includes(d.getUTCDay());
  // Standard cron: when both day fields are restricted, either may match.
  if (p.domStar && p.dowStar) return true;
  if (p.domStar) return dowHit;
  if (p.dowStar) return domHit;
  return domHit || dowHit;
}

/** Upcoming fire times after `from` (exclusive), in UTC. Searches up to a
 *  one-year horizon — enough for any standard cron — then stops. */
export function nextRuns(expr: string, from: Date, count: number): Date[] {
  const p = parseCron(expr);
  if (!p) return [];
  const out: Date[] = [];
  const d = new Date(from.getTime());
  d.setUTCSeconds(0, 0);
  d.setUTCMinutes(d.getUTCMinutes() + 1);
  const HORIZON_MINUTES = 366 * 24 * 60;
  for (let i = 0; i < HORIZON_MINUTES && out.length < count; i++) {
    if (matches(p, d)) out.push(new Date(d.getTime()));
    d.setUTCMinutes(d.getUTCMinutes() + 1);
  }
  return out;
}

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const pad = (n: number) => String(n).padStart(2, "0");

export function formatUtc(d: Date): string {
  return `${DAYS[d.getUTCDay()]} ${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(
    d.getUTCDate(),
  )} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())} UTC`;
}

const isSingleInt = (s: string) => /^\d+$/.test(s.trim());

/** A best-effort human summary of a cron expression. */
export function describeCron(expr: string): string {
  if (!isValidCron(expr)) return "Invalid cron expression";
  const preset = CRON_PRESETS.find((p) => p.expr === normalizeCron(expr));
  if (preset) return preset.label;
  const f = cronToFields(expr);
  const time =
    isSingleInt(f.minute) && isSingleInt(f.hour)
      ? `at ${pad(Number(f.hour))}:${pad(Number(f.minute))}`
      : `at minute "${f.minute}" of hour "${f.hour}"`;
  const parts = [time];
  if (f.dom !== "*") parts.push(`on day-of-month ${f.dom}`);
  if (f.month !== "*") parts.push(`in month ${f.month}`);
  if (f.dow !== "*") parts.push(`on weekday ${f.dow}`);
  else if (f.dom === "*") parts.push("every day");
  return parts.join(", ");
}

// ===========================================================================
// Component — a tabular field editor with presets + a live preview. Fully
// controlled by the cron string in the store (no local React state).
// ===========================================================================

export function CronEditor({ value, onChange }: { value: string; onChange: (expr: string) => void }) {
  const fields = cronToFields(value);
  const valid = isValidCron(value);
  const activePreset = CRON_PRESETS.find((p) => p.expr === normalizeCron(value))?.id ?? "custom";

  const upcoming = useMemo(() => (valid ? nextRuns(value, new Date(), 3) : []), [value, valid]);

  const setField = (key: keyof CronFields, raw: string) =>
    onChange(fieldsToCron({ ...fields, [key]: raw }));

  return (
    <div className="space-y-4">
      {/* Preset picker */}
      <label className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">Preset</span>
        <select
          value={activePreset}
          onChange={(e) => {
            const p = CRON_PRESETS.find((x) => x.id === e.target.value);
            if (p) onChange(p.expr);
          }}
          className="h-8 rounded-md border border-input bg-surface-input px-2 text-[12px] outline-none"
        >
          {CRON_PRESETS.map((p) => (
            <option key={p.id} value={p.id} className="bg-surface-panel">
              {p.label}
            </option>
          ))}
          <option value="custom" disabled className="bg-surface-panel">
            Custom…
          </option>
        </select>
      </label>

      {/* Tabular field editor */}
      <div className="overflow-hidden rounded-md border border-outline-subtle">
        <div className="grid grid-cols-[1fr_minmax(0,1.4fr)_auto] items-center gap-2 border-b border-outline-subtle bg-surface-toolbar px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          <span>Field</span>
          <span>Value</span>
          <span>Allowed</span>
        </div>
        {CRON_FIELDS.map((def) => {
          const raw = fields[def.key];
          const fieldOk = parseField(raw.trim() || "*", def.min, def.max) !== null;
          return (
            <div
              key={def.key}
              className="grid grid-cols-[1fr_minmax(0,1.4fr)_auto] items-center gap-2 border-b border-outline-subtle px-3 py-1.5 last:border-b-0"
            >
              <span className="text-[12px] text-foreground/90">{def.label}</span>
              <Input
                value={raw}
                onChange={(e) => setField(def.key, e.target.value)}
                placeholder="*"
                className={cn("h-7 font-mono text-[12px]", !fieldOk && "border-destructive/70")}
              />
              <span className="font-mono text-[10px] text-muted-foreground/75">{def.hint}</span>
            </div>
          );
        })}
      </div>

      {/* Live preview */}
      <div className="rounded-md border border-outline-subtle bg-surface-panel p-3">
        <div className="flex items-center gap-2">
          <code className="rounded bg-surface-input px-2 py-0.5 font-mono text-[12px] text-foreground/90">
            {normalizeCron(value) || "* * * * *"}
          </code>
          <span className="text-[11px] text-muted-foreground">{describeCron(value)}</span>
        </div>
        {!valid ? (
          <div className="mt-2 flex items-center gap-1.5 text-[11px] text-destructive">
            <AlertTriangle className="size-3.5" /> Not a valid 5-field cron expression.
          </div>
        ) : (
          <div className="mt-2 flex flex-col gap-1">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              <CalendarClock className="size-3" /> Next runs
            </div>
            {upcoming.length === 0 ? (
              <span className="text-[11px] text-muted-foreground">No upcoming runs within a year.</span>
            ) : (
              upcoming.map((d) => (
                <span key={d.getTime()} className="font-mono text-[11px] text-foreground/80">
                  {formatUtc(d)}
                </span>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}
