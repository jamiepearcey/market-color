import { useState } from "react";
import { useSnapshot } from "valtio";
import {
  ArrowLeft,
  Download,
  Loader2,
  CircleCheck,
  CircleX,
  Copy,
  Check,
  Github,
  ExternalLink,
  Mail,
  User,
  CalendarClock,
  Tag,
  Scale,
  ArrowDownToLine,
  ArrowUpFromLine,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { PageHeader } from "@/components/ui/page-header";
import { actions, store } from "@/lib/store";
import { cn } from "@/lib/utils";
import { iconForCategory } from "@/lib/connector-icons";
import type { RegistryPackage, RegistrySetting } from "@/lib/types";

export function PackageDetailView({ pkg }: { pkg: RegistryPackage }) {
  const snap = useSnapshot(store);
  const Icon = iconForCategory(pkg.category);
  const isSource = pkg.type === "source";
  const installing = Boolean(snap.installing[pkg.name]);
  const result = snap.installResult?.name === pkg.name ? snap.installResult : null;
  const settings = pkg.manifest.settings ?? [];
  const capabilities = pkg.manifest.capabilities ?? [];
  const packagedAt = formatDate(pkg.packagedAt);

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <PageHeader
        icon={
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2"
            onClick={() => actions.selectPackage(null)}
            aria-label="Back to the store"
          >
            <ArrowLeft className="size-4" /> Store
          </Button>
        }
        title={pkg.title}
        meta={
          <>
            <Badge variant={isSource ? "accent" : "default"} className="gap-1 text-[10px]">
              {isSource ? <ArrowDownToLine className="size-2.5" /> : <ArrowUpFromLine className="size-2.5" />}
              {isSource ? "source · import" : "target · export"}
            </Badge>
            <Badge variant="outline" className="text-[10px]">{pkg.category}</Badge>
            <span className="font-mono text-[11px] text-muted-foreground">v{pkg.version}</span>
            {pkg.installed && (
              <Badge variant="success" className="gap-1 text-[10px]">
                <CircleCheck className="size-2.5" /> Installed
              </Badge>
            )}
          </>
        }
        actions={
          <Button
            size="sm"
            variant={pkg.installed ? "outline" : "default"}
            disabled={installing || pkg.installed}
            onClick={() => actions.installPackage(pkg.name)}
          >
            {pkg.installed ? (
              <>
                <CircleCheck className="size-4" /> Installed
              </>
            ) : installing ? (
              <>
                <Loader2 className="size-4 animate-spin" /> Installing…
              </>
            ) : (
              <>
                <Download className="size-4" /> Install
              </>
            )}
          </Button>
        }
      />

      <ScrollArea className="min-h-0 flex-1">
        <div className="mx-auto max-w-3xl space-y-6 p-6">
          {/* Title block */}
          <div className="flex items-start gap-3">
            <span className="grid size-10 shrink-0 place-items-center rounded-md border border-outline-subtle bg-primary/[0.14]">
              <Icon className="size-5 text-icon-tile-foreground" />
            </span>
            <div className="min-w-0">
              <div className="text-base font-semibold tracking-[-0.01em]">{pkg.title}</div>
              <div className="font-mono text-[11px] text-muted-foreground">{pkg.name}</div>
              <p className="mt-1 text-[12.5px] leading-relaxed text-muted-foreground">{pkg.summary}</p>
            </div>
          </div>

          {/* Install status / log */}
          {result && (
            <div
              className={cn(
                "rounded-md border px-3 py-2 text-[12px]",
                result.status === "installed"
                  ? "border-emerald-500/30 bg-emerald-500/[0.07]"
                  : "border-destructive/30 bg-destructive/[0.06]",
              )}
            >
              <div className="flex items-start gap-2">
                {result.status === "installed" ? (
                  <CircleCheck className="mt-0.5 size-4 shrink-0 text-emerald-400" />
                ) : (
                  <CircleX className="mt-0.5 size-4 shrink-0 text-destructive" />
                )}
                <span className="min-w-0 flex-1 leading-relaxed">{result.message}</span>
              </div>
              {result.log && (
                <pre className="mt-2 max-h-48 overflow-auto rounded border border-outline-subtle bg-surface-input p-2 font-mono text-[11px] leading-5 text-muted-foreground">
                  {result.log}
                </pre>
              )}
            </div>
          )}

          {/* Install command */}
          <Section title="Package / install command">
            <CopyBlock value={pkg.installCommand} label="install command" />
            <div className="mt-2 text-[11px] text-muted-foreground">
              Publisher <span className="font-mono text-foreground">{pkg.artifact.publisher}</span>{" "}
              · package <span className="font-mono text-foreground">{pkg.artifact.package}</span>
            </div>
          </Section>

          {/* Provenance */}
          <Section title="Provenance">
            <dl className="grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
              <Field icon={User} label="Packaged by">
                <span className="text-foreground">{pkg.packagedBy.name}</span>
                {pkg.packagedBy.email && (
                  <a
                    href={`mailto:${pkg.packagedBy.email}`}
                    className="ml-2 inline-flex items-center gap-1 text-primary hover:underline"
                  >
                    <Mail className="size-3" /> {pkg.packagedBy.email}
                  </a>
                )}
                {pkg.packagedBy.url && (
                  <a
                    href={pkg.packagedBy.url}
                    target="_blank"
                    rel="noreferrer"
                    className="ml-2 inline-flex items-center gap-1 text-primary hover:underline"
                  >
                    <ExternalLink className="size-3" /> site
                  </a>
                )}
              </Field>
              <Field icon={CalendarClock} label="Packaged at">
                <span className="text-foreground">{packagedAt}</span>
              </Field>
              <Field icon={Tag} label="Version">
                <span className="font-mono text-foreground">{pkg.version}</span>
              </Field>
              <Field icon={Scale} label="License">
                <span className="text-foreground">{pkg.license || "—"}</span>
              </Field>
              <Field icon={Github} label="Repository">
                {pkg.repository ? (
                  <a
                    href={pkg.repository}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 text-primary hover:underline"
                  >
                    <span className="truncate">{repoLabel(pkg.repository)}</span>
                    <ExternalLink className="size-3 shrink-0" />
                  </a>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </Field>
              <Field icon={isSource ? ArrowDownToLine : ArrowUpFromLine} label="Type">
                <span className="text-foreground">{isSource ? "source (import)" : "target (export)"}</span>
                <span className="ml-2 text-muted-foreground">· {pkg.category}</span>
              </Field>
            </dl>
          </Section>

          {/* Description */}
          <Section title="Description">
            <p className="whitespace-pre-wrap text-[12.5px] leading-relaxed text-muted-foreground">
              {pkg.description || pkg.summary}
            </p>
          </Section>

          {/* Capabilities */}
          {capabilities.length > 0 && (
            <Section title="Capabilities">
              <div className="flex flex-wrap gap-1.5">
                {capabilities.map((c) => (
                  <Badge key={c} variant="secondary" className="text-[10px]">{c}</Badge>
                ))}
              </div>
            </Section>
          )}

          {/* Tags */}
          {pkg.tags.length > 0 && (
            <Section title="Tags">
              <div className="flex flex-wrap gap-1.5">
                {pkg.tags.map((t) => (
                  <Badge key={t} variant="outline" className="gap-1 text-[10px]">
                    <Tag className="size-2.5" /> {t}
                  </Badge>
                ))}
              </div>
            </Section>
          )}

          {/* Settings schema */}
          <Section title="Settings">
            {settings.length === 0 ? (
              <div className="text-[12px] text-muted-foreground">
                This connector publishes no settings schema.
              </div>
            ) : (
              <div className="overflow-x-auto rounded-md border border-outline-subtle">
                <table className="w-full border-collapse text-left text-[11.5px]">
                  <thead>
                    <tr className="border-b border-outline-subtle bg-surface-header text-[10px] uppercase tracking-wider text-muted-foreground">
                      <th className="px-3 py-1.5 font-semibold">Name</th>
                      <th className="px-3 py-1.5 font-semibold">Kind</th>
                      <th className="px-3 py-1.5 font-semibold">Required</th>
                      <th className="px-3 py-1.5 font-semibold">Default</th>
                      <th className="px-3 py-1.5 font-semibold">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {settings.map((s, i) => (
                      <SettingRow key={`${s.name}-${i}`} setting={s} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Section>

          <Separator />
          <p className="text-[11px] text-muted-foreground">
            {pkg.installed
              ? "Installed — this connector is available under Sources / Targets."
              : "Install adds this connector to the local engine; it then appears under Sources / Targets."}
          </p>
        </div>
      </ScrollArea>
    </div>
  );
}

function SettingRow({ setting }: { setting: RegistrySetting }) {
  const secret = isSecret(setting);
  const def =
    setting.default === undefined || setting.default === null || setting.default === ""
      ? "—"
      : secret
        ? "••••••"
        : String(setting.default);
  return (
    <tr className="border-b border-outline-subtle last:border-0">
      <td className="px-3 py-1.5 font-mono text-foreground">{setting.name}</td>
      <td className="px-3 py-1.5">
        <span className="font-mono text-muted-foreground">{setting.kind ?? "string"}</span>
        {secret && (
          <Badge variant="outline" className="ml-1.5 text-[9px] text-amber-300/90">secret</Badge>
        )}
      </td>
      <td className="px-3 py-1.5">
        {setting.required ? (
          <span className="text-amber-300/90">required</span>
        ) : (
          <span className="text-muted-foreground">optional</span>
        )}
      </td>
      <td className="px-3 py-1.5 font-mono text-muted-foreground">{def}</td>
      <td className="px-3 py-1.5 text-muted-foreground">
        {setting.description ?? "—"}
        {setting.options && setting.options.length > 0 && (
          <span className="ml-1 font-mono text-[10px] text-muted-foreground/80">
            ({setting.options.join(", ")})
          </span>
        )}
      </td>
    </tr>
  );
}

function isSecret(s: RegistrySetting): boolean {
  const k = (s.kind ?? "").toLowerCase();
  if (k === "secret") return true;
  return /\b(password|secret|token|api[_ -]?key|access[_ -]?key|private[_ -]?key|credential)\b/.test(
    s.name.toLowerCase(),
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {title}
      </div>
      {children}
    </section>
  );
}

function Field({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof User;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="min-w-0">
      <dt className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
        <Icon className="size-3" /> {label}
      </dt>
      <dd className="mt-0.5 min-w-0 truncate text-[12px]">{children}</dd>
    </div>
  );
}

function CopyBlock({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <div className="flex items-stretch gap-2">
      <code className="min-w-0 flex-1 overflow-x-auto whitespace-pre rounded-md border border-outline-subtle bg-surface-input px-3 py-2 font-mono text-[12px] text-foreground">
        {value || "—"}
      </code>
      <Button
        type="button"
        size="icon"
        variant="outline"
        className="h-auto shrink-0"
        onClick={copy}
        disabled={!value}
        aria-label={`Copy ${label}`}
        title={`Copy ${label}`}
      >
        {copied ? <Check className="size-3.5 text-emerald-400" /> : <Copy className="size-3.5" />}
      </Button>
    </div>
  );
}

function repoLabel(url: string): string {
  return url.replace(/^https?:\/\//, "").replace(/\.git$/, "");
}

function formatDate(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}
