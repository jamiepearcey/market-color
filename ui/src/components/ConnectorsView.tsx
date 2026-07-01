import { useEffect, useState } from "react";
import { useSnapshot } from "valtio";
import { motion } from "framer-motion";
import {
  Plus,
  Save,
  Trash2,
  Cable,
  FlaskConical,
  Loader2,
  CircleCheck,
  CircleX,
  AlertTriangle,
  Play,
  ArrowDownToLine,
  ArrowUpFromLine,
  Power,
  Search,
  Ban,
  Github,
  FileSearch,
  FolderOpen,
  KeyRound,
  ShieldCheck,
  Store,
  Eye,
  EyeOff,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { SimulatedBadge, SimulatedNotice } from "@/components/ui/simulated";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { PageHeader } from "@/components/ui/page-header";
import {
  ActionBar,
  ActionBarButton,
  ActionBarSeparator,
} from "@/components/ui/action-bar";
import { actions, store } from "@/lib/store";
import { pickDataFile, pickDirectory } from "@/lib/api";
import { isValidCron } from "@/components/CronEditor";
import { cn } from "@/lib/utils";
import { iconForCategory } from "@/lib/connector-icons";
import type {
  ConnectorInstance,
  ConnectorParam,
  ConnectorSpec,
} from "@/lib/types";
import { EmptyPanel, ErrorPanel, SkeletonRows } from "@/components/AsyncStates";
import { formatInlineFieldError } from "@/lib/ui-errors";

export function ConnectorsView() {
  const snap = useSnapshot(store);
  const draft = snap.connectorDraft;
  const spec = snap.connectorDraftSpec;
  const role = snap.connectorRole;
  const isSource = role === "source";
  const instances = (
    isSource ? snap.sourceInstances : snap.targetInstances
  ) as ConnectorInstance[];

  return (
    <div className="flex min-h-0 flex-1">
      {/* Secondary sidebar: configured instances for the active role */}
      <aside className="flex w-[266px] shrink-0 flex-col border-r border-outline-subtle bg-surface-sidebar">
        <div className="flex h-9 items-center justify-between border-b border-outline-subtle bg-surface-header px-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            {isSource ? "Sources" : "Targets"}
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2"
            onClick={() => actions.closeConnectorDraft()}
            title="Browse the connector catalog"
          >
            <Plus className="size-4" /> Add
          </Button>
        </div>
        <ScrollArea className="flex-1 px-2 py-2">
          <InstanceGroup
            label={isSource ? "Sources" : "Targets"}
            icon={isSource ? ArrowDownToLine : ArrowUpFromLine}
            instances={instances}
            activeId={draft?.id ?? null}
            empty={
              isSource
                ? "No import connectors yet."
                : "No export connectors yet."
            }
          />
        </ScrollArea>
      </aside>

      {/* Main area: wizard when a draft is open, otherwise the catalog browser */}
      {draft && spec ? (
        <ConnectorWizard
          draft={draft as ConnectorInstance}
          spec={spec as ConnectorSpec}
        />
      ) : (
        <CatalogBrowser role={role} />
      )}
    </div>
  );
}

function InstanceGroup({
  label,
  icon: Icon,
  instances,
  activeId,
  empty,
}: {
  label: string;
  icon: typeof Cable;
  instances: ConnectorInstance[];
  activeId: string | null;
  empty: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 px-1.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        <Icon className="size-3" /> {label}
        <Badge variant="secondary" className="ml-auto text-[10px]">
          {instances.length}
        </Badge>
      </div>
      {instances.length === 0 ? (
        <div className="px-2 py-3 text-[11px] text-muted-foreground">
          {empty}
        </div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {instances.map((c) => {
            const Icon = iconForCategory(
              c.kind === "source" ? specCategory(c) : specCategory(c),
            );
            const unavailable = !specAvailable(c);
            return (
              <button
                key={c.id}
                onClick={() => actions.editConnector(c.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left text-[12.5px] tracking-[-0.015em] transition-colors",
                  activeId === c.id
                    ? "border-outline-subtle bg-surface-active text-foreground"
                    : "border-transparent text-muted-foreground hover:bg-surface-hover hover:text-foreground",
                  unavailable && activeId !== c.id && "opacity-55",
                )}
              >
                <Icon
                  className={cn(
                    "size-4 shrink-0",
                    activeId === c.id && "text-icon-tile-foreground",
                  )}
                />
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{c.name}</span>
                  <span className="block truncate text-[10px] text-muted-foreground">
                    {specCategory(c)} · {c.driver}
                    {c.lock?.status && c.lock.status !== "locked"
                      ? ` · ${c.lock.status}`
                      : ""}
                  </span>
                </span>
                {unavailable && (
                  <span
                    title="No installed driver — unavailable"
                    className="shrink-0"
                  >
                    <Ban className="size-3 text-muted-foreground/70" />
                  </span>
                )}
                {c.lock?.status && c.lock.status !== "locked" && (
                  <span
                    title={c.lock.message ?? `Lock status: ${c.lock.status}`}
                    className={cn(
                      "shrink-0 rounded border px-1 py-0 text-[9px]",
                      c.lock.status === "stale" || c.lock.status === "missing"
                        ? "border-amber-500/30 bg-amber-500/[0.08] text-amber-200"
                        : "border-outline-subtle bg-surface-toolbar text-muted-foreground",
                    )}
                  >
                    {c.lock.status}
                  </span>
                )}
                <span
                  title={c.enabled ? "Enabled" : "Disabled"}
                  className={cn(
                    "size-1.5 shrink-0 rounded-md",
                    c.enabled ? "bg-emerald-400" : "bg-muted-foreground/40",
                  )}
                />
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function specCategory(c: ConnectorInstance): string {
  return store.connectorSpecs.find((s) => s.name === c.spec)?.category ?? "—";
}

/** Whether the spec backing an instance has an installed driver. Unknown spec
 *  (e.g. removed from the library) is treated as unavailable. */
function specAvailable(c: ConnectorInstance): boolean {
  return (
    store.connectorSpecs.find((s) => s.name === c.spec)?.available ?? false
  );
}

function connectorSpecLabel(spec: ConnectorSpec): string {
  return spec.name
    .replace(/_export$/, "")
    .replace(/[_-]+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** The catalog browser shown when no connector is being edited. */
function CatalogBrowser({ role }: { role: "source" | "target" }) {
  const snap = useSnapshot(store);
  const [searchDraft, setSearchDraft] = useState(snap.connectorSearch);
  const isSource = role === "source";
  const catalog = (
    snap.connectorCatalog as {
      kind: "source" | "target";
      category: string;
      specs: ConnectorSpec[];
    }[]
  ).filter((g) => g.kind === role);
  const total = snap.connectorSpecs.filter((s) => s.kind === role).length;
  const available = snap.connectorSpecs.filter(
    (s) => s.kind === role && s.available,
  ).length;
  const shown = catalog.reduce((n, g) => n + g.specs.length, 0);
  const filtered = total > 0 && shown === 0;

  useEffect(() => setSearchDraft(snap.connectorSearch), [snap.connectorSearch]);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (searchDraft !== snap.connectorSearch)
        actions.setConnectorSearch(searchDraft);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [searchDraft, snap.connectorSearch]);

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <PageHeader
        icon={
          isSource ? (
            <ArrowDownToLine className="size-4 text-primary" />
          ) : (
            <ArrowUpFromLine className="size-4 text-primary" />
          )
        }
        title={isSource ? "Import connectors" : "Export connectors"}
        meta={
          <span className="truncate text-[11px] text-muted-foreground">
            {available} of {total} have an installed driver · pick one to
            configure, test, save, then enable.
          </span>
        }
        actions={
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => actions.openStore(role)}
              title="Discover more connectors in the store"
            >
              <Store className="size-4" /> Search the store
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => actions.linkGithubConnectorCatalog()}
            >
              <Github className="size-4" /> Link GitHub
            </Button>
          </div>
        }
      />

      {snap.connectorCatalogMessage && (
        <div className="border-b border-outline-subtle bg-surface-header px-3 py-2">
          <div className="flex items-center gap-2 rounded-md border border-outline-subtle bg-surface-panel px-3 py-2 text-[12px] text-muted-foreground">
            <AlertTriangle className="size-3.5 text-amber-300/90" />
            {snap.connectorCatalogMessage}
          </div>
        </div>
      )}

      {/* Filter toolbar */}
      <div className="flex items-center gap-2 border-b border-outline-subtle bg-surface-header px-3 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-1.5 rounded-md border border-outline-subtle bg-surface-input px-2">
          <Search className="size-3.5 shrink-0 text-muted-foreground" />
          <input
            value={searchDraft}
            onChange={(e) => setSearchDraft(e.target.value)}
            placeholder="Filter connectors by name, category, or driver"
            aria-label="Filter connectors"
            className="h-7 min-w-0 flex-1 bg-transparent text-[12px] outline-none placeholder:text-muted-foreground"
          />
          {searchDraft && (
            <button
              onClick={() => setSearchDraft("")}
              className="text-muted-foreground hover:text-foreground"
              title="Clear filter"
            >
              <CircleX className="size-3.5" />
            </button>
          )}
        </div>
        <button
          onClick={() => actions.toggleConnectorHideUnavailable()}
          aria-pressed={snap.connectorHideUnavailable}
          className={cn(
            "flex shrink-0 items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-colors",
            snap.connectorHideUnavailable
              ? "border-primary/40 bg-primary/[0.12] text-foreground"
              : "border-outline-subtle text-muted-foreground hover:bg-surface-hover hover:text-foreground",
          )}
          title="Hide connectors that have no installed driver"
        >
          <Ban className="size-3.5" /> Available only
        </button>
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-6 p-6">
          <SimulatedNotice>
            Preview mode: connector catalog and driver actions may use
            representative sample data instead of a live engine.
          </SimulatedNotice>
          {snap.connectorLoading && total === 0 ? (
            <SkeletonRows count={6} />
          ) : snap.connectorError ? (
            <ErrorPanel
              title="Could not load connectors"
              message={snap.connectorError}
              onRetry={() => actions.loadConnectors()}
            />
          ) : (
            <>
              <CatalogSection
                title={isSource ? "Sources (import)" : "Targets (export)"}
                icon={isSource ? ArrowDownToLine : ArrowUpFromLine}
                groups={catalog}
              />
              {total === 0 && (
                <EmptyPanel
                  title="No connector specs found"
                  message="Install or link a connector package, then refresh this catalog."
                  action={
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => actions.loadConnectors()}
                    >
                      <RefreshCw className="size-3.5" /> Refresh
                    </Button>
                  }
                />
              )}
              {filtered && (
                <EmptyPanel
                  title="No connectors match"
                  message="Clear the filter or include unavailable drivers to widen the catalog."
                />
              )}
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

function CatalogSection({
  title,
  icon: Icon,
  groups,
}: {
  title: string;
  icon: typeof Cable;
  groups: {
    kind: "source" | "target";
    category: string;
    specs: ConnectorSpec[];
  }[];
}) {
  if (groups.length === 0) return null;
  return (
    <div>
      <div className="mb-3 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        <Icon className="size-3.5" /> {title}
      </div>
      <div className="space-y-4">
        {groups.map((g) => {
          const CatIcon = iconForCategory(g.category);
          return (
            <div key={`${g.kind}-${g.category}`}>
              <div className="mb-2 flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                <CatIcon className="size-3.5" /> {g.category}
              </div>
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                {g.specs.map((s) => (
                  <button
                    key={s.name}
                    onClick={() => actions.newConnector(s.name)}
                    title={
                      s.available ? undefined : unavailableDriverMessage(s)
                    }
                    className={cn(
                      "group rounded-md border border-outline-subtle bg-surface-panel p-3 text-left transition-colors hover:border-primary/[0.25] hover:bg-surface-hover",
                      !s.available && "opacity-55 hover:opacity-100",
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <span className="grid size-7 shrink-0 place-items-center rounded border border-transparent bg-primary/[0.14]">
                        <CatIcon className="size-3.5 text-icon-tile-foreground" />
                      </span>
                      <span className="min-w-0 flex-1 truncate text-[13px] font-semibold tracking-[-0.015em]">
                        {connectorSpecLabel(s)}
                      </span>
                      {!s.available && (
                        <Badge
                          variant="outline"
                          className="shrink-0 gap-1 text-[9px] text-muted-foreground"
                        >
                          <Ban className="size-2.5" /> unavailable
                        </Badge>
                      )}
                      {s.available && s.exampleUnresolved && (
                        <Badge
                          variant="outline"
                          className="shrink-0 text-[9px] text-amber-300/90"
                        >
                          example
                        </Badge>
                      )}
                    </div>
                    <p className="mt-1.5 line-clamp-2 text-[11.5px] leading-relaxed text-muted-foreground">
                      {s.description}
                    </p>
                    <div className="mt-2 flex items-center gap-2 text-[10px] text-muted-foreground">
                      <span className="truncate font-mono">{s.name}</span>
                      <span className="font-mono">{s.driver}</span>
                      <span className="ml-auto inline-flex items-center gap-1 text-icon-tile-foreground opacity-0 transition-opacity group-hover:opacity-100">
                        Configure <Plus className="size-3" />
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The setup wizard: fill params → Test → Save → Enable/Disable + custom actions. */
function ConnectorWizard({
  draft,
  spec,
}: {
  draft: ConnectorInstance;
  spec: ConnectorSpec;
}) {
  const snap = useSnapshot(store);
  const saved = snap.connectorInstances.some((c) => c.id === draft.id);
  const test = snap.connectorTestResult;
  const result = snap.connectorActionResult;
  const Icon = iconForCategory(spec.category);
  const intent = snap.connectorReturnIntent;
  const saveLabel = intent?.bind
    ? "Save and bind"
    : intent
      ? "Save and return"
      : "Save";
  const validation = connectorDraftValidation(
    spec,
    draft,
    snap.connectorPreviewJobParams,
  );
  const hasErrors = validation.connectionInvalid || validation.defaultsInvalid;
  const hasActionErrors =
    validation.connectionInvalid || validation.previewInvalid;

  useEffect(() => () => actions.clearConnectorDraftSecrets(), []);

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <PageHeader
        icon={<Icon className="size-4 text-primary" />}
        title={
          <Input
            value={draft.name}
            onChange={(e) => actions.setConnectorName(e.target.value)}
            className="h-8 w-56 text-sm font-semibold"
          />
        }
        meta={
          <>
            <Badge
              variant={spec.kind === "source" ? "accent" : "default"}
              className="text-[10px]"
            >
              {spec.kind === "source" ? "source · import" : "target · export"}
            </Badge>
            <Badge variant="outline" className="text-[10px]">
              {spec.category}
            </Badge>
            <span className="font-mono text-[11px] text-muted-foreground">
              {spec.driver}
            </span>
            {!spec.available && (
              <Badge
                variant="outline"
                className="gap-1 text-[10px] text-muted-foreground"
              >
                <Ban className="size-2.5" /> unavailable
              </Badge>
            )}
            {spec.available &&
              (spec.exampleUnresolved || test?.status === "example") && (
                <Badge
                  variant="outline"
                  className="text-[10px] text-amber-300/90"
                >
                  example
                </Badge>
              )}
          </>
        }
        actions={
          <ActionBar>
            <ActionBarButton
              onClick={() => actions.testConnectorDraft()}
              disabled={snap.connectorBusy || validation.connectionInvalid}
              title={
                validation.connectionInvalid
                  ? "Fix connection field errors first"
                  : undefined
              }
            >
              {snap.connectorBusy ? (
                <Loader2 className="animate-spin" />
              ) : (
                <FlaskConical />
              )}
              Test
            </ActionBarButton>
            <ActionBarButton
              tone="primary"
              onClick={() =>
                actions.saveConnector({ completeReturn: Boolean(intent) })
              }
              disabled={hasErrors}
              title={
                hasErrors
                  ? "Fix required or invalid fields before saving"
                  : undefined
              }
            >
              <Save /> {saveLabel}
            </ActionBarButton>
            <ActionBarSeparator />
            <ActionBarButton
              tone={draft.enabled ? "primary" : "default"}
              onClick={() => {
                actions.saveConnector();
                actions.toggleConnectorEnabled(draft.id);
              }}
              disabled={hasErrors}
              title={
                hasErrors
                  ? "Fix required or invalid fields before enabling"
                  : draft.enabled
                    ? "Disable (hide from job pickers)"
                    : "Enable after a passing test, or confirm an explicit override"
              }
            >
              <Power /> {draft.enabled ? "Enabled" : "Disabled"}
            </ActionBarButton>
            <ActionBarSeparator />
            <ActionBarButton
              tone="danger"
              onClick={() =>
                actions.requestConfirm(
                  {
                    title: "Delete connector",
                    message: `Delete the “${draft.name}” connector configuration? This cannot be undone.`,
                    confirmLabel: "Delete",
                    tone: "danger",
                  },
                  () => actions.deleteConnector(draft.id),
                )
              }
              disabled={!saved}
            >
              <Trash2 /> Delete
            </ActionBarButton>
          </ActionBar>
        }
      />

      <ScrollArea className="min-h-0 flex-1">
        <div className="mx-auto max-w-3xl space-y-6 p-6">
          {intent && (
            <div className="flex items-center gap-2 rounded-md border border-primary/25 bg-primary/[0.08] px-3 py-2 text-[12px] text-muted-foreground">
              <Cable className="size-4 text-icon-tile-foreground" />
              <span className="min-w-0 flex-1">
                {intent.bind
                  ? `Configuring ${intent.bind.role} connector for the open job.`
                  : intent.pack
                    ? "Configuring a connector required by this pack."
                    : "Configuring connector for setup."}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="h-7 px-2"
                onClick={() => actions.cancelConnectorReturn()}
              >
                Cancel
              </Button>
            </div>
          )}
          <p className="text-[12.5px] leading-relaxed text-muted-foreground">
            {spec.description}
          </p>

          {!spec.available && <UnavailableDriverPanel spec={spec} />}
          {spec.available && <ReliabilityPanel draft={draft} />}

          {/* Connection params */}
          <section>
            <SectionTitle>Connection</SectionTitle>
            <p className="mb-2 text-[11px] text-muted-foreground">
              Saved values land in the connector's manifest config block.
              Runtime process env and project{" "}
              <span className="font-mono">.env</span> still win if the same
              setting is defined there, and secret inputs are persisted as
              references instead of plaintext values.
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              {spec.params.map((p) => (
                <ParamField
                  key={p.name}
                  param={p}
                  value={draft.params[p.name] ?? ""}
                  error={validation.connection[p.name] ?? null}
                  onChange={(v) => actions.setConnectorParam(p.name, v)}
                />
              ))}
              {spec.params.length === 0 && (
                <div className="text-[12px] text-muted-foreground">
                  This connector needs no connection parameters.
                </div>
              )}
            </div>
            {test && (
              <ResultBanner
                status={test.status}
                message={test.message}
                elapsedMs={test.elapsedMs}
                className="mt-3"
              />
            )}
          </section>

          {/* Custom actions */}
          {spec.actions.length > 0 && (
            <section>
              <SectionTitle>Actions</SectionTitle>
              <p className="mb-2 text-[11px] text-muted-foreground">
                Run the driver with a flag and the current configuration —
                extensibility hooks defined by the connector.
              </p>
              {spec.jobParams.length > 0 && (
                <div className="mb-3 rounded-md border border-outline-subtle bg-surface-panel/60 p-3">
                  <div className="mb-2 text-[11px] font-medium text-muted-foreground">
                    Preview inputs
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {spec.jobParams.map((p) => (
                      <ParamField
                        key={p.name}
                        param={optionalParam(p)}
                        value={snap.connectorPreviewJobParams[p.name] ?? ""}
                        error={validation.preview[p.name] ?? null}
                        onChange={(v) =>
                          actions.setConnectorPreviewJobParam(p.name, v)
                        }
                      />
                    ))}
                  </div>
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                {spec.actions.map((a) => (
                  <Button
                    key={a.name}
                    size="sm"
                    variant="outline"
                    disabled={snap.connectorBusy || hasActionErrors}
                    title={a.description ?? undefined}
                    onClick={() => actions.runConnectorActionDraft({ ...a })}
                  >
                    <Play className="size-3.5" /> {a.label}
                    <span className="ml-1 font-mono text-[10px] text-muted-foreground">
                      {a.flag}
                    </span>
                  </Button>
                ))}
              </div>
              {result && (
                <ResultBanner
                  status={result.status}
                  message={result.message}
                  elapsedMs={result.elapsedMs}
                  className="mt-3"
                />
              )}
            </section>
          )}

          {/* Per-job defaults */}
          {spec.jobParams.length > 0 && (
            <section>
              <SectionTitle>Per-job defaults</SectionTitle>
              <p className="mb-2 text-[11px] text-muted-foreground">
                Optional defaults copied into a job when this connector is
                selected. Leave blank to collect the value on each job.
              </p>
              <div className="grid gap-3 md:grid-cols-2">
                {spec.jobParams.map((p) => (
                  <ParamField
                    key={p.name}
                    param={optionalParam(p)}
                    value={draft.jobParamDefaults?.[p.name] ?? p.default ?? ""}
                    error={validation.defaults[p.name] ?? null}
                    onChange={(v) =>
                      actions.setConnectorJobParamDefault(p.name, v)
                    }
                  />
                ))}
              </div>
            </section>
          )}

          <Separator />
          <p className="text-[11px] text-muted-foreground">
            {draft.enabled
              ? "Enabled — this connector is selectable as a job's source/target."
              : "Save and enable this connector to use it when creating a job."}
          </p>
        </div>
      </ScrollArea>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

function optionalParam(param: ConnectorParam): ConnectorParam {
  return { ...param, required: false };
}

function unavailableDriverMessage(spec: ConnectorSpec): string {
  return `No installed driver for “${spec.driver}”. Build or install the matching celeritas extension, or point Celeritas at it, then return here and run Test.`;
}

function UnavailableDriverPanel({ spec }: { spec: ConnectorSpec }) {
  return (
    <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/[0.07] px-3 py-2 text-[12px] text-muted-foreground">
      <Ban className="mt-0.5 size-4 shrink-0 text-amber-300" />
      <div className="min-w-0">
        <div className="font-medium text-foreground">Driver unavailable</div>
        <div className="mt-0.5 leading-relaxed">
          {unavailableDriverMessage(spec)}
        </div>
        <div className="mt-1 font-mono text-[11px] text-muted-foreground">
          driver: {spec.driver}
        </div>
      </div>
    </div>
  );
}

function ReliabilityPanel({ draft }: { draft: ConnectorInstance }) {
  const last = draft.lastTest;
  const passed = last?.status === "pass";
  const unverified = !last || last.status === "example";
  if (draft.lock && draft.lock.status !== "locked") {
    return (
      <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/[0.07] px-3 py-2 text-[12px] text-muted-foreground">
        <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />
        <div className="min-w-0 leading-relaxed">
          <span className="font-medium text-foreground">
            Reproducibility warning.
          </span>{" "}
          {draft.lock.message ??
            `Connector lock status is ${draft.lock.status}.`}
        </div>
      </div>
    );
  }
  if (draft.enabled && passed) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-emerald-500/25 bg-emerald-500/[0.07] px-3 py-2 text-[12px] text-muted-foreground">
        <CircleCheck className="size-4 shrink-0 text-emerald-400" />
        Enabled with a passing test
        {last?.at ? ` from ${new Date(last.at).toLocaleString()}` : ""}.
      </div>
    );
  }
  return (
    <div className="flex items-start gap-2 rounded-md border border-outline-subtle bg-surface-panel/70 px-3 py-2 text-[12px] text-muted-foreground">
      <ShieldCheck className="mt-0.5 size-4 shrink-0 text-icon-tile-foreground" />
      <div className="min-w-0 leading-relaxed">
        <span className="font-medium text-foreground">
          Enablement is reliability-aware.
        </span>{" "}
        {passed
          ? "The last test passed, so this connector can be enabled without an override."
          : unverified
            ? "Run Test for a verified connection before enabling, or use Enable and confirm the override."
            : "The last test failed. Fix the connection and test again, or explicitly override when enabling."}
      </div>
    </div>
  );
}

/** A single connection/config field rendered by param kind. Exported so the job
 *  editor can reuse it for per-job connector inputs. */
export function ParamField({
  param,
  value,
  error,
  onChange,
}: {
  param: ConnectorParam;
  value: string;
  error?: string | null;
  onChange: (v: string) => void;
}) {
  const label = param.label || param.name;
  const input = param.input ?? (param.kind === "path" ? "file" : "text");
  const secretBearing = isSecretBearingParam(param);
  const secretRef = isSecretReferenceValue(value);
  const validationError =
    error ??
    (secretRef
      ? validationErrorFor("secret_ref", value)
      : validationErrorFor(input, value));
  const showStatus =
    Boolean(value) &&
    (secretRef ||
      (input !== "text" &&
        input !== "file" &&
        input !== "directory" &&
        input !== "multiline"));
  const isCodeLike = [
    "sql",
    "json",
    "toml",
    "yaml",
    "multiline",
    "columns",
  ].includes(input);
  const canBrowse = input === "directory" || input === "file";
  const [revealed, setRevealed] = useState(false);
  const placeholder =
    param.placeholder ??
    (secretBearing
      ? `inline value or secrets-keeper://secret/${param.name}#value`
      : undefined);
  const browse = async () => {
    const picked =
      input === "directory" ? await pickDirectory() : await pickDataFile();
    if (picked) onChange(picked);
  };
  return (
    <label className="space-y-1">
      <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
        {label}
        {param.required && <span className="text-amber-300/90">*</span>}
        <span className="font-mono text-[10px] opacity-60">
          {input !== "text" ? input : param.kind}
        </span>
        {param.advanced && (
          <Badge variant="outline" className="text-[9px]">
            advanced
          </Badge>
        )}
        {param.group && (
          <span className="text-[10px] opacity-60">{param.group}</span>
        )}
        {secretBearing && (
          <Badge
            variant="outline"
            className="ml-auto gap-1 text-[9px] text-muted-foreground"
          >
            <KeyRound className="size-2.5" /> secret ref ok
          </Badge>
        )}
        {showStatus && (
          <span
            className={cn(
              secretBearing ? "text-[10px]" : "ml-auto text-[10px]",
              validationError ? "text-destructive" : "text-emerald-400",
            )}
          >
            {validationError ? "invalid" : "valid"}
          </span>
        )}
      </span>
      {param.kind === "bool" ? (
        <select
          value={value || "false"}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-full rounded-md border border-input bg-surface-input px-2 text-[12px] outline-none"
        >
          <option value="true" className="bg-surface-panel">
            true
          </option>
          <option value="false" className="bg-surface-panel">
            false
          </option>
        </select>
      ) : param.kind === "enum" ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-full rounded-md border border-input bg-surface-input px-2 text-[12px] outline-none"
        >
          <option value="" className="bg-surface-panel">
            —
          </option>
          {param.options.map((o) => (
            <option key={o} value={o} className="bg-surface-panel">
              {o}
            </option>
          ))}
        </select>
      ) : (
        <>
          {isCodeLike ? (
            <textarea
              value={value}
              onChange={(e) => onChange(e.target.value)}
              placeholder={placeholder}
              aria-invalid={validationError ? true : undefined}
              rows={input === "columns" ? 2 : input === "multiline" ? 4 : 5}
              className={cn(
                "min-h-16 w-full resize-y rounded-md border border-input bg-surface-input px-2 py-1.5 font-mono text-[11.5px] leading-5 outline-none focus-visible:ring-2 focus-visible:ring-ring/60",
                validationError &&
                  "border-destructive/60 focus-visible:ring-destructive/30",
              )}
            />
          ) : (
            <span className="flex items-center gap-1.5">
              <Input
                type={
                  input === "date"
                    ? "date"
                    : input === "datetime"
                      ? "datetime-local"
                      : secretBearing && !secretRef
                        ? revealed
                          ? "text"
                          : "password"
                        : param.kind === "number"
                          ? "number"
                          : "text"
                }
                min={param.min != null ? String(param.min) : undefined}
                max={param.max != null ? String(param.max) : undefined}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                aria-invalid={validationError ? true : undefined}
                className={cn(
                  "h-8 min-w-0 text-[12px]",
                  (param.kind === "path" ||
                    param.kind === "url" ||
                    secretBearing ||
                    input !== "text") &&
                    "font-mono text-[11.5px]",
                  validationError &&
                    "border-destructive/60 focus-visible:ring-destructive/30",
                )}
              />
              {secretBearing && !secretRef && (
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  className="h-8 w-8 shrink-0"
                  onClick={(e) => {
                    e.preventDefault();
                    setRevealed((current) => !current);
                  }}
                  title={revealed ? "Hide secret" : "Reveal secret"}
                >
                  {revealed ? (
                    <EyeOff className="size-3.5" />
                  ) : (
                    <Eye className="size-3.5" />
                  )}
                </Button>
              )}
              {secretBearing && !secretRef && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-8 shrink-0 px-2 text-[11px]"
                  onClick={(e) => {
                    e.preventDefault();
                    onChange(`secrets-keeper://secret/${param.name}#value`);
                  }}
                  title="Use a secrets-keeper reference instead of an inline secret"
                >
                  <KeyRound className="size-3.5" /> Ref
                </Button>
              )}
              {param.unit && (
                <span className="shrink-0 text-[10px] text-muted-foreground">
                  {param.unit}
                </span>
              )}
              {canBrowse && (
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  className="h-8 w-8 shrink-0"
                  onClick={(e) => {
                    e.preventDefault();
                    void browse();
                  }}
                  title={
                    input === "directory" ? "Choose folder" : "Choose file"
                  }
                >
                  {input === "directory" ? (
                    <FolderOpen className="size-3.5" />
                  ) : (
                    <FileSearch className="size-3.5" />
                  )}
                </Button>
              )}
            </span>
          )}
          {["regex", "glob", "columns"].includes(input) &&
            (value || validationError) && (
              <span
                className={cn(
                  "block min-h-7 overflow-hidden rounded-md border bg-surface-input px-2 py-1 font-mono text-[11px] leading-5",
                  validationError
                    ? "border-destructive/40"
                    : "border-outline-subtle",
                )}
              >
                {value ? (
                  <InputPreview input={input} value={value} />
                ) : (
                  <span className="text-muted-foreground">
                    {param.placeholder}
                  </span>
                )}
              </span>
            )}
        </>
      )}
      {(param.help || validationError || param.dependsOn) && (
        <span
          className={cn(
            "block text-[10px]",
            validationError ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {validationError ??
            param.help ??
            (param.dependsOn ? `Depends on ${param.dependsOn}` : null)}
        </span>
      )}
      {secretBearing && !validationError && (
        <span className="block text-[10px] text-muted-foreground">
          Accepts inline values or{" "}
          <span className="font-mono">secrets-keeper://mount/path#field</span>.
          Configure the backend in Settings.
        </span>
      )}
    </label>
  );
}

function isSecretBearingParam(param: ConnectorParam): boolean {
  const haystack =
    `${param.name} ${param.label ?? ""} ${param.help ?? ""}`.toLowerCase();
  return (
    param.kind === "secret" ||
    param.sensitive === true ||
    param.input === "secret_ref" ||
    /\b(password|passwd|secret|token|api[_ -]?key|access[_ -]?key|private[_ -]?key|credential)\b/.test(
      haystack,
    )
  );
}

function isSecretReferenceValue(value: string): boolean {
  const trimmed = value.trim();
  return (
    trimmed.startsWith("secrets-keeper://") ||
    trimmed.startsWith("secret-session://")
  );
}

function connectorFieldError(
  param: ConnectorParam,
  value: string,
): string | null {
  const trimmed = value.trim();
  if (!trimmed) return param.required ? "Required." : null;
  if (
    param.kind === "enum" &&
    param.options.length > 0 &&
    !param.options.includes(value)
  ) {
    return `Choose one of: ${param.options.join(", ")}`;
  }
  if (param.kind === "bool" && value !== "true" && value !== "false") {
    return "Choose true or false.";
  }
  if (param.kind === "number" && Number.isNaN(Number(value))) {
    return "Enter a valid number.";
  }
  if (
    param.min != null &&
    !Number.isNaN(Number(value)) &&
    Number(value) < param.min
  ) {
    return `Must be at least ${param.min}.`;
  }
  if (
    param.max != null &&
    !Number.isNaN(Number(value)) &&
    Number(value) > param.max
  ) {
    return `Must be at most ${param.max}.`;
  }
  const input = param.input ?? (param.kind === "path" ? "file" : "text");
  if (isSecretBearingParam(param) && isSecretReferenceValue(value)) {
    return formatInlineFieldError(
      {
        code: "VALIDATION",
        message: validationErrorFor("secret_ref", value),
        field: param.name,
      },
      param.name,
    );
  }
  return validationErrorFor(input, value);
}

function connectorDraftValidation(
  spec: ConnectorSpec,
  draft: ConnectorInstance,
  previewJobParams: Record<string, string>,
) {
  const connection = Object.fromEntries(
    spec.params.map((param) => [
      param.name,
      connectorFieldError(param, draft.params[param.name] ?? ""),
    ]),
  ) as Record<string, string | null>;
  const defaults = Object.fromEntries(
    spec.jobParams.map((param) => [
      param.name,
      connectorFieldError(
        optionalParam(param),
        draft.jobParamDefaults?.[param.name] ?? "",
      ),
    ]),
  ) as Record<string, string | null>;
  const preview = Object.fromEntries(
    spec.jobParams.map((param) => [
      param.name,
      connectorFieldError(
        optionalParam(param),
        previewJobParams[param.name] ?? "",
      ),
    ]),
  ) as Record<string, string | null>;
  return {
    connection,
    defaults,
    preview,
    connectionInvalid: Object.values(connection).some(Boolean),
    defaultsInvalid: Object.values(defaults).some(Boolean),
    previewInvalid: Object.values(preview).some(Boolean),
  };
}

function validationErrorFor(input: string, value: string): string | null {
  if (!value.trim()) return null;
  switch (input) {
    case "regex":
      try {
        new RegExp(value);
        return null;
      } catch (e) {
        return e instanceof Error ? e.message : "Invalid regular expression";
      }
    case "glob":
      return validGlob(value) ? null : "Invalid glob pattern";
    case "json":
      try {
        JSON.parse(value);
        return null;
      } catch (e) {
        return e instanceof Error ? e.message : "Invalid JSON";
      }
    case "cron":
      return isValidCron(value) ? null : "Invalid 5-field cron expression";
    case "date":
      return /^\d{4}-\d{2}-\d{2}$/.test(value) ? null : "Use YYYY-MM-DD";
    case "datetime":
      return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(value) ||
        !Number.isNaN(Date.parse(value))
        ? null
        : "Use an ISO date/time";
    case "duration":
      return /^(\d+(\.\d+)?\s*(ms|s|m|h|d|w|mo|y)|P(T?\d+[YMWDHS]|[\dYMWDTHMS.]+))$/i.test(
        value.trim(),
      )
        ? null
        : "Use a duration like 30d, 6m, 1y, or PT5M";
    case "identifier":
      return /^[A-Za-z_][\w.:-]*$/.test(value.trim())
        ? null
        : "Use letters, numbers, _, ., :, or -; start with a letter or _";
    case "secret_ref":
      return isValidSecretsKeeperRef(value.trim()) ||
        isValidSecretSessionRef(value.trim())
        ? null
        : "Use secrets-keeper://mount/path#field";
    case "columns":
      return value
        .split(",")
        .map((v) => v.trim())
        .filter(Boolean)
        .every((v) => /^[A-Za-z_][\w.]*$/.test(v))
        ? null
        : "Use comma-separated column identifiers";
    case "toml":
    case "yaml":
      return validKeyValueBlock(value)
        ? null
        : `Invalid ${input.toUpperCase()}-style key/value block`;
    default:
      return null;
  }
}

function isValidSecretsKeeperRef(value: string): boolean {
  const match = /^secrets-keeper:\/\/([^/#\s]+)\/([^#\s]+)#([^#/\s]+)$/.exec(
    value,
  );
  return Boolean(match?.[1] && match?.[2] && match?.[3]);
}

function isValidSecretSessionRef(value: string): boolean {
  return /^secret-session:\/\/[A-Za-z0-9._/-]+$/.test(value);
}

function validGlob(value: string): boolean {
  let depth = 0;
  for (const ch of value) {
    if (ch === "[") depth++;
    if (ch === "]") depth--;
    if (depth < 0) return false;
  }
  return depth === 0 && !/\0/.test(value);
}

function validKeyValueBlock(value: string): boolean {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .every(
      (line) =>
        line.includes(":") || line.includes("=") || line.startsWith("-"),
    );
}

function InputPreview({ input, value }: { input: string; value: string }) {
  if (input === "columns") {
    return (
      <>
        {value.split(",").map((part, index) => (
          <span
            key={`${part}-${index}`}
            className="mr-1 inline-flex rounded border border-outline-subtle bg-surface-panel px-1 text-foreground"
          >
            {part.trim() || "?"}
          </span>
        ))}
      </>
    );
  }
  const pieces =
    value.match(/(\\.|[[\]{}()|^$+*?.]|[^\\[\]{}()|^$+*?.]+)/g) ?? [];
  return (
    <>
      {pieces.map((part, index) => {
        const cls = part.startsWith("\\")
          ? "text-sky-300"
          : /^[[\]{}()]$/.test(part)
            ? "text-amber-300"
            : /^[|^$+*?.]$/.test(part) ||
                (input === "glob" && /[*?]/.test(part))
              ? "text-fuchsia-300"
              : "text-foreground";
        return (
          <span key={`${part}-${index}`} className={cls}>
            {part}
          </span>
        );
      })}
    </>
  );
}

export function ResultBanner({
  status,
  message,
  elapsedMs,
  className,
}: {
  status: string;
  message: string;
  elapsedMs?: number | undefined;
  className?: string | undefined;
}) {
  const tone =
    status === "pass" || status === "complete"
      ? {
          icon: CircleCheck,
          color: "text-emerald-400",
          border: "border-emerald-500/30",
        }
      : status === "example"
        ? {
            icon: AlertTriangle,
            color: "text-amber-300",
            border: "border-amber-500/30",
          }
        : {
            icon: CircleX,
            color: "text-destructive",
            border: "border-destructive/30",
          };
  const Icon = tone.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 3 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "flex items-start gap-2 rounded-md border bg-surface-panel px-3 py-2 text-[12px]",
        tone.border,
        className,
      )}
    >
      <Icon className={cn("mt-0.5 size-4 shrink-0", tone.color)} />
      <span className="min-w-0 flex-1 leading-relaxed">{message}</span>
      <SimulatedBadge className="mt-0.5 shrink-0" />
      {elapsedMs != null && (
        <span className="shrink-0 text-[10px] text-muted-foreground">
          {elapsedMs} ms
        </span>
      )}
    </motion.div>
  );
}
