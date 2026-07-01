import { useEffect, useRef, useState, type ReactNode } from "react";
import { proxy, useSnapshot } from "valtio";
import { motion } from "framer-motion";
import {
  Braces,
  Plus,
  Play,
  Save,
  Trash2,
  Workflow,
  Loader2,
  ChevronRight,
  CircleCheck,
  CircleX,
  Clock,
  GripVertical,
  Code2,
  FileUp,
  FlaskConical,
  Gauge,
  Power,
  CalendarClock,
  History,
  Settings2,
  Eye,
  Boxes,
  GitBranch,
  GitCommit,
  RefreshCw,
  Pin,
  ServerCog,
  SquareArrowOutUpRight,
  Type,
  ClipboardCopy,
  FileSearch,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ButtonGroup, ButtonGroupItem } from "@/components/ui/button-group";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowDownToLine, ArrowUpFromLine } from "lucide-react";
import { OutputTable } from "./OutputTable";
import { SimulatedBadge, SimulatedNotice } from "@/components/ui/simulated";
import { TemplateCombobox } from "./TemplateCombobox";
import { CodeEditor } from "./CodeEditor";
import { CronEditor } from "./CronEditor";
import { ConnectorCombobox } from "./ConnectorCombobox";
import { FilePathDropzone } from "./FilePathDropzone";
import { ParamField } from "./ConnectorsView";
import { actions, store, connectorInstance, connectorSpec } from "@/lib/store";
import { deriveJobReadiness, type JobReadiness } from "@/lib/job-readiness";
import { listBindingConformance } from "@/lib/api";
import { cn } from "@/lib/utils";
import type {
  Binding,
  BindingConformance,
  ConnectorInstance,
  Job,
  JobRun,
  JobRunStep,
  RunResult,
} from "@/lib/types";
import { EmptyPanel, ErrorPanel, SkeletonRows } from "@/components/AsyncStates";

const JOB_API_BASE_KEY = "jobs:api-base-url";
const DEFAULT_JOB_API_BASE = "http://127.0.0.1:7700";

type JobApiEndpoint = {
  method: "GET" | "POST" | "PUT" | "DELETE";
  path: string;
  summary: string;
  request?: string;
  response: string;
  body?: unknown;
};

type JobSetupSection =
  | "source"
  | "steps"
  | "output"
  | "validate"
  | "schedule"
  | "runs";
type RunDiagnosticsScope = "run" | "steps";

const sampleJobBody = {
  id: "job-nightly-risk",
  name: "Nightly risk rollup",
  description: "Runs a chained flow over the configured source.",
  enabled: true,
  definition: {
    source: "/data/returns.csv",
    steps: [
      {
        kind: "unit",
        template: "mean_return",
        args: {
          value_column: "ret",
          group_column: "symbol",
        },
      },
      {
        kind: "sql",
        template: "inline_sql",
        args: {},
        sql: "SELECT symbol, avg(ret) AS mean_ret FROM input GROUP BY symbol",
      },
    ],
    schedule: "0 2 * * *",
    sink: "s3://bucket/qf-intermediates",
    sourceConnectorId: "filesystem-source",
    sourceConnectorParams: {
      filename_regex: ".*\\.csv$",
    },
    targetConnectorId: "filesystem-target",
    targetConnectorParams: {
      output_path: "/data/out/nightly-risk.parquet",
    },
  },
};

const sampleRunBody = {
  targetId: "embedded",
};

const jobApiEndpoints: JobApiEndpoint[] = [
  {
    method: "GET",
    path: "/healthz",
    summary: "Check API process health.",
    response: "200 application/json: { status, service }",
  },
  {
    method: "GET",
    path: "/v1/jobs",
    summary: "List saved jobs.",
    response: "200 application/json: Job[]",
  },
  {
    method: "POST",
    path: "/v1/jobs",
    summary: "Create or replace a job.",
    request: "application/json: Job",
    response: "201 application/json: Job",
    body: sampleJobBody,
  },
  {
    method: "GET",
    path: "/v1/jobs/{jobId}",
    summary: "Read one job.",
    response: "200 application/json: Job; 404 when unknown.",
  },
  {
    method: "PUT",
    path: "/v1/jobs/{jobId}",
    summary: "Update one job. The path id is authoritative.",
    request: "application/json: Job",
    response: "200 application/json: Job",
    body: sampleJobBody,
  },
  {
    method: "DELETE",
    path: "/v1/jobs/{jobId}",
    summary: "Delete one job and its run history.",
    response: "204 no content.",
  },
  {
    method: "POST",
    path: "/v1/jobs/{jobId}/run",
    summary: "Trigger a job run.",
    request: "application/json: RunJobRequest",
    response: "200 application/json: { runId }",
    body: sampleRunBody,
  },
  {
    method: "GET",
    path: "/v1/runs?jobId=&status=running",
    summary: "List recent runs, optionally filtered by job id and status.",
    response: "200 application/json: JobRun[]",
  },
  {
    method: "GET",
    path: "/v1/runs/{runId}/steps",
    summary: "List recorded run step results.",
    response: "200 application/json: JobRunStep[]",
  },
];

const jobApiOpenApiSpec = {
  openapi: "3.0.3",
  info: {
    title: "Celeritas Job API",
    version: "0.1.0",
    description:
      "Opt-in REST API exposed by celeritas-scheduler on a port separate from the coordinator.",
  },
  paths: {
    "/healthz": {
      get: {
        summary: "Health check",
        responses: { "200": { description: "API is reachable" } },
      },
    },
    "/v1/jobs": {
      get: {
        summary: "List jobs",
        responses: { "200": { description: "Jobs" } },
      },
      post: {
        summary: "Create or replace a job",
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/Job" },
            },
          },
        },
        responses: { "201": { description: "Created job" } },
      },
    },
    "/v1/jobs/{jobId}": {
      get: {
        summary: "Get one job",
        parameters: [
          {
            name: "jobId",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: {
          "200": { description: "Job" },
          "404": { description: "Not found" },
        },
      },
      put: {
        summary: "Update one job",
        parameters: [
          {
            name: "jobId",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/Job" },
            },
          },
        },
        responses: { "200": { description: "Updated job" } },
      },
      delete: {
        summary: "Delete one job",
        parameters: [
          {
            name: "jobId",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: { "204": { description: "Deleted" } },
      },
    },
    "/v1/jobs/{jobId}/run": {
      post: {
        summary: "Trigger a job run",
        parameters: [
          {
            name: "jobId",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        requestBody: {
          required: true,
          content: {
            "application/json": {
              schema: { $ref: "#/components/schemas/RunJobRequest" },
            },
          },
        },
        responses: { "200": { description: "Run id" } },
      },
    },
    "/v1/runs": {
      get: {
        summary: "List runs",
        parameters: [
          {
            name: "jobId",
            in: "query",
            required: false,
            schema: { type: "string" },
          },
          {
            name: "status",
            in: "query",
            required: false,
            schema: {
              type: "string",
              enum: ["queued", "running", "complete", "failed"],
            },
          },
        ],
        responses: { "200": { description: "Runs" } },
      },
    },
    "/v1/runs/{runId}/steps": {
      get: {
        summary: "List run steps",
        parameters: [
          {
            name: "runId",
            in: "path",
            required: true,
            schema: { type: "string" },
          },
        ],
        responses: { "200": { description: "Run steps" } },
      },
    },
  },
  components: {
    schemas: {
      Job: {
        type: "object",
        required: ["id", "name", "definition", "enabled"],
        properties: {
          id: { type: "string", example: "job-nightly-risk" },
          name: { type: "string", example: "Nightly risk rollup" },
          description: { type: "string" },
          enabled: { type: "boolean" },
          definition: { $ref: "#/components/schemas/JobDefinition" },
        },
      },
      JobDefinition: {
        type: "object",
        required: ["source", "steps"],
        properties: {
          source: { type: "string", example: "/data/returns.csv" },
          steps: {
            type: "array",
            items: { $ref: "#/components/schemas/JobStep" },
          },
          schedule: { type: "string", example: "0 2 * * *" },
          sink: { type: "string", example: "s3://bucket/qf-intermediates" },
          sourceConnectorId: { type: "string" },
          sourceBindingId: { type: "string" },
          sourceConnectorParams: {
            type: "object",
            additionalProperties: true,
          },
          targetConnectorId: { type: "string" },
          targetBindingId: { type: "string" },
          targetConnectorParams: {
            type: "object",
            additionalProperties: true,
          },
        },
      },
      JobStep: {
        type: "object",
        required: ["template", "args"],
        properties: {
          kind: { type: "string", enum: ["unit", "sql", "markdown"] },
          template: { type: "string", example: "mean_return" },
          args: { type: "object", additionalProperties: true },
          sql: { type: "string" },
          markdown: { type: "string" },
        },
      },
      RunJobRequest: {
        type: "object",
        required: ["targetId"],
        properties: {
          targetId: { type: "string", example: "embedded" },
        },
      },
    },
  },
};

export function JobsView() {
  const snap = useSnapshot(store);
  const local = useRef(proxy({ advancedOpen: false, tab: "pipeline" })).current;
  const localSnap = useSnapshot(local);
  const draft = snap.jobDraft;
  const readiness = draft
    ? deriveJobReadiness({
        draft: draft as Job,
        jobSourceMode: snap.jobSourceMode,
        templates: store.templates,
        connectorSpecs: store.connectorSpecs,
        connectorInstances: store.connectorInstances,
        bindings: store.bindings,
        sourceColumns: store.sourceColumns,
        stepColumns: store.stepColumns,
        sourcePreview: store.sourcePreview,
        dryResult: store.dryResult,
        activeTargetId: store.activeTargetId,
        targets: store.targets,
      })
    : null;
  const primary = readiness?.nextAction;
  const focusSetupSection = (section: JobSetupSection) => {
    const tab =
      section === "schedule"
        ? "schedule"
        : section === "runs"
          ? "runs"
          : "pipeline";
    local.tab = tab;
    window.setTimeout(() => {
      const el = document.querySelector<HTMLElement>(
        `[data-job-section="${section}"]`,
      );
      el?.scrollIntoView({ block: "start", behavior: "smooth" });
      el?.focus({ preventScroll: true });
    }, 0);
  };
  const focusFirstBlocker = () => {
    const blocker = readiness?.blockers[0];
    focusSetupSection(sectionForReadinessItem(blocker?.id));
  };
  const runPrimaryAction = () => {
    if (!primary || primary.disabled) return;
    if (primary.kind === "chooseSource") {
      focusSetupSection("source");
      return;
    }
    if (primary.kind === "testSource") {
      void actions.testSource();
      focusSetupSection("source");
      return;
    }
    if (primary.kind === "addStep") {
      local.tab = "pipeline";
      actions.addStep();
      window.setTimeout(() => focusSetupSection("steps"), 0);
      return;
    }
    if (primary.kind === "fillParams" || primary.kind === "resolveBlockers") {
      focusFirstBlocker();
      return;
    }
    if (primary.kind === "dryRun") {
      void actions.dryRun();
      focusSetupSection("validate");
      return;
    }
    if (primary.kind === "runNow") void actions.runDraft();
  };
  const sourceBinding = draft?.definition.sourceBindingId
    ? snap.bindings.find((b) => b.id === draft.definition.sourceBindingId)
    : null;
  const sourceConnectorId =
    sourceBinding?.instanceId ?? draft?.definition.sourceConnectorId ?? null;
  const targetBinding = draft?.definition.targetBindingId
    ? snap.bindings.find((b) => b.id === draft.definition.targetBindingId)
    : null;
  const targetConnectorId =
    targetBinding?.instanceId ?? draft?.definition.targetConnectorId ?? null;
  const copyRunDiagnostics = (run: JobRun, scope: RunDiagnosticsScope) => {
    const steps =
      snap.openRunId === run.id ? (snap.runSteps as JobRunStep[]) : [];
    const text = formatRunDiagnostics({
      job: draft as Job,
      run,
      steps,
      scope,
      sourceMode: snap.jobSourceMode,
      targetId: snap.activeTargetId,
    });
    void navigator.clipboard?.writeText(text);
    actions.notify({
      tone: "ok",
      message:
        scope === "steps"
          ? "Copied step diagnostics."
          : "Copied run diagnostics.",
    });
  };
  return (
    <div className="flex min-h-0 flex-1">
      {/* Jobs list */}
      <aside className="flex w-[266px] shrink-0 flex-col border-r border-outline-subtle bg-surface-sidebar">
        <div className="flex h-9 items-center justify-between border-b border-outline-subtle bg-surface-header px-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Jobs · bound flows
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2"
            onClick={() => actions.newJob()}
          >
            <Plus className="size-4" /> New
          </Button>
        </div>
        <ScrollArea className="flex-1 pb-3 pl-1.5 pr-2 pt-2">
          {snap.jobsLoading && snap.jobs.length === 0 ? (
            <SkeletonRows count={5} />
          ) : null}
          {snap.jobsError ? (
            <ErrorPanel
              title="Could not load jobs"
              message={snap.jobsError}
              onRetry={() => actions.loadJobs()}
            />
          ) : null}
          {!snap.jobsLoading && !snap.jobsError && snap.jobs.length === 0 ? (
            <EmptyPanel
              title="No jobs yet"
              message="Create a bound flow to chain steps, test a source, and schedule runs."
              action={
                <Button size="sm" onClick={() => actions.newJob()}>
                  <Plus className="size-3.5" /> New job
                </Button>
              }
            />
          ) : null}
          {!snap.jobsError &&
            snap.jobs.map((j) => (
              <button
                key={j.id}
                onClick={() => actions.editJob(j.id)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md border px-2 py-2 text-left text-[12.5px] font-medium tracking-[-0.015em] transition-colors",
                  draft?.id === j.id
                    ? "border-outline-subtle bg-surface-active text-foreground"
                    : "border-transparent text-muted-foreground hover:bg-surface-hover hover:text-foreground",
                )}
              >
                <Workflow
                  className={cn(
                    "size-4 shrink-0",
                    draft?.id === j.id && "text-icon-tile-foreground",
                  )}
                />
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{j.name}</span>
                  <span className="block text-[10px] text-muted-foreground">
                    {j.definition.steps.length} steps
                  </span>
                </span>
                <span
                  className={cn(
                    "size-1.5 rounded-md",
                    j.enabled ? "bg-emerald-400" : "bg-muted-foreground/40",
                  )}
                />
              </button>
            ))}
        </ScrollArea>
      </aside>

      {/* Editor */}
      {!draft ? (
        <div className="grid flex-1 place-items-center p-6 text-sm text-muted-foreground">
          <div className="w-full max-w-sm rounded-md border border-outline-subtle bg-surface-panel p-5 text-center">
            <div className="mx-auto mb-3 grid size-9 place-items-center rounded-md border border-transparent bg-primary/[0.14]">
              <Workflow className="size-4 text-icon-tile-foreground" />
            </div>
            <div className="text-sm font-semibold text-foreground">
              No job selected
            </div>
            <p className="mt-1 text-[12px] leading-relaxed text-muted-foreground">
              Create a compact workflow to chain templates, dry run against a
              source file, and schedule execution.
            </p>
            <Button size="sm" className="mt-4" onClick={() => actions.newJob()}>
              <Plus className="size-3.5" /> New job
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex min-w-0 flex-1 flex-col">
          <SimulatedNotice className="mx-6 mt-4">
            Preview mode: run history, dry runs, and source tests may show
            simulated output rather than live engine results.
          </SimulatedNotice>
          <div className="flex items-center gap-2 border-b border-outline-subtle px-6 py-3">
            <Input
              value={draft.name}
              onChange={(e) => actions.setJobField({ name: e.target.value })}
              className="h-8 max-w-xs text-sm font-semibold"
            />
            <Button
              size="sm"
              variant={draft.enabled ? "default" : "outline"}
              onClick={() => actions.toggleDraftEnabled()}
              title={
                draft.enabled
                  ? "Job is enabled — click to disable"
                  : "Job is disabled — click to enable"
              }
            >
              <Power className="size-4" />{" "}
              {draft.enabled ? "Enabled" : "Disabled"}
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => actions.saveDraft()}
            >
              <Save className="size-4" /> Save
            </Button>
            <Button
              size="sm"
              onClick={runPrimaryAction}
              disabled={
                snap.jobBusy ||
                snap.dryBusy ||
                snap.sourcePreviewBusy ||
                primary?.disabled
              }
              title={readiness?.blockers[0]?.message ?? "Next setup action"}
            >
              {snap.jobBusy || snap.dryBusy || snap.sourcePreviewBusy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <PrimaryActionIcon action={primary?.kind} />
              )}
              {primary?.label ?? "Run now"}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              className="ml-auto"
              title="Delete job"
              onClick={() =>
                actions.requestConfirm(
                  {
                    title: "Delete job",
                    message: `Delete “${draft.name}”? This removes the job and its run history and cannot be undone.`,
                    confirmLabel: "Delete",
                    tone: "danger",
                  },
                  () => actions.deleteJob(draft.id),
                )
              }
            >
              <Trash2 className="size-4" />
            </Button>
          </div>
          {draft.definition.packBinding && (
            <PackBindingStrip job={draft as Job} />
          )}
          {readiness && <JobReadinessChecklist readiness={readiness} />}
          {readiness && (
            <CreateJobWizardShell
              draft={draft as Job}
              readiness={readiness}
              activeTab={localSnap.tab}
              setTab={(tab) => (local.tab = tab)}
              focusSection={focusSetupSection}
              runPrimaryAction={runPrimaryAction}
              busy={snap.jobBusy || snap.dryBusy || snap.sourcePreviewBusy}
            />
          )}

          <Tabs
            key={draft.id}
            value={localSnap.tab}
            onValueChange={(value) => (local.tab = value)}
            className="flex min-h-0 flex-1 flex-col"
          >
            <div className="border-b border-outline-subtle px-6 py-1.5">
              <div className="flex items-center gap-2">
                <TabsList>
                  <TabsTrigger value="pipeline">
                    <Settings2 className="size-4" /> Pipeline
                  </TabsTrigger>
                  <TabsTrigger value="schedule">
                    <CalendarClock className="size-4" /> Schedule
                  </TabsTrigger>
                  <TabsTrigger value="runs">
                    <History className="size-4" /> Run history
                  </TabsTrigger>
                  {localSnap.advancedOpen && (
                    <TabsTrigger value="api">
                      <ServerCog className="size-4" /> REST API
                    </TabsTrigger>
                  )}
                </TabsList>
                <Button
                  size="sm"
                  variant={localSnap.advancedOpen ? "secondary" : "ghost"}
                  className="h-8 px-2"
                  onClick={() => {
                    const next = !local.advancedOpen;
                    local.advancedOpen = next;
                    if (!next && local.tab === "api") local.tab = "pipeline";
                  }}
                  title={
                    localSnap.advancedOpen
                      ? "Hide advanced job controls"
                      : "Show advanced job controls"
                  }
                >
                  <ServerCog className="size-4" /> Advanced
                </Button>
              </div>
            </div>

            <TabsContent value="pipeline" className="min-h-0 flex-1">
              <ScrollArea className="h-full">
                <div className="space-y-4 p-5">
                  {/* Source: a data file XOR a source connector (never both). */}
                  <div
                    data-job-section="source"
                    tabIndex={-1}
                    className="scroll-mt-3 focus:outline-none"
                  >
                    <div className="mb-2 flex items-center gap-3">
                      <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        Source
                      </span>
                      <ButtonGroup>
                        <ButtonGroupItem
                          active={snap.jobSourceMode === "file"}
                          onClick={() => actions.setJobSourceMode("file")}
                        >
                          <FileUp /> CSV / file
                        </ButtonGroupItem>
                        <ButtonGroupItem
                          active={snap.jobSourceMode === "connector"}
                          onClick={() => actions.setJobSourceMode("connector")}
                        >
                          <ArrowDownToLine /> Connector
                        </ButtonGroupItem>
                      </ButtonGroup>
                      <Button
                        size="sm"
                        variant="secondary"
                        className="ml-auto"
                        onClick={() => actions.testSource()}
                        disabled={
                          snap.sourcePreviewBusy ||
                          (snap.jobSourceMode === "file"
                            ? !draft.definition.source.trim()
                            : !draft.definition.sourceBindingId &&
                              !draft.definition.sourceConnectorId)
                        }
                        title="Sample rows from the source without running the steps"
                      >
                        {snap.sourcePreviewBusy ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          <Eye className="size-4" />
                        )}
                        Test source
                      </Button>
                    </div>
                    {snap.jobSourceMode === "file" ? (
                      <FilePathDropzone
                        value={draft.definition.source}
                        onChange={(path) => actions.setDraftSource(path)}
                        onBrowse={() => actions.browseSource()}
                        placeholder="/data/returns.csv"
                        status={
                          snap.sourceInspecting
                            ? "inspecting..."
                            : snap.sourceColumns.length
                              ? `${snap.sourceColumns.length} cols`
                              : "no columns yet"
                        }
                      />
                    ) : (
                      <div className="grid gap-2">
                        <SourceBindingPicker
                          contractRef={requiredSourceContract(
                            draft as Job,
                            snap.templates,
                          )}
                          selectedId={draft.definition.sourceBindingId}
                        />
                        {!draft.definition.sourceBindingId && (
                          <ConnectorBind
                            role="source"
                            label=""
                            icon={ArrowDownToLine}
                            instances={
                              snap.enabledSources as ConnectorInstance[]
                            }
                            selectedId={draft.definition.sourceConnectorId}
                            params={draft.definition.sourceConnectorParams}
                            onSelect={(id) => actions.setJobSourceConnector(id)}
                            onParam={(k, v) => actions.setJobSourceParam(k, v)}
                            onConfigure={() =>
                              actions.configureJobConnector(
                                "source",
                                requiredSourceSpec(draft as Job),
                              )
                            }
                          />
                        )}
                      </div>
                    )}
                    <SourcePreview />
                  </div>

                  {/* Output: where final results are exported. The shared sink
                      (remote inter-step staging) is only meaningful on a remote
                      target — the embedded engine chains in-process and ignores
                      it — so it's hidden otherwise. */}
                  <div
                    data-job-section="output"
                    tabIndex={-1}
                    className="grid scroll-mt-3 gap-3 focus:outline-none md:grid-cols-2"
                  >
                    <ConnectorBind
                      role="target"
                      label="Target connector (optional)"
                      icon={ArrowUpFromLine}
                      instances={snap.enabledTargets as ConnectorInstance[]}
                      selectedId={draft.definition.targetConnectorId}
                      params={draft.definition.targetConnectorParams}
                      onSelect={(id) => actions.setJobTargetConnector(id)}
                      onParam={(k, v) => actions.setJobTargetParam(k, v)}
                      onConfigure={() =>
                        actions.configureJobConnector("target")
                      }
                    />
                    {snap.targets.find((t) => t.id === snap.activeTargetId)
                      ?.kind === "remote" &&
                      (localSnap.advancedOpen ? (
                        <Labeled label="Shared sink (remote inter-step staging)">
                          <Input
                            value={draft.definition.sink ?? ""}
                            onChange={(e) =>
                              actions.setDraftSink(e.target.value)
                            }
                            placeholder="s3://bucket/qf-intermediates  (or a shared dir)"
                            className="font-mono text-[12px]"
                          />
                          <div className="mt-1 text-[10px] text-muted-foreground">
                            A URI all workers can reach; each step writes here
                            for the next to read.
                          </div>
                        </Labeled>
                      ) : (
                        <div className="flex items-center gap-2 rounded-md border border-outline-subtle bg-surface-panel px-3 py-2 text-[11px] text-muted-foreground">
                          <ServerCog className="size-3.5" />
                          <span className="min-w-0 flex-1">
                            Remote staging settings are hidden in Advanced.
                          </span>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 px-2 text-[10px]"
                            onClick={() => {
                              local.advancedOpen = true;
                              local.tab = "pipeline";
                            }}
                          >
                            Show
                          </Button>
                        </div>
                      ))}
                  </div>

                  {/* Steps */}
                  <div
                    data-job-section="steps"
                    tabIndex={-1}
                    className="scroll-mt-3 focus:outline-none"
                  >
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        ETL steps
                      </span>
                      <Button
                        size="sm"
                        variant="secondary"
                        className="h-7"
                        onClick={() => actions.addStep()}
                      >
                        <Plus className="size-3.5" /> Unit
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7"
                        onClick={() => actions.addSqlStep()}
                      >
                        <Code2 className="size-3.5" /> SQL
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7"
                        onClick={() => actions.addMarkdownStep()}
                      >
                        <Type className="size-3.5" /> Runbook
                      </Button>
                    </div>
                    <div className="space-y-3">
                      {draft.definition.steps.map((step, idx) => (
                        <StepCard
                          key={idx}
                          idx={idx}
                          kind={step.kind}
                          template={step.template}
                          args={step.args}
                          sql={step.sql}
                          markdown={step.markdown}
                        />
                      ))}
                      {draft.definition.steps.length === 0 && (
                        <div className="rounded-md border border-dashed border-outline-subtle p-6 text-center text-sm text-muted-foreground">
                          Add a template step — each runs over the source and
                          records to history.
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Dry-run output (ephemeral — not recorded to history) */}
                  {(snap.dryBusy || snap.dryResult) && (
                    <div
                      data-job-section="validate"
                      tabIndex={-1}
                      className="scroll-mt-3 focus:outline-none"
                    >
                      <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        <Gauge className="size-3.5" /> Dry-run output
                      </div>
                      {snap.dryBusy ? (
                        <div className="grid h-28 place-items-center rounded-md border border-outline-subtle bg-surface-panel text-sm text-muted-foreground">
                          <span className="flex items-center gap-2">
                            <Loader2 className="size-4 animate-spin" /> running
                            over source…
                          </span>
                        </div>
                      ) : snap.dryResult ? (
                        <div className="overflow-hidden rounded-md border border-outline-subtle bg-surface-panel">
                          <div className="flex items-center gap-2 border-b border-outline-subtle px-4 py-2 text-sm">
                            {snap.dryResult.status === "complete" ? (
                              <CircleCheck className="size-4 text-emerald-400" />
                            ) : (
                              <CircleX className="size-4 text-destructive" />
                            )}
                            <span>{snap.dryResult.status}</span>
                            <SimulatedBadge />
                            {snap.dryResult.elapsedMs != null && (
                              <span className="ml-auto text-xs text-muted-foreground">
                                {snap.dryResult.elapsedMs} ms ·{" "}
                                {snap.dryResult.rowCount ??
                                  snap.dryResult.rows.length}{" "}
                                rows
                              </span>
                            )}
                          </div>
                          {snap.dryResult.error ? (
                            <pre className="whitespace-pre-wrap p-4 font-mono text-xs text-destructive">
                              {snap.dryResult.error}
                            </pre>
                          ) : (
                            <OutputTable rows={snap.dryResult.rows} />
                          )}
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="schedule" className="min-h-0 flex-1">
              <ScrollArea className="h-full">
                <ScheduleTab draft={draft as Job} />
              </ScrollArea>
            </TabsContent>

            <TabsContent value="runs" className="min-h-0 flex-1">
              <ScrollArea className="h-full">
                <div
                  data-job-section="runs"
                  tabIndex={-1}
                  className="space-y-1.5 p-6 focus:outline-none"
                >
                  {snap.runsLoading && snap.runs.length === 0 ? (
                    <SkeletonRows count={4} />
                  ) : null}
                  {snap.runsError ? (
                    <ErrorPanel
                      title="Could not load runs"
                      message={snap.runsError}
                      onRetry={() => draft && actions.refreshRuns(draft.id)}
                    />
                  ) : null}
                  {!snap.runsLoading &&
                  !snap.runsError &&
                  snap.runs.length === 0 ? (
                    <EmptyPanel
                      title="No runs yet"
                      message="Run this job once to capture history and diagnostics here."
                    />
                  ) : (
                    <>
                      {snap.runs.map((r) => (
                        <div key={r.id}>
                        <button
                          onClick={() => actions.openRun(r.id)}
                          className="flex w-full items-center gap-2 rounded-md border border-outline-subtle px-3 py-2 text-left text-[13px] hover:bg-surface-panel"
                        >
                          <ChevronRight
                            className={cn(
                              "size-3.5 transition-transform",
                              snap.openRunId === r.id && "rotate-90",
                            )}
                          />
                          <RunStatus status={r.status} />
                          <span className="font-mono text-[11px] text-muted-foreground">
                            {r.startedAt}
                          </span>
                          {r.metadata?.determinism && (
                            <Badge
                              variant={
                                r.metadata.determinism === "deterministic"
                                  ? "secondary"
                                  : "outline"
                              }
                              className="shrink-0 text-[9px]"
                              title={
                                r.metadata.determinism === "deterministic"
                                  ? "Reproducible: same inputs + same flow version ⇒ same output"
                                  : "Not reproducible: a non-deterministic step was used"
                              }
                            >
                              {r.metadata.determinism === "deterministic"
                                ? "reproducible"
                                : "non-reproducible"}
                            </Badge>
                          )}
                          <span className="ml-auto text-[11px] text-muted-foreground">
                            {r.trigger}
                          </span>
                        </button>
                        {r.status === "failed" && (
                          <div className="ml-5 mt-1 flex flex-wrap items-center gap-2 rounded-md border border-destructive/25 bg-destructive/[0.06] px-3 py-2 text-[11px] text-muted-foreground">
                            <CircleX className="size-3.5 text-destructive" />
                            <span className="min-w-0 flex-1">
                              {firstFailure(
                                snap.openRunId === r.id
                                  ? (snap.runSteps as JobRunStep[])
                                  : [],
                              ) ??
                                "Run failed. Open diagnostics, fix the source or step, then rerun the current job."}
                            </span>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 px-2 text-[10px]"
                              onClick={() => actions.openRun(r.id)}
                            >
                              <FileSearch className="size-3.5" /> Diagnostics
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 px-2 text-[10px]"
                              onClick={() =>
                                copyRunDiagnostics(r as JobRun, "run")
                              }
                            >
                              <ClipboardCopy className="size-3.5" /> Copy
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="h-7 px-2 text-[10px]"
                              onClick={() => focusSetupSection("source")}
                            >
                              <ArrowDownToLine className="size-3.5" /> Source
                            </Button>
                            {sourceBinding && (
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-7 px-2 text-[10px]"
                                onClick={() =>
                                  actions.openBinding(sourceBinding.id)
                                }
                              >
                                <Type className="size-3.5" /> Binding
                              </Button>
                            )}
                            {sourceConnectorId && (
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-7 px-2 text-[10px]"
                                onClick={() =>
                                  actions.openConnector(sourceConnectorId)
                                }
                              >
                                <ArrowDownToLine className="size-3.5" />{" "}
                                Connector
                              </Button>
                            )}
                            <Button
                              size="sm"
                              className="h-7 px-2 text-[10px]"
                              onClick={() => actions.runDraft()}
                              disabled={
                                snap.jobBusy ||
                                snap.dryBusy ||
                                snap.sourcePreviewBusy ||
                                !readiness?.canRun
                              }
                              title={
                                readiness?.canRun
                                  ? "Run this job again"
                                  : (readiness?.blockers[0]?.message ??
                                    "Resolve setup blockers first")
                              }
                            >
                              {snap.jobBusy ? (
                                <Loader2 className="size-3.5 animate-spin" />
                              ) : (
                                <RefreshCw className="size-3.5" />
                              )}
                              Rerun current job
                            </Button>
                          </div>
                        )}
                        {snap.openRunId === r.id && (
                          <div className="ml-5 mt-1 space-y-2 border-l border-outline-subtle pl-3">
                            <RunDiagnosticsPanel
                              run={r}
                              steps={snap.runSteps}
                              sourceMode={snap.jobSourceMode}
                              targetId={snap.activeTargetId}
                              canRun={!!readiness?.canRun}
                              busy={
                                snap.jobBusy ||
                                snap.dryBusy ||
                                snap.sourcePreviewBusy
                              }
                              onCopyRun={() => copyRunDiagnostics(r, "run")}
                              onCopySteps={() => copyRunDiagnostics(r, "steps")}
                              onOpenSource={() => focusSetupSection("source")}
                              onOpenSteps={() => focusSetupSection("steps")}
                              onOpenBinding={
                                sourceBinding
                                  ? () => actions.openBinding(sourceBinding.id)
                                  : undefined
                              }
                              onOpenSourceConnector={
                                sourceConnectorId
                                  ? () =>
                                      actions.openConnector(sourceConnectorId)
                                  : undefined
                              }
                              onOpenTargetConnector={
                                targetConnectorId
                                  ? () =>
                                      actions.openConnector(targetConnectorId)
                                  : undefined
                              }
                              onRerun={() => actions.runDraft()}
                              rerunTitle={
                                readiness?.canRun
                                  ? "Run the current saved job definition again"
                                  : (readiness?.blockers[0]?.message ??
                                    "Resolve setup blockers first")
                              }
                            />
                            {r.metadata?.packBinding && (
                              <div className="flex flex-wrap items-center gap-2 rounded-md bg-surface-toolbar px-3 py-1.5 text-[11px] text-muted-foreground">
                                <Boxes className="size-3.5" />
                                <span>
                                  Pack{" "}
                                  <span className="font-medium text-foreground">
                                    {r.metadata.packBinding.packName}
                                  </span>
                                </span>
                                {r.metadata.packBinding.refName && (
                                  <span className="inline-flex items-center gap-1 font-mono">
                                    <GitBranch className="size-3" />{" "}
                                    {r.metadata.packBinding.refName}
                                  </span>
                                )}
                                {r.metadata.packBinding.commitSha && (
                                  <span className="inline-flex items-center gap-1 font-mono">
                                    <GitCommit className="size-3" />{" "}
                                    {r.metadata.packBinding.commitSha.slice(
                                      0,
                                      12,
                                    )}
                                  </span>
                                )}
                                {r.metadata.packBinding.contentHash && (
                                  <span
                                    className="ml-auto max-w-[220px] truncate font-mono text-[10px]"
                                    title={r.metadata.packBinding.contentHash}
                                  >
                                    sha256:
                                    {r.metadata.packBinding.contentHash.slice(
                                      0,
                                      16,
                                    )}
                                  </span>
                                )}
                              </div>
                            )}
                            {snap.runSteps.map((s) => (
                              <div
                                key={s.stepIdx}
                                className="flex items-center gap-2 rounded-md bg-surface-panel px-3 py-1.5 text-[12px]"
                              >
                                <RunStatus status={s.status} small />
                                <span className="font-mono">{s.template}</span>
                                <span className="ml-auto text-muted-foreground">
                                  {s.rowCount ?? "—"} rows
                                </span>
                                {s.log && (
                                  <span
                                    className="max-w-[40%] truncate text-destructive"
                                    title={s.log}
                                  >
                                    {s.log}
                                  </span>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                        </div>
                      ))}
                      {draft && snap.runs.length < snap.runsTotal && (
                        <div className="pt-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => actions.loadMoreRuns(draft.id)}
                            disabled={snap.runsLoading}
                          >
                            {snap.runsLoading ? (
                              <Loader2 className="size-3.5 animate-spin" />
                            ) : (
                              <History className="size-3.5" />
                            )}
                            Load more runs
                          </Button>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>

            {localSnap.advancedOpen && (
              <TabsContent value="api" className="min-h-0 flex-1">
                <ScrollArea className="h-full">
                  <JobApiTab />
                </ScrollArea>
              </TabsContent>
            )}
          </Tabs>
        </div>
      )}
    </div>
  );
}

function RunDiagnosticsPanel({
  run,
  steps,
  sourceMode,
  targetId,
  canRun,
  busy,
  onCopyRun,
  onCopySteps,
  onOpenSource,
  onOpenSteps,
  onOpenBinding,
  onOpenSourceConnector,
  onOpenTargetConnector,
  onRerun,
  rerunTitle,
}: {
  run: JobRun;
  steps: readonly JobRunStep[];
  sourceMode: "file" | "connector";
  targetId: string | null;
  canRun: boolean;
  busy: boolean;
  onCopyRun: () => void;
  onCopySteps: () => void;
  onOpenSource: () => void;
  onOpenSteps: () => void;
  onOpenBinding?: (() => void) | undefined;
  onOpenSourceConnector?: (() => void) | undefined;
  onOpenTargetConnector?: (() => void) | undefined;
  onRerun: () => void;
  rerunTitle: string;
}) {
  const failedSteps = steps.filter((s) => s.status === "failed");
  const headline =
    failedSteps[0]?.log ??
    run.metadata?.diagnostics?.message ??
    (run.status === "failed"
      ? "No step log was recorded for this failed run."
      : "Run diagnostics");

  return (
    <section className="rounded-md border border-outline-subtle bg-surface-panel">
      <div className="flex flex-wrap items-center gap-2 border-b border-outline-subtle px-3 py-2 text-[11px]">
        <FileSearch className="size-3.5 text-muted-foreground" />
        <span className="font-medium text-foreground">Run diagnostics</span>
        <span className="font-mono text-muted-foreground">{run.id}</span>
        <span className="ml-auto text-muted-foreground">
          {steps.length} step{steps.length === 1 ? "" : "s"} · target{" "}
          {run.targetId ?? targetId ?? "unknown"}
        </span>
      </div>
      <div className="grid gap-3 p-3 md:grid-cols-[minmax(0,1fr)_minmax(220px,0.38fr)]">
        <div className="min-w-0 space-y-2">
          <div className="rounded-md bg-surface-toolbar px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Most actionable signal
            </div>
            <div
              className={cn(
                "mt-1 break-words text-[12px]",
                run.status === "failed"
                  ? "text-destructive"
                  : "text-foreground",
              )}
            >
              {String(headline)}
            </div>
          </div>
          <div className="grid gap-1.5">
            {steps.length === 0 ? (
              <div className="rounded-md border border-dashed border-outline-subtle px-3 py-2 text-[11px] text-muted-foreground">
                Step details have not been loaded or this run recorded no steps.
              </div>
            ) : (
              steps.map((step) => (
                <div
                  key={step.stepIdx}
                  className={cn(
                    "rounded-md border px-3 py-2 text-[11px]",
                    step.status === "failed"
                      ? "border-destructive/30 bg-destructive/[0.06]"
                      : "border-outline-subtle bg-surface-toolbar",
                  )}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <RunStatus status={step.status} small />
                    <span className="font-mono text-foreground">
                      {step.stepIdx + 1}. {step.template || "inline step"}
                    </span>
                    <span className="ml-auto text-muted-foreground">
                      {step.rowCount ?? "—"} rows
                    </span>
                  </div>
                  {step.log && (
                    <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap rounded bg-background/40 p-2 font-mono text-[10px] text-destructive">
                      {step.log}
                    </pre>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
        <div className="space-y-2">
          <Button
            size="sm"
            className="w-full justify-start"
            onClick={onRerun}
            disabled={busy || !canRun}
            title={rerunTitle}
          >
            {busy ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            Rerun current job
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="w-full justify-start"
            onClick={onCopyRun}
          >
            <ClipboardCopy className="size-4" /> Copy run diagnostics
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="w-full justify-start"
            onClick={onCopySteps}
          >
            <ClipboardCopy className="size-4" /> Copy step logs
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="w-full justify-start"
            onClick={onOpenSource}
          >
            {sourceMode === "connector" ? (
              <ArrowDownToLine className="size-4" />
            ) : (
              <FileUp className="size-4" />
            )}
            Open source setup
          </Button>
          {onOpenBinding && (
            <Button
              size="sm"
              variant="outline"
              className="w-full justify-start"
              onClick={onOpenBinding}
            >
              <Type className="size-4" /> Open source binding
            </Button>
          )}
          {onOpenSourceConnector && (
            <Button
              size="sm"
              variant="outline"
              className="w-full justify-start"
              onClick={onOpenSourceConnector}
            >
              <ArrowDownToLine className="size-4" /> Open source connector
            </Button>
          )}
          {onOpenTargetConnector && (
            <Button
              size="sm"
              variant="outline"
              className="w-full justify-start"
              onClick={onOpenTargetConnector}
            >
              <ArrowUpFromLine className="size-4" /> Open target connector
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            className="w-full justify-start"
            onClick={onOpenSteps}
          >
            <Workflow className="size-4" /> Open step setup
          </Button>
          <div className="rounded-md bg-surface-toolbar px-3 py-2 text-[10px] text-muted-foreground">
            Rerun uses the current job definition. Copy diagnostics first if you
            need the failed run's exact context.
          </div>
        </div>
      </div>
    </section>
  );
}

function firstFailure(steps: JobRunStep[]): string | null {
  const failed = steps.find((s) => s.status === "failed");
  if (!failed) return null;
  return (
    failed.log ||
    `Step ${failed.stepIdx + 1} failed: ${failed.template || "inline step"}`
  );
}

function formatRunDiagnostics({
  job,
  run,
  steps,
  scope,
  sourceMode,
  targetId,
}: {
  job: Job;
  run: JobRun;
  steps: JobRunStep[];
  scope: RunDiagnosticsScope;
  sourceMode: "file" | "connector";
  targetId: string | null;
}) {
  const payload = {
    scope,
    copiedAt: new Date().toISOString(),
    job: {
      id: job.id,
      name: job.name,
      enabled: job.enabled,
      sourceMode,
      definition: job.definition,
    },
    run,
    activeTargetId: targetId,
    failedSteps: steps.filter((s) => s.status === "failed"),
    steps: scope === "steps" ? steps : steps.map((s) => summarizeStep(s)),
  };
  return JSON.stringify(payload, null, 2);
}

function summarizeStep(step: JobRunStep) {
  return {
    stepIdx: step.stepIdx,
    template: step.template,
    status: step.status,
    rowCount: step.rowCount,
    log: step.log,
  };
}

function JobApiTab() {
  const snap = useSnapshot(store);
  const [baseUrl, setBaseUrl] = useState(
    () => localStorage.getItem(JOB_API_BASE_KEY) || DEFAULT_JOB_API_BASE,
  );
  const cleanBase = baseUrl.replace(/\/+$/, "");
  const docsUrl = `${cleanBase}/docs`;
  const openApiUrl = `${cleanBase}/openapi.json`;
  const jobId = snap.jobDraft?.id ?? "job-id";
  const runId = snap.openRunId ?? snap.runs[0]?.id ?? "run-id";

  useEffect(() => {
    localStorage.setItem(JOB_API_BASE_KEY, baseUrl);
  }, [baseUrl]);

  return (
    <div className="space-y-5 p-6">
      <section className="rounded-md border border-outline-subtle bg-surface-panel">
        <div className="flex flex-wrap items-center gap-3 border-b border-outline-subtle bg-surface-header px-4 py-3">
          <span className="grid size-8 place-items-center rounded-md bg-primary/[0.14] text-icon-tile-foreground">
            <ServerCog className="size-4" />
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-[12px] font-semibold uppercase tracking-[0.08em] text-foreground/90">
              Job REST API
            </div>
            <div className="mt-0.5 text-[11px] text-muted-foreground">
              Opt in from the headless scheduler with{" "}
              <span className="font-mono text-foreground">
                CELERITAS_JOB_API_BIND=127.0.0.1:7700
              </span>
              .
            </div>
          </div>
          <Button size="sm" variant="outline" asChild>
            <a href={docsUrl} target="_blank" rel="noreferrer">
              <SquareArrowOutUpRight className="size-4" /> Docs
            </a>
          </Button>
          <Button size="sm" variant="secondary" asChild>
            <a href={openApiUrl} target="_blank" rel="noreferrer">
              <Braces className="size-4" /> OpenAPI
            </a>
          </Button>
        </div>
        <div className="grid gap-3 p-4 md:grid-cols-[minmax(0,1fr)_minmax(260px,0.55fr)]">
          <Labeled label="API base URL">
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              className="font-mono text-[12px]"
            />
          </Labeled>
          <div className="rounded-md bg-surface-toolbar p-3">
            <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Run command
            </div>
            <code className="mt-1 block whitespace-pre-wrap break-all text-[11px] text-foreground">
              CELERITAS_JOB_API_BIND=127.0.0.1:7700 cargo run --bin
              celeritas-scheduler
            </code>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          <Braces className="size-3.5" /> Operations
        </div>
        <div className="space-y-3">
          {jobApiEndpoints.map((endpoint) => {
            const path = endpointPath(endpoint.path, jobId, runId);
            const curl = curlExample(endpoint, cleanBase, path);
            return (
              <div
                key={`${endpoint.method}-${endpoint.path}`}
                className="rounded-md border border-outline-subtle bg-surface-panel"
              >
                <div className="flex flex-wrap items-center gap-2 border-b border-outline-subtle bg-surface-header px-3 py-2">
                  <Badge
                    variant={methodBadgeVariant(endpoint.method)}
                    className="min-w-14 justify-center font-mono text-[10px]"
                  >
                    {endpoint.method}
                  </Badge>
                  <code className="min-w-0 flex-1 truncate text-[12px] text-foreground">
                    {path}
                  </code>
                  <a
                    href={`${cleanBase}${path.split("?")[0]}`}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex size-7 items-center justify-center rounded-md text-muted-foreground hover:bg-surface-hover hover:text-foreground"
                    title="Open endpoint"
                  >
                    <SquareArrowOutUpRight className="size-3.5" />
                  </a>
                </div>
                <div className="grid gap-3 px-3 py-3 lg:grid-cols-[minmax(0,0.72fr)_minmax(280px,0.45fr)]">
                  <div className="space-y-2">
                    <div className="text-[12px] leading-relaxed text-muted-foreground">
                      {endpoint.summary}
                    </div>
                    <div className="grid gap-2 md:grid-cols-2">
                      <SpecField label="Request">
                        {endpoint.request ?? "No request body."}
                      </SpecField>
                      <SpecField label="Response">
                        {endpoint.response}
                      </SpecField>
                    </div>
                    {endpoint.body != null && (
                      <div>
                        <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                          JSON body
                        </div>
                        <pre className="max-h-[260px] overflow-auto rounded-md border border-outline-subtle bg-surface-input p-3 text-[11px] leading-relaxed text-foreground">
                          {JSON.stringify(endpoint.body, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                  <div>
                    <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                      curl
                    </div>
                    <pre className="h-full max-h-[360px] overflow-auto rounded-md border border-outline-subtle bg-surface-input p-3 text-[11px] leading-relaxed text-foreground">
                      {curl}
                    </pre>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            <Braces className="size-3.5" /> Full OpenAPI JSON
          </div>
          <Button size="sm" variant="secondary" asChild>
            <a href={openApiUrl} target="_blank" rel="noreferrer">
              <SquareArrowOutUpRight className="size-3.5" /> Open raw spec
            </a>
          </Button>
        </div>
        <pre className="max-h-[560px] overflow-auto rounded-md border border-outline-subtle bg-surface-input p-4 text-[11px] leading-relaxed text-foreground">
          {JSON.stringify(jobApiOpenApiSpec, null, 2)}
        </pre>
      </section>
    </div>
  );
}

function SpecField({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="rounded-md border border-outline-subtle bg-surface-toolbar p-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono text-[11px] leading-relaxed text-foreground">
        {children}
      </div>
    </div>
  );
}

function curlExample(
  endpoint: JobApiEndpoint,
  cleanBase: string,
  path: string,
) {
  const url = `${cleanBase}${path}`;
  const lines = [`curl -fsS -X ${endpoint.method} ${JSON.stringify(url)}`];
  if (endpoint.body) {
    lines.push('  -H "Content-Type: application/json"');
    lines.push(`  --data '${JSON.stringify(endpoint.body, null, 2)}'`);
  }
  return lines.join(" \\\n");
}

function methodBadgeVariant(
  method: string,
): "default" | "secondary" | "outline" | "accent" {
  if (method === "GET") return "secondary";
  if (method === "POST") return "default";
  if (method === "DELETE") return "accent";
  return "outline";
}

function endpointPath(path: string, jobId: string, runId: string) {
  return path
    .replace("{jobId}", encodeURIComponent(jobId))
    .replace("{runId}", encodeURIComponent(runId))
    .replace("jobId=", `jobId=${encodeURIComponent(jobId)}`);
}

function PrimaryActionIcon({
  action,
}: {
  action?: JobReadiness["nextAction"]["kind"] | undefined;
}) {
  if (action === "chooseSource") return <ArrowDownToLine className="size-4" />;
  if (action === "testSource") return <Eye className="size-4" />;
  if (action === "addStep") return <Plus className="size-4" />;
  if (action === "fillParams" || action === "resolveBlockers") {
    return <Settings2 className="size-4" />;
  }
  if (action === "dryRun") return <FlaskConical className="size-4" />;
  return <Play className="size-4" />;
}

function JobReadinessChecklist({ readiness }: { readiness: JobReadiness }) {
  const completeCount = readiness.items.filter(
    (item) => item.state === "complete",
  ).length;
  const warningCount = readiness.warnings.length;
  const issue = readiness.blockers[0] ?? readiness.warnings[0];

  return (
    <div className="flex items-center gap-2 border-b border-outline-subtle bg-surface-toolbar px-6 py-2 text-[11px]">
      {readiness.canRun ? (
        <CircleCheck className="size-3.5 text-emerald-400" />
      ) : readiness.blockers.length > 0 ? (
        <CircleX className="size-3.5 text-destructive" />
      ) : (
        <Clock className="size-3.5 text-amber-200" />
      )}
      <span className="font-medium text-foreground">
        {readiness.canRun
          ? "Ready to run"
          : `${completeCount}/${readiness.items.length} ready`}
      </span>
      <span className="text-muted-foreground">
        {readiness.blockers.length > 0
          ? `${readiness.blockers.length} blocker${readiness.blockers.length === 1 ? "" : "s"}`
          : warningCount > 0
            ? `${warningCount} optional warning${warningCount === 1 ? "" : "s"}`
            : "All setup checks passed"}
      </span>
      {issue && (
        <span className="min-w-0 flex-1 truncate text-muted-foreground">
          {issue.message}
        </span>
      )}
    </div>
  );
}

function sectionForReadinessItem(
  id?: JobReadiness["items"][number]["id"],
): JobSetupSection {
  if (id === "source" || id === "sourceTest") return "source";
  if (id === "steps" || id === "params") return "steps";
  if (id === "output" || id === "compute") return "output";
  if (id === "dryRun") return "validate";
  if (id === "schedule") return "schedule";
  return "source";
}

function CreateJobWizardShell({
  draft,
  readiness,
  activeTab,
  setTab,
  focusSection,
  runPrimaryAction,
  busy,
}: {
  draft: Job;
  readiness: JobReadiness;
  activeTab: string;
  setTab: (tab: string) => void;
  focusSection: (section: JobSetupSection) => void;
  runPrimaryAction: () => void;
  busy: boolean;
}) {
  const byId = (id: JobReadiness["items"][number]["id"]) =>
    readiness.items.find((item) => item.id === id);
  const source = combineWizardState(
    byId("source")?.state,
    byId("sourceTest")?.state,
  );
  const steps = combineWizardState(byId("steps")?.state, byId("params")?.state);
  const validate = combineWizardState(
    byId("dryRun")?.state,
    byId("compute")?.state,
  );
  const schedule = byId("schedule")?.state ?? "blocked";
  const output = byId("output")?.state ?? "warning";
  const startLabel = draft.definition.packBinding
    ? `Pack: ${draft.definition.packBinding.packName}`
    : "Blank job";
  const reviewState = readiness.canRun
    ? "complete"
    : readiness.blockers.length > 0
      ? "blocked"
      : "warning";

  const stages: WizardStage[] = [
    {
      id: "start",
      title: "Start",
      message: startLabel,
      state: "complete",
      icon: Boxes,
      tab: "pipeline",
      action: (
        <div className="flex items-center gap-1">
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2 text-[10px]"
            onClick={() => actions.setView("packs")}
          >
            Packs
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2 text-[10px]"
            onClick={() => actions.setView("explore")}
          >
            Explore
          </Button>
        </div>
      ),
    },
    {
      id: "source",
      title: "Bind source",
      message: byId("sourceTest")?.message ?? byId("source")?.message ?? "",
      state: source,
      icon: ArrowDownToLine,
      tab: "pipeline",
      action: (
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2 text-[10px]"
          onClick={() => focusSection("source")}
        >
          Open
        </Button>
      ),
    },
    {
      id: "steps",
      title: "Configure steps",
      message: byId("params")?.message ?? byId("steps")?.message ?? "",
      state: steps,
      icon: Workflow,
      tab: "pipeline",
      action: (
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2 text-[10px]"
          onClick={() => {
            if (draft.definition.steps.length === 0) actions.addStep();
            window.setTimeout(() => focusSection("steps"), 0);
          }}
        >
          {draft.definition.steps.length === 0 ? "Add" : "Open"}
        </Button>
      ),
    },
    {
      id: "output",
      title: "Choose output",
      message: byId("output")?.message ?? "",
      state: output,
      icon: ArrowUpFromLine,
      tab: "pipeline",
      action: (
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2 text-[10px]"
          onClick={() => focusSection("output")}
        >
          Open
        </Button>
      ),
    },
    {
      id: "validate",
      title: "Validate",
      message: byId("dryRun")?.message ?? byId("compute")?.message ?? "",
      state: validate,
      icon: FlaskConical,
      tab: "pipeline",
      action: (
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2 text-[10px]"
          onClick={() => {
            if (readiness.nextAction.kind === "dryRun") {
              runPrimaryAction();
              return;
            }
            focusSection(sectionForReadinessItem(readiness.blockers[0]?.id));
          }}
          disabled={
            busy ||
            readiness.nextAction.disabled ||
            readiness.nextAction.kind === "runNow"
          }
        >
          {readiness.nextAction.kind === "dryRun" ? "Dry run" : "Next"}
        </Button>
      ),
    },
    {
      id: "schedule",
      title: "Schedule",
      message: byId("schedule")?.message ?? "",
      state: schedule,
      icon: CalendarClock,
      tab: "schedule",
      action: (
        <Button
          size="sm"
          variant="outline"
          className="h-7 px-2 text-[10px]"
          onClick={() => focusSection("schedule")}
        >
          Open
        </Button>
      ),
    },
    {
      id: "review",
      title: "Review",
      message: readiness.canRun
        ? "Ready to run."
        : (readiness.blockers[0]?.message ??
          readiness.warnings[0]?.message ??
          ""),
      state: reviewState,
      icon: Play,
      tab: "runs",
      action: (
        <Button
          size="sm"
          className="h-7 px-2 text-[10px]"
          onClick={
            readiness.canRun
              ? runPrimaryAction
              : () =>
                  focusSection(
                    sectionForReadinessItem(readiness.blockers[0]?.id),
                  )
          }
          disabled={busy}
        >
          {readiness.canRun ? "Run" : "Fix"}
        </Button>
      ),
    },
  ];

  return (
    <div className="border-b border-outline-subtle bg-surface-panel px-6 py-3">
      <div className="mb-2 flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Bound flow setup
        </span>
        <Badge
          variant={readiness.canRun ? "success" : "outline"}
          className="text-[10px]"
        >
          {readiness.canRun ? "ready" : `${readiness.blockers.length} blockers`}
        </Badge>
      </div>
      <div className="grid gap-2 xl:grid-cols-7">
        {stages.map((stage) => (
          <WizardStageCard
            key={stage.id}
            stage={stage}
            active={stage.tab === activeTab}
            onOpen={() => setTab(stage.tab)}
          />
        ))}
      </div>
    </div>
  );
}

interface WizardStage {
  id: string;
  title: string;
  message: string;
  state: "complete" | "blocked" | "warning";
  icon: typeof Workflow;
  tab: string;
  action: React.ReactNode;
}

function WizardStageCard({
  stage,
  active,
  onOpen,
}: {
  stage: WizardStage;
  active: boolean;
  onOpen: () => void;
}) {
  const Icon = stage.icon;
  return (
    <div
      className={cn(
        "min-w-0 rounded-md border bg-surface-toolbar p-2",
        active ? "border-outline-strong" : "border-outline-subtle",
      )}
    >
      <button
        type="button"
        className="flex w-full min-w-0 items-center gap-2 text-left"
        onClick={onOpen}
      >
        <span
          className={cn(
            "grid size-7 shrink-0 place-items-center rounded-md",
            stage.state === "complete" &&
              "bg-emerald-500/[0.12] text-emerald-200",
            stage.state === "blocked" &&
              "bg-destructive/[0.12] text-destructive",
            stage.state === "warning" && "bg-amber-400/[0.12] text-amber-200",
          )}
        >
          <Icon className="size-3.5" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[12px] font-semibold">
            {stage.title}
          </span>
          <span className="block truncate text-[10.5px] text-muted-foreground">
            {stage.message}
          </span>
        </span>
      </button>
      <div className="mt-2 flex items-center justify-between gap-2">
        <WizardState state={stage.state} />
        {stage.action}
      </div>
    </div>
  );
}

function WizardState({ state }: { state: WizardStage["state"] }) {
  if (state === "complete") {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] text-emerald-200">
        <CircleCheck className="size-3" /> done
      </span>
    );
  }
  if (state === "warning") {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] text-amber-200">
        <Clock className="size-3" /> optional
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-destructive">
      <CircleX className="size-3" /> needed
    </span>
  );
}

function combineWizardState(
  ...states: Array<WizardStage["state"] | undefined>
): WizardStage["state"] {
  if (states.includes("blocked")) return "blocked";
  if (states.includes("warning")) return "warning";
  return "complete";
}

function requiredSourceSpec(job: Job) {
  const binding = job.definition.packBinding;
  if (!binding) return undefined;
  const pack = store.packs.find(
    (p) => p.id === binding.packId || p.name === binding.packName,
  );
  const jobTemplate = pack?.jobTemplates.find(
    (jt) => jt.name === binding.jobTemplate,
  );
  return jobTemplate?.source.connector ?? undefined;
}

function PackBindingStrip({ job }: { job: Job }) {
  const snap = useSnapshot(store);
  const binding = job.definition.packBinding;
  if (!binding) return null;
  const shortCommit = binding.commitSha ? binding.commitSha.slice(0, 12) : null;
  const canUpdate = binding.origin === "github" && !!binding.sourceId;
  const latest = snap.packUpdateResolution?.latest;
  const updateLabel =
    snap.packUpdateError ??
    (snap.packUpdateResolution
      ? snap.packUpdateResolution.status === "current"
        ? "current"
        : latest?.commitSha
          ? `${snap.packUpdateResolution.status} @ ${latest.commitSha.slice(0, 12)}`
          : snap.packUpdateResolution.status
      : null);
  return (
    <div className="flex items-center gap-2 border-b border-outline-subtle bg-surface-toolbar px-6 py-1.5 text-[11px] text-muted-foreground">
      <Boxes className="size-3.5 text-icon-muted" />
      <span className="min-w-0 truncate">
        From{" "}
        <span className="font-medium text-foreground">{binding.packName}</span>
        <span className="mx-1 text-muted-foreground/60">/</span>
        <span className="font-mono">{binding.jobTemplate}</span>
      </span>
      <Badge
        variant={binding.origin === "github" ? "outline" : "secondary"}
        className="shrink-0 text-[9px]"
      >
        {binding.origin}
      </Badge>
      {binding.refName && (
        <span className="hidden shrink-0 items-center gap-1 font-mono md:inline-flex">
          <GitBranch className="size-3" /> {binding.refName}
        </span>
      )}
      {shortCommit && (
        <span className="hidden shrink-0 items-center gap-1 font-mono md:inline-flex">
          <GitCommit className="size-3" /> {shortCommit}
        </span>
      )}
      {canUpdate && (
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {updateLabel && (
            <span
              className={cn(
                "mr-1 max-w-[180px] truncate font-mono text-[10px]",
                snap.packUpdateError && "text-destructive",
              )}
            >
              {updateLabel}
            </span>
          )}
          <button
            onClick={() => actions.setJobPackUpdatePolicy("pinned")}
            title="Pin this job to the selected commit"
            className={cn(
              "inline-flex h-6 items-center gap-1 rounded border px-1.5 font-mono text-[10px]",
              binding.updatePolicy === "pinned"
                ? "border-outline-strong bg-surface-active text-foreground"
                : "border-outline-subtle text-muted-foreground hover:text-foreground",
            )}
          >
            <Pin className="size-3" /> pinned
          </button>
          <button
            onClick={() => actions.setJobPackUpdatePolicy("track")}
            title="Track compatible updates from the linked ref"
            className={cn(
              "inline-flex h-6 items-center gap-1 rounded border px-1.5 font-mono text-[10px]",
              binding.updatePolicy === "track"
                ? "border-outline-strong bg-surface-active text-foreground"
                : "border-outline-subtle text-muted-foreground hover:text-foreground",
            )}
          >
            <GitBranch className="size-3" /> track
          </button>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 px-1.5 text-[10px]"
            onClick={() => actions.checkJobPackUpdate()}
            disabled={snap.packUpdateBusy}
          >
            {snap.packUpdateBusy ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RefreshCw className="size-3" />
            )}
            Check
          </Button>
          <Button
            size="sm"
            variant="secondary"
            className="h-6 px-1.5 text-[10px]"
            onClick={() => actions.applyCompatibleJobPackUpdate(false)}
            disabled={snap.packUpdateBusy}
          >
            Apply
          </Button>
        </div>
      )}
      {!canUpdate && (
        <span className="ml-auto shrink-0 font-mono text-[10px]">
          {binding.updatePolicy}
        </span>
      )}
    </div>
  );
}

/** Compact "Test source" result — a one-line status, expanding to a table only
 *  when the source actually returned rows. */
function SourcePreview() {
  const snap = useSnapshot(store);
  if (snap.sourcePreviewBusy) {
    return (
      <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin" /> reading source…
      </div>
    );
  }
  const res = snap.sourcePreview as RunResult | null;
  if (!res) return null;
  const ok = res.status === "complete";
  const count = res.rowCount ?? res.rows.length;
  const hasRows = ok && res.rows.length > 0;
  return (
    <div className="mt-1.5">
      <div className="flex items-center gap-1.5 text-[11px]">
        {ok ? (
          <CircleCheck className="size-3.5 text-emerald-400" />
        ) : (
          <CircleX className="size-3.5 text-destructive" />
        )}
        <SimulatedBadge />
        {!ok ? (
          <span className="text-destructive">
            {res.error || "source error"}
          </span>
        ) : hasRows ? (
          <span className="text-muted-foreground">
            resolved · {count} sample row{count === 1 ? "" : "s"}
          </span>
        ) : (
          // complete but empty — surface the backend note (e.g. mock-driver message) honestly.
          <span className="text-amber-300/90">
            {res.error || "resolved — no rows returned"}
          </span>
        )}
      </div>
      {hasRows && (
        <div className="mt-1.5 overflow-hidden rounded-md border border-outline-subtle bg-surface-panel">
          <OutputTable rows={res.rows as Record<string, unknown>[]} />
        </div>
      )}
    </div>
  );
}

/** Schedule page: opt a job into scheduled execution and build its cron with a
 *  real, tabular cron editor. On-demand jobs simply have no schedule. */
function ScheduleTab({ draft }: { draft: Job }) {
  const scheduled = !!draft.definition.schedule?.trim();
  return (
    <div
      data-job-section="schedule"
      tabIndex={-1}
      className="max-w-2xl scroll-mt-3 space-y-5 p-6 focus:outline-none"
    >
      <div>
        <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Execution mode
        </div>
        <ButtonGroup>
          <ButtonGroupItem
            active={!scheduled}
            onClick={() => actions.setDraftScheduled(false)}
          >
            <Play /> On demand
          </ButtonGroupItem>
          <ButtonGroupItem
            active={scheduled}
            onClick={() => actions.setDraftScheduled(true)}
          >
            <CalendarClock /> Scheduled
          </ButtonGroupItem>
        </ButtonGroup>
        <p className="mt-2 text-[11px] leading-relaxed text-muted-foreground">
          {scheduled
            ? "Runs automatically on the cron schedule below (embedded target, UTC). Missed runs catch up on next launch."
            : "Runs only when you trigger it (Run now) or via the API — no automatic schedule."}
        </p>
      </div>
      {scheduled && (
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Schedule (cron, UTC)
          </div>
          <CronEditor
            value={draft.definition.schedule ?? ""}
            onChange={(expr) => actions.setDraftSchedule(expr)}
          />
        </div>
      )}
    </div>
  );
}

function StepCard({
  idx,
  kind,
  template,
  args,
  sql,
  markdown,
}: {
  idx: number;
  kind?: "unit" | "sql" | "markdown" | undefined;
  template: string;
  args: Record<string, string | number>;
  sql?: string | undefined;
  markdown?: string | undefined;
}) {
  const snap = useSnapshot(store);
  // Per-step "show SQL" toggle (project rule: Valtio over React state).
  const local = useRef(proxy({ sqlOpen: false })).current;
  const lsnap = useSnapshot(local);
  const stepKind = kind ?? "unit";
  const tpl = snap.templates.find((t) => t.name === template);
  const inputs = (tpl?.inputs ?? []).filter((i) => !isRelationInput(i.kind)); // relation = job source
  const pack = snap.templatePack[template];
  const kindLabel =
    stepKind === "sql"
      ? "inline SQL"
      : stepKind === "markdown"
        ? "runbook"
        : "unit";
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      className="group/step overflow-hidden rounded-md border border-outline-subtle bg-surface-panel"
    >
      {/* Cell-style header: the step is "what runs"; params are the body. */}
      <div className="flex items-center gap-2 border-b border-outline-subtle bg-surface-header px-2 py-1">
        <GripVertical className="size-4 shrink-0 text-muted-foreground/40" />
        <span className="grid size-5 shrink-0 place-items-center rounded bg-primary/[0.14]">
          <Workflow className="size-3 text-icon-tile-foreground" />
        </span>
        <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
          step {idx + 1}
        </span>
        {stepKind === "unit" ? (
          <div className="flex min-w-0 flex-1 items-center">
            <TemplateCombobox
              value={template}
              onChange={(name) => actions.setStepTemplate(idx, name)}
            />
          </div>
        ) : (
          <div className="min-w-0 flex-1 truncate text-[12px] font-medium">
            {kindLabel}
          </div>
        )}
        {pack && (
          <Badge
            variant="outline"
            className="shrink-0 gap-1 text-[9px]"
            title={`From pack ${pack}`}
          >
            <Boxes className="size-2.5" /> {pack}
          </Badge>
        )}
        {stepKind !== "unit" ? (
          <Badge
            variant={stepKind === "sql" ? "default" : "secondary"}
            className="shrink-0 text-[10px]"
          >
            {kindLabel}
          </Badge>
        ) : (
          tpl && (
            <Badge
              variant={tpl.kind === "aggregate" ? "accent" : "default"}
              className="shrink-0 text-[10px]"
            >
              {tpl.kind}
            </Badge>
          )
        )}
        <Button
          size="icon"
          variant="ghost"
          className={cn(
            "size-6",
            lsnap.sqlOpen && "bg-surface-toolbar text-foreground",
          )}
          title={lsnap.sqlOpen ? "Hide SQL" : "Show SQL"}
          disabled={stepKind !== "unit" || !tpl?.source}
          onClick={() => (local.sqlOpen = !local.sqlOpen)}
        >
          <Code2 className="size-3.5" />
        </Button>
        <Button
          size="icon"
          variant="ghost"
          className="size-6 hover:text-destructive"
          title="Remove step"
          onClick={() => actions.removeStep(idx)}
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>

      {/* Named params — the default, notebook-like view. */}
      <div className="p-3">
        {stepKind === "sql" ? (
          <div className="overflow-hidden rounded-md border border-outline-subtle bg-surface-input">
            <CodeEditor
              path={`job-step-${idx + 1}.sql`}
              value={sql ?? ""}
              onChange={(value) => actions.setStepSql(idx, value)}
            />
          </div>
        ) : stepKind === "markdown" ? (
          <div className="overflow-hidden rounded-md border border-outline-subtle bg-surface-input">
            <CodeEditor
              path={`job-step-${idx + 1}.md`}
              value={markdown ?? ""}
              onChange={(value) => actions.setStepMarkdown(idx, value)}
            />
          </div>
        ) : !tpl ? (
          <div className="text-[11px] text-muted-foreground">
            Choose a template for this step.
          </div>
        ) : inputs.length > 0 ? (
          <div className="grid grid-cols-2 gap-2">
            {inputs.map((inp) => {
              const val = String(args[inp.name] ?? "");
              // Columns available to this step: the prior step's output (chained
              // schema), or the source's columns for the first step.
              const cols =
                idx <= 0
                  ? snap.sourceColumns.map((c) => c.name)
                  : snap.stepColumns[idx - 1]?.length
                    ? [...snap.stepColumns[idx - 1]]
                    : snap.sourceColumns.map((c) => c.name);
              return (
                <label key={inp.name} className="space-y-1">
                  <span className="text-[10px] text-muted-foreground">
                    {inp.name} <span className="opacity-60">({inp.kind})</span>
                  </span>
                  {inp.kind === "Column" && cols.length ? (
                    <select
                      value={val}
                      onChange={(e) =>
                        actions.setStepArg(idx, inp.name, e.target.value)
                      }
                      className="h-7 w-full rounded-md border border-input bg-surface-input px-2 font-mono text-[11px] outline-none"
                    >
                      <option value="">—</option>
                      {cols.map((c) => (
                        <option key={c} value={c} className="bg-surface-panel">
                          {c}
                        </option>
                      ))}
                    </select>
                  ) : inp.kind === "Columns" && cols.length ? (
                    <>
                      <Input
                        list={`cols-${idx}-${inp.name}`}
                        value={val}
                        onChange={(e) =>
                          actions.setStepArg(idx, inp.name, e.target.value)
                        }
                        placeholder="col_a, col_b"
                        className="h-7 font-mono text-[11px]"
                      />
                      <datalist id={`cols-${idx}-${inp.name}`}>
                        {cols.map((c) => (
                          <option key={c} value={c} />
                        ))}
                      </datalist>
                    </>
                  ) : (
                    <Input
                      value={val}
                      onChange={(e) =>
                        actions.setStepArg(idx, inp.name, e.target.value)
                      }
                      className="h-7 font-mono text-[11px]"
                    />
                  )}
                </label>
              );
            })}
          </div>
        ) : (
          <div className="text-[11px] text-muted-foreground">
            No parameters — this step runs over the chained input.
          </div>
        )}
      </div>

      {/* SQL — hidden by default, revealed on toggle (the template definition). */}
      {lsnap.sqlOpen && tpl?.source && (
        <div className="border-t border-outline-subtle bg-surface-input/40">
          <CodeEditor
            path={`${template}.toml`}
            value={tpl.source}
            readOnly
            onChange={() => {}}
          />
        </div>
      )}
    </motion.div>
  );
}

function isRelationInput(kind: string) {
  return kind === "Table" || kind === "TypedRelation";
}

function requiredSourceContract(
  job: Job,
  templates: readonly {
    name: string;
    inputs: readonly { kind: string; contract?: string | null }[];
  }[],
) {
  for (const step of job.definition.steps) {
    if (step.kind && step.kind !== "unit") continue;
    const template = templates.find((t) => t.name === step.template);
    const typed = template?.inputs.find(
      (input) => input.kind === "TypedRelation",
    );
    if (typed?.contract) return typed.contract;
  }
  return null;
}

function cloneBinding(binding: Binding): Binding {
  return {
    ...binding,
    runParams: { ...binding.runParams },
    ...(binding.mapping ? { mapping: { ...binding.mapping } } : {}),
    ...(binding.schema
      ? {
          schema: {
            columns: binding.schema.columns.map((column) => ({ ...column })),
            preview: binding.schema.preview.map((row) => ({ ...row })),
          },
        }
      : {}),
  };
}

function cloneBindingConformance(
  result: BindingConformance,
): BindingConformance {
  return {
    binding: cloneBinding(result.binding),
    conformance: {
      conforms: result.conformance.conforms,
      missing: [...result.conformance.missing],
      mismatches: result.conformance.mismatches.map((mismatch) => ({
        ...mismatch,
      })),
    },
  };
}

function SourceBindingPicker({
  contractRef,
  selectedId,
}: {
  contractRef: string | null;
  selectedId?: string | undefined;
}) {
  const snap = useSnapshot(store);
  const local = useRef(
    proxy({
      results: [] as BindingConformance[],
      loading: false,
      error: null as string | null,
    }),
  ).current;
  const lsnap = useSnapshot(local);
  const bindingsVersion = snap.bindings
    .map((binding) => `${binding.id}:${binding.schema?.columns.length ?? 0}`)
    .join("|");

  useEffect(() => {
    let cancelled = false;
    if (!contractRef) {
      local.results = snap.bindings.map((binding) => ({
        binding: cloneBinding(binding),
        conformance: { conforms: true, missing: [], mismatches: [] },
      }));
      local.loading = false;
      local.error = null;
      return;
    }
    local.loading = true;
    local.error = null;
    listBindingConformance(contractRef)
      .then((results) => {
        if (!cancelled) local.results = results.map(cloneBindingConformance);
      })
      .catch((err) => {
        if (!cancelled) {
          local.results = [];
          local.error = String(err);
        }
      })
      .finally(() => {
        if (!cancelled) local.loading = false;
      });
    return () => {
      cancelled = true;
    };
  }, [contractRef, bindingsVersion, local, snap.bindings]);

  const matching = lsnap.results
    .filter((result) => result.conformance.conforms)
    .map((result) => result.binding);
  const excluded = contractRef
    ? lsnap.results.filter((result) => !result.conformance.conforms)
    : [];
  const selected = selectedId
    ? (matching.find((binding) => binding.id === selectedId) ??
      snap.bindings.find((binding) => binding.id === selectedId))
    : null;
  const disabled =
    lsnap.loading ||
    (contractRef ? matching.length === 0 : snap.bindings.length === 0);
  const helper = lsnap.error
    ? lsnap.error
    : contractRef
      ? lsnap.loading
        ? `Checking ${contractRef} bindings...`
        : matching.length
          ? `${matching.length} binding${matching.length === 1 ? "" : "s"} conform to ${contractRef}.`
          : `No saved bindings conform to ${contractRef}.`
      : snap.bindings.length
        ? "Use a saved binding, or choose a connector below."
        : "No saved bindings yet; choose a connector below.";

  return (
    <Labeled
      label={contractRef ? `Saved binding (${contractRef})` : "Saved binding"}
    >
      <div className="flex items-center gap-2">
        <Boxes className="size-4 shrink-0 text-muted-foreground" />
        <select
          value={selectedId ?? ""}
          disabled={disabled}
          onChange={(event) => actions.setJobSourceBinding(event.target.value)}
          className="h-8 min-w-0 flex-1 rounded-md border border-input bg-surface-input px-2 text-[12px] outline-none"
        >
          <option value="">Select a saved binding...</option>
          {matching.map((binding) => (
            <option
              key={binding.id}
              value={binding.id}
              className="bg-surface-panel"
            >
              {binding.name}
            </option>
          ))}
        </select>
        {selected && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-8 shrink-0 px-2 text-[11px]"
            onClick={() => actions.setJobSourceBinding("")}
          >
            Clear
          </Button>
        )}
      </div>
      <div
        className={cn(
          "mt-1 text-[10.5px]",
          lsnap.error ? "text-destructive" : "text-muted-foreground",
        )}
      >
        {selected
          ? `${selected.name} resolves through ${connectorInstance(selected.instanceId)?.name ?? selected.instanceId}.`
          : helper}
      </div>
      {contractRef && excluded.length > 0 && (
        <details className="mt-1 rounded border border-outline-subtle bg-surface-panel px-2 py-1 text-[10.5px] text-muted-foreground">
          <summary className="cursor-pointer">
            {excluded.length} excluded binding{excluded.length === 1 ? "" : "s"}
          </summary>
          <div className="mt-1 space-y-1">
            {excluded.map((result) => (
              <div key={result.binding.id} className="flex gap-1">
                <span className="min-w-0 flex-1 truncate">
                  {result.binding.name}
                </span>
                <span
                  className="min-w-0 max-w-[58%] truncate text-amber-300/90"
                  title={conformanceReason(result)}
                >
                  {conformanceReason(result)}
                </span>
              </div>
            ))}
          </div>
        </details>
      )}
    </Labeled>
  );
}

function conformanceReason(result: {
  conformance: {
    missing: readonly string[];
    mismatches: readonly {
      field: string;
      expected: string;
      found: string;
    }[];
  };
}): string {
  const { missing, mismatches } = result.conformance;
  const parts = [
    ...missing.map((field) => `missing ${field}`),
    ...mismatches.map(
      (mismatch) =>
        `${mismatch.field}: expected ${mismatch.expected}, found ${mismatch.found}`,
    ),
  ];
  return parts.length ? parts.join("; ") : "does not conform";
}

/** Pick an enabled source/target connector for a job and bind its per-job
 *  params. The "source/target connector components" of the job editor. */
function ConnectorBind({
  role,
  label,
  icon: Icon,
  instances,
  selectedId,
  params,
  onSelect,
  onParam,
  onConfigure,
}: {
  role: "source" | "target";
  label: string;
  icon: typeof ArrowDownToLine;
  instances: ConnectorInstance[];
  selectedId?: string | undefined;
  params?: Record<string, string> | undefined;
  onSelect: (id: string) => void;
  onParam: (key: string, value: string) => void;
  onConfigure?: (() => void) | undefined;
}) {
  const selected = connectorInstance(selectedId ?? null);
  const spec = selected ? connectorSpec(selected.spec) : null;
  return (
    <Labeled label={label}>
      <div className="flex items-center gap-2">
        <Icon className="size-4 shrink-0 text-muted-foreground" />
        {instances.length === 0 ? (
          <div className="flex min-h-8 flex-1 items-center gap-2 rounded-md border border-dashed border-outline-subtle px-2 py-1 text-[11px] text-muted-foreground">
            <span className="min-w-0 flex-1">
              No enabled {role} connectors.
            </span>
            {onConfigure && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                className="h-6 shrink-0 px-2 text-[10px]"
                onClick={onConfigure}
              >
                <Plus className="size-3" /> Create
              </Button>
            )}
          </div>
        ) : (
          <ConnectorCombobox
            instances={instances}
            selectedId={selectedId}
            onSelect={onSelect}
            placeholder={`Select a ${role} connector…`}
          />
        )}
      </div>
      {selected && spec && spec.jobParams.length > 0 && (
        <div className="mt-2 grid gap-2 rounded-md border border-outline-subtle bg-surface-panel p-2.5">
          {spec.jobParams.map((p) => (
            <ParamField
              key={p.name}
              param={p}
              value={params?.[p.name] ?? ""}
              onChange={(v) => onParam(p.name, v)}
            />
          ))}
        </div>
      )}
    </Labeled>
  );
}

function Labeled({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      {label && <div className="text-xs text-muted-foreground">{label}</div>}
      {children}
    </div>
  );
}

function RunStatus({
  status,
  small = false,
}: {
  status: string;
  small?: boolean;
}) {
  const sz = small ? "size-3.5" : "size-4";
  if (status === "complete")
    return <CircleCheck className={cn(sz, "text-emerald-400")} />;
  if (status === "failed")
    return <CircleX className={cn(sz, "text-destructive")} />;
  return <Clock className={cn(sz, "text-amber-400")} />;
}
