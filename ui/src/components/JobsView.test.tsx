import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Job, JobRun, JobRunStep } from "@/lib/types";

const { mockDeriveJobReadiness } = vi.hoisted(() => ({
  mockDeriveJobReadiness: vi.fn(),
}));

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: Record<string, unknown>) => (
      <div {...props}>{children as React.ReactNode}</div>
    ),
  },
}));

vi.mock("@/lib/job-readiness", () => ({
  deriveJobReadiness: mockDeriveJobReadiness,
}));

vi.mock("@/lib/store", async () => {
  const { proxy } = await import("valtio");

  const actions = new Proxy({} as Record<string, ReturnType<typeof vi.fn>>, {
    get(target, prop: string) {
      if (!(prop in target)) target[prop] = vi.fn();
      return target[prop];
    },
  });

  const store = proxy({
    jobsLoading: false,
    jobsError: null as string | null,
    jobs: [] as Job[],
    jobDraft: null as Job | null,
    jobBusy: false,
    jobSourceMode: "file" as "file" | "connector",
    dryBusy: false,
    dryResult: null,
    sourcePreviewBusy: false,
    sourcePreview: null,
    sourceInspecting: false,
    sourceColumns: [],
    stepColumns: [],
    bindings: [],
    templates: [],
    connectorSpecs: [],
    connectorInstances: [],
    enabledSources: [],
    enabledTargets: [],
    targets: [
      {
        id: "embedded",
        name: "Embedded",
        kind: "embedded",
        url: "http://127.0.0.1:7000",
      },
    ],
    activeTargetId: "embedded",
    runsLoading: false,
    runsError: null as string | null,
    runs: [] as JobRun[],
    openRunId: null as string | null,
    runSteps: [] as JobRunStep[],
    templatePack: {} as Record<string, string>,
    packs: [],
    packUpdateResolution: null,
    packUpdateBusy: false,
    packUpdateError: null as string | null,
    runtime: {
      engineMode: "preview" as const,
    },
  });

  const connectorInstance = vi.fn(() => null);
  const connectorSpec = vi.fn(() => null);

  return { actions, store, connectorInstance, connectorSpec };
});

vi.mock("@/lib/api", () => ({
  listBindingConformance: vi.fn().mockResolvedValue([]),
}));

vi.mock("./TemplateCombobox", () => ({
  TemplateCombobox: ({
    value,
    onChange,
  }: {
    value: string;
    onChange: (value: string) => void;
  }) => (
    <input
      aria-label="Template"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("./CodeEditor", () => ({
  CodeEditor: ({
    value,
    onChange,
  }: {
    value: string;
    onChange: (value: string) => void;
  }) => (
    <textarea
      aria-label="Code editor"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("./ConnectorCombobox", () => ({
  ConnectorCombobox: ({
    selectedId,
    onSelect,
  }: {
    selectedId?: string;
    onSelect: (id: string) => void;
  }) => (
    <select
      aria-label="Connector"
      value={selectedId ?? ""}
      onChange={(event) => onSelect(event.target.value)}
    >
      <option value="">Select connector</option>
    </select>
  ),
}));

vi.mock("./FilePathDropzone", () => ({
  FilePathDropzone: ({
    value,
    onChange,
  }: {
    value: string;
    onChange: (value: string) => void;
  }) => (
    <input
      aria-label="Source path"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("./ConnectorsView", () => ({
  ParamField: ({
    param,
    value,
    onChange,
  }: {
    param: { name: string; label?: string | null };
    value: string;
    onChange: (value: string) => void;
  }) => (
    <input
      aria-label={param.label ?? param.name}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("./OutputTable", () => ({
  OutputTable: ({ rows }: { rows: readonly Record<string, unknown>[] }) => (
    <div>rows:{rows.length}</div>
  ),
}));

vi.mock("@/components/ui/tabs", () => ({
  Tabs: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsList: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  TabsTrigger: ({
    children,
    value,
    onClick,
  }: {
    children: React.ReactNode;
    value: string;
    onClick?: () => void;
  }) => (
    <button role="tab" type="button" data-value={value} onClick={onClick}>
      {children}
    </button>
  ),
  TabsContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/components/ui/simulated", () => ({
  SimulatedBadge: () => <span>Simulated</span>,
  SimulatedNotice: ({
    children,
    className,
  }: {
    children?: React.ReactNode;
    className?: string;
  }) => <div className={className}>{children}</div>,
}));

import { JobsView } from "./JobsView";
import { actions, store } from "@/lib/store";

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-nightly-risk",
    name: "Nightly risk rollup",
    description: "",
    enabled: true,
    definition: {
      source: "/data/returns.csv",
      steps: [],
      schedule: undefined,
    },
    ...overrides,
  };
}

function makeRun(overrides: Partial<JobRun> = {}): JobRun {
  return {
    id: "run-1",
    jobId: "job-nightly-risk",
    status: "complete",
    trigger: "manual",
    startedAt: "2026-06-15T09:00:00Z",
    ...overrides,
  };
}

function makeStep(overrides: Partial<JobRunStep> = {}): JobRunStep {
  return {
    runId: "run-1",
    stepIdx: 0,
    template: "mean_return",
    args: {},
    status: "complete",
    rowCount: 42,
    log: "step completed",
    ...overrides,
  };
}

function setReadyReadiness() {
  mockDeriveJobReadiness.mockReturnValue({
    items: [
      { id: "source", label: "Source", state: "complete", message: "ready" },
      {
        id: "sourceTest",
        label: "Source test",
        state: "complete",
        message: "ready",
      },
      { id: "steps", label: "Steps", state: "complete", message: "ready" },
      { id: "params", label: "Params", state: "complete", message: "ready" },
      { id: "output", label: "Output", state: "warning", message: "optional" },
      { id: "dryRun", label: "Dry run", state: "complete", message: "ready" },
      {
        id: "schedule",
        label: "Schedule",
        state: "warning",
        message: "optional",
      },
      {
        id: "compute",
        label: "Compute",
        state: "complete",
        message: "ready",
      },
    ],
    blockers: [],
    warnings: [{ id: "output", label: "Output", state: "warning", message: "optional" }],
    canRun: true,
    nextAction: {
      kind: "runNow",
      label: "Run now",
      disabled: false,
    },
  });
}

describe("JobsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setReadyReadiness();
    store.jobsLoading = false;
    store.jobsError = null;
    store.jobs = [];
    store.jobDraft = null;
    store.jobBusy = false;
    store.jobSourceMode = "file";
    store.dryBusy = false;
    store.dryResult = null;
    store.sourcePreviewBusy = false;
    store.sourcePreview = null;
    store.sourceInspecting = false;
    store.sourceColumns = [];
    store.stepColumns = [];
    store.bindings = [];
    store.templates = [];
    store.connectorSpecs = [];
    store.connectorInstances = [];
    store.enabledSources = [];
    store.enabledTargets = [];
    store.targets = [
      {
        id: "embedded",
        name: "Embedded",
        kind: "embedded",
        url: "http://127.0.0.1:7000",
      },
    ];
    store.activeTargetId = "embedded";
    store.runsLoading = false;
    store.runsError = null;
    store.runs = [];
    store.openRunId = null;
    store.runSteps = [];
    store.templatePack = {};
    store.packs = [];
    store.packUpdateResolution = null;
    store.packUpdateBusy = false;
    store.packUpdateError = null;
    store.runtime.engineMode = "preview";
  });

  it("renders the empty state and creates a new job from the affordance", () => {
    render(<JobsView />);

    expect(screen.getByText("No job selected")).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: /new job|new/i })[0]);

    expect(actions.newJob).toHaveBeenCalled();
  });

  it("runs the current draft and shows existing run steps in run history", () => {
    store.jobs = [makeJob()];
    store.jobDraft = makeJob();
    store.runs = [makeRun()];
    store.openRunId = "run-1";
    store.runSteps = [makeStep()];

    render(<JobsView />);

    fireEvent.click(screen.getByRole("button", { name: "Run now" }));
    expect(actions.runDraft).toHaveBeenCalled();

    expect(screen.getAllByText("Run diagnostics")).toHaveLength(2);
    expect(screen.getAllByText("step completed")).toHaveLength(2);
    expect(screen.getByText("mean_return")).toBeInTheDocument();
  });

  it("renders preview notices and badges for simulated offline run output", () => {
    store.runtime.engineMode = "preview";
    store.jobs = [makeJob()];
    store.jobDraft = makeJob();
    store.dryResult = {
      status: "complete",
      rows: [{ region: "emea" }],
      rowCount: 1,
      elapsedMs: 12,
      error: null,
    };

    render(<JobsView />);

    expect(
      screen.getByText(
        "Preview mode: run history, dry runs, and source tests may show simulated output rather than live engine results.",
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Simulated").length).toBeGreaterThan(0);
  });

  it("renders schedule validation and wires cron changes back through the store", () => {
    store.jobs = [makeJob()];
    store.jobDraft = makeJob({
      definition: {
        source: "/data/returns.csv",
        steps: [],
        schedule: "bogus",
      },
    });

    render(<JobsView />);

    expect(
      screen.getByText("Not a valid 5-field cron expression."),
    ).toBeInTheDocument();

    const minuteInput = screen.getByDisplayValue("bogus");
    fireEvent.change(minuteInput, { target: { value: "0" } });

    expect(actions.setDraftSchedule).toHaveBeenCalledWith("0 * * * *");
  });
});
