import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { Job, JobRun, JobRunStep, StagedRelation } from "@/lib/types";

vi.mock("@/lib/store", async () => {
  const { proxy } = await import("valtio");

  const actions = new Proxy(
    {
      loadObservability: vi.fn(),
      setView: vi.fn(),
      editJob: vi.fn(),
      openRun: vi.fn(),
    } as Record<string, ReturnType<typeof vi.fn>>,
    {
      get(target, prop: string) {
        if (!(prop in target)) target[prop] = vi.fn();
        return target[prop];
      },
    },
  );

  const store = proxy({
    jobs: [] as Job[],
    jobsLoading: false,
    jobsError: null as string | null,
    allRuns: [] as JobRun[],
    allRunsLoading: false,
    allRunsError: null as string | null,
    stagedRelations: [] as StagedRelation[],
    stagedBusy: false,
    stagedMessage: null as string | null,
    openRunId: null as string | null,
    runSteps: [] as JobRunStep[],
  });

  return { actions, store };
});

import { ObservabilityView } from "./ObservabilityView";
import { actions, store } from "@/lib/store";

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "job-daily-orders",
    name: "Daily orders",
    description: "",
    enabled: true,
    definition: {
      source: "/tmp/orders.csv",
      steps: [],
      schedule: "0 2 * * *",
      targetConnectorId: "filesystem-target",
      targetConnectorParams: {
        output_path: "/tmp/out/orders.parquet",
      },
    },
    ...overrides,
  };
}

function makeRun(overrides: Partial<JobRun> = {}): JobRun {
  return {
    id: "run-1",
    jobId: "job-daily-orders",
    status: "complete",
    trigger: "schedule",
    startedAt: "2026-06-25T02:00:00Z",
    finishedAt: "2026-06-25T02:05:00Z",
    metadata: {
      syncMode: "incremental",
      summary: {
        totalRowCount: 4200,
        stepCount: 2,
        successfulSteps: 2,
        failedSteps: 0,
      },
      output: {
        targetPath: "/tmp/out/orders.parquet",
        stagedRelationId: "rel-orders",
      },
    },
    ...overrides,
  };
}

function makeRelation(overrides: Partial<StagedRelation> = {}): StagedRelation {
  return {
    relationId: "rel-orders",
    name: "Daily orders",
    contentHash: "abc123",
    contentType: "application/x-parquet",
    sizeBytes: 1024,
    rowCount: 4200,
    columns: ["order_id", "updated_at"],
    producer: {
      flowRunId: "run-1",
    },
    dependsOn: [],
    retention: "ephemeral",
    createdMs: Date.parse("2026-06-25T02:05:00Z"),
    freshness: "fresh",
    recomputable: true,
    ...overrides,
  };
}

function makeSecondJob(): Job {
  return {
    id: "job-risk-scan",
    name: "Risk scan",
    description: "",
    enabled: true,
    definition: {
      source: "/tmp/risk.csv",
      steps: [],
      targetConnectorId: "filesystem-target",
      targetConnectorParams: {
        output_path: "/tmp/out/risk.parquet",
      },
    },
  };
}

describe("ObservabilityView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    store.jobs = [];
    store.jobsLoading = false;
    store.jobsError = null;
    store.allRuns = [];
    store.allRunsLoading = false;
    store.allRunsError = null;
    store.stagedRelations = [];
    store.stagedBusy = false;
    store.stagedMessage = null;
    store.openRunId = null;
    store.runSteps = [];
  });

  it("renders operational metrics, schedules, and outputs", () => {
    store.jobs = [makeJob()];
    store.allRuns = [makeRun()];
    store.stagedRelations = [makeRelation()];

    render(<ObservabilityView />);

    expect(screen.getByText("ETL Observability")).toBeInTheDocument();
    expect(screen.getByText("Success rate")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("Scheduled jobs")).toBeInTheDocument();
    expect(screen.getByText("Slowest recent runs")).toBeInTheDocument();
    expect(screen.getByText("Throughput leaders")).toBeInTheDocument();
    expect(screen.getByText("Schedule watchlist")).toBeInTheDocument();
    expect(screen.getAllByText("Output freshness").length).toBeGreaterThan(0);
    expect(screen.getByText("Runtime distribution")).toBeInTheDocument();
    expect(screen.getByText("Failure signatures")).toBeInTheDocument();
    expect(screen.getByText("Volume anomalies")).toBeInTheDocument();
    expect(screen.getByText("Failure streaks")).toBeInTheDocument();
    expect(screen.getByText("Determinism coverage")).toBeInTheDocument();
    expect(screen.getByText("Observability gaps")).toBeInTheDocument();
    expect(screen.getByText("Execution shape")).toBeInTheDocument();
    expect(screen.getByText("Operating drift")).toBeInTheDocument();
    expect(screen.getByText("Output estate")).toBeInTheDocument();
    expect(screen.getByText("Control posture")).toBeInTheDocument();
    expect(screen.getByText("Governance scorecards")).toBeInTheDocument();
    expect(screen.getByText("Metadata coverage")).toBeInTheDocument();
    expect(screen.getAllByText("Daily orders").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText("/tmp/out/orders.parquet").length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("rel-orders").length).toBeGreaterThan(0);
  });

  it("routes a failed run back to the jobs diagnostics flow", () => {
    store.jobs = [makeJob()];
    store.allRuns = [
      makeRun({
        id: "run-fail",
        status: "failed",
        metadata: {
          diagnostics: {
            message: "Warehouse write timed out",
          },
          summary: {
            totalRowCount: 0,
          },
        },
      }),
    ];

    render(<ObservabilityView />);

    const alertsPanel = screen
      .getByText("Recent alerts")
      .closest("section") as HTMLElement;
    fireEvent.click(
      within(alertsPanel).getByRole("button", { name: /job-daily-orders/i }),
    );

    expect(actions.setView).toHaveBeenCalledWith("jobs");
    expect(actions.editJob).toHaveBeenCalledWith("job-daily-orders");
    expect(actions.openRun).toHaveBeenCalledWith("run-fail");
  });

  it("focuses a job card to drive the drilldown timeline", () => {
    store.jobs = [makeJob(), makeSecondJob()];
    store.allRuns = [
      makeRun(),
      makeRun({
        id: "run-2",
        jobId: "job-risk-scan",
        status: "complete",
        startedAt: "2026-06-25T04:00:00Z",
        metadata: {
          syncMode: "append",
          summary: {
            totalRowCount: 88,
            stepCount: 1,
            successfulSteps: 1,
            failedSteps: 0,
          },
          output: {
            targetPath: "/tmp/out/risk.parquet",
          },
        },
      }),
    ];

    render(<ObservabilityView />);

    const riskScanCard = screen
      .getAllByText("Risk scan")
      .find((node) => node.closest("div.rounded-2xl"));
    expect(riskScanCard).toBeTruthy();
    fireEvent.click(
      within(riskScanCard!.closest("div.rounded-2xl") as HTMLElement).getByRole(
        "button",
        { name: "Focus" },
      ),
    );

    expect(screen.getAllByText("Risk scan").length).toBeGreaterThan(0);
    expect(screen.getAllByText("/tmp/out/risk.parquet").length).toBeGreaterThan(
      0,
    );
  });

  it("selects a run from the timeline without leaving observability", () => {
    store.jobs = [makeJob()];
    store.allRuns = [
      makeRun(),
      makeRun({
        id: "run-older",
        startedAt: "2026-06-25T01:00:00Z",
        metadata: {
          syncMode: "append",
          summary: {
            totalRowCount: 120,
            stepCount: 1,
            successfulSteps: 1,
            failedSteps: 0,
          },
          output: {
            targetPath: "/tmp/out/older.parquet",
            stagedRelationId: "rel-older",
          },
        },
      }),
    ];

    render(<ObservabilityView />);

    const timelinePanel = screen
      .getByText("Job timeline")
      .closest("section") as HTMLElement;
    fireEvent.click(
      within(timelinePanel).getByRole("button", { name: /run-older/i }),
    );

    expect(screen.getByText("Run spotlight")).toBeInTheDocument();
    expect(
      screen.getAllByText("/tmp/out/older.parquet").length,
    ).toBeGreaterThan(0);
    expect(actions.openRun).toHaveBeenCalled();
    expect(actions.setView).not.toHaveBeenCalledWith("jobs");
  });
});
