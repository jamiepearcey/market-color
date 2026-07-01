import type { ReactNode } from "react";
import { useEffect, useRef } from "react";
import {
  Activity,
  AlertTriangle,
  CalendarClock,
  CheckCheck,
  CheckCircle2,
  Clock3,
  Database,
  Filter,
  Gauge,
  MoveRight,
  RefreshCw,
  Shapes,
  Siren,
  Waves,
} from "lucide-react";
import { proxy, useSnapshot } from "valtio";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyPanel, ErrorPanel, SkeletonRows } from "@/components/AsyncStates";
import { formatUtc, nextRuns } from "@/components/CronEditor";
import { actions, store } from "@/lib/store";
import type { Job, JobRun, JobRunStep, StagedRelation } from "@/lib/types";
import { cn } from "@/lib/utils";

type TrendPoint = {
  label: string;
  total: number;
  complete: number;
  failed: number;
};

export function ObservabilityView() {
  const snap = useSnapshot(store);
  const local = useRef(
    proxy({
      range: "7d" as "24h" | "7d" | "30d" | "all",
      status: "all" as "all" | "complete" | "failed",
      jobId: "all",
      selectedRunId: "",
    }),
  ).current;
  const localSnap = useSnapshot(local);
  const jobs = snap.jobs.map((job) => JSON.parse(JSON.stringify(job)) as Job);
  const runs = snap.allRuns.map(
    (run) => JSON.parse(JSON.stringify(run)) as JobRun,
  );
  const staged = snap.stagedRelations.map(
    (relation) => JSON.parse(JSON.stringify(relation)) as StagedRelation,
  );
  const loading = snap.jobsLoading || snap.allRunsLoading || snap.stagedBusy;
  const error =
    snap.jobsError ?? snap.allRunsError ?? (snap.stagedMessage || null);

  if (
    loading &&
    jobs.length === 0 &&
    runs.length === 0 &&
    staged.length === 0
  ) {
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-outline-subtle px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-2xl bg-[linear-gradient(135deg,rgba(56,189,248,0.18),rgba(45,212,191,0.12))] text-sky-200">
              <Activity className="size-5" />
            </div>
            <div>
              <div className="text-lg font-semibold tracking-[-0.02em] text-foreground">
                Observability
              </div>
              <div className="text-sm text-muted-foreground">
                Run telemetry, schedule visibility, staged outputs, and job
                health.
              </div>
            </div>
          </div>
        </div>
        <div className="space-y-4 p-6">
          <SkeletonRows count={12} />
        </div>
      </div>
    );
  }

  if (error && jobs.length === 0 && runs.length === 0) {
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-outline-subtle px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-2xl bg-[linear-gradient(135deg,rgba(56,189,248,0.18),rgba(45,212,191,0.12))] text-sky-200">
              <Activity className="size-5" />
            </div>
            <div>
              <div className="text-lg font-semibold tracking-[-0.02em] text-foreground">
                Observability
              </div>
              <div className="text-sm text-muted-foreground">
                Run telemetry, schedule visibility, staged outputs, and job
                health.
              </div>
            </div>
          </div>
        </div>
        <div className="p-6">
          <ErrorPanel
            title="Could not load observability data"
            message={error}
            onRetry={() => void actions.loadObservability()}
          />
        </div>
      </div>
    );
  }

  if (jobs.length === 0 && runs.length === 0) {
    return (
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-outline-subtle px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-2xl bg-[linear-gradient(135deg,rgba(56,189,248,0.18),rgba(45,212,191,0.12))] text-sky-200">
              <Activity className="size-5" />
            </div>
            <div>
              <div className="text-lg font-semibold tracking-[-0.02em] text-foreground">
                Observability
              </div>
              <div className="text-sm text-muted-foreground">
                Run telemetry, schedule visibility, staged outputs, and job
                health.
              </div>
            </div>
          </div>
        </div>
        <div className="p-6">
          <EmptyPanel
            title="No telemetry yet"
            message="Create a job and run it once to populate operational metrics, timelines, and staged outputs."
          />
        </div>
      </div>
    );
  }

  const now = new Date();
  const filteredRuns = runs.filter((run) => {
    if (localSnap.jobId !== "all" && run.jobId !== localSnap.jobId)
      return false;
    if (localSnap.status !== "all" && run.status !== localSnap.status)
      return false;
    const ts = parseTimestamp(run.startedAt);
    if (ts == null) return localSnap.range === "all";
    if (localSnap.range === "all") return true;
    const windowMs =
      localSnap.range === "24h"
        ? 24 * 60 * 60 * 1000
        : localSnap.range === "7d"
          ? 7 * 24 * 60 * 60 * 1000
          : 30 * 24 * 60 * 60 * 1000;
    return now.getTime() - ts <= windowMs;
  });
  const filteredJobs =
    localSnap.jobId === "all"
      ? jobs
      : jobs.filter((job) => job.id === localSnap.jobId);
  const scheduledJobs = filteredJobs.filter((job) => hasSchedule(job));
  const enabledJobs = filteredJobs.filter((job) => job.enabled);
  const completeRuns = filteredRuns.filter((run) => run.status === "complete");
  const failedRuns = filteredRuns.filter((run) => run.status === "failed");
  const activeAlerts = failedRuns.slice(0, 4);
  const terminalRuns = filteredRuns.filter(
    (run) => run.status === "complete" || run.status === "failed",
  );
  const successRate = terminalRuns.length
    ? Math.round((completeRuns.length / terminalRuns.length) * 100)
    : 0;
  const last24h = filteredRuns.filter((run) => {
    const ts = parseTimestamp(run.startedAt);
    return ts != null && now.getTime() - ts <= 24 * 60 * 60 * 1000;
  });
  const last72h = filteredRuns.filter((run) => {
    const ts = parseTimestamp(run.startedAt);
    return ts != null && now.getTime() - ts <= 72 * 60 * 60 * 1000;
  });
  const freshRelations = staged.filter(
    (relation) => relation.freshness === "fresh",
  );
  const latestRun = filteredRuns[0] ?? null;
  const latestRelation =
    staged.slice().sort((a, b) => b.createdMs - a.createdMs)[0] ?? null;
  const trend = buildDailyTrend(
    filteredRuns,
    localSnap.range === "24h" ? 2 : localSnap.range === "30d" ? 30 : 14,
  );
  const hourly = buildHourlyTrend(filteredRuns);
  const syncModes = buildSyncModeDistribution(filteredJobs, filteredRuns);
  const jobCards = buildJobCards(filteredJobs, filteredRuns, staged, now);
  const noisyJobs = buildNoisyJobs(filteredJobs, filteredRuns);
  const slowRuns = buildSlowRuns(filteredJobs, filteredRuns).slice(0, 6);
  const jobPerformance = buildJobPerformance(filteredJobs, filteredRuns).slice(
    0,
    6,
  );
  const scheduleWatch = buildScheduleWatch(
    filteredJobs,
    filteredRuns,
    staged,
    now,
  );
  const atRiskJobs = scheduleWatch.filter(
    (entry) => entry.status !== "healthy",
  );
  const staleRelations = buildStaleRelations(staged, now).slice(0, 6);
  const runtimeStats = buildRuntimeStats(filteredRuns);
  const failureSignatures = buildFailureSignatures(filteredRuns).slice(0, 6);
  const volumeAnomalies = buildVolumeAnomalies(
    filteredJobs,
    filteredRuns,
  ).slice(0, 6);
  const failureStreaks = buildFailureStreaks(filteredJobs, filteredRuns).slice(
    0,
    6,
  );
  const determinismMix = buildDeterminismMix(filteredJobs, filteredRuns);
  const observabilityGaps = buildObservabilityGaps(
    filteredJobs,
    filteredRuns,
    staged,
  ).slice(0, 6);
  const executionModeMix = buildExecutionModeMix(filteredJobs, filteredRuns);
  const sourceModeMix = buildSourceModeMix(filteredJobs, filteredRuns);
  const operatingDrift = buildOperatingDrift(filteredJobs, filteredRuns).slice(
    0,
    6,
  );
  const outputEstateMix = buildOutputEstateMix(filteredJobs, filteredRuns);
  const controlPosture = buildControlPosture(filteredJobs, filteredRuns);
  const governanceScorecards = buildGovernanceScorecards(
    filteredJobs,
    filteredRuns,
  ).slice(0, 6);
  const metadataCoverage = buildMetadataCoverage(filteredJobs, filteredRuns);
  const telemetryConfidence = buildTelemetryConfidence(
    filteredJobs,
    filteredRuns,
  ).slice(0, 6);
  const spotlightJob =
    (localSnap.jobId !== "all"
      ? (filteredJobs.find((job) => job.id === localSnap.jobId) ?? null)
      : (filteredJobs[0] ?? null)) || null;
  const spotlightRuns = spotlightJob
    ? filteredRuns.filter((run) => run.jobId === spotlightJob.id).slice(0, 8)
    : [];
  const spotlightRun =
    spotlightRuns.find((run) => run.id === localSnap.selectedRunId) ??
    spotlightRuns[0] ??
    null;
  const spotlightSteps =
    snap.openRunId === spotlightRun?.id ? snap.runSteps.slice() : [];

  useEffect(() => {
    if (!spotlightRun) return;
    if (snap.openRunId === spotlightRun.id) return;
    void actions.openRun(spotlightRun.id);
  }, [spotlightRun?.id, snap.openRunId]);

  return (
    <div className="flex min-w-0 flex-1 flex-col">
      <div className="border-b border-outline-subtle bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.12),transparent_34%),linear-gradient(180deg,rgba(13,18,29,0.7),rgba(13,18,29,0.18))] px-6 py-5">
        <div className="flex flex-wrap items-start gap-4">
          <div className="flex min-w-0 flex-1 items-start gap-3">
            <div className="grid size-11 shrink-0 place-items-center rounded-2xl border border-sky-400/20 bg-[linear-gradient(135deg,rgba(56,189,248,0.22),rgba(45,212,191,0.12))] text-sky-100 shadow-[0_0_40px_rgba(56,189,248,0.12)]">
              <Activity className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-xl font-semibold tracking-[-0.03em] text-foreground">
                  ETL Observability
                </h1>
                <Badge
                  variant="secondary"
                  className="border border-sky-400/20 bg-sky-400/10 text-[10px] uppercase tracking-[0.12em] text-sky-100"
                >
                  Live control plane
                </Badge>
              </div>
              <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted-foreground">
                Operational view across schedules, run health, sync modes,
                recent failures, and staged outputs.
              </p>
            </div>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void actions.loadObservability()}
          >
            <RefreshCw className="size-4" /> Refresh
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="space-y-6 p-6">
          <section className="rounded-2xl border border-outline-subtle bg-surface-panel shadow-[0_8px_30px_rgba(7,11,19,0.18)]">
            <div className="flex flex-wrap items-center gap-3 border-b border-outline-subtle px-4 py-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                <Filter className="size-4 text-muted-foreground" />
                Filters
              </div>
              <div className="min-w-0 text-xs text-muted-foreground">
                Scope the dashboard to a time window, run outcome, or single
                job.
              </div>
            </div>
            <div className="flex flex-wrap gap-4 px-4 py-4">
              <FilterGroup label="Window">
                {(["24h", "7d", "30d", "all"] as const).map((range) => (
                  <button
                    key={range}
                    type="button"
                    onClick={() => {
                      local.range = range;
                    }}
                    className={cn(
                      "rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors",
                      localSnap.range === range
                        ? "border-sky-400/30 bg-sky-400/10 text-sky-100"
                        : "border-outline-subtle bg-background/40 text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {range}
                  </button>
                ))}
              </FilterGroup>
              <FilterGroup label="Outcome">
                {(["all", "complete", "failed"] as const).map((status) => (
                  <button
                    key={status}
                    type="button"
                    onClick={() => {
                      local.status = status;
                    }}
                    className={cn(
                      "rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors",
                      localSnap.status === status
                        ? "border-teal-400/30 bg-teal-400/10 text-teal-100"
                        : "border-outline-subtle bg-background/40 text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {status}
                  </button>
                ))}
              </FilterGroup>
              <FilterGroup label="Job">
                <select
                  aria-label="Filter by job"
                  value={localSnap.jobId}
                  onChange={(event) => {
                    local.jobId = event.target.value;
                    local.selectedRunId = "";
                  }}
                  className="h-8 rounded-md border border-outline-subtle bg-background px-2 text-[12px] text-foreground"
                >
                  <option value="all">All jobs</option>
                  {jobs.map((job) => (
                    <option key={job.id} value={job.id}>
                      {job.name}
                    </option>
                  ))}
                </select>
              </FilterGroup>
            </div>
          </section>

          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
            <MetricCard
              icon={Gauge}
              label="Success rate"
              value={`${successRate}%`}
              note={`${completeRuns.length} / ${terminalRuns.length || 0} terminal runs complete`}
              accent="sky"
            />
            <MetricCard
              icon={CalendarClock}
              label="Scheduled jobs"
              value={`${scheduledJobs.length}`}
              note={`${enabledJobs.filter((job) => hasSchedule(job)).length} enabled on cron`}
              accent="teal"
            />
            <MetricCard
              icon={Waves}
              label="Runs in 24h"
              value={`${last24h.length}`}
              note={`${last72h.length} over the last 72 hours`}
              accent="amber"
            />
            <MetricCard
              icon={Database}
              label="Staged outputs"
              value={`${staged.length}`}
              note={`${freshRelations.length} fresh relations available`}
              accent="violet"
            />
            <MetricCard
              icon={AlertTriangle}
              label="At-risk jobs"
              value={`${atRiskJobs.length}`}
              note={`${scheduleWatch.length} scheduled jobs evaluated for lateness and freshness`}
              accent="amber"
            />
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.9fr)]">
            <Panel
              title="Run volume"
              subtitle="Last 14 days across the latest 200 recorded runs."
              icon={Activity}
            >
              <StackedRunChart data={trend} />
            </Panel>
            <Panel
              title="Sync mode mix"
              subtitle="Grouped from run metadata first, then job definitions when no run exists yet."
              icon={Shapes}
            >
              <ModeDistributionChart data={syncModes} />
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.92fr)]">
            <Panel
              title="Run pulse"
              subtitle="Hourly activity over the last 24 hours."
              icon={Clock3}
            >
              <HourlyPulseChart data={hourly} />
            </Panel>
            <Panel
              title="Upcoming schedules"
              subtitle="Next fire times computed from each job's cron expression in UTC."
              icon={CalendarClock}
            >
              {scheduledJobs.length === 0 ? (
                <EmptyPanel
                  title="No scheduled jobs"
                  message="Jobs without cron schedules remain on-demand only."
                />
              ) : (
                <div className="space-y-2">
                  {scheduledJobs.slice(0, 6).map((job) => (
                    <ScheduleRow key={job.id} job={job} />
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Recent alerts"
              subtitle="Latest failed runs with direct links back to run diagnostics."
              icon={Siren}
            >
              {activeAlerts.length === 0 ? (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.08] px-4 py-5 text-sm text-emerald-100">
                  No recent failed runs in the loaded telemetry window.
                </div>
              ) : (
                <div className="space-y-2">
                  {activeAlerts.map((run) => (
                    <FailureRow key={run.id} run={run} />
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Failure concentration"
              subtitle="Jobs with the most failed runs in the active filter window."
              icon={AlertTriangle}
            >
              {noisyJobs.length === 0 ? (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.08] px-4 py-5 text-sm text-emerald-100">
                  No failed runs in the active filter window.
                </div>
              ) : (
                <div className="space-y-3">
                  {noisyJobs.map((entry) => (
                    <div key={entry.job.id} className="space-y-1.5">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="min-w-0 flex-1 truncate font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.failures} failed / {entry.total} total
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-surface-toolbar">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,rgba(244,63,94,0.9),rgba(251,146,60,0.9))]"
                          style={{
                            width: `${Math.max(entry.ratio * 100, 8)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Execution mix"
              subtitle="Manual versus scheduled triggers in the active filter window."
              icon={CheckCheck}
            >
              <ExecutionMix runs={filteredRuns} />
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Slowest recent runs"
              subtitle="Recent runs with the highest recorded elapsed time in the active filter window."
              icon={Clock3}
            >
              {slowRuns.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No completed runs with elapsed-time telemetry are in scope.
                </div>
              ) : (
                <div className="space-y-2">
                  {slowRuns.map((entry) => (
                    <button
                      key={entry.run.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.run.jobId;
                        local.selectedRunId = entry.run.id;
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-2 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <RunStatusPill status={entry.run.status} />
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.jobName}
                        </span>
                        <span className="text-[11px] text-amber-100">
                          {formatDurationMs(entry.elapsedMs)}
                        </span>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-3">
                        <MiniStat label="Run" value={entry.run.id} />
                        <MiniStat
                          label="Rows"
                          value={formatNumber(entry.rows)}
                        />
                        <MiniStat
                          label="Started"
                          value={shortWhen(entry.run.startedAt)}
                        />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Throughput leaders"
              subtitle="Jobs ranked by average rows per second across completed runs with duration telemetry."
              icon={Waves}
            >
              {jobPerformance.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No throughput telemetry is available in the active filter
                  window.
                </div>
              ) : (
                <div className="space-y-3">
                  {jobPerformance.map((entry) => (
                    <div
                      key={entry.job.id}
                      className="rounded-xl border border-outline-subtle bg-background/30 px-3 py-3"
                    >
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            local.jobId = entry.job.id;
                            local.selectedRunId = "";
                          }}
                        >
                          Focus
                        </Button>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-4">
                        <MiniStat
                          label="Rows / sec"
                          value={formatRate(entry.avgRowsPerSecond)}
                        />
                        <MiniStat
                          label="Avg elapsed"
                          value={formatDurationMs(entry.avgElapsedMs)}
                        />
                        <MiniStat
                          label="Completed runs"
                          value={formatNumber(entry.completedRuns)}
                        />
                        <MiniStat
                          label="Rows moved"
                          value={formatNumber(entry.totalRows)}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Schedule watchlist"
              subtitle="Enabled scheduled jobs ranked by lateness, recent failure state, and missing output freshness."
              icon={CalendarClock}
            >
              {scheduleWatch.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No enabled scheduled jobs are currently in scope.
                </div>
              ) : (
                <div className="space-y-2">
                  {scheduleWatch.slice(0, 6).map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = "";
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-2 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex flex-wrap items-center gap-2">
                        <span
                          className={cn(
                            "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em]",
                            entry.status === "healthy"
                              ? "border-emerald-500/30 bg-emerald-500/[0.08] text-emerald-100"
                              : entry.status === "late"
                                ? "border-amber-500/30 bg-amber-500/[0.08] text-amber-100"
                                : "border-rose-500/30 bg-rose-500/[0.08] text-rose-100",
                          )}
                        >
                          {humanizeMode(entry.status)}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.nextRunLabel}
                        </span>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-4">
                        <MiniStat
                          label="Last run"
                          value={entry.lastRunLabel}
                          tone={
                            entry.status === "failed" ? "danger" : "default"
                          }
                        />
                        <MiniStat
                          label="Output freshness"
                          value={entry.freshnessLabel}
                          tone={entry.status === "stale" ? "danger" : "default"}
                        />
                        <MiniStat
                          label="Schedule interval"
                          value={entry.intervalLabel}
                        />
                        <MiniStat
                          label="Reason"
                          value={entry.reason}
                          tone={
                            entry.status === "healthy" ? "default" : "danger"
                          }
                        />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Output freshness"
              subtitle="Newest staged relations first, with stale and aging outputs surfaced for quick inspection."
              icon={Database}
            >
              {staleRelations.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No staged relations are currently marked stale or aging.
                </div>
              ) : (
                <div className="space-y-2">
                  {staleRelations.map((relation) => (
                    <div
                      key={relation.relationId}
                      className="rounded-xl border border-outline-subtle bg-background/30 px-3 py-2"
                    >
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {relation.name}
                        </span>
                        <Badge
                          variant="outline"
                          className={cn(
                            "text-[9px] uppercase tracking-[0.12em]",
                            relation.freshness === "fresh"
                              ? "border-emerald-500/30 bg-emerald-500/[0.08] text-emerald-100"
                              : "border-amber-500/30 bg-amber-500/[0.08] text-amber-100",
                          )}
                        >
                          {relation.freshness ?? "unknown"}
                        </Badge>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-4">
                        <MiniStat
                          label="Relation"
                          value={relation.relationId}
                        />
                        <MiniStat
                          label="Rows"
                          value={formatNumber(relation.rowCount ?? 0)}
                        />
                        <MiniStat
                          label="Age"
                          value={formatAgeMs(
                            now.getTime() - relation.createdMs,
                          )}
                        />
                        <MiniStat
                          label="Produced"
                          value={shortWhen(
                            new Date(relation.createdMs).toISOString(),
                          )}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Runtime distribution"
              subtitle="Percentiles and spread across completed runs with elapsed-time telemetry in the active filter window."
              icon={Clock3}
            >
              {runtimeStats.count === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No completed runs with elapsed-time telemetry are in scope.
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid gap-2 md:grid-cols-4">
                    <MiniStat
                      label="Median"
                      value={formatDurationMs(runtimeStats.p50)}
                    />
                    <MiniStat
                      label="P95"
                      value={formatDurationMs(runtimeStats.p95)}
                    />
                    <MiniStat
                      label="Fastest"
                      value={formatDurationMs(runtimeStats.min)}
                    />
                    <MiniStat
                      label="Slowest"
                      value={formatDurationMs(runtimeStats.max)}
                    />
                  </div>
                  <RuntimeSpread stats={runtimeStats} />
                </div>
              )}
            </Panel>
            <Panel
              title="Failure signatures"
              subtitle="Normalized failure messages clustered from failed runs in the active filter window."
              icon={AlertTriangle}
            >
              {failureSignatures.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No failed-run signatures are available in the active filter
                  window.
                </div>
              ) : (
                <div className="space-y-3">
                  {failureSignatures.map((entry) => (
                    <div key={entry.signature} className="space-y-1.5">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="min-w-0 flex-1 truncate font-medium text-foreground">
                          {entry.label}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.count} runs
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-surface-toolbar">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,rgba(244,63,94,0.9),rgba(251,146,60,0.9))]"
                          style={{
                            width: `${Math.max(entry.ratio * 100, 10)}%`,
                          }}
                        />
                      </div>
                      <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                        <span>Latest: {shortWhen(entry.latestAt)}</span>
                        <span>Jobs: {entry.jobCount}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Volume anomalies"
              subtitle="Jobs whose latest row count deviates materially from their own recent baseline."
              icon={Waves}
            >
              {volumeAnomalies.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No row-volume anomalies are detectable in the active filter
                  window.
                </div>
              ) : (
                <div className="space-y-3">
                  {volumeAnomalies.map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = entry.latestRun.id;
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-3 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className={cn(
                            "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em]",
                            entry.direction === "up"
                              ? "border-amber-500/30 bg-amber-500/[0.08] text-amber-100"
                              : "border-sky-500/30 bg-sky-500/[0.08] text-sky-100",
                          )}
                        >
                          {entry.direction === "up" ? "Spike" : "Drop"}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.deltaPercent > 0 ? "+" : ""}
                          {entry.deltaPercent}%
                        </span>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-4">
                        <MiniStat
                          label="Latest rows"
                          value={formatNumber(entry.latestRows)}
                        />
                        <MiniStat
                          label="Baseline"
                          value={formatNumber(entry.baselineRows)}
                        />
                        <MiniStat label="Run" value={entry.latestRun.id} />
                        <MiniStat
                          label="Observed"
                          value={shortWhen(entry.latestRun.startedAt)}
                        />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Failure streaks"
              subtitle="Jobs currently failing back-to-back, ordered by streak length and recency."
              icon={Siren}
            >
              {failureStreaks.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No active consecutive failure streaks are in the active filter
                  window.
                </div>
              ) : (
                <div className="space-y-3">
                  {failureStreaks.map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = entry.latestRun.id;
                      }}
                      className="w-full rounded-xl border border-rose-500/20 bg-rose-500/[0.08] px-3 py-3 text-left transition-colors hover:bg-rose-500/[0.12]"
                    >
                      <div className="flex items-center gap-2">
                        <span className="inline-flex items-center rounded-full border border-rose-500/30 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] text-rose-100">
                          {entry.streak}x failed
                        </span>
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {shortWhen(entry.latestRun.startedAt)}
                        </span>
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-3">
                        <MiniStat
                          label="Latest run"
                          value={entry.latestRun.id}
                          tone="danger"
                        />
                        <MiniStat
                          label="Latest error"
                          value={truncateLabel(
                            failureMessage(entry.latestRun),
                            40,
                          )}
                          tone="danger"
                        />
                        <MiniStat
                          label="Total runs in scope"
                          value={formatNumber(entry.totalRuns)}
                        />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Determinism coverage"
              subtitle="Latest known reproducibility class by job from run metadata."
              icon={Shapes}
            >
              {determinismMix.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No determinism metadata is available in the active filter
                  window.
                </div>
              ) : (
                <div className="space-y-3">
                  {determinismMix.map((entry, index) => (
                    <div key={entry.label} className="space-y-1.5">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="min-w-0 flex-1 truncate font-medium text-foreground">
                          {entry.label}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.count} jobs
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-surface-toolbar">
                        <div
                          className={cn(
                            "h-full rounded-full bg-[linear-gradient(90deg,var(--tw-gradient-stops))]",
                            index % 3 === 0
                              ? "from-emerald-400 to-teal-300"
                              : index % 3 === 1
                                ? "from-amber-400 to-orange-300"
                                : "from-slate-400 to-slate-300",
                          )}
                          style={{
                            width: `${Math.max(entry.ratio * 100, 8)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Observability gaps"
              subtitle="Jobs missing successful runs, staged outputs, or schedule coverage in the active scope."
              icon={Gauge}
            >
              {observabilityGaps.length === 0 ? (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.08] px-4 py-5 text-sm text-emerald-100">
                  No obvious control-plane coverage gaps are visible in the
                  active scope.
                </div>
              ) : (
                <div className="space-y-2">
                  {observabilityGaps.map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = "";
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-3 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.gaps.length} gaps
                        </span>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {entry.gaps.map((gap) => (
                          <Badge
                            key={gap}
                            variant="outline"
                            className="border-amber-500/30 bg-amber-500/[0.08] text-[9px] uppercase tracking-[0.12em] text-amber-100"
                          >
                            {gap}
                          </Badge>
                        ))}
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-3">
                        <MiniStat
                          label="Runs in scope"
                          value={formatNumber(entry.runCount)}
                        />
                        <MiniStat
                          label="Last success"
                          value={entry.lastSuccessLabel}
                        />
                        <MiniStat
                          label="Execution mix"
                          value={entry.executionLabel}
                        />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Execution shape"
              subtitle="Fleet mix across execution and source modes from the latest recorded run metadata."
              icon={Activity}
            >
              <div className="grid gap-4 lg:grid-cols-2">
                <ModeSummaryList
                  title="Execution mode"
                  emptyMessage="No execution mode metadata is available yet."
                  data={executionModeMix}
                />
                <ModeSummaryList
                  title="Source mode"
                  emptyMessage="No source mode metadata is available yet."
                  data={sourceModeMix}
                />
              </div>
            </Panel>
            <Panel
              title="Operating drift"
              subtitle="Jobs whose recent operating shape suggests manual-only or ad hoc behavior."
              icon={Filter}
            >
              {operatingDrift.length === 0 ? (
                <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.08] px-4 py-5 text-sm text-emerald-100">
                  No operating-pattern drift is visible in the active scope.
                </div>
              ) : (
                <div className="space-y-2">
                  {operatingDrift.map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = "";
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-3 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.flags.length} flags
                        </span>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {entry.flags.map((flag) => (
                          <Badge
                            key={flag}
                            variant="outline"
                            className="border-sky-500/30 bg-sky-500/[0.08] text-[9px] uppercase tracking-[0.12em] text-sky-100"
                          >
                            {flag}
                          </Badge>
                        ))}
                      </div>
                      <div className="mt-2 grid gap-2 md:grid-cols-3">
                        <MiniStat
                          label="Runs in scope"
                          value={formatNumber(entry.runCount)}
                        />
                        <MiniStat
                          label="Execution"
                          value={entry.executionLabel}
                        />
                        <MiniStat label="Source" value={entry.sourceLabel} />
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Output estate"
              subtitle="Latest known destination types across the fleet from recorded run output metadata."
              icon={Database}
            >
              <ModeSummaryList
                title="Destination type"
                emptyMessage="No output destination metadata is available yet."
                data={outputEstateMix}
              />
            </Panel>
            <Panel
              title="Control posture"
              subtitle="Fleet classification by how managed each job is in practice."
              icon={CheckCheck}
            >
              <ModeSummaryList
                title="Posture"
                emptyMessage="No jobs are in scope for posture classification."
                data={controlPosture}
              />
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Governance scorecards"
              subtitle="Per-job control/readiness scores from scheduling, determinism, source, success, and durable output evidence."
              icon={Gauge}
            >
              {governanceScorecards.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No jobs are in scope for governance scoring.
                </div>
              ) : (
                <div className="space-y-3">
                  {governanceScorecards.map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = "";
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-3 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.score}/{entry.maxScore}
                        </span>
                      </div>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-toolbar">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,rgba(45,212,191,0.95),rgba(56,189,248,0.85))]"
                          style={{
                            width: `${Math.max(
                              (entry.score / entry.maxScore) * 100,
                              8,
                            )}%`,
                          }}
                        />
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {entry.criteria.map((criterion) => (
                          <Badge
                            key={criterion.label}
                            variant="outline"
                            className={cn(
                              "text-[9px] uppercase tracking-[0.12em]",
                              criterion.met
                                ? "border-emerald-500/30 bg-emerald-500/[0.08] text-emerald-100"
                                : "border-outline-subtle bg-background/40 text-muted-foreground",
                            )}
                          >
                            {criterion.label}
                          </Badge>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Metadata coverage"
              subtitle="How many jobs have each key observability field populated on their latest known run."
              icon={CheckCheck}
            >
              <div className="space-y-3">
                {metadataCoverage.map((entry, index) => (
                  <div key={entry.label} className="space-y-1.5">
                    <div className="flex items-center gap-2 text-sm">
                      <span className="min-w-0 flex-1 truncate font-medium text-foreground">
                        {entry.label}
                      </span>
                      <span className="text-[11px] text-muted-foreground">
                        {entry.count}/{entry.total}
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-surface-toolbar">
                      <div
                        className={cn(
                          "h-full rounded-full bg-[linear-gradient(90deg,var(--tw-gradient-stops))]",
                          index % 4 === 0
                            ? "from-sky-400 to-cyan-300"
                            : index % 4 === 1
                              ? "from-emerald-400 to-teal-300"
                              : index % 4 === 2
                                ? "from-amber-400 to-orange-300"
                                : "from-violet-400 to-fuchsia-300",
                        )}
                        style={{ width: `${Math.max(entry.ratio * 100, 8)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
            <Panel
              title="Recent run ledger"
              subtitle="Fleet-wide latest runs with provenance, determinism, and output posture at a glance."
              icon={Activity}
            >
              {filteredRuns.length === 0 ? (
                <EmptyPanel
                  title="No runs in scope"
                  message="Adjust the active filters to inspect recent run provenance."
                />
              ) : (
                <div className="space-y-2">
                  {filteredRuns.slice(0, 8).map((run) => (
                    <RunLedgerRow
                      key={run.id}
                      run={run}
                      onSelect={() => {
                        local.jobId = run.jobId;
                        local.selectedRunId = run.id;
                      }}
                    />
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Telemetry confidence"
              subtitle="Jobs with the weakest latest-run metadata completeness across key observability fields."
              icon={Gauge}
            >
              {telemetryConfidence.length === 0 ? (
                <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
                  No jobs are in scope for telemetry-confidence scoring.
                </div>
              ) : (
                <div className="space-y-3">
                  {telemetryConfidence.map((entry) => (
                    <button
                      key={entry.job.id}
                      type="button"
                      onClick={() => {
                        local.jobId = entry.job.id;
                        local.selectedRunId = "";
                      }}
                      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-3 text-left transition-colors hover:bg-background/45"
                    >
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                          {entry.job.name}
                        </span>
                        <span className="text-[11px] text-muted-foreground">
                          {entry.score}/{entry.maxScore}
                        </span>
                      </div>
                      <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-toolbar">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,rgba(251,191,36,0.95),rgba(249,115,22,0.85))]"
                          style={{
                            width: `${Math.max(
                              (entry.score / entry.maxScore) * 100,
                              8,
                            )}%`,
                          }}
                        />
                      </div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {entry.missing.map((field) => (
                          <Badge
                            key={field}
                            variant="outline"
                            className="border-amber-500/30 bg-amber-500/[0.08] text-[9px] uppercase tracking-[0.12em] text-amber-100"
                          >
                            {field}
                          </Badge>
                        ))}
                        {entry.missing.length === 0 ? (
                          <Badge
                            variant="outline"
                            className="border-emerald-500/30 bg-emerald-500/[0.08] text-[9px] uppercase tracking-[0.12em] text-emerald-100"
                          >
                            Fully covered
                          </Badge>
                        ) : null}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
            <Panel
              title="Job fleet"
              subtitle="Per-job operating state, last run, schedule, and output posture."
              icon={Gauge}
            >
              <div className="space-y-3">
                {jobCards.length === 0 ? (
                  <EmptyPanel
                    title="No jobs defined"
                    message="Create a job to start collecting per-job health and output summaries."
                  />
                ) : (
                  jobCards.map((card) => (
                    <JobHealthCard
                      key={card.job.id}
                      card={card}
                      active={localSnap.jobId === card.job.id}
                      onFocus={() => {
                        local.jobId = card.job.id;
                        local.selectedRunId = "";
                      }}
                      onOpen={() => {
                        actions.setView("jobs");
                        actions.editJob(card.job.id);
                      }}
                    />
                  ))
                )}
              </div>
            </Panel>
            <Panel
              title="Latest outputs"
              subtitle="Newest staged relations and the runs that produced them."
              icon={Database}
            >
              {latestRelation == null && latestRun == null ? (
                <EmptyPanel
                  title="No outputs recorded"
                  message="Successful runs will surface their staged relations and output destinations here."
                />
              ) : (
                <div className="space-y-3">
                  {latestRelation ? (
                    <OutputSummary relation={latestRelation} />
                  ) : null}
                  {filteredRuns
                    .filter((run) => run.metadata?.output)
                    .slice(0, 4)
                    .map((run) => (
                      <RunOutputRow key={run.id} run={run} />
                    ))}
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Job timeline"
              subtitle={
                spotlightJob
                  ? `Recent runs, outputs, and row counts for ${spotlightJob.name}.`
                  : "Select a job filter or focus a job card to inspect recent runs."
              }
              icon={Clock3}
            >
              {!spotlightJob ? (
                <EmptyPanel
                  title="No job selected"
                  message="Choose a job from the filter bar or click Focus on a job card to inspect the recent timeline."
                />
              ) : spotlightRuns.length === 0 ? (
                <EmptyPanel
                  title="No runs in scope"
                  message="The active filters exclude this job's recent runs."
                />
              ) : (
                <div className="space-y-2">
                  {spotlightRuns.map((run) => (
                    <RunTimelineRow
                      key={run.id}
                      run={run}
                      active={spotlightRun?.id === run.id}
                      onSelect={() => {
                        local.selectedRunId = run.id;
                      }}
                    />
                  ))}
                </div>
              )}
            </Panel>
            <Panel
              title="Run spotlight"
              subtitle={
                spotlightRun
                  ? `Selected run ${spotlightRun.id} with inline output and summary detail.`
                  : "Pick a run in the job timeline to inspect it here."
              }
              icon={Database}
            >
              {!spotlightRun ? (
                <EmptyPanel
                  title="No run selected"
                  message="Select a run from the timeline to inspect its summary, outputs, and operational metadata."
                />
              ) : (
                <div className="space-y-4">
                  <RunSpotlight run={spotlightRun} />
                  <div className="grid gap-4 xl:grid-cols-2">
                    <RowTrendChart
                      runs={spotlightRuns}
                      selectedRunId={spotlightRun.id}
                    />
                    <DurationTrendChart
                      runs={spotlightRuns}
                      selectedRunId={spotlightRun.id}
                    />
                  </div>
                  <StepMetricsList steps={spotlightSteps} />
                </div>
              )}
            </Panel>
          </section>

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Panel
              title="Filter summary"
              subtitle="Current dashboard scope so operators can sanity-check what they are looking at."
              icon={Filter}
            >
              <div className="grid gap-2 md:grid-cols-2">
                <MiniStat
                  label="Window"
                  value={
                    localSnap.range === "all"
                      ? "All recorded runs"
                      : localSnap.range
                  }
                />
                <MiniStat
                  label="Outcome"
                  value={
                    localSnap.status === "all"
                      ? "All statuses"
                      : humanizeMode(localSnap.status)
                  }
                />
                <MiniStat
                  label="Job"
                  value={spotlightJob?.name ?? "All jobs"}
                />
                <MiniStat
                  label="Runs in scope"
                  value={formatNumber(filteredRuns.length)}
                />
              </div>
            </Panel>
          </section>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  note,
  accent,
}: {
  icon: typeof Gauge;
  label: string;
  value: string;
  note: string;
  accent: "sky" | "teal" | "amber" | "violet";
}) {
  const tone =
    accent === "sky"
      ? "from-sky-400/18 to-cyan-300/8 text-sky-100 border-sky-400/20"
      : accent === "teal"
        ? "from-teal-400/18 to-emerald-300/8 text-teal-100 border-teal-400/20"
        : accent === "amber"
          ? "from-amber-400/18 to-orange-300/8 text-amber-100 border-amber-400/20"
          : "from-violet-400/18 to-fuchsia-300/8 text-violet-100 border-violet-400/20";

  return (
    <div className="rounded-2xl border border-outline-subtle bg-surface-panel p-4 shadow-[0_8px_30px_rgba(7,11,19,0.18)]">
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "grid size-10 shrink-0 place-items-center rounded-2xl border bg-[linear-gradient(135deg,var(--tw-gradient-stops))]",
            tone,
          )}
        >
          <Icon className="size-4" />
        </div>
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            {label}
          </div>
          <div className="mt-2 text-2xl font-semibold tracking-[-0.04em] text-foreground">
            {value}
          </div>
          <div className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {note}
          </div>
        </div>
      </div>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  icon: Icon,
  children,
}: {
  title: string;
  subtitle: string;
  icon: typeof Activity;
  children: ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-outline-subtle bg-surface-panel shadow-[0_8px_30px_rgba(7,11,19,0.18)]">
      <div className="flex items-start gap-3 border-b border-outline-subtle px-4 py-3">
        <div className="grid size-9 shrink-0 place-items-center rounded-xl bg-surface-toolbar text-muted-foreground">
          <Icon className="size-4" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-semibold text-foreground">{title}</div>
          <div className="text-xs leading-relaxed text-muted-foreground">
            {subtitle}
          </div>
        </div>
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

function FilterGroup({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </div>
      <div className="flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}

function StackedRunChart({ data }: { data: TrendPoint[] }) {
  const max = Math.max(...data.map((point) => point.total), 1);
  return (
    <div className="space-y-3">
      <div className="grid h-52 grid-cols-14 items-end gap-2">
        {data.map((point) => {
          const complete =
            point.complete === 0
              ? 0
              : Math.max((point.complete / max) * 100, 6);
          const failed =
            point.failed === 0 ? 0 : Math.max((point.failed / max) * 100, 6);
          return (
            <div
              key={point.label}
              className="flex h-full flex-col justify-end gap-1"
            >
              <div className="group relative flex h-full flex-col justify-end">
                <div className="absolute bottom-[calc(100%+8px)] left-1/2 z-10 hidden w-28 -translate-x-1/2 rounded-lg border border-outline-subtle bg-background/95 px-2 py-1 text-[10px] text-muted-foreground shadow-lg group-hover:block">
                  <div className="font-medium text-foreground">
                    {point.label}
                  </div>
                  <div>{point.total} total runs</div>
                  <div>{point.complete} complete</div>
                  <div>{point.failed} failed</div>
                </div>
                {failed > 0 ? (
                  <div
                    className="rounded-t-md bg-gradient-to-t from-rose-500/70 to-orange-300/90"
                    style={{ height: `${failed}%` }}
                  />
                ) : null}
                {complete > 0 ? (
                  <div
                    className={cn(
                      "bg-gradient-to-t from-sky-500/70 to-cyan-300/90",
                      failed > 0 ? "" : "rounded-t-md",
                    )}
                    style={{ height: `${complete}%` }}
                  />
                ) : null}
                {point.total === 0 ? (
                  <div className="h-1 rounded-full bg-surface-toolbar" />
                ) : null}
              </div>
              <div className="text-center text-[10px] text-muted-foreground">
                {point.label.slice(5)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-4 text-[11px] text-muted-foreground">
        <LegendSwatch
          className="from-sky-500/70 to-cyan-300/90"
          label="Complete"
        />
        <LegendSwatch
          className="from-rose-500/70 to-orange-300/90"
          label="Failed"
        />
      </div>
    </div>
  );
}

function ModeDistributionChart({
  data,
}: {
  data: { mode: string; count: number; ratio: number }[];
}) {
  if (data.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
        No sync-mode telemetry has been recorded yet.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {data.map((entry, index) => (
        <div key={entry.mode} className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm">
            <span className="min-w-0 flex-1 truncate font-medium text-foreground">
              {humanizeMode(entry.mode)}
            </span>
            <span className="text-[11px] text-muted-foreground">
              {entry.count} jobs
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-toolbar">
            <div
              className={cn(
                "h-full rounded-full bg-[linear-gradient(90deg,var(--tw-gradient-stops))]",
                index % 4 === 0
                  ? "from-sky-400 to-cyan-300"
                  : index % 4 === 1
                    ? "from-emerald-400 to-teal-300"
                    : index % 4 === 2
                      ? "from-amber-400 to-orange-300"
                      : "from-violet-400 to-fuchsia-300",
              )}
              style={{ width: `${Math.max(entry.ratio * 100, 6)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function HourlyPulseChart({ data }: { data: TrendPoint[] }) {
  const max = Math.max(...data.map((point) => point.total), 1);
  return (
    <div className="space-y-3">
      <div className="flex h-28 items-end gap-1.5">
        {data.map((point) => (
          <div
            key={point.label}
            className="group flex min-w-0 flex-1 flex-col justify-end"
          >
            <div className="relative">
              <div className="absolute bottom-[calc(100%+8px)] left-1/2 z-10 hidden w-24 -translate-x-1/2 rounded-lg border border-outline-subtle bg-background/95 px-2 py-1 text-[10px] text-muted-foreground shadow-lg group-hover:block">
                <div className="font-medium text-foreground">{point.label}</div>
                <div>{point.total} runs</div>
              </div>
              <div
                className="rounded-t-md bg-[linear-gradient(180deg,rgba(45,212,191,0.9),rgba(56,189,248,0.45))]"
                style={{
                  height:
                    point.total === 0
                      ? "6px"
                      : `${Math.max((point.total / max) * 100, 10)}px`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>24h ago</span>
        <span>Now</span>
      </div>
    </div>
  );
}

function ScheduleRow({ job }: { job: Job }) {
  const expr = job.definition.schedule?.trim() ?? "";
  const next = nextRuns(expr, new Date(), 3);
  return (
    <div className="rounded-xl border border-outline-subtle bg-surface-toolbar px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
          {job.name}
        </span>
        <Badge
          variant={job.enabled ? "secondary" : "outline"}
          className="text-[9px] uppercase tracking-[0.12em]"
        >
          {job.enabled ? "enabled" : "paused"}
        </Badge>
      </div>
      <div className="mt-1 font-mono text-[11px] text-muted-foreground">
        {expr}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {next.length === 0 ? (
          <span className="text-[11px] text-amber-200">
            Invalid or empty cron
          </span>
        ) : (
          next.map((value) => (
            <span
              key={value.toISOString()}
              className="rounded-full border border-outline-subtle bg-background/50 px-2 py-1 text-[10px] text-muted-foreground"
            >
              {formatUtc(value)}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function FailureRow({ run }: { run: JobRun }) {
  const message =
    String(run.metadata?.diagnostics?.message || "").trim() ||
    String(
      (run.metadata as Record<string, unknown> | undefined)?.error || "",
    ).trim() ||
    "Open the run diagnostics for step-level detail.";
  return (
    <button
      type="button"
      onClick={() => {
        actions.setView("jobs");
        actions.editJob(run.jobId);
        void actions.openRun(run.id);
      }}
      className="w-full rounded-xl border border-rose-500/20 bg-rose-500/[0.08] px-3 py-2 text-left transition-colors hover:bg-rose-500/[0.12]"
    >
      <div className="flex items-center gap-2">
        <AlertTriangle className="size-4 shrink-0 text-rose-200" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
          {run.jobId}
        </span>
        <span className="font-mono text-[10px] text-muted-foreground">
          {shortWhen(run.startedAt)}
        </span>
      </div>
      <div className="mt-1 text-[12px] leading-relaxed text-rose-100/90">
        {message}
      </div>
    </button>
  );
}

function JobHealthCard({
  card,
  active,
  onFocus,
  onOpen,
}: {
  card: ReturnType<typeof buildJobCards>[number];
  active: boolean;
  onFocus: () => void;
  onOpen: () => void;
}) {
  const lastRun = card.lastRun;
  const hasProblems =
    card.failedRuns > 0 && (!lastRun || lastRun.status === "failed");
  return (
    <div
      className={cn(
        "rounded-2xl border border-outline-subtle bg-[linear-gradient(180deg,rgba(18,24,35,0.78),rgba(18,24,35,0.42))] p-4",
        active && "border-sky-400/30 shadow-[0_0_0_1px_rgba(56,189,248,0.18)]",
      )}
    >
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="min-w-0 truncate text-base font-semibold tracking-[-0.02em] text-foreground">
              {card.job.name}
            </div>
            <Badge
              variant={card.job.enabled ? "secondary" : "outline"}
              className="text-[9px] uppercase tracking-[0.12em]"
            >
              {card.job.enabled ? "enabled" : "disabled"}
            </Badge>
            {hasProblems ? (
              <Badge
                variant="outline"
                className="border-rose-500/30 bg-rose-500/[0.08] text-[9px] uppercase tracking-[0.12em] text-rose-100"
              >
                degraded
              </Badge>
            ) : (
              <Badge
                variant="outline"
                className="border-emerald-500/30 bg-emerald-500/[0.08] text-[9px] uppercase tracking-[0.12em] text-emerald-100"
              >
                healthy
              </Badge>
            )}
          </div>
          <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-muted-foreground">
            <span>Mode: {humanizeMode(card.syncMode)}</span>
            <span>Runs: {card.runs.length}</span>
            <span>Success: {card.successRate}%</span>
            <span>Avg rows: {formatNumber(card.avgRows)}</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={active ? "secondary" : "ghost"}
            onClick={onFocus}
          >
            Focus
          </Button>
          <Button size="sm" variant="ghost" onClick={onOpen}>
            Open job <MoveRight className="size-4" />
          </Button>
        </div>
      </div>
      <div className="mt-4 grid gap-2 md:grid-cols-4">
        <MiniStat
          label="Last run"
          value={lastRun ? shortWhen(lastRun.startedAt) : "Never"}
          tone={lastRun?.status === "failed" ? "danger" : "default"}
        />
        <MiniStat
          label="Schedule"
          value={
            card.nextRun ?? (hasSchedule(card.job) ? "Pending" : "On demand")
          }
        />
        <MiniStat label="Output" value={card.outputLabel} />
        <MiniStat label="Staged" value={card.relationLabel} />
      </div>
    </div>
  );
}

function RunTimelineRow({
  run,
  active,
  onSelect,
}: {
  run: JobRun;
  active: boolean;
  onSelect: () => void;
}) {
  const summary = run.metadata?.summary;
  const output = run.metadata?.output;
  return (
    <button
      type="button"
      onClick={() => {
        onSelect();
      }}
      className={cn(
        "w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-2 text-left transition-colors hover:bg-background/45",
        active && "border-sky-400/30 bg-sky-400/[0.08]",
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <RunStatusPill status={run.status} />
        <span className="font-mono text-[11px] text-foreground">{run.id}</span>
        <span className="font-mono text-[11px] text-muted-foreground">
          {shortWhen(run.startedAt)}
        </span>
        <span className="ml-auto text-[11px] text-muted-foreground">
          {humanizeMode(run.metadata?.syncMode || "ad_hoc")}
        </span>
      </div>
      <div className="mt-2 grid gap-2 md:grid-cols-3">
        <MiniStat
          label="Rows"
          value={formatNumber(summary?.totalRowCount ?? 0)}
        />
        <MiniStat
          label="Steps"
          value={`${summary?.successfulSteps ?? 0}/${summary?.stepCount ?? 0} ok`}
        />
        <MiniStat
          label="Output"
          value={
            output?.targetPath ||
            output?.stagedRelationId ||
            output?.targetConnectorId ||
            "Recorded"
          }
        />
      </div>
    </button>
  );
}

function RunSpotlight({ run }: { run: JobRun }) {
  const summary = run.metadata?.summary;
  const output = run.metadata?.output;
  const elapsedMs = elapsedBetween(run.startedAt, run.finishedAt);
  return (
    <div className="rounded-2xl border border-outline-subtle bg-surface-toolbar p-4">
      <div className="flex flex-wrap items-center gap-2">
        <RunStatusPill status={run.status} />
        <span className="font-mono text-[11px] text-muted-foreground">
          {run.id}
        </span>
        <span className="text-[11px] text-muted-foreground">
          {shortWhen(run.startedAt)}
        </span>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto"
          onClick={() => {
            actions.setView("jobs");
            actions.editJob(run.jobId);
            void actions.openRun(run.id);
          }}
        >
          Open diagnostics <MoveRight className="size-4" />
        </Button>
      </div>
      <div className="mt-4 grid gap-2 md:grid-cols-2">
        <MiniStat
          label="Trigger"
          value={humanizeMode(run.trigger || "manual")}
        />
        <MiniStat
          label="Sync mode"
          value={humanizeMode(run.metadata?.syncMode || "ad_hoc")}
        />
        <MiniStat
          label="Rows"
          value={formatNumber(summary?.totalRowCount ?? 0)}
        />
        <MiniStat
          label="Steps"
          value={`${summary?.successfulSteps ?? 0}/${summary?.stepCount ?? 0} ok`}
        />
        <MiniStat
          label="Elapsed"
          value={
            elapsedMs == null ? "In progress" : formatDurationMs(elapsedMs)
          }
        />
        <MiniStat
          label="Output path"
          value={output?.targetPath || output?.sinkUri || "Not recorded"}
        />
        <MiniStat
          label="Staged relation"
          value={output?.stagedRelationId || "Not recorded"}
        />
      </div>
    </div>
  );
}

function RowTrendChart({
  runs,
  selectedRunId,
}: {
  runs: readonly JobRun[];
  selectedRunId: string;
}) {
  const max = Math.max(
    ...runs.map((run) => run.metadata?.summary?.totalRowCount ?? 0),
    1,
  );
  return (
    <div className="space-y-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        Row-count trend
      </div>
      <div className="flex h-28 items-end gap-2">
        {runs.map((run) => {
          const rows = run.metadata?.summary?.totalRowCount ?? 0;
          const selected = run.id === selectedRunId;
          return (
            <div
              key={run.id}
              className="group flex min-w-0 flex-1 flex-col justify-end"
            >
              <div className="relative">
                <div className="absolute bottom-[calc(100%+8px)] left-1/2 z-10 hidden w-28 -translate-x-1/2 rounded-lg border border-outline-subtle bg-background/95 px-2 py-1 text-[10px] text-muted-foreground shadow-lg group-hover:block">
                  <div className="font-medium text-foreground">{run.id}</div>
                  <div>{formatNumber(rows)} rows</div>
                </div>
                <div
                  className={cn(
                    "rounded-t-md bg-[linear-gradient(180deg,rgba(45,212,191,0.9),rgba(56,189,248,0.45))]",
                    selected &&
                      "ring-2 ring-sky-400/40 ring-offset-1 ring-offset-background",
                  )}
                  style={{
                    height:
                      rows === 0
                        ? "6px"
                        : `${Math.max((rows / max) * 100, 12)}px`,
                  }}
                />
              </div>
              <div className="mt-1 truncate text-center text-[10px] text-muted-foreground">
                {shortWhen(run.startedAt)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DurationTrendChart({
  runs,
  selectedRunId,
}: {
  runs: readonly JobRun[];
  selectedRunId: string;
}) {
  const durations = runs.map(
    (run) => elapsedBetween(run.startedAt, run.finishedAt) ?? 0,
  );
  const max = Math.max(...durations, 1);
  return (
    <div className="space-y-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        Duration trend
      </div>
      <div className="flex h-28 items-end gap-2">
        {runs.map((run) => {
          const durationMs = elapsedBetween(run.startedAt, run.finishedAt) ?? 0;
          const selected = run.id === selectedRunId;
          return (
            <div
              key={run.id}
              className="group flex min-w-0 flex-1 flex-col justify-end"
            >
              <div className="relative">
                <div className="absolute bottom-[calc(100%+8px)] left-1/2 z-10 hidden w-32 -translate-x-1/2 rounded-lg border border-outline-subtle bg-background/95 px-2 py-1 text-[10px] text-muted-foreground shadow-lg group-hover:block">
                  <div className="font-medium text-foreground">{run.id}</div>
                  <div>{formatDurationMs(durationMs)}</div>
                </div>
                <div
                  className={cn(
                    "rounded-t-md bg-[linear-gradient(180deg,rgba(251,191,36,0.95),rgba(249,115,22,0.45))]",
                    selected &&
                      "ring-2 ring-amber-400/40 ring-offset-1 ring-offset-background",
                  )}
                  style={{
                    height:
                      durationMs === 0
                        ? "6px"
                        : `${Math.max((durationMs / max) * 100, 12)}px`,
                  }}
                />
              </div>
              <div className="mt-1 truncate text-center text-[10px] text-muted-foreground">
                {shortWhen(run.startedAt)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StepMetricsList({ steps }: { steps: readonly JobRunStep[] }) {
  if (steps.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
        No step metrics loaded yet for this run.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        Step metrics
      </div>
      <div className="space-y-2">
        {steps.map((step) => {
          const metrics = step.metrics ?? {};
          const durationMs =
            typeof metrics.durationMs === "number" ? metrics.durationMs : null;
          return (
            <div
              key={`${step.runId}-${step.stepIdx}`}
              className="rounded-xl border border-outline-subtle bg-background/30 px-3 py-2"
            >
              <div className="flex flex-wrap items-center gap-2">
                <RunStatusPill status={step.status} />
                <span className="font-mono text-[11px] text-foreground">
                  {step.stepIdx + 1}. {step.template || "inline step"}
                </span>
              </div>
              <div className="mt-2 grid gap-2 md:grid-cols-5">
                <MiniStat
                  label="Rows"
                  value={formatNumber(step.rowCount ?? 0)}
                />
                <MiniStat
                  label="Duration"
                  value={
                    durationMs == null
                      ? "Not recorded"
                      : formatDurationMs(durationMs)
                  }
                />
                <MiniStat
                  label="Columns"
                  value={String(
                    (metrics.columnCount as number | undefined) ?? 0,
                  )}
                />
                <MiniStat
                  label="Error class"
                  value={humanizeMode(String(metrics.errorClass || "none"))}
                />
                <MiniStat
                  label="Has log"
                  value={(metrics.hasLog as boolean | undefined) ? "Yes" : "No"}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RunStatusPill({ status }: { status: string }) {
  const ok = status === "complete";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em]",
        ok
          ? "border-emerald-500/30 bg-emerald-500/[0.08] text-emerald-100"
          : status === "failed"
            ? "border-rose-500/30 bg-rose-500/[0.08] text-rose-100"
            : "border-outline-subtle bg-surface-toolbar text-muted-foreground",
      )}
    >
      {status}
    </span>
  );
}

function MiniStat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "danger";
}) {
  return (
    <div className="rounded-xl border border-outline-subtle bg-background/30 px-3 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          "mt-1 text-sm font-medium",
          tone === "danger" ? "text-rose-100" : "text-foreground",
        )}
      >
        {value}
      </div>
    </div>
  );
}

function OutputSummary({ relation }: { relation: Readonly<StagedRelation> }) {
  return (
    <div className="rounded-2xl border border-outline-subtle bg-surface-toolbar p-4">
      <div className="flex items-center gap-2">
        <Badge
          variant="secondary"
          className="text-[9px] uppercase tracking-[0.12em]"
        >
          latest relation
        </Badge>
        <span className="font-mono text-[11px] text-muted-foreground">
          {relation.relationId}
        </span>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        <MiniStat label="Rows" value={formatNumber(relation.rowCount ?? 0)} />
        <MiniStat label="Freshness" value={relation.freshness ?? "unknown"} />
        <MiniStat
          label="Created"
          value={shortWhen(new Date(relation.createdMs).toISOString())}
        />
        <MiniStat
          label="Flow run"
          value={relation.producer.flowRunId ?? "Unknown"}
        />
      </div>
    </div>
  );
}

function RuntimeSpread({
  stats,
}: {
  stats: ReturnType<typeof buildRuntimeStats>;
}) {
  const points = [
    { label: "P50", value: stats.p50 },
    { label: "P75", value: stats.p75 },
    { label: "P95", value: stats.p95 },
    { label: "Max", value: stats.max },
  ];
  const max = Math.max(...points.map((point) => point.value), 1);
  return (
    <div className="space-y-3">
      <div className="flex h-28 items-end gap-2">
        {points.map((point) => (
          <div
            key={point.label}
            className="group flex min-w-0 flex-1 flex-col justify-end"
          >
            <div className="relative">
              <div className="absolute bottom-[calc(100%+8px)] left-1/2 z-10 hidden w-24 -translate-x-1/2 rounded-lg border border-outline-subtle bg-background/95 px-2 py-1 text-[10px] text-muted-foreground shadow-lg group-hover:block">
                <div className="font-medium text-foreground">{point.label}</div>
                <div>{formatDurationMs(point.value)}</div>
              </div>
              <div
                className="rounded-t-md bg-[linear-gradient(180deg,rgba(248,113,113,0.9),rgba(251,191,36,0.45))]"
                style={{
                  height: `${Math.max((point.value / max) * 100, 12)}px`,
                }}
              />
            </div>
            <div className="mt-1 truncate text-center text-[10px] text-muted-foreground">
              {point.label}
            </div>
          </div>
        ))}
      </div>
      <div className="text-[11px] text-muted-foreground">
        {stats.count} completed runs with elapsed telemetry
      </div>
    </div>
  );
}

function ModeSummaryList({
  title,
  emptyMessage,
  data,
}: {
  title: string;
  emptyMessage: string;
  data: { label: string; count: number; ratio: number }[];
}) {
  if (data.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-outline-subtle px-4 py-5 text-sm text-muted-foreground">
        {emptyMessage}
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        {title}
      </div>
      {data.map((entry, index) => (
        <div key={entry.label} className="space-y-1.5">
          <div className="flex items-center gap-2 text-sm">
            <span className="min-w-0 flex-1 truncate font-medium text-foreground">
              {entry.label}
            </span>
            <span className="text-[11px] text-muted-foreground">
              {entry.count} jobs
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-surface-toolbar">
            <div
              className={cn(
                "h-full rounded-full bg-[linear-gradient(90deg,var(--tw-gradient-stops))]",
                index % 4 === 0
                  ? "from-sky-400 to-cyan-300"
                  : index % 4 === 1
                    ? "from-emerald-400 to-teal-300"
                    : index % 4 === 2
                      ? "from-amber-400 to-orange-300"
                      : "from-violet-400 to-fuchsia-300",
              )}
              style={{ width: `${Math.max(entry.ratio * 100, 8)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function ExecutionMix({ runs }: { runs: readonly JobRun[] }) {
  const scheduled = runs.filter((run) => run.trigger === "schedule").length;
  const manual = runs.filter((run) => run.trigger !== "schedule").length;
  const total = Math.max(runs.length, 1);
  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-outline-subtle bg-surface-toolbar p-4">
        <div className="flex h-3 overflow-hidden rounded-full bg-background/60">
          <div
            className="h-full bg-[linear-gradient(90deg,rgba(45,212,191,0.95),rgba(56,189,248,0.85))]"
            style={{ width: `${(scheduled / total) * 100}%` }}
          />
          <div
            className="h-full bg-[linear-gradient(90deg,rgba(251,191,36,0.95),rgba(249,115,22,0.85))]"
            style={{ width: `${(manual / total) * 100}%` }}
          />
        </div>
        <div className="mt-4 grid gap-2 md:grid-cols-2">
          <MiniStat label="Scheduled" value={`${scheduled} runs`} />
          <MiniStat label="Manual / API" value={`${manual} runs`} />
        </div>
      </div>
    </div>
  );
}

function RunOutputRow({ run }: { run: JobRun }) {
  const output = run.metadata?.output;
  const label =
    output?.targetPath ||
    output?.sinkUri ||
    output?.stagedRelationId ||
    output?.targetConnectorId ||
    "Output recorded";
  return (
    <button
      type="button"
      onClick={() => {
        actions.setView("jobs");
        actions.editJob(run.jobId);
        void actions.openRun(run.id);
      }}
      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-2 text-left transition-colors hover:bg-background/45"
    >
      <div className="flex items-center gap-2">
        <CheckCircle2 className="size-4 shrink-0 text-emerald-200" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
          {run.jobId}
        </span>
        <span className="text-[11px] text-muted-foreground">
          {shortWhen(run.startedAt)}
        </span>
      </div>
      <div className="mt-1 truncate font-mono text-[11px] text-muted-foreground">
        {label}
      </div>
    </button>
  );
}

function RunLedgerRow({
  run,
  onSelect,
}: {
  run: JobRun;
  onSelect: () => void;
}) {
  const output = run.metadata?.output;
  return (
    <button
      type="button"
      onClick={onSelect}
      className="w-full rounded-xl border border-outline-subtle bg-background/30 px-3 py-3 text-left transition-colors hover:bg-background/45"
    >
      <div className="flex flex-wrap items-center gap-2">
        <RunStatusPill status={run.status} />
        <span className="font-mono text-[11px] text-foreground">{run.id}</span>
        <span className="text-[11px] text-muted-foreground">
          {shortWhen(run.startedAt)}
        </span>
        <span className="ml-auto text-[11px] text-muted-foreground">
          {run.jobId}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <LedgerBadge
          label={humanizeMode(run.metadata?.executionMode || "unknown")}
        />
        <LedgerBadge
          label={humanizeMode(run.metadata?.sourceMode || "unknown")}
        />
        <LedgerBadge
          label={humanizeMode(run.metadata?.syncMode || "unknown")}
        />
        <LedgerBadge
          label={humanizeMode(run.metadata?.determinism || "unknown")}
        />
        <LedgerBadge
          label={
            output?.targetPath
              ? "Path output"
              : output?.targetConnectorId
                ? "Connector output"
                : output?.sinkUri
                  ? "Sink output"
                  : "No output"
          }
        />
      </div>
    </button>
  );
}

function LedgerBadge({ label }: { label: string }) {
  return (
    <span className="rounded-full border border-outline-subtle bg-background/40 px-2 py-1 text-[10px] text-muted-foreground">
      {label}
    </span>
  );
}

function LegendSwatch({
  className,
  label,
}: {
  className: string;
  label: string;
}) {
  return (
    <span className="flex items-center gap-2">
      <span
        className={cn(
          "inline-block h-2.5 w-6 rounded-full bg-[linear-gradient(90deg,var(--tw-gradient-stops))]",
          className,
        )}
      />
      <span>{label}</span>
    </span>
  );
}

function buildDailyTrend(runs: readonly JobRun[], days: number): TrendPoint[] {
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);
  const series: TrendPoint[] = [];
  for (let i = days - 1; i >= 0; i -= 1) {
    const day = new Date(today);
    day.setUTCDate(today.getUTCDate() - i);
    const label = day.toISOString().slice(0, 10);
    series.push({ label, total: 0, complete: 0, failed: 0 });
  }
  const index = new Map(series.map((point) => [point.label, point]));
  for (const run of runs) {
    const ts = parseTimestamp(run.startedAt);
    if (ts == null) continue;
    const day = new Date(ts).toISOString().slice(0, 10);
    const point = index.get(day);
    if (!point) continue;
    point.total += 1;
    if (run.status === "complete") point.complete += 1;
    if (run.status === "failed") point.failed += 1;
  }
  return series;
}

function buildHourlyTrend(runs: readonly JobRun[]): TrendPoint[] {
  const points: TrendPoint[] = [];
  const now = new Date();
  now.setUTCMinutes(0, 0, 0);
  for (let i = 23; i >= 0; i -= 1) {
    const hour = new Date(now);
    hour.setUTCHours(now.getUTCHours() - i);
    points.push({
      label: `${String(hour.getUTCHours()).padStart(2, "0")}:00`,
      total: 0,
      complete: 0,
      failed: 0,
    });
  }
  const index = new Map(points.map((point) => [point.label, point]));
  for (const run of runs) {
    const ts = parseTimestamp(run.startedAt);
    if (ts == null) continue;
    const date = new Date(ts);
    const key = `${String(date.getUTCHours()).padStart(2, "0")}:00`;
    const point = index.get(key);
    if (!point) continue;
    point.total += 1;
    if (run.status === "complete") point.complete += 1;
    if (run.status === "failed") point.failed += 1;
  }
  return points;
}

function buildSyncModeDistribution(
  jobs: readonly Job[],
  runs: readonly JobRun[],
) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const counts = new Map<string, number>();
  for (const job of jobs) {
    const mode =
      latestByJob.get(job.id)?.metadata?.syncMode || inferJobSyncMode(job);
    counts.set(mode, (counts.get(mode) ?? 0) + 1);
  }
  const total = jobs.length || 1;
  return [...counts.entries()]
    .map(([mode, count]) => ({ mode, count, ratio: count / total }))
    .sort((a, b) => b.count - a.count);
}

function buildJobCards(
  jobs: readonly Job[],
  runs: readonly JobRun[],
  staged: readonly StagedRelation[],
  now: Date,
) {
  const byJob = new Map<string, JobRun[]>();
  for (const run of runs) {
    const list = byJob.get(run.jobId) ?? [];
    list.push(run);
    byJob.set(run.jobId, list);
  }
  const relationByRun = new Map<string, StagedRelation>();
  for (const relation of staged) {
    const runId = relation.producer.flowRunId;
    if (runId && !relationByRun.has(runId)) relationByRun.set(runId, relation);
  }
  return jobs.map((job) => {
    const jobRuns = byJob.get(job.id) ?? [];
    const lastRun = jobRuns[0] ?? null;
    const rows = jobRuns
      .map((run) => run.metadata?.summary?.totalRowCount ?? 0)
      .filter((value) => value > 0);
    const successRate = jobRuns.length
      ? Math.round(
          (jobRuns.filter((run) => run.status === "complete").length /
            jobRuns.length) *
            100,
        )
      : 0;
    const nextRun = nextRunLabel(job, now);
    const latestRelation =
      (lastRun && relationByRun.get(lastRun.id)) ||
      staged.find((relation) => relation.name === job.name) ||
      null;
    const outputLabel =
      lastRun?.metadata?.output?.targetPath ||
      lastRun?.metadata?.output?.sinkUri ||
      lastRun?.metadata?.output?.targetConnectorId ||
      (job.definition.targetConnectorId ?? "None");
    return {
      job,
      runs: jobRuns,
      lastRun,
      failedRuns: jobRuns.filter((run) => run.status === "failed").length,
      successRate,
      avgRows:
        rows.length > 0
          ? Math.round(
              rows.reduce((sum, value) => sum + value, 0) / rows.length,
            )
          : 0,
      syncMode: lastRun?.metadata?.syncMode || inferJobSyncMode(job),
      nextRun,
      outputLabel,
      relationLabel: latestRelation
        ? latestRelation.relationId
        : lastRun?.metadata?.output?.stagedRelationId || "None",
    };
  });
}

function buildNoisyJobs(jobs: readonly Job[], runs: readonly JobRun[]) {
  return jobs
    .map((job) => {
      const jobRuns = runs.filter((run) => run.jobId === job.id);
      const failures = jobRuns.filter((run) => run.status === "failed").length;
      return {
        job,
        failures,
        total: jobRuns.length,
        ratio: jobRuns.length === 0 ? 0 : failures / jobRuns.length,
      };
    })
    .filter((entry) => entry.failures > 0)
    .sort((a, b) => b.failures - a.failures || b.ratio - a.ratio)
    .slice(0, 6);
}

function buildSlowRuns(jobs: readonly Job[], runs: readonly JobRun[]) {
  const nameById = new Map(jobs.map((job) => [job.id, job.name]));
  return runs
    .map((run) => ({
      run,
      jobName: nameById.get(run.jobId) ?? run.jobId,
      rows: run.metadata?.summary?.totalRowCount ?? 0,
      elapsedMs: elapsedBetween(run.startedAt, run.finishedAt),
    }))
    .filter(
      (entry): entry is typeof entry & { elapsedMs: number } =>
        entry.elapsedMs != null,
    )
    .sort((a, b) => b.elapsedMs - a.elapsedMs)
    .slice(0, 12);
}

function buildJobPerformance(jobs: readonly Job[], runs: readonly JobRun[]) {
  return jobs
    .map((job) => {
      const completedRuns = runs.filter(
        (run) =>
          run.jobId === job.id &&
          run.status === "complete" &&
          elapsedBetween(run.startedAt, run.finishedAt) != null,
      );
      if (completedRuns.length === 0) {
        return null;
      }
      const elapsedValues = completedRuns
        .map((run) => elapsedBetween(run.startedAt, run.finishedAt) ?? 0)
        .filter((value) => value > 0);
      if (elapsedValues.length === 0) {
        return null;
      }
      const totalRows = completedRuns.reduce(
        (sum, run) => sum + (run.metadata?.summary?.totalRowCount ?? 0),
        0,
      );
      const totalElapsedMs = elapsedValues.reduce(
        (sum, value) => sum + value,
        0,
      );
      return {
        job,
        completedRuns: completedRuns.length,
        totalRows,
        avgElapsedMs: Math.round(totalElapsedMs / elapsedValues.length),
        avgRowsPerSecond:
          totalElapsedMs === 0 ? 0 : totalRows / (totalElapsedMs / 1000),
      };
    })
    .filter(
      (
        entry,
      ): entry is {
        job: Job;
        completedRuns: number;
        totalRows: number;
        avgElapsedMs: number;
        avgRowsPerSecond: number;
      } => entry != null,
    )
    .sort(
      (a, b) =>
        b.avgRowsPerSecond - a.avgRowsPerSecond ||
        a.avgElapsedMs - b.avgElapsedMs,
    );
}

function buildScheduleWatch(
  jobs: readonly Job[],
  runs: readonly JobRun[],
  staged: readonly StagedRelation[],
  now: Date,
) {
  const runsByJob = new Map<string, JobRun[]>();
  for (const run of runs) {
    const list = runsByJob.get(run.jobId) ?? [];
    list.push(run);
    runsByJob.set(run.jobId, list);
  }
  const relationByName = new Map<string, StagedRelation>();
  for (const relation of staged) {
    if (relation.name && !relationByName.has(relation.name)) {
      relationByName.set(relation.name, relation);
    }
  }
  return jobs
    .filter((job) => job.enabled && hasSchedule(job))
    .map((job) => {
      const jobRuns = runsByJob.get(job.id) ?? [];
      const lastRun = jobRuns[0] ?? null;
      const latestSuccess =
        jobRuns.find((run) => run.status === "complete") ?? null;
      const relation = relationByName.get(job.name) ?? null;
      const intervalMs = inferScheduleInterval(job, now);
      const lastRunMs = parseTimestamp(lastRun?.startedAt);
      const lastSuccessMs = parseTimestamp(latestSuccess?.startedAt);
      const overdue =
        intervalMs != null &&
        lastRunMs != null &&
        now.getTime() - lastRunMs > intervalMs * 1.5;
      let status: "healthy" | "late" | "failed" | "stale" | "no_run" =
        "healthy";
      let reason = "Within expected schedule window";
      if (!lastRun) {
        status = "no_run";
        reason = "No recorded run yet";
      } else if (lastRun.status === "failed") {
        status = "failed";
        reason = "Latest run failed";
      } else if (overdue) {
        status = "late";
        reason = "No run within expected schedule interval";
      } else if (
        relation == null ||
        relation.freshness !== "fresh" ||
        (intervalMs != null &&
          now.getTime() - relation.createdMs > intervalMs * 2)
      ) {
        status = "stale";
        reason =
          relation == null
            ? "No staged output recorded"
            : "Latest staged output is aging or stale";
      }
      return {
        job,
        status,
        reason,
        nextRunLabel: nextRunLabel(job, now) ?? "On demand",
        intervalLabel:
          intervalMs == null ? "Unknown" : formatDurationMs(intervalMs),
        lastRunLabel: lastRun ? shortWhen(lastRun.startedAt) : "Never",
        freshnessLabel: relation
          ? `${relation.freshness ?? "unknown"} · ${formatAgeMs(
              now.getTime() - relation.createdMs,
            )}`
          : "No relation",
        lastSuccessMs,
      };
    })
    .sort((a, b) => riskWeight(b.status) - riskWeight(a.status))
    .sort((a, b) => {
      if (riskWeight(b.status) !== riskWeight(a.status)) return 0;
      return (a.lastSuccessMs ?? 0) - (b.lastSuccessMs ?? 0);
    });
}

function buildStaleRelations(relations: readonly StagedRelation[], now: Date) {
  return relations
    .map((relation) => ({
      ...relation,
      ageMs: Math.max(0, now.getTime() - relation.createdMs),
    }))
    .filter(
      (relation) =>
        relation.freshness !== "fresh" || relation.ageMs > 6 * 60 * 60 * 1000,
    )
    .sort((a, b) => {
      if (a.freshness === b.freshness) return b.ageMs - a.ageMs;
      if (a.freshness === "fresh") return 1;
      if (b.freshness === "fresh") return -1;
      return b.ageMs - a.ageMs;
    });
}

function buildRuntimeStats(runs: readonly JobRun[]) {
  const durations = runs
    .filter((run) => run.status === "complete")
    .map((run) => elapsedBetween(run.startedAt, run.finishedAt))
    .filter((value): value is number => value != null)
    .sort((a, b) => a - b);
  if (durations.length === 0) {
    return { count: 0, min: 0, p50: 0, p75: 0, p95: 0, max: 0 };
  }
  return {
    count: durations.length,
    min: durations[0],
    p50: percentile(durations, 0.5),
    p75: percentile(durations, 0.75),
    p95: percentile(durations, 0.95),
    max: durations[durations.length - 1],
  };
}

function buildFailureSignatures(runs: readonly JobRun[]) {
  const failures = runs.filter((run) => run.status === "failed");
  const counts = new Map<
    string,
    { count: number; label: string; latestAt: string; jobs: Set<string> }
  >();
  for (const run of failures) {
    const message = failureMessage(run);
    const signature = normalizeFailureMessage(message);
    const existing = counts.get(signature);
    if (existing) {
      existing.count += 1;
      existing.jobs.add(run.jobId);
      const runStartedAt = parseTimestamp(run.startedAt) ?? 0;
      const latestSeenAt = parseTimestamp(existing.latestAt) ?? 0;
      if (runStartedAt > latestSeenAt) {
        existing.latestAt = run.startedAt;
      }
      continue;
    }
    counts.set(signature, {
      count: 1,
      label: message,
      latestAt: run.startedAt,
      jobs: new Set([run.jobId]),
    });
  }
  const total = Math.max(failures.length, 1);
  return [...counts.entries()]
    .map(([signature, entry]) => ({
      signature,
      label: truncateLabel(entry.label, 68),
      count: entry.count,
      ratio: entry.count / total,
      latestAt: entry.latestAt,
      jobCount: entry.jobs.size,
    }))
    .sort((a, b) => b.count - a.count || b.jobCount - a.jobCount);
}

function buildVolumeAnomalies(jobs: readonly Job[], runs: readonly JobRun[]) {
  return jobs
    .map((job) => {
      const completedRuns = runs.filter(
        (run) => run.jobId === job.id && run.status === "complete",
      );
      if (completedRuns.length < 3) return null;
      const latestRun = completedRuns[0];
      const latestRows = latestRun.metadata?.summary?.totalRowCount ?? 0;
      const history = completedRuns
        .slice(1, 5)
        .map((run) => run.metadata?.summary?.totalRowCount ?? 0)
        .filter((value) => value > 0);
      if (history.length < 2 || latestRows <= 0) return null;
      const baselineRows = Math.round(
        history.reduce((sum, value) => sum + value, 0) / history.length,
      );
      if (baselineRows <= 0) return null;
      const deltaPercent = Math.round(
        ((latestRows - baselineRows) / baselineRows) * 100,
      );
      if (Math.abs(deltaPercent) < 35) return null;
      return {
        job,
        latestRun,
        latestRows,
        baselineRows,
        deltaPercent,
        direction: latestRows >= baselineRows ? "up" : "down",
      };
    })
    .filter(
      (
        entry,
      ): entry is {
        job: Job;
        latestRun: JobRun;
        latestRows: number;
        baselineRows: number;
        deltaPercent: number;
        direction: "up" | "down";
      } => entry != null,
    )
    .sort((a, b) => Math.abs(b.deltaPercent) - Math.abs(a.deltaPercent));
}

function buildFailureStreaks(jobs: readonly Job[], runs: readonly JobRun[]) {
  return jobs
    .map((job) => {
      const jobRuns = runs.filter((run) => run.jobId === job.id);
      if (jobRuns.length === 0) return null;
      let streak = 0;
      for (const run of jobRuns) {
        if (run.status !== "failed") break;
        streak += 1;
      }
      if (streak === 0) return null;
      return {
        job,
        streak,
        latestRun: jobRuns[0],
        totalRuns: jobRuns.length,
      };
    })
    .filter(
      (
        entry,
      ): entry is {
        job: Job;
        streak: number;
        latestRun: JobRun;
        totalRuns: number;
      } => entry != null,
    )
    .sort(
      (a, b) =>
        b.streak - a.streak ||
        (parseTimestamp(b.latestRun.startedAt) ?? 0) -
          (parseTimestamp(a.latestRun.startedAt) ?? 0),
    );
}

function buildDeterminismMix(jobs: readonly Job[], runs: readonly JobRun[]) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const counts = new Map<string, number>();
  for (const job of jobs) {
    const determinism =
      latestByJob.get(job.id)?.metadata?.determinism ?? "unknown";
    counts.set(determinism, (counts.get(determinism) ?? 0) + 1);
  }
  const total = jobs.length || 1;
  return [...counts.entries()]
    .map(([label, count]) => ({
      label: humanizeMode(label),
      count,
      ratio: count / total,
    }))
    .sort((a, b) => b.count - a.count);
}

function buildObservabilityGaps(
  jobs: readonly Job[],
  runs: readonly JobRun[],
  staged: readonly StagedRelation[],
) {
  const runsByJob = new Map<string, JobRun[]>();
  for (const run of runs) {
    const list = runsByJob.get(run.jobId) ?? [];
    list.push(run);
    runsByJob.set(run.jobId, list);
  }
  const relationNames = new Set(
    staged.map((relation) => relation.name).filter(Boolean),
  );
  return jobs
    .map((job) => {
      const jobRuns = runsByJob.get(job.id) ?? [];
      const successfulRun = jobRuns.find((run) => run.status === "complete");
      const scheduledRuns = jobRuns.filter((run) => run.trigger === "schedule");
      const manualRuns = jobRuns.filter((run) => run.trigger !== "schedule");
      const gaps: string[] = [];
      if (jobRuns.length === 0) gaps.push("No runs");
      if (!successfulRun) gaps.push("No successful run");
      if (!relationNames.has(job.name)) gaps.push("No staged output");
      if (hasSchedule(job) && scheduledRuns.length === 0)
        gaps.push("No scheduled execution");
      if (
        !hasSchedule(job) &&
        manualRuns.length === jobRuns.length &&
        jobRuns.length > 0
      ) {
        gaps.push("Manual only");
      }
      return {
        job,
        gaps,
        runCount: jobRuns.length,
        lastSuccessLabel: successfulRun
          ? shortWhen(successfulRun.startedAt)
          : "Never",
        executionLabel:
          scheduledRuns.length > 0 && manualRuns.length > 0
            ? "Mixed"
            : scheduledRuns.length > 0
              ? "Scheduled"
              : manualRuns.length > 0
                ? "Manual"
                : "None",
      };
    })
    .filter((entry) => entry.gaps.length > 0)
    .sort((a, b) => b.gaps.length - a.gaps.length || a.runCount - b.runCount);
}

function buildExecutionModeMix(jobs: readonly Job[], runs: readonly JobRun[]) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const counts = new Map<string, number>();
  for (const job of jobs) {
    const mode = latestByJob.get(job.id)?.metadata?.executionMode ?? "unknown";
    counts.set(mode, (counts.get(mode) ?? 0) + 1);
  }
  const total = jobs.length || 1;
  return [...counts.entries()]
    .map(([label, count]) => ({
      label: humanizeMode(label),
      count,
      ratio: count / total,
    }))
    .sort((a, b) => b.count - a.count);
}

function buildSourceModeMix(jobs: readonly Job[], runs: readonly JobRun[]) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const counts = new Map<string, number>();
  for (const job of jobs) {
    const mode = latestByJob.get(job.id)?.metadata?.sourceMode ?? "unknown";
    counts.set(mode, (counts.get(mode) ?? 0) + 1);
  }
  const total = jobs.length || 1;
  return [...counts.entries()]
    .map(([label, count]) => ({
      label: humanizeMode(label),
      count,
      ratio: count / total,
    }))
    .sort((a, b) => b.count - a.count);
}

function buildOperatingDrift(jobs: readonly Job[], runs: readonly JobRun[]) {
  return jobs
    .map((job) => {
      const jobRuns = runs.filter((run) => run.jobId === job.id);
      if (jobRuns.length === 0) return null;
      const executionModes = new Set(
        jobRuns
          .map((run) => run.metadata?.executionMode)
          .filter((value): value is string => Boolean(value)),
      );
      const sourceModes = new Set(
        jobRuns
          .map((run) => run.metadata?.sourceMode)
          .filter((value): value is string => Boolean(value)),
      );
      const flags: string[] = [];
      if (
        executionModes.size === 1 &&
        executionModes.has("on_demand") &&
        hasSchedule(job)
      ) {
        flags.push("Scheduled job running manual-only");
      }
      if (sourceModes.size === 1 && sourceModes.has("file")) {
        flags.push("File-only source path");
      }
      if (
        jobRuns.every(
          (run) =>
            !run.metadata?.output?.stagedRelationId &&
            !run.metadata?.output?.targetPath,
        )
      ) {
        flags.push("No durable output trail");
      }
      if (flags.length === 0) return null;
      return {
        job,
        flags,
        runCount: jobRuns.length,
        executionLabel:
          executionModes.size === 0
            ? "Unknown"
            : [...executionModes].map(humanizeMode).join(", "),
        sourceLabel:
          sourceModes.size === 0
            ? "Unknown"
            : [...sourceModes].map(humanizeMode).join(", "),
      };
    })
    .filter(
      (
        entry,
      ): entry is {
        job: Job;
        flags: string[];
        runCount: number;
        executionLabel: string;
        sourceLabel: string;
      } => entry != null,
    )
    .sort((a, b) => b.flags.length - a.flags.length || a.runCount - b.runCount);
}

function buildOutputEstateMix(jobs: readonly Job[], runs: readonly JobRun[]) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const counts = new Map<string, number>();
  for (const job of jobs) {
    const output = latestByJob.get(job.id)?.metadata?.output;
    const label = classifyOutputEstate(output);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const total = jobs.length || 1;
  return [...counts.entries()]
    .map(([label, count]) => ({
      label,
      count,
      ratio: count / total,
    }))
    .sort((a, b) => b.count - a.count);
}

function buildControlPosture(jobs: readonly Job[], runs: readonly JobRun[]) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const counts = new Map<string, number>();
  for (const job of jobs) {
    const latestRun = latestByJob.get(job.id);
    const posture = classifyControlPosture(job, latestRun);
    counts.set(posture, (counts.get(posture) ?? 0) + 1);
  }
  const total = jobs.length || 1;
  return [...counts.entries()]
    .map(([label, count]) => ({
      label,
      count,
      ratio: count / total,
    }))
    .sort((a, b) => b.count - a.count);
}

function buildGovernanceScorecards(
  jobs: readonly Job[],
  runs: readonly JobRun[],
) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  return jobs
    .map((job) => {
      const latestRun = latestByJob.get(job.id);
      const criteria = [
        { label: "scheduled", met: hasSchedule(job) },
        { label: "successful", met: latestRun?.status === "complete" },
        {
          label: "deterministic",
          met: latestRun?.metadata?.determinism === "deterministic",
        },
        {
          label: "managed source",
          met:
            latestRun?.metadata?.sourceMode != null &&
            latestRun.metadata.sourceMode !== "file",
        },
        {
          label: "durable output",
          met: Boolean(
            latestRun?.metadata?.output?.targetPath ||
            latestRun?.metadata?.output?.targetConnectorId ||
            latestRun?.metadata?.output?.stagedRelationId,
          ),
        },
      ];
      return {
        job,
        criteria,
        score: criteria.filter((criterion) => criterion.met).length,
        maxScore: criteria.length,
      };
    })
    .sort((a, b) => b.score - a.score || a.job.name.localeCompare(b.job.name));
}

function buildMetadataCoverage(jobs: readonly Job[], runs: readonly JobRun[]) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const total = jobs.length || 1;
  const checks = [
    {
      label: "Execution mode",
      predicate: (run: JobRun | undefined) =>
        Boolean(run?.metadata?.executionMode),
    },
    {
      label: "Source mode",
      predicate: (run: JobRun | undefined) =>
        Boolean(run?.metadata?.sourceMode),
    },
    {
      label: "Sync mode",
      predicate: (run: JobRun | undefined) => Boolean(run?.metadata?.syncMode),
    },
    {
      label: "Determinism",
      predicate: (run: JobRun | undefined) =>
        Boolean(run?.metadata?.determinism),
    },
    {
      label: "Output metadata",
      predicate: (run: JobRun | undefined) => Boolean(run?.metadata?.output),
    },
    {
      label: "Run summary",
      predicate: (run: JobRun | undefined) => Boolean(run?.metadata?.summary),
    },
  ];
  return checks.map((check) => {
    const count = jobs.filter((job) =>
      check.predicate(latestByJob.get(job.id)),
    ).length;
    return {
      label: check.label,
      count,
      total,
      ratio: count / total,
    };
  });
}

function buildTelemetryConfidence(
  jobs: readonly Job[],
  runs: readonly JobRun[],
) {
  const latestByJob = new Map<string, JobRun>();
  for (const run of runs) {
    if (!latestByJob.has(run.jobId)) latestByJob.set(run.jobId, run);
  }
  const checks = [
    {
      label: "execution mode",
      predicate: (run: JobRun | undefined) =>
        Boolean(run?.metadata?.executionMode),
    },
    {
      label: "source mode",
      predicate: (run: JobRun | undefined) =>
        Boolean(run?.metadata?.sourceMode),
    },
    {
      label: "sync mode",
      predicate: (run: JobRun | undefined) => Boolean(run?.metadata?.syncMode),
    },
    {
      label: "determinism",
      predicate: (run: JobRun | undefined) =>
        Boolean(run?.metadata?.determinism),
    },
    {
      label: "output metadata",
      predicate: (run: JobRun | undefined) => Boolean(run?.metadata?.output),
    },
    {
      label: "run summary",
      predicate: (run: JobRun | undefined) => Boolean(run?.metadata?.summary),
    },
  ];
  return jobs
    .map((job) => {
      const latestRun = latestByJob.get(job.id);
      const missing = checks
        .filter((check) => !check.predicate(latestRun))
        .map((check) => check.label);
      return {
        job,
        missing,
        score: checks.length - missing.length,
        maxScore: checks.length,
      };
    })
    .sort((a, b) => a.score - b.score || a.job.name.localeCompare(b.job.name));
}

function nextRunLabel(job: Job, now: Date): string | null {
  const expr = job.definition.schedule?.trim();
  if (!expr) return null;
  const next = nextRuns(expr, now, 1)[0];
  return next ? formatUtc(next) : "Invalid cron";
}

function inferScheduleInterval(job: Job, now: Date): number | null {
  const expr = job.definition.schedule?.trim();
  if (!expr) return null;
  const next = nextRuns(expr, now, 2);
  if (next.length < 2) return null;
  return Math.max(0, next[1].getTime() - next[0].getTime());
}

function hasSchedule(job: Job): boolean {
  return Boolean(job.definition.schedule?.trim());
}

function inferJobSyncMode(job: Job): string {
  if (
    (job.definition as unknown as { fullRefresh?: boolean }).fullRefresh ===
    true
  ) {
    return "full_refresh";
  }
  const targetParams = job.definition.targetConnectorParams ?? {};
  if (typeof targetParams.mode === "string" && targetParams.mode.trim()) {
    return targetParams.mode.trim();
  }
  if (job.definition.sourceBindingId || job.definition.sourceConnectorId) {
    return "incremental";
  }
  return "ad_hoc";
}

function humanizeMode(mode: string): string {
  return mode
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (match) => match.toUpperCase());
}

function parseTimestamp(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function elapsedBetween(
  startedAt: string | null | undefined,
  finishedAt: string | null | undefined,
): number | null {
  const startedMs = parseTimestamp(startedAt);
  const finishedMs = parseTimestamp(finishedAt);
  if (startedMs == null || finishedMs == null) return null;
  return Math.max(0, finishedMs - startedMs);
}

function shortWhen(value: string): string {
  const ts = parseTimestamp(value);
  if (ts == null) return value || "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(ts));
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function formatRate(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 /s";
  return `${new Intl.NumberFormat(undefined, {
    maximumFractionDigits: value >= 100 ? 0 : value >= 10 ? 1 : 2,
  }).format(value)} /s`;
}

function formatAgeMs(ageMs: number): string {
  if (!Number.isFinite(ageMs) || ageMs <= 0) return "Just now";
  const hours = Math.floor(ageMs / (60 * 60 * 1000));
  if (hours < 1) return "<1h";
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

function formatDurationMs(durationMs: number): string {
  if (!Number.isFinite(durationMs) || durationMs < 0) return "0s";
  if (durationMs < 1000) return `${durationMs} ms`;
  const totalSeconds = Math.round(durationMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes === 0) return `${totalSeconds}s`;
  return `${minutes}m ${seconds}s`;
}

function percentile(values: readonly number[], ratio: number): number {
  if (values.length === 0) return 0;
  const index = Math.min(
    values.length - 1,
    Math.max(0, Math.ceil(values.length * ratio) - 1),
  );
  return values[index];
}

function failureMessage(run: JobRun): string {
  return (
    String(run.metadata?.diagnostics?.message || "").trim() ||
    String(
      (run.metadata as Record<string, unknown> | undefined)?.error || "",
    ).trim() ||
    "Unclassified failure"
  );
}

function normalizeFailureMessage(message: string): string {
  return message
    .toLowerCase()
    .replace(/\b\d+\b/g, "#")
    .replace(/`[^`]+`/g, "`value`")
    .replace(/\s+/g, " ")
    .trim();
}

function truncateLabel(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, Math.max(0, max - 1)).trimEnd()}…`;
}

function classifyOutputEstate(
  output:
    | {
        sinkUri?: string | undefined;
        targetConnectorId?: string | undefined;
        targetPath?: string | undefined;
      }
    | null
    | undefined,
): string {
  if (!output) return "No recorded output";
  if (output.targetPath) return "Filesystem path";
  if (output.sinkUri) return "Sink URI";
  if (output.targetConnectorId) return "Connector target";
  return "Recorded output";
}

function classifyControlPosture(
  job: Job,
  latestRun: JobRun | undefined,
): string {
  const executionMode = latestRun?.metadata?.executionMode;
  const sourceMode = latestRun?.metadata?.sourceMode;
  const determinism = latestRun?.metadata?.determinism;
  const hasDurableOutput = Boolean(
    latestRun?.metadata?.output?.targetPath ||
    latestRun?.metadata?.output?.stagedRelationId ||
    latestRun?.metadata?.output?.targetConnectorId,
  );
  if (
    executionMode === "scheduled" &&
    sourceMode !== "file" &&
    determinism === "deterministic" &&
    hasDurableOutput
  ) {
    return "Managed";
  }
  if (
    executionMode === "on_demand" ||
    sourceMode === "file" ||
    !hasDurableOutput
  ) {
    return "Ad hoc";
  }
  if (!latestRun && hasSchedule(job)) return "Partial";
  return "Partial";
}

function riskWeight(status: string): number {
  switch (status) {
    case "failed":
      return 4;
    case "late":
      return 3;
    case "stale":
      return 2;
    case "no_run":
      return 1;
    default:
      return 0;
  }
}
