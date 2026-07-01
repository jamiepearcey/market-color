import type {
  ConnectorInstance,
  ConnectorParam,
  ConnectorSpec,
  Binding,
  Job,
  RunResult,
  SourceColumn,
  Target,
  Template,
} from "./types";

export type JobReadinessState = "complete" | "blocked" | "warning";

export type JobReadinessAction =
  | "chooseSource"
  | "testSource"
  | "addStep"
  | "fillParams"
  | "dryRun"
  | "runNow"
  | "resolveBlockers";

export interface JobReadinessItem {
  id:
    | "source"
    | "sourceTest"
    | "steps"
    | "params"
    | "output"
    | "dryRun"
    | "schedule"
    | "compute";
  label: string;
  state: JobReadinessState;
  message: string;
}

export interface JobReadiness {
  items: JobReadinessItem[];
  blockers: JobReadinessItem[];
  warnings: JobReadinessItem[];
  canRun: boolean;
  nextAction: {
    kind: JobReadinessAction;
    label: string;
    disabled?: boolean;
  };
}

export interface JobReadinessInput {
  draft: Job;
  jobSourceMode: "file" | "connector";
  templates: readonly Template[];
  connectorSpecs: readonly ConnectorSpec[];
  connectorInstances: readonly ConnectorInstance[];
  bindings?: readonly Binding[];
  sourceColumns: readonly SourceColumn[];
  stepColumns: readonly (readonly string[])[];
  sourcePreview: RunResult | null;
  dryResult: RunResult | null;
  activeTargetId: string | null;
  targets: readonly Target[];
}

export function deriveJobReadiness(input: JobReadinessInput): JobReadiness {
  const { draft } = input;
  const activeTarget = input.targets.find((t) => t.id === input.activeTargetId);
  const fileSourceSelected = Boolean(draft.definition.source.trim());
  const connectorSource = findConnector(input, draft.definition.sourceConnectorId);
  const sourceBinding = input.bindings?.find((b) => b.id === draft.definition.sourceBindingId);
  const bindingSource = sourceBinding ? findConnector(input, sourceBinding.instanceId) : null;
  const effectiveSource = connectorSource ?? bindingSource;
  const sourceSelected =
    input.jobSourceMode === "file" ? fileSourceSelected : Boolean(effectiveSource);
  const sourceParamsMissing =
    input.jobSourceMode === "connector" && effectiveSource
      ? missingJobParams(
          input,
          effectiveSource,
          draft.definition.sourceBindingId
            ? sourceBinding?.runParams
            : draft.definition.sourceConnectorParams,
        )
      : [];
  const targetConnector = findConnector(input, draft.definition.targetConnectorId);
  const targetParamsMissing = targetConnector
    ? missingJobParams(input, targetConnector, draft.definition.targetConnectorParams)
    : [];
  const stepProblems = stepArgProblems(input);
  const stepsPresent = draft.definition.steps.length > 0;
  const stepsValid =
    stepsPresent &&
    draft.definition.steps.every((step) => stepIsRunnable(step, input.templates));
  const sourceTested = input.sourcePreview?.status === "complete";
  const dryRunPassed = input.dryResult?.status === "complete";
  const scheduled = Boolean(draft.definition.schedule?.trim());
  const scheduleValid = !scheduled || isValidCronExpression(draft.definition.schedule ?? "");
  const remoteTargetSelected = activeTarget?.kind === "remote";

  const items: JobReadinessItem[] = [
    {
      id: "source",
      label: "Source",
      state: sourceSelected && sourceParamsMissing.length === 0 ? "complete" : "blocked",
      message: sourceMessage(input, sourceSelected, sourceParamsMissing),
    },
    {
      id: "sourceTest",
      label: "Source test",
      state: sourceTested ? "complete" : "blocked",
      message: sourceTested
        ? "Source resolved successfully."
        : sourceSelected
          ? "Test the source before running."
          : "Select a source first.",
    },
    {
      id: "steps",
      label: "Steps",
      state: stepsValid ? "complete" : "blocked",
      message: stepsValid
        ? `${draft.definition.steps.length} step${draft.definition.steps.length === 1 ? "" : "s"} configured.`
        : stepsPresent
          ? "Choose a valid template for every step."
          : "Add at least one ETL step.",
    },
    {
      id: "params",
      label: "Parameters",
      state: stepProblems.length === 0 && targetParamsMissing.length === 0 ? "complete" : "blocked",
      message:
        stepProblems.length === 0 && targetParamsMissing.length === 0
          ? "Required step and target parameters are filled."
          : [...stepProblems, ...targetParamsMissing].slice(0, 2).join("; "),
    },
    {
      id: "output",
      label: "Output",
      state: targetConnector ? "complete" : "warning",
      message: targetConnector
        ? `Exporting to ${targetConnector.name}.`
        : "No target connector selected; results stay in run history.",
    },
    {
      id: "dryRun",
      label: "Dry run",
      state:
        input.jobSourceMode === "connector"
          ? "warning"
          : dryRunPassed
            ? "complete"
            : "blocked",
      message:
        input.jobSourceMode === "connector"
          ? "Connector-backed dry run is not available yet; use source preview."
          : dryRunPassed
            ? "Dry run passed."
            : "Run a dry run before execution.",
    },
    {
      id: "schedule",
      label: "Schedule",
      state: scheduleValid ? "complete" : "blocked",
      message: scheduleValid
        ? scheduled
          ? "Scheduled cron is valid."
          : "Runs on demand."
        : "Use a valid 5-field cron expression.",
    },
    {
      id: "compute",
      label: "Compute",
      state: activeTarget && !remoteTargetSelected ? "complete" : "blocked",
      message: activeTarget
        ? remoteTargetSelected
          ? "Remote targets are coming soon. Switch back to the embedded local engine."
          : `${activeTarget.name} is selected.`
        : "Select an execution target.",
    },
  ];

  const blockers = items.filter((item) => item.state === "blocked");
  const warnings = items.filter((item) => item.state === "warning");
  const fileModeDryRunOk = input.jobSourceMode === "connector" || dryRunPassed;
  const canRun =
    sourceSelected &&
    sourceParamsMissing.length === 0 &&
    sourceTested &&
    stepsValid &&
    stepProblems.length === 0 &&
    targetParamsMissing.length === 0 &&
    scheduleValid &&
    Boolean(activeTarget) &&
    !remoteTargetSelected &&
    fileModeDryRunOk;

  return {
    items,
    blockers,
    warnings,
    canRun,
    nextAction: nextAction({
      sourceSelected,
      sourceTested,
      stepsValid,
      stepProblems,
      targetParamsMissing,
      dryRunPassed,
      connectorMode: input.jobSourceMode === "connector",
      canRun,
    }),
  };
}

function sourceMessage(
  input: JobReadinessInput,
  sourceSelected: boolean,
  sourceParamsMissing: string[],
) {
  if (!sourceSelected) {
    return input.jobSourceMode === "file"
      ? "Choose a CSV/file source."
      : "Select a saved binding or source connector.";
  }
  if (sourceParamsMissing.length > 0) {
    return sourceParamsMissing.slice(0, 2).join("; ");
  }
  return input.jobSourceMode === "file"
    ? "File source selected."
    : "Source connector selected.";
}

function nextAction(input: {
  sourceSelected: boolean;
  sourceTested: boolean;
  stepsValid: boolean;
  stepProblems: string[];
  targetParamsMissing: string[];
  dryRunPassed: boolean;
  connectorMode: boolean;
  canRun: boolean;
}): JobReadiness["nextAction"] {
  if (!input.sourceSelected || !input.sourceTested) {
    return {
      kind: input.sourceSelected ? "testSource" : "chooseSource",
      label: input.sourceSelected ? "Test source" : "Choose source",
    };
  }
  if (!input.stepsValid) return { kind: "addStep", label: "Add step" };
  if (input.stepProblems.length > 0 || input.targetParamsMissing.length > 0) {
    return { kind: "fillParams", label: "Fill parameters" };
  }
  if (!input.connectorMode && !input.dryRunPassed) {
    return { kind: "dryRun", label: "Dry run" };
  }
  if (!input.canRun) return { kind: "resolveBlockers", label: "Resolve blockers" };
  return { kind: "runNow", label: "Run now" };
}

function stepArgProblems(input: JobReadinessInput) {
  const problems: string[] = [];
  input.draft.definition.steps.forEach((step, index) => {
    if (step.kind === "markdown") return;
    if (step.kind === "sql") {
      if (!step.sql?.trim()) problems.push(`Step ${index + 1}: enter SQL.`);
      return;
    }
    const template = input.templates.find((t) => t.name === step.template);
    if (!template) {
      problems.push(`Step ${index + 1}: choose a template.`);
      return;
    }
    for (const templateInput of template.inputs.filter((i) => !isRelationInput(i.kind))) {
      const value = String(step.args[templateInput.name] ?? "").trim();
      if (!value) {
        problems.push(`Step ${index + 1}: ${templateInput.name} is required.`);
        continue;
      }
      if (templateInput.kind === "Column" && availableColumns(input, index).length > 0) {
        const cols = availableColumns(input, index);
        if (!cols.includes(value)) {
          problems.push(`Step ${index + 1}: ${value} is not in the current schema.`);
        }
      }
    }
  });
  return problems;
}

function isRelationInput(kind: string) {
  return kind === "Table" || kind === "TypedRelation";
}

function stepIsRunnable(
  step: JobReadinessInput["draft"]["definition"]["steps"][number],
  templates: readonly Template[],
) {
  if (step.kind === "markdown") return true;
  if (step.kind === "sql") return Boolean(step.sql?.trim());
  return Boolean(step.template && templates.some((t) => t.name === step.template));
}

function availableColumns(input: JobReadinessInput, stepIndex: number) {
  if (stepIndex > 0 && input.stepColumns[stepIndex - 1]?.length) {
    return [...input.stepColumns[stepIndex - 1]];
  }
  return input.sourceColumns.map((col) => col.name);
}

function findConnector(input: JobReadinessInput, id: string | undefined) {
  if (!id) return null;
  return input.connectorInstances.find((c) => c.id === id) ?? null;
}

function missingJobParams(
  input: JobReadinessInput,
  instance: ConnectorInstance,
  values: Record<string, string> | undefined,
) {
  const spec = input.connectorSpecs.find((s) => s.name === instance.spec);
  if (!spec) return [`${instance.name}: connector spec is missing.`];
  return spec.jobParams
    .filter((param) => param.required)
    .filter((param) => !paramValue(param, values, instance).trim())
    .map((param) => `${instance.name}: ${param.label || param.name} is required.`);
}

function paramValue(
  param: ConnectorParam,
  values: Record<string, string> | undefined,
  instance: ConnectorInstance,
) {
  return String(
    values?.[param.name] ??
      instance.jobParamDefaults?.[param.name] ??
      param.default ??
      "",
  );
}

function isValidCronExpression(value: string) {
  const parts = value.trim().split(/\s+/);
  return parts.length === 5 && parts.every(Boolean);
}
