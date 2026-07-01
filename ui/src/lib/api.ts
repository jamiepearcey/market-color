// Data layer. In the Tauri shell it calls Rust commands that drive the real
// `quant-fabric local` engine; in a plain browser (vite dev / preview) it falls
// back to the bundled catalog snapshot + a mock run, so the UI is fully
// explorable without the desktop backend.
import type {
  ChatMessage,
  BindingConformance,
  Conformance,
  ContractInfo,
  ClusterPlan,
  ClusterStatus,
  CommandBlock,
  Binding,
  ConnectorActionResult,
  ConnectorInstanceRecord,
  ConnectorSpec,
  ConnectorTestResult,
  DataTreeNode,
  EmbeddedConfig,
  ExternalEndpoint,
  RaftMember,
  RaftMembershipResponse,
  RegistryListResult,
  RegistryPackage,
  RegistryInstallResult,
  Job,
  JobPackUpdateResolution,
  JobRun,
  JobRunListResult,
  JobRunStep,
  JobStep,
  LinkGithubPackRequest,
  NotebookFile,
  Pack,
  PackFile,
  PackSource,
  PackSourceMonitor,
  PackSyncEvent,
  PackVersion,
  PackVersionCompare,
  RunResult,
  SharedConnectorBinding,
  SourceSchema,
  StagedRelation,
  Target,
  Template,
  WorkerStatus,
} from "./types";
import catalog from "@/data/catalog.json";
import { serverReady, srv, srvGet, srvPost, srvDelete } from "./server";

type TauriInvoke = <T>(
  cmd: string,
  args?: Record<string, unknown>,
) => Promise<T>;

function hasTauriInternals(
  value: Window & typeof globalThis,
): value is Window & typeof globalThis & { __TAURI_INTERNALS__: unknown } {
  return "__TAURI_INTERNALS__" in value;
}

function parseCatalogTemplates(input: unknown): Template[] {
  if (!Array.isArray(input)) return [];
  return input.filter(
    (item): item is Template =>
      typeof item === "object" &&
      item !== null &&
      typeof item.name === "string" &&
      typeof item.description === "string" &&
      typeof item.kind === "string" &&
      typeof item.category === "string" &&
      Array.isArray(item.inputs) &&
      Array.isArray(item.requiredAggregateKinds),
  );
}

const catalogTemplates = parseCatalogTemplates(catalog);

function tauri(): TauriInvoke | null {
  if (!hasTauriInternals(window)) return null;
  return (cmd, args) =>
    import("@tauri-apps/api/core").then((m) =>
      m.invoke(cmd, args),
    ) as Promise<never>;
}

export const isDesktop = () => tauri() !== null;

/**
 * True in the browser preview build, where there is no Tauri engine and results
 * (template/job runs, connector probes, cluster topology) are *synthesised* by
 * the mocks below. UI that shows such a result must label it as simulated so a
 * green outcome is never mistaken for a real one. See `<SimulatedBadge/>`.
 */
export const isPreview = () => !isDesktop();

export async function listTemplates(): Promise<Template[]> {
  const inv = tauri();
  if (inv) {
    try {
      return (await inv<Template[]>("list_templates")).map(normalizeTemplate);
    } catch {
      /* fall through to snapshot */
    }
  }
  return catalogTemplates.map(normalizeTemplate);
}

function normalizeTemplate(template: Template): Template {
  const displayName =
    template.displayName || titleFromTemplateName(template.name);
  return {
    ...template,
    displayName,
    shortName: template.shortName || displayName,
  };
}

function titleFromTemplateName(name: string): string {
  const acronyms = new Set([
    "adf",
    "alm",
    "bps",
    "capm",
    "cds",
    "cpr",
    "cs01",
    "cva",
    "dv01",
    "ead",
    "esg",
    "etf",
    "ev",
    "ff5",
    "fra",
    "fx",
    "lgd",
    "lcr",
    "ois",
    "ols",
    "pfe",
    "pnl",
    "pv01",
    "r2",
    "raroc",
    "roi",
    "rwa",
    "smm",
    "var",
    "ytm",
  ]);
  return name
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((part) => {
      const lower = part.toLowerCase();
      if (acronyms.has(lower)) return lower.toUpperCase();
      if (/^\d+$/.test(part)) return part;
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

export async function runTemplate(
  name: string,
  args: Record<string, string>,
): Promise<RunResult> {
  const inv = tauri();
  if (inv) {
    return inv<RunResult>("run_template", { name, args });
  }
  // Browser mock: synthesize a plausible result so the run flow is demoable.
  await new Promise((r) => setTimeout(r, 650));
  const tpl = catalogTemplates.find((t) => t.name === name);
  const outCols =
    tpl?.kind === "aggregate"
      ? ["group", "result_1", "result_2"]
      : ["row", "value", "output"];
  const rows = Array.from({ length: 6 }, (_, i) =>
    Object.fromEntries(
      outCols.map((c, j) => [
        c,
        j === 0 ? `r${i}` : +(Math.random() * 10).toFixed(4),
      ]),
    ),
  );
  return {
    status: "complete",
    rows,
    rowCount: rows.length,
    elapsedMs: 14 + Math.floor(Math.random() * 20),
    error: null,
  };
}

// ---- Connectors (ADR-0026). Catalog is real (parsed from the TOML library on
//      the desktop side); test/run are mocked until drivers land. ----

export async function listConnectors(): Promise<ConnectorSpec[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ConnectorSpec[]>("list_connectors");
    } catch {
      /* fall through to server / preview snapshot */
    }
  }
  if (await serverReady()) {
    return await srvGet<ConnectorSpec[]>("/connectors");
  }
  return BROWSER_CONNECTORS;
}

/// Cohesive packs (ADR-0027). Desktop only; browser preview has no packs.
export async function listPacks(): Promise<Pack[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<Pack[]>("list_packs");
    } catch {
      /* fall through */
    }
  }
  return [];
}

export async function readPackFiles(packId: string): Promise<PackFile[]> {
  const inv = tauri();
  if (inv) return inv<PackFile[]>("read_pack_files", { packId });
  return [];
}

export async function writePackFile(
  packId: string,
  relPath: string,
  contents: string,
): Promise<void> {
  const inv = tauri();
  if (inv) return inv("write_pack_file", { packId, relPath, contents });
  console.info("[mock] writePackFile", { packId, relPath });
}

export async function duplicatePack(
  packId: string,
  newId: string,
  newName: string,
): Promise<Pack | null> {
  const inv = tauri();
  if (inv) return inv<Pack>("duplicate_pack", { packId, newId, newName });
  return null;
}

export async function createPack(
  id: string,
  name: string,
  description: string,
  files: PackFile[],
): Promise<Pack | null> {
  const inv = tauri();
  if (inv) return inv<Pack>("create_pack", { id, name, description, files });
  console.info("[mock] createPack", { id, name, files: files.length });
  return null;
}

export async function exportPack(
  packId: string,
  destPath: string,
): Promise<void> {
  const inv = tauri();
  if (inv) return inv("export_pack", { packId, destPath });
}

export async function importPack(srcPath: string): Promise<Pack | null> {
  const inv = tauri();
  if (inv) return inv<Pack>("import_pack", { srcPath });
  return null;
}

export async function deletePack(packId: string): Promise<void> {
  const inv = tauri();
  if (inv) return inv("delete_pack", { packId });
}

export async function linkGithubPack(
  request: LinkGithubPackRequest,
): Promise<PackSource | null> {
  const inv = tauri();
  if (inv) return inv<PackSource>("link_github_pack", { request });
  return null;
}

export async function syncPackSource(
  sourceId: string,
): Promise<PackVersion | null> {
  const inv = tauri();
  if (inv) return inv<PackVersion>("sync_pack_source", { sourceId });
  return null;
}

export async function updatePackSourceMonitor(
  sourceId: string,
  monitor: PackSourceMonitor,
): Promise<void> {
  const inv = tauri();
  if (inv) return inv("update_pack_source_monitor", { sourceId, monitor });
}

export async function listPackSources(): Promise<PackSource[]> {
  const inv = tauri();
  if (inv) return inv<PackSource[]>("list_pack_sources");
  return [];
}

export async function listPackVersions(
  sourceId?: string,
): Promise<PackVersion[]> {
  const inv = tauri();
  if (inv)
    return inv<PackVersion[]>("list_pack_versions", {
      sourceId: sourceId ?? null,
    });
  return [];
}

export async function listPackSyncEvents(
  sourceId?: string,
): Promise<PackSyncEvent[]> {
  const inv = tauri();
  if (inv)
    return inv<PackSyncEvent[]>("list_pack_sync_events", {
      sourceId: sourceId ?? null,
    });
  return [];
}

export async function comparePackVersions(
  fromVersionId: string,
  toVersionId: string,
): Promise<PackVersionCompare | null> {
  const inv = tauri();
  if (inv)
    return inv<PackVersionCompare>("compare_pack_versions", {
      fromVersionId,
      toVersionId,
    });
  return null;
}

export async function resolveJobPackUpdate(
  sourceId: string,
  currentVersionId?: string,
): Promise<JobPackUpdateResolution | null> {
  const inv = tauri();
  if (inv)
    return inv<JobPackUpdateResolution>("resolve_job_pack_update", {
      sourceId,
      currentVersionId: currentVersionId ?? null,
    });
  return null;
}

export async function pickPackBundle(): Promise<string | null> {
  const inv = tauri();
  if (inv) return inv<string | null>("pick_pack_bundle");
  return null;
}

export async function pickSavePath(
  defaultName: string,
): Promise<string | null> {
  const inv = tauri();
  if (inv) return inv<string | null>("pick_save_path", { defaultName });
  return null;
}

export async function testConnector(
  kind: string,
  driver: string,
  params: Record<string, string>,
): Promise<ConnectorTestResult> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ConnectorTestResult>("test_connector", {
        kind,
        driver,
        params,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return await srvPost<ConnectorTestResult>("/connectors/test", {
      spec: driver,
      driver,
      kind,
      params,
      jobParams: {},
    });
  }
  await new Promise((r) => setTimeout(r, 500));
  const pathy = Object.values(params).some(
    (v) => /^(\.\/|\/|~\/)/.test(v) && !v.includes("://"),
  );
  return pathy
    ? {
        status: "example",
        message:
          "(preview) configured path is an example — connect the desktop backend to test for real.",
        elapsedMs: 12,
      }
    : {
        status: "pass",
        message: "(preview) mock connectivity check passed.",
        elapsedMs: 12,
      };
}

export async function runConnectorAction(
  driver: string,
  flag: string,
  params: Record<string, string>,
  jobParams: Record<string, string>,
): Promise<ConnectorActionResult> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ConnectorActionResult>("run_connector_action", {
        driver,
        flag,
        params,
        jobParams,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return await srvPost<ConnectorActionResult>("/connectors/action", {
      spec: driver,
      driver,
      action: flag,
      params,
      jobParams,
    });
  }
  await new Promise((r) => setTimeout(r, 500));
  // Browser preview: synthesize a small representative sample so discovery /
  // preview (and the binding contract-conformance panel) are explorable without
  // the desktop backend. Labeled `(preview)` so it is never mistaken for real.
  const sample = [
    {
      symbol: "AAPL",
      price: 191.24,
      bid: 191.2,
      ask: 191.28,
      ts: "2026-06-10",
    },
    {
      symbol: "MSFT",
      price: 438.11,
      bid: 438.05,
      ask: 438.18,
      ts: "2026-06-10",
    },
    {
      symbol: "NVDA",
      price: 121.4,
      bid: 121.36,
      ask: 121.45,
      ts: "2026-06-10",
    },
  ];
  return {
    status: "complete",
    message: `(preview) sample of ${driver} ${flag}. Connect the desktop backend for real data.`,
    rows: sample,
    elapsedMs: 12,
  };
}

const BROWSER_BINDINGS: Binding[] = [];

const CONTRACT_FIELDS: Record<
  string,
  { name: string; kind: string; required?: boolean }[]
> = {
  "Position.v1": [
    { name: "instrument", kind: "string" },
    { name: "quantity", kind: "number" },
    { name: "book", kind: "string", required: false },
  ],
  "Trading/Position.v1": [
    { name: "instrument", kind: "string" },
    { name: "quantity", kind: "number" },
    { name: "book", kind: "string", required: false },
  ],
  "MarketSnapshot.v1": [
    { name: "instrument", kind: "string" },
    { name: "price", kind: "number" },
    { name: "as_of", kind: "date", required: false },
  ],
  "MarketData/MarketSnapshot.v1": [
    { name: "instrument", kind: "string" },
    { name: "price", kind: "number" },
    { name: "as_of", kind: "date", required: false },
  ],
  "TopOfBookQuote.v1": [
    { name: "instrument", kind: "string" },
    { name: "bid", kind: "number" },
    { name: "ask", kind: "number" },
  ],
  "MarketData/TopOfBookQuote.v1": [
    { name: "instrument", kind: "string" },
    { name: "bid", kind: "number" },
    { name: "ask", kind: "number" },
  ],
  "OptionValuationInput.v1": [
    { name: "instrument", kind: "string" },
    { name: "spot", kind: "number" },
    { name: "strike", kind: "number" },
    { name: "ttm", kind: "number" },
    { name: "rate", kind: "number" },
    { name: "vol", kind: "number" },
  ],
  "Derivatives/OptionValuationInput.v1": [
    { name: "instrument", kind: "string" },
    { name: "spot", kind: "number" },
    { name: "strike", kind: "number" },
    { name: "ttm", kind: "number" },
    { name: "rate", kind: "number" },
    { name: "vol", kind: "number" },
  ],
};

export async function listBindings(): Promise<Binding[]> {
  const inv = tauri();
  if (inv) return inv<Binding[]>("list_bindings");
  return BROWSER_BINDINGS.slice();
}

export async function listSharedConnectorBindings(): Promise<
  SharedConnectorBinding[]
> {
  const inv = tauri();
  if (inv)
    return inv<SharedConnectorBinding[]>("list_shared_connector_bindings");
  return [];
}

export async function runSharedConnectorBinding(
  name: string,
): Promise<ConnectorActionResult> {
  const inv = tauri();
  if (inv)
    return inv<ConnectorActionResult>("run_shared_connector_binding", { name });
  return {
    status: "example",
    message: `Browser preview cannot run the repo connector binding "${name}". Open the desktop app to execute it.`,
    rows: [],
  };
}

export async function listConformingBindings(
  contractRef: string,
): Promise<BindingConformance[]> {
  const inv = tauri();
  if (inv)
    return inv<BindingConformance[]>("list_conforming_bindings", {
      contractRef,
    });
  return listBindingConformance(contractRef).then((results) =>
    results.filter((result) => result.conformance.conforms),
  );
}

export async function listBindingConformance(
  contractRef: string,
): Promise<BindingConformance[]> {
  const inv = tauri();
  if (inv)
    return inv<BindingConformance[]>("list_binding_conformance", {
      contractRef,
    });
  const fields = CONTRACT_FIELDS[contractRef] ?? [];
  return BROWSER_BINDINGS.map((binding) => {
    const columns = binding.schema?.columns ?? [];
    const missing = binding.schema
      ? fields
          .filter((field) => field.required !== false)
          .filter((field) => {
            const source = binding.mapping?.[field.name] ?? field.name;
            return !columns.some((column) => column.name === source);
          })
          .map((field) => field.name)
      : fields
          .filter((field) => field.required !== false)
          .map((field) => field.name);
    return {
      binding,
      conformance: { conforms: missing.length === 0, missing, mismatches: [] },
    };
  });
}

export async function listContracts(): Promise<ContractInfo[]> {
  const inv = tauri();
  if (inv) return inv<ContractInfo[]>("list_contracts");
  // Browser fallback: derive from the hardcoded CONTRACT_FIELDS (bare refs only).
  return Object.entries(CONTRACT_FIELDS)
    .filter(([ref]) => !ref.includes("/"))
    .map(([ref, fields]) => {
      const dot = ref.lastIndexOf(".");
      return {
        reference: ref,
        domain: "",
        name: dot > 0 ? ref.slice(0, dot) : ref,
        version: dot > 0 ? ref.slice(dot + 1) : "v1",
        description: "",
        fields: fields.map((f) => ({
          name: f.name,
          kind: f.kind,
          required: f.required !== false,
        })),
      };
    });
}

// Single-binding conformance check (authoritative — runs the contract registry
// over the saved binding's discovered schema + mapping). The live editor uses a
// client-side presence preview; this is the server-side full validation.
export async function validateBindingConformance(
  bindingId: string,
  contractRef: string,
): Promise<Conformance> {
  const inv = tauri();
  if (inv)
    return inv<Conformance>("validate_binding_conformance", {
      bindingId,
      contractRef,
    });
  const fields = CONTRACT_FIELDS[contractRef] ?? [];
  const binding = BROWSER_BINDINGS.find((b) => b.id === bindingId);
  const columns = binding?.schema?.columns ?? [];
  const missing = fields
    .filter((f) => f.required !== false)
    .filter((f) => {
      const source = binding?.mapping?.[f.name] ?? f.name;
      return !columns.some((c) => c.name === source);
    })
    .map((f) => f.name);
  return { conforms: missing.length === 0, missing, mismatches: [] };
}

export async function saveBinding(binding: Binding): Promise<void> {
  const inv = tauri();
  if (inv) return inv("save_binding", { binding });
  const idx = BROWSER_BINDINGS.findIndex((b) => b.id === binding.id);
  const copy = JSON.parse(JSON.stringify(binding)) as Binding;
  if (idx >= 0) BROWSER_BINDINGS[idx] = copy;
  else BROWSER_BINDINGS.unshift(copy);
}

export async function deleteBinding(id: string): Promise<void> {
  const inv = tauri();
  if (inv) return inv("delete_binding", { id });
  const idx = BROWSER_BINDINGS.findIndex((b) => b.id === id);
  if (idx >= 0) BROWSER_BINDINGS.splice(idx, 1);
}

export async function saveConnectorInstanceRecord(
  instance: ConnectorInstanceRecord,
): Promise<ConnectorInstanceRecord | null> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ConnectorInstanceRecord>("save_connector_instance", {
        instance,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return await srvPost<ConnectorInstanceRecord>(
      "/connector-instances",
      instance,
    );
  }
  return null;
}

export async function deleteConnectorInstanceRecord(id: string): Promise<void> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv("delete_connector_instance", { id });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    await srvDelete<{ ok: boolean }>(
      `/connector-instances/${encodeURIComponent(id)}`,
    );
    return;
  }
}

/** Persisted connector instances (server mode; preview has none). */
export async function listConnectorInstances(): Promise<
  ConnectorInstanceRecord[]
> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ConnectorInstanceRecord[]>("list_connector_instances");
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return await srvGet<ConnectorInstanceRecord[]>("/connector-instances");
  }
  return [];
}

export async function listStagedRelations(): Promise<StagedRelation[]> {
  const inv = tauri();
  if (inv) return inv<StagedRelation[]>("list_staged_relations");
  return [];
}

export async function readStagedRelation(
  relationId: string,
): Promise<Record<string, unknown>[]> {
  const inv = tauri();
  if (inv)
    return inv<Record<string, unknown>[]>("read_staged_relation", {
      relationId,
    });
  return [];
}

export async function pinStagedRelation(relationId: string): Promise<void> {
  const inv = tauri();
  if (inv) return inv("pin_staged_relation", { relationId });
}

export async function forgetStagedRelation(relationId: string): Promise<void> {
  const inv = tauri();
  if (inv) return inv("forget_staged_relation", { relationId });
}

export async function gcStagedRelations(): Promise<number> {
  const inv = tauri();
  if (inv) return inv<number>("gc_staged_relations");
  return 0;
}

// A tiny representative snapshot so the Connectors UI is explorable in a plain
// browser (vite dev / preview). The desktop app uses the full TOML library.
const BROWSER_CONNECTORS: ConnectorSpec[] = [
  {
    name: "filesystem_csv",
    description: "Read CSV files from a local or mounted directory.",
    kind: "source",
    driver: "filesystem.csv",
    icon: "file-text",
    category: "Filesystem",
    params: [
      {
        name: "root_path",
        label: "Root directory",
        kind: "path",
        input: "directory",
        required: true,
        placeholder: "/data/marketdata",
        options: [],
      },
      {
        name: "delimiter",
        label: "Delimiter",
        kind: "string",
        required: false,
        default: ",",
        options: [],
      },
    ],
    jobParams: [
      {
        name: "pattern",
        label: "File glob",
        kind: "string",
        input: "glob",
        required: true,
        default: "*.csv",
        options: [],
      },
      {
        name: "filename_regex",
        label: "Filename regex",
        kind: "string",
        input: "regex",
        required: false,
        advanced: true,
        options: [],
      },
    ],
    actions: [{ name: "preview", label: "Preview rows", flag: "--preview" }],
    example: { root_path: "./sample-data" },
    exampleUnresolved: true,
    available: false,
    path: null,
    source: null,
  },
  {
    name: "s3_parquet",
    description: "Read Parquet objects from S3-compatible storage.",
    kind: "source",
    driver: "s3.parquet",
    icon: "cloud",
    category: "Object Store",
    params: [
      {
        name: "bucket",
        label: "Bucket",
        kind: "string",
        required: true,
        options: [],
      },
      {
        name: "region",
        label: "Region",
        kind: "string",
        required: false,
        default: "us-east-1",
        options: [],
      },
      {
        name: "secret_access_key",
        label: "Secret access key",
        kind: "secret",
        required: true,
        placeholder: "secrets-keeper://secret/aws#secret_access_key",
        help: "Inline secret or secrets-keeper://mount/path#field.",
        options: [],
      },
    ],
    jobParams: [
      {
        name: "prefix",
        label: "Key prefix",
        kind: "string",
        required: true,
        default: "parquet/",
        options: [],
      },
    ],
    actions: [{ name: "schema", label: "Show schema", flag: "--schema" }],
    example: {},
    exampleUnresolved: false,
    available: false,
    path: null,
    source: null,
  },
  {
    name: "postgres_export",
    description: "Write job output to a PostgreSQL table.",
    kind: "target",
    driver: "postgres",
    icon: "database",
    category: "Databases",
    params: [
      {
        name: "host",
        label: "Host",
        kind: "string",
        required: true,
        options: [],
      },
      {
        name: "database",
        label: "Database",
        kind: "string",
        required: true,
        options: [],
      },
      {
        name: "password",
        label: "Password",
        kind: "secret",
        required: true,
        placeholder: "secrets-keeper://secret/postgres#password",
        help: "Inline password or secrets-keeper://mount/path#field.",
        options: [],
      },
    ],
    jobParams: [
      {
        name: "table",
        label: "Target table",
        kind: "string",
        required: true,
        default: "public.results",
        options: [],
      },
      {
        name: "write_mode",
        label: "Write mode",
        kind: "enum",
        required: true,
        default: "append",
        options: ["append", "overwrite", "upsert"],
      },
    ],
    actions: [{ name: "create_ddl", label: "Show DDL", flag: "--show-ddl" }],
    example: {},
    exampleUnresolved: false,
    available: false,
    path: null,
    source: null,
  },
];

export interface ClusterOverview {
  coordinatorUrl: string;
  workerUrls: string[];
  workerCount: number;
}

export async function clusterOverview(): Promise<ClusterOverview | null> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ClusterOverview>("cluster_overview");
    } catch {
      return null; // cluster still starting
    }
  }
  return {
    coordinatorUrl: "http://127.0.0.1:7000",
    workerUrls: ["http://127.0.0.1:7101", "http://127.0.0.1:7102"],
    workerCount: 2,
  };
}

export async function embeddedConfig(): Promise<EmbeddedConfig> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<EmbeddedConfig>("embedded_config");
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    try {
      return await srvGet<EmbeddedConfig>("/engine");
    } catch {
      /* fall through to preview */
    }
  }
  // Browser preview: persist the chosen core count in localStorage so the
  // control behaves end-to-end without the desktop backend.
  const max =
    (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 8;
  const saved = Number(localStorage.getItem("embedded.cores"));
  const cores =
    Number.isFinite(saved) && saved > 0
      ? Math.min(saved, max)
      : Math.min(2, max);
  return { cores, maxCores: max, running: cores, envOverride: false };
}

export async function setEmbeddedCores(cores: number): Promise<EmbeddedConfig> {
  const inv = tauri();
  if (inv) return inv<EmbeddedConfig>("set_embedded_cores", { cores });
  const max =
    (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 8;
  const clamped = Math.min(Math.max(1, Math.round(cores)), max);
  localStorage.setItem("embedded.cores", String(clamped));
  return {
    cores: clamped,
    maxCores: max,
    running: clamped,
    envOverride: false,
  };
}

export async function clusterStatusAt(url: string): Promise<ClusterStatus> {
  const inv = tauri();
  if (inv) return inv<ClusterStatus>("cluster_status_at", { url });
  // Browser mock: a plausible single-node-embedded topology.
  await new Promise((r) => setTimeout(r, 300));
  return {
    generated_ms: Date.now(),
    flight_mode: "disabled",
    workers: {
      total: 2,
      healthy: 2,
      stale: 0,
      slots: 4,
      in_flight: 0,
      flight_capable: 0,
      healthy_flight_capable: 0,
    },
    queries: { total: 12, running: 0, complete: 12, failed: 0 },
    tasks: {
      total: 24,
      pending: 0,
      leased: 0,
      running: 0,
      complete: 24,
      failed: 0,
      retrying: 0,
      abandoned: 0,
    },
    scheduler: { health: "ok", locality_entries: 0 },
    consensus: {
      mode: "local",
      is_leader: true,
      node_id: 1,
      leader_id: 1,
      current_term: 1,
      last_applied_index: 42,
    },
    cost_observations: 24,
  };
}

export async function listWorkers(): Promise<WorkerStatus[]> {
  const inv = tauri();
  if (inv) return inv<WorkerStatus[]>("workers_list");
  await new Promise((r) => setTimeout(r, 180));
  return [
    {
      worker_id: "preview-worker-1",
      name: "preview-worker-1",
      base_url: "http://127.0.0.1:7101",
      flight_endpoint: null,
      slots: 2,
      in_flight: 0,
      healthy: true,
      last_seen_ms: Date.now(),
      stale: false,
      labels: { runtime: "preview", zone: "local" },
    },
    {
      worker_id: "preview-worker-2",
      name: "preview-worker-2",
      base_url: "http://127.0.0.1:7102",
      flight_endpoint: null,
      slots: 2,
      in_flight: 0,
      healthy: true,
      last_seen_ms: Date.now(),
      stale: false,
      labels: { runtime: "preview", zone: "local" },
    },
  ];
}

// P6 cluster bootstrap wizard. The desktop backend owns the generator
// (`bootstrap.rs`, cargo-tested); this browser fallback mirrors it for preview.
export async function generateBootstrap(
  plan: ClusterPlan,
): Promise<CommandBlock[]> {
  const inv = tauri();
  if (inv) return inv<CommandBlock[]>("generate_bootstrap", { plan });
  const mtlsEnv = (certBase: string): CommandBlock["env"] => {
    if (!plan.mtls.enabled) return [];
    const e = [{ key: "QUANT_FABRIC_TLS_MODE", value: "files" }];
    if (plan.mtls.caBundle)
      e.push({ key: "QUANT_FABRIC_TLS_CA_BUNDLE", value: plan.mtls.caBundle });
    e.push({ key: "QUANT_FABRIC_TLS_CERT", value: `${certBase}.pem` });
    e.push({ key: "QUANT_FABRIC_TLS_KEY", value: `${certBase}-key.pem` });
    if (plan.mtls.required)
      e.push({ key: "QUANT_FABRIC_TLS_REQUIRED", value: "true" });
    return e;
  };
  let coord = `quant-fabric coordinator --bind ${plan.coordinatorBind}`;
  if (plan.raft.enabled) {
    if (plan.raft.dataDir) coord += ` --raft-data-dir ${plan.raft.dataDir}`;
    if (plan.raft.nodeId != null)
      coord += ` --raft-node-id ${plan.raft.nodeId}`;
    if (plan.raft.peers.length)
      coord += ` --raft-peers ${plan.raft.peers.join(",")}`;
    if (plan.raft.init) coord += " --raft-init";
  }
  const blocks: CommandBlock[] = [
    {
      role: "coordinator",
      title: "Coordinator",
      env: mtlsEnv("coordinator"),
      command: coord,
    },
  ];
  plan.workers.forEach((w, i) => {
    const name = w.name?.trim() || `worker-${i + 1}`;
    let cmd = `quant-fabric worker --coordinator ${plan.coordinatorUrl} --bind ${w.bind} --advertise ${w.advertise} --pull`;
    if (w.slots != null) cmd += ` --slots ${w.slots}`;
    cmd += ` --name ${name}`;
    blocks.push({
      role: "worker",
      title: name,
      env: mtlsEnv(name),
      command: cmd,
    });
  });
  return blocks;
}

// P6 live node ops: set a coordinator's Raft voter membership (full desired set).
export async function setClusterMembership(
  url: string,
  members: RaftMember[],
): Promise<RaftMembershipResponse> {
  const inv = tauri();
  if (inv)
    return inv<RaftMembershipResponse>("set_cluster_membership", {
      url,
      members,
    });
  await new Promise((r) => setTimeout(r, 400));
  return { voters: members.map((m) => m.node_id) };
}

// Persisted UI preferences (template favorites / recents). Desktop → control-plane
// store; browser preview → localStorage, so both persist.
export async function getPref(key: string): Promise<string | null> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<string | null>("get_pref", { key });
    } catch {
      return null;
    }
  }
  if (await serverReady()) {
    try {
      const res = await srvGet<{ value: string | null }>(
        `/prefs/${encodeURIComponent(key)}`,
      );
      return res.value ?? null;
    } catch {
      /* fall through to localStorage */
    }
  }
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export async function setPref(key: string, value: string): Promise<void> {
  const inv = tauri();
  if (inv) {
    try {
      await inv("set_pref", { key, value });
      return;
    } catch {
      /* ignore */
    }
  }
  if (await serverReady()) {
    try {
      await srv<{ ok: boolean }>(`/prefs/${encodeURIComponent(key)}`, {
        method: "PUT",
        body: JSON.stringify({ value }),
      });
      return;
    } catch {
      /* fall through to localStorage */
    }
  }
  try {
    localStorage.setItem(key, value);
  } catch {
    /* ignore */
  }
}

// Control-plane store (ADR-0021): persisted control targets.
export async function listTargets(): Promise<Target[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<Target[]>("list_targets");
    } catch {
      return [];
    }
  }
  if (await serverReady()) {
    try {
      return await srvGet<Target[]>("/targets");
    } catch {
      /* fall through */
    }
  }
  return []; // browser preview: not persisted
}

export async function saveTarget(target: Target): Promise<void> {
  const inv = tauri();
  if (inv) {
    try {
      await inv("save_target", { target });
      return;
    } catch (e) {
      console.warn("save_target failed", e);
    }
  }
  if (await serverReady()) {
    try {
      await srvPost<Target>("/targets", target);
    } catch (e) {
      console.warn("save_target (server) failed", e);
    }
  }
}

export async function removeTarget(id: string): Promise<void> {
  const inv = tauri();
  if (inv) {
    try {
      await inv("delete_target", { id });
      return;
    } catch (e) {
      console.warn("delete_target failed", e);
    }
  }
  if (await serverReady()) {
    try {
      await srvDelete<{ ok: boolean }>(`/targets/${encodeURIComponent(id)}`);
    } catch (e) {
      console.warn("delete_target (server) failed", e);
    }
  }
}

// Jobs + run history (control-plane store).
export async function listJobs(): Promise<Job[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<Job[]>("list_jobs");
    } catch {
      return [];
    }
  }
  if (await serverReady()) {
    return await srvGet<Job[]>("/jobs");
  }
  return [];
}

export async function saveJob(job: Job): Promise<void> {
  const inv = tauri();
  if (inv) {
    await inv("save_job", { job });
    return;
  }
  if (await serverReady()) {
    await srvPost<Job>("/jobs", job);
  }
}

export async function deleteJobRemote(id: string): Promise<void> {
  const inv = tauri();
  if (inv) {
    await inv("delete_job", { id });
    return;
  }
  if (await serverReady()) {
    await srvDelete<{ ok: boolean }>(`/jobs/${encodeURIComponent(id)}`);
  }
}

export async function runJob(jobId: string, targetId: string): Promise<string> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<string>("run_job", { jobId, targetId });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    const res = await srvPost<{ run: JobRun; steps: JobRunStep[] }>(
      `/jobs/${encodeURIComponent(jobId)}/run`,
      {},
    );
    return res.run.id;
  }
  await new Promise((r) => setTimeout(r, 600));
  return `run-preview-${Date.now()}`;
}

export async function dryRunJob(
  source: string,
  steps: JobStep[],
): Promise<RunResult> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<RunResult>("dry_run_job", { source, steps });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    try {
      const res = await srvPost<{ run: JobRun; steps: JobRunStep[] }>(
        "/jobs/dry-run",
        {
          source,
          steps,
        },
      );
      const rowCount = res.steps.reduce((n, s) => n + (s.rowCount ?? 0), 0);
      return {
        status: res.run.status,
        rows: [],
        rowCount,
        elapsedMs: 0,
        error:
          res.run.status === "failed"
            ? res.steps
                .map((s) => s.log)
                .filter(Boolean)
                .join("\n") || "dry run failed"
            : null,
      };
    } catch {
      /* fall through to preview */
    }
  }
  await new Promise((r) => setTimeout(r, 600));
  return {
    status: "complete",
    rows: [{ note: "preview mode — dry run executes in the desktop app" }],
    rowCount: 1,
    elapsedMs: 12,
    error: null,
  };
}

// Per-step output columns (run the chain embedded) so step i's column pickers
// can offer the columns produced by step i-1. Browser preview can't run it.
export async function describeSteps(
  source: string,
  steps: JobStep[],
): Promise<string[][]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<string[][]>("describe_steps", { source, steps });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    try {
      // Server reports planned steps; column inference isn't exposed, so map to
      // empty column lists (preview parity) rather than fabricate columns.
      const planned = await srvPost<JobRunStep[]>("/jobs/describe-steps", {
        job: { definition: { source, steps } },
      });
      return planned.map(() => []);
    } catch {
      /* fall through to preview */
    }
  }
  return steps.map(() => []);
}

export async function listRuns({
  jobId,
  status,
  since,
  limit = 50,
  offset = 0,
}: {
  jobId?: string;
  status?: string;
  since?: string;
  limit?: number;
  offset?: number;
} = {}): Promise<JobRunListResult> {
  const inv = tauri();
  if (inv) {
    try {
      const runs = await inv<JobRun[]>("list_runs", { jobId: jobId ?? null });
      const filtered = status ? runs.filter((run) => run.status === status) : runs;
      const sinceFiltered = since
        ? filtered.filter((run) => String(run.startedAt || "") >= since)
        : filtered;
      return {
        runs: sinceFiltered.slice(offset, offset + limit),
        total: sinceFiltered.length,
        limit,
        offset,
      };
    } catch {
      return { runs: [], total: 0, limit, offset };
    }
  }
  if (await serverReady()) {
    const params = new URLSearchParams();
    if (jobId) params.set("jobId", jobId);
    if (status) params.set("status", status);
    if (since) params.set("since", since);
    params.set("limit", String(limit));
    params.set("offset", String(offset));
    return await srvGet<JobRunListResult>(`/runs?${params.toString()}`);
  }
  return { runs: [], total: 0, limit, offset };
}

export async function listRunSteps(runId: string): Promise<JobRunStep[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<JobRunStep[]>("list_run_steps", { runId });
    } catch {
      return [];
    }
  }
  if (await serverReady()) {
    return await srvGet<JobRunStep[]>(
      `/runs/${encodeURIComponent(runId)}/steps`,
    );
  }
  return [];
}

export async function saveTemplate(
  category: string,
  name: string,
  source: string,
): Promise<void> {
  const inv = tauri();
  if (inv) return inv("save_template", { category, name, source });
  console.info("[mock] saveTemplate", {
    category,
    name,
    length: source.length,
  });
}

export async function inspectSource(path: string): Promise<SourceSchema> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<SourceSchema>("inspect_source", { path });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    try {
      return await srvPost<SourceSchema>("/connectors/inspect", { path });
    } catch {
      /* fall through to preview */
    }
  }
  // Browser preview: a plausible schema so the column pickers demo.
  await new Promise((r) => setTimeout(r, 200));
  return {
    columns: [
      "date",
      "symbol",
      "sector",
      "ret",
      "mkt",
      "smb",
      "hml",
      "price0",
      "price1",
      "vol",
    ].map((name) => ({
      name,
      dataType:
        name === "date"
          ? "DATE"
          : name === "symbol" || name === "sector"
            ? "VARCHAR"
            : "DOUBLE",
    })),
    preview: [],
  };
}

export async function pickDataFile(): Promise<string | null> {
  const inv = tauri();
  if (inv) return inv<string | null>("pick_data_file");
  return null;
}

/**
 * Walk the configured data roots (default: the in-repo `data/` directory; pass
 * `roots` to override) and return a pruned tree of queryable datasets for the
 * Explore data sidebar. Desktop-only — the browser preview has no filesystem,
 * so this returns an empty tree.
 */
export async function listDataTree(
  roots: string[] = [],
): Promise<DataTreeNode[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<DataTreeNode[]>("list_data_tree", { roots });
    } catch {
      return [];
    }
  }
  return [];
}

/** A custom DuckDB quant scalar function, for pack-editor SQL completions. */
export type QuantFunctionInfo = {
  name: string;
  args: string[];
  returns?: string | null;
  category?: string | null;
};

/**
 * The full catalog of custom DuckDB quant functions registered on the engine,
 * for `source_sql` IntelliSense. Desktop-only — in the browser preview there is
 * no engine to enumerate, so this returns an empty list (built-in keywords and
 * aggregates still complete).
 */
export async function listQuantFunctions(): Promise<QuantFunctionInfo[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<QuantFunctionInfo[]>("list_quant_functions");
    } catch {
      /* fall through to empty */
    }
  }
  return [];
}

// ---- Explore notebook (ADR-0029) ----

/** Run a raw DuckDB SQL cell on the warm engine, with an optional data source
 *  materialized as the `input` table. Browser preview synthesizes rows so the
 *  notebook is demoable without the desktop backend. */
export async function runQuery(
  sql: string,
  source: string | null,
): Promise<RunResult> {
  const inv = tauri();
  if (inv) return inv<RunResult>("run_query", { sql, source });
  await new Promise((r) => setTimeout(r, 350));
  if (!sql.trim()) return { status: "failed", rows: [], error: "empty query" };
  const cols = ["group", "value", "n"];
  const rows = Array.from({ length: 5 }, (_, i) =>
    Object.fromEntries(
      cols.map((c, j) => [
        c,
        j === 0 ? `g${i}` : +(Math.random() * 10).toFixed(4),
      ]),
    ),
  );
  return {
    status: "complete",
    rows,
    rowCount: rows.length,
    elapsedMs: 8 + Math.floor(Math.random() * 12),
    error: null,
  };
}

/** Postgres-wire coordinates of the embedded control-plane server, so an
 *  external notebook can attach. Desktop-only; the browser preview returns a
 *  representative loopback example. */
export async function externalEndpointInfo(): Promise<ExternalEndpoint | null> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<ExternalEndpoint>("external_endpoint_info");
    } catch {
      return null;
    }
  }
  return {
    url: "postgresql://postgres@127.0.0.1:5544/template1?sslmode=disable",
    host: "127.0.0.1",
    port: 5544,
    database: "template1",
    user: "postgres",
    pinned: false,
  };
}

// ---- Explore notebook library (ADR-0029) — file-backed notebooks ----
// Desktop → notebook directories on disk; browser preview → a localStorage map
// so the library is still usable (one virtual "Browser storage" directory).

const BROWSER_NOTEBOOK_DIR = "Browser storage";
const BROWSER_NOTEBOOK_KEY = "explore:notebook_files";

function browserNotebooks(): Record<string, string> {
  try {
    const raw = localStorage.getItem(BROWSER_NOTEBOOK_KEY);
    const v = raw ? JSON.parse(raw) : {};
    return v && typeof v === "object" ? (v as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function writeBrowserNotebooks(map: Record<string, string>): void {
  try {
    localStorage.setItem(BROWSER_NOTEBOOK_KEY, JSON.stringify(map));
  } catch {
    /* ignore */
  }
}

/** The writable default notebook directory new notebooks save to. */
export async function defaultNotebookDir(): Promise<string> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<string>("default_notebook_dir");
    } catch {
      /* fall through */
    }
  }
  return BROWSER_NOTEBOOK_DIR;
}

/** List saved notebooks across the default dir plus the extra `dirs` provided. */
export async function listNotebooks(dirs: string[]): Promise<NotebookFile[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<NotebookFile[]>("list_notebooks", { dirs });
    } catch {
      return [];
    }
  }
  const map = browserNotebooks();
  return Object.entries(map)
    .map(([path, contents]) => {
      let name =
        path
          .split("/")
          .pop()
          ?.replace(/\.json$/, "") ?? path;
      let cellCount = 0;
      try {
        const nb = JSON.parse(contents);
        if (nb?.name) name = String(nb.name);
        if (Array.isArray(nb?.cells)) cellCount = nb.cells.length;
      } catch {
        /* keep filename */
      }
      const dir = path.slice(0, path.lastIndexOf("/")) || BROWSER_NOTEBOOK_DIR;
      return { path, dir, name, cellCount, modifiedMs: 0 };
    })
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** Read a saved notebook's raw JSON contents. */
export async function readNotebook(path: string): Promise<string> {
  const inv = tauri();
  if (inv) return inv<string>("read_notebook", { path });
  const map = browserNotebooks();
  if (!(path in map)) throw new Error(`notebook not found: ${path}`);
  return map[path];
}

/** Write a notebook to `<dir>/<name>.json`, returning the absolute path. */
export async function saveNotebook(
  dir: string,
  name: string,
  contents: string,
): Promise<string> {
  const inv = tauri();
  if (inv) return inv<string>("save_notebook", { dir, name, contents });
  const path = `${dir}/${name}.json`;
  const map = browserNotebooks();
  map[path] = contents;
  writeBrowserNotebooks(map);
  return path;
}

/** Delete a saved notebook file. */
export async function deleteNotebook(path: string): Promise<void> {
  const inv = tauri();
  if (inv) return inv("delete_notebook", { path });
  const map = browserNotebooks();
  delete map[path];
  writeBrowserNotebooks(map);
}

/** Native folder picker for adding a directory to the notebook library. */
export async function pickDirectory(): Promise<string | null> {
  const inv = tauri();
  if (inv) return inv<string | null>("pick_directory");
  return null; // browser preview: no filesystem to browse
}

export async function aiChat(
  messages: ChatMessage[],
  context: string,
): Promise<string> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<string>("ai_chat", { messages, context });
    } catch (e) {
      return `AI backend error: ${e}. Set ANTHROPIC_API_KEY for the desktop app.`;
    }
  }
  await new Promise((r) => setTimeout(r, 500));
  const last = messages[messages.length - 1]?.content ?? "";
  return `**(preview mode)** I'd help you with: _${last}_.\n\nConnect the desktop backend (and set \`ANTHROPIC_API_KEY\`) to edit and generate templates with a live model. I can scaffold a \`[template.sql]\` or \`[template.plan]\`, bind inputs, and add a \`[template.example]\` so it passes the execution gate.`;
}

export type SecretsKeeperStatus = {
  ok: boolean;
  initialized: boolean;
  sealed: boolean;
  tokenValid: boolean;
  message: string;
};

/** Probe a secrets-keeper backend (reachability + seal status + token validity).
 *  Desktop only — the browser preview cannot reach a TLS backend, so it returns
 *  a clearly-simulated result. */
export async function secretsKeeperTest(
  url: string,
  token: string,
): Promise<SecretsKeeperStatus> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<SecretsKeeperStatus>("secretskeeper_test", {
        url,
        token,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    try {
      const res = await srvPost<{ status: string; message: string }>(
        "/secrets-keeper/test",
        {
          url,
          token: token || undefined,
        },
      );
      const ok = res.status === "ok" || res.status === "pass";
      return {
        ok,
        initialized: ok,
        sealed: false,
        tokenValid: ok && Boolean(token),
        message: res.message,
      };
    } catch {
      /* fall through to preview */
    }
  }
  await new Promise((r) => setTimeout(r, 300));
  if (!url.trim()) throw new Error("Enter the secrets backend URL.");
  return {
    ok: false,
    initialized: false,
    sealed: false,
    tokenValid: false,
    message: "(preview) Run the desktop app to reach a secrets backend.",
  };
}

/** Store the secrets-keeper bearer token in the OS keychain (desktop, option A);
 *  an empty value clears it. Browser preview keeps it in localStorage so the
 *  flow stays demoable. The raw token is write-only — never read back. */
export async function setSecretsKeeperToken(token: string): Promise<void> {
  const inv = tauri();
  if (inv) {
    await inv("set_secretskeeper_token", { token });
    return;
  }
  if (token.trim()) localStorage.setItem("secretskeeper.token", token.trim());
  else localStorage.removeItem("secretskeeper.token");
}

/** Whether a secrets-keeper token is configured (without revealing it). */
export async function hasSecretsKeeperToken(): Promise<boolean> {
  const inv = tauri();
  if (inv) return inv<boolean>("has_secretskeeper_token");
  return Boolean(localStorage.getItem("secretskeeper.token"));
}

// ---- Connector Store / remote registry (ADR-0016) ----------------------------
// Server-mode functions: the sidecar fetches the online registry manifest and
// cross-references local `hub list`. In the browser preview (no sidecar) these
// fall back to a small bundled seed so the Store still renders offline.

/** Default registry URL used only by the preview seed for display/provenance. */
const REGISTRY_SEED_URL = "https://celeritas.dev/registry.manifest.json";

/** A tiny offline seed so the Store renders in browser preview (no sidecar).
 *  Mirrors the `registry.manifest.json` publication records, normalized. */
const REGISTRY_SEED: RegistryPackage[] = [
  {
    name: "source-postgres-cdc",
    title: "Postgres CDC",
    summary: "Stream change data capture from a PostgreSQL database.",
    description:
      "Logical-replication based change data capture for PostgreSQL. Tails the write-ahead log to emit inserts, updates and deletes as a streaming source with at-least-once delivery and resumable slots.",
    type: "source",
    version: "0.4.1",
    author: {
      name: "Celeritas Labs",
      email: "connectors@celeritas.dev",
      url: "https://celeritas.dev",
    },
    repository: "https://github.com/celeritas-dev/connector-postgres-cdc",
    license: "Apache-2.0",
    artifact: {
      publisher: "cargo",
      package: "celeritas-source-postgres-cdc",
      install_command: "cargo install celeritas-source-postgres-cdc",
    },
    manifest: {
      capabilities: ["incremental", "cdc", "resumable"],
      settings: [
        {
          name: "dsn",
          kind: "secret",
          scope: "connection",
          required: true,
          description: "PostgreSQL connection string.",
        },
        {
          name: "slot",
          kind: "string",
          scope: "connection",
          required: true,
          default: "celeritas",
          description: "Logical replication slot name.",
        },
        {
          name: "publication",
          kind: "string",
          scope: "connection",
          required: false,
          default: "celeritas_pub",
          description: "Publication to subscribe to.",
        },
      ],
    },
    tags: ["postgres", "cdc", "database", "streaming"],
    category: "databases",
    installed: false,
    installCommand: "cargo install celeritas-source-postgres-cdc",
    packagedBy: {
      name: "Celeritas Labs",
      email: "connectors@celeritas.dev",
      url: "https://celeritas.dev",
    },
    packagedAt: "2026-05-20T00:00:00Z",
  },
  {
    name: "target-snowflake",
    title: "Snowflake Loader",
    summary: "Bulk-load tables and streams into Snowflake.",
    description:
      "High-throughput loader for Snowflake using staged file uploads and COPY INTO. Supports schema evolution, MERGE upserts on a key, and warehouse auto-suspend awareness.",
    type: "target",
    version: "1.2.0",
    author: {
      name: "Acme Data",
      email: "oss@acme.example",
      url: "https://acme.example",
    },
    repository: "https://github.com/acme/celeritas-target-snowflake",
    license: "MIT",
    artifact: {
      publisher: "pip",
      package: "celeritas-target-snowflake",
      install_command: "pip install celeritas-target-snowflake",
    },
    manifest: {
      capabilities: ["upsert", "schema-evolution", "bulk-load"],
      settings: [
        {
          name: "account",
          kind: "string",
          scope: "connection",
          required: true,
          description: "Snowflake account identifier.",
        },
        {
          name: "user",
          kind: "string",
          scope: "connection",
          required: true,
          description: "Login user.",
        },
        {
          name: "password",
          kind: "secret",
          scope: "connection",
          required: true,
          description: "Login password or key.",
        },
        {
          name: "warehouse",
          kind: "string",
          scope: "connection",
          required: true,
          description: "Warehouse to run loads on.",
        },
        {
          name: "database",
          kind: "string",
          scope: "connection",
          required: true,
          description: "Target database.",
        },
        {
          name: "merge_key",
          kind: "string",
          scope: "run",
          required: false,
          description: "Column to upsert on.",
        },
      ],
    },
    tags: ["snowflake", "warehouse", "loader"],
    category: "databases",
    installed: false,
    installCommand: "pip install celeritas-target-snowflake",
    packagedBy: {
      name: "Acme Data",
      email: "oss@acme.example",
      url: "https://acme.example",
    },
    packagedAt: "2026-04-02T00:00:00Z",
  },
  {
    name: "source-stripe",
    title: "Stripe",
    summary: "Extract charges, customers and invoices from the Stripe API.",
    description:
      "REST source for the Stripe API with incremental sync by updated timestamp, automatic pagination and rate-limit backoff. Streams charges, customers, invoices, subscriptions and events.",
    type: "source",
    version: "0.9.3",
    author: {
      name: "Fieldsync",
      email: "hello@fieldsync.example",
      url: "https://fieldsync.example",
    },
    repository: "https://github.com/fieldsync/celeritas-source-stripe",
    license: "Apache-2.0",
    artifact: {
      publisher: "pip",
      package: "celeritas-source-stripe",
      install_command: "celeritas hub add celeritas-source-stripe",
    },
    manifest: {
      capabilities: ["incremental", "rest", "pagination"],
      settings: [
        {
          name: "api_key",
          kind: "secret",
          scope: "connection",
          required: true,
          description: "Stripe secret API key.",
        },
        {
          name: "start_date",
          kind: "string",
          scope: "run",
          required: false,
          description: "ISO date to begin sync from.",
        },
        {
          name: "streams",
          kind: "string",
          scope: "run",
          required: false,
          default: "charges,customers,invoices",
          description: "Comma-separated streams.",
        },
      ],
    },
    tags: ["stripe", "api", "payments", "saas"],
    category: "apis",
    installed: false,
    installCommand: "celeritas hub add celeritas-source-stripe",
    packagedBy: {
      name: "Fieldsync",
      email: "hello@fieldsync.example",
      url: "https://fieldsync.example",
    },
    packagedAt: "2026-03-11T00:00:00Z",
  },
  {
    name: "target-s3-parquet",
    title: "S3 Parquet",
    summary: "Write partitioned Parquet datasets to Amazon S3.",
    description:
      "Object-storage loader that writes Hive-partitioned Parquet to an S3 bucket (or any S3-compatible endpoint). Configurable compression, row-group size and partition columns.",
    type: "target",
    version: "0.6.0",
    author: {
      name: "Celeritas Labs",
      email: "connectors@celeritas.dev",
      url: "https://celeritas.dev",
    },
    repository: "https://github.com/celeritas-dev/connector-s3-parquet",
    license: "Apache-2.0",
    artifact: {
      publisher: "cargo",
      package: "celeritas-target-s3-parquet",
      install_command: "cargo install celeritas-target-s3-parquet",
    },
    manifest: {
      capabilities: ["partitioning", "parquet", "compression"],
      settings: [
        {
          name: "bucket",
          kind: "string",
          scope: "connection",
          required: true,
          description: "Destination S3 bucket.",
        },
        {
          name: "prefix",
          kind: "string",
          scope: "connection",
          required: false,
          description: "Key prefix for the dataset.",
        },
        {
          name: "access_key_id",
          kind: "secret",
          scope: "connection",
          required: true,
          description: "AWS access key id.",
        },
        {
          name: "secret_access_key",
          kind: "secret",
          scope: "connection",
          required: true,
          description: "AWS secret access key.",
        },
        {
          name: "partition_by",
          kind: "string",
          scope: "run",
          required: false,
          description: "Comma-separated partition columns.",
        },
        {
          name: "compression",
          kind: "enum",
          scope: "run",
          required: false,
          default: "zstd",
          options: ["zstd", "snappy", "gzip", "none"],
          description: "Parquet compression codec.",
        },
      ],
    },
    tags: ["s3", "parquet", "object-storage", "lake"],
    category: "object-storage",
    installed: false,
    installCommand: "cargo install celeritas-target-s3-parquet",
    packagedBy: {
      name: "Celeritas Labs",
      email: "connectors@celeritas.dev",
      url: "https://celeritas.dev",
    },
    packagedAt: "2026-05-28T00:00:00Z",
  },
];

function seedRegistryResult(opts?: {
  type?: string;
  q?: string;
  category?: string;
}): RegistryListResult {
  const type = opts?.type;
  const q = (opts?.q ?? "").trim().toLowerCase();
  const category = opts?.category;
  const packages = REGISTRY_SEED.filter((p) => {
    if (type && type !== "all" && p.type !== type) return false;
    if (category && category !== "all" && p.category !== category) return false;
    if (q) {
      const hay =
        `${p.name} ${p.title} ${p.summary} ${p.description} ${p.category} ${p.tags.join(" ")}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  return {
    registry: {
      name: "Celeritas Registry (offline seed)",
      url: REGISTRY_SEED_URL,
    },
    packages,
  };
}

function registryQuery(opts?: {
  url?: string;
  type?: string;
  q?: string;
  category?: string;
}): string {
  const params = new URLSearchParams();
  if (opts?.url) params.set("url", opts.url);
  if (opts?.type && opts.type !== "all") params.set("type", opts.type);
  if (opts?.q?.trim()) params.set("q", opts.q.trim());
  if (opts?.category && opts.category !== "all")
    params.set("category", opts.category);
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

/** List packages from the remote registry (sources AND targets). Server mode;
 *  preview falls back to the bundled seed so the Store still renders. */
export async function listRegistry(opts?: {
  url?: string;
  type?: string;
  q?: string;
  category?: string;
}): Promise<RegistryListResult> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<RegistryListResult>("list_registry", {
        url: opts?.url ?? null,
        type: opts?.type ?? null,
        q: opts?.q ?? null,
        category: opts?.category ?? null,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return await srvGet<RegistryListResult>(`/registry${registryQuery(opts)}`);
  }
  return seedRegistryResult(opts);
}

/** Fetch the full detail record for a single package. */
export async function getRegistryPackage(
  name: string,
  url?: string,
): Promise<RegistryPackage | null> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<RegistryPackage | null>("get_registry_package", {
        name,
        url: url ?? null,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    const params = new URLSearchParams({ name });
    if (url) params.set("url", url);
    return await srvGet<RegistryPackage>(
      `/registry/package?${params.toString()}`,
    );
  }
  return REGISTRY_SEED.find((p) => p.name === name) ?? null;
}

/** Install a package by name. Server mode runs the artifact's install command;
 *  preview returns an honest "unavailable" result (nothing is actually run). */
export async function installRegistryPackage(
  name: string,
  url?: string,
): Promise<RegistryInstallResult> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<RegistryInstallResult>("install_registry_package", {
        name,
        url: url ?? null,
      });
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return srvPost<RegistryInstallResult>(
      "/registry/install",
      url ? { name, url } : { name },
    );
  }
  const pkg = REGISTRY_SEED.find((p) => p.name === name);
  return {
    status: "failed",
    message:
      "Install needs the local engine. Start the Celeritas sidecar (or use the desktop app) to install from the store; this is a browser preview.",
    installCommand: pkg?.installCommand ?? "",
    log: "",
  };
}

/** Configured registry URLs (prefs-backed on the server). */
export async function getRegistryUrls(): Promise<string[]> {
  const inv = tauri();
  if (inv) {
    try {
      return await inv<string[]>("get_registry_urls");
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    return await srvGet<string[]>("/registry/urls");
  }
  return [REGISTRY_SEED_URL];
}

/** Persist the configured registry URLs (prefs-backed on the server). */
export async function setRegistryUrls(urls: string[]): Promise<void> {
  const inv = tauri();
  if (inv) {
    try {
      await inv("set_registry_urls", { urls });
      return;
    } catch {
      /* fall through */
    }
  }
  if (await serverReady()) {
    await srv<{ ok: boolean }>("/registry/urls", {
      method: "PUT",
      body: JSON.stringify({ urls }),
    });
  }
}
