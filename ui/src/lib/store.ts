import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import {
  aiChat,
  clusterOverview,
  clusterStatusAt,
  embeddedConfig,
  setEmbeddedCores,
  deleteBinding as deleteBindingRemote,
  deleteConnectorInstanceRecord,
  deleteJobRemote,
  describeSteps,
  dryRunJob,
  generateBootstrap,
  getPref,
  isDesktop,
  setPref,
  setClusterMembership,
  inspectSource,
  linkGithubPack,
  listBindings,
  listContracts,
  listConnectors,
  listJobs,
  listPacks,
  listPackSources,
  listPackSyncEvents,
  listPackVersions,
  listRegistry,
  getRegistryPackage,
  installRegistryPackage,
  getRegistryUrls,
  setRegistryUrls as setRegistryUrlsRemote,
  listRuns,
  listRunSteps,
  listSharedConnectorBindings,
  listTargets,
  listTemplates,
  listWorkers,
  pickDataFile,
  pickDirectory,
  pickPackBundle,
  pickSavePath,
  defaultNotebookDir as defaultNotebookDirRemote,
  listNotebooks,
  readNotebook,
  saveNotebook as saveNotebookRemote,
  deleteNotebook as deleteNotebookRemote,
  readPackFiles,
  writePackFile,
  duplicatePack as duplicatePackRemote,
  createPack as createPackRemote,
  exportPack as exportPackRemote,
  externalEndpointInfo,
  listDataTree,
  importPack as importPackRemote,
  forgetStagedRelation as forgetStagedRelationRemote,
  gcStagedRelations,
  deletePack as deletePackRemote,
  listStagedRelations,
  pinStagedRelation as pinStagedRelationRemote,
  readStagedRelation as readStagedRelationRemote,
  removeTarget,
  runQuery,
  saveTemplate,
  resolveJobPackUpdate as resolveJobPackUpdateRemote,
  runConnectorAction as runConnectorActionRemote,
  runSharedConnectorBinding as runSharedConnectorBindingRemote,
  runJob,
  saveBinding as saveBindingRemote,
  saveConnectorInstanceRecord,
  runTemplate,
  saveJob,
  secretsKeeperTest,
  setSecretsKeeperToken,
  hasSecretsKeeperToken,
  saveTarget,
  syncPackSource as syncPackSourceRemote,
  testConnector as testConnectorRemote,
  updatePackSourceMonitor as updatePackSourceMonitorRemote,
} from "./api";
import {
  defaultRegistryUrl,
  defaultServerBase,
  getServerBase,
  serverHealth,
  setServerBase,
} from "./server";
import {
  newNotebook,
  newCell,
  notebookParamNames,
  substitute,
  toTemplateToml,
  toPackBundle,
  toNotebookJobDraft,
  notebookFromPackJobTemplate,
  slugify,
} from "./notebook";
import type {
  Binding,
  ContractInfo,
  CellKind,
  ClusterPlan,
  ClusterStatus,
  CommandBlock,
  ConnectorAction,
  ConnectorActionResult,
  ConnectorInstanceRecord,
  ConnectorKind,
  ConnectorInstance,
  ConnectorSpec,
  ConnectorTestResult,
  DataTreeNode,
  EmbeddedConfig,
  ExternalEndpoint,
  Notebook,
  NotebookFile,
  NotebookParam,
  JobPackUpdateResolution,
  LinkGithubPackRequest,
  RaftMember,
  RegistryPackage,
  Job,
  JobRun,
  JobRunStep,
  Pack,
  PackFile,
  PackSource,
  PackSourceMonitor,
  PackSyncEvent,
  PackVersion,
  RunResult,
  SharedConnectorBinding,
  SourceColumn,
  StagedRelation,
  Target,
  Template,
  WorkerStatus,
  WorkerPlan,
} from "./types";
import { deriveJobReadiness } from "./job-readiness";
import { describeUiError } from "./ui-errors";

export type View =
  | "explore"
  | "packs"
  | "studio"
  | "observability"
  | "store"
  | "jobs"
  | "connectors"
  | "bindings"
  | "staged"
  | "settings";
export type KindFilter = "all" | "aggregate" | "projection";
export type ConnectorBindRole = "source" | "target";
export type RegistryTypeFilter = "all" | "source" | "target";

interface ConnectorReturnIntent {
  id: string;
  from: "pack-job-template" | "pack-connector-row" | "job-empty-state";
  returnView: "packs" | "jobs";
  specName: string;
  kind: ConnectorKind;
  pack?: {
    packId: string;
    jobTemplateName?: string;
  };
  bind?: {
    jobDraftId: string;
    role: ConnectorBindRole;
  };
}

const FAV_KEY = "templates:favorites";
const RECENT_KEY = "templates:recent";
const RECENT_MAX = 15;
const PANELS_KEY = "studio:panels";
const CONNECTORS_KEY = "connectors:instances";
const NOTEBOOK_KEY = "explore:notebook";
/** Persisted extra notebook directories (beyond the default) for the library. */
const NOTEBOOK_PATHS_KEY = "explore:notebook_paths";
const SERVER_URL_PREF = "ui.server_url";
const THEME_PREF = "ui.theme";
/** Persisted credential read by the desktop `ai_chat` command (OpenAI path). */
const OPENAI_KEY_PREF = "openai.api_key";
/** secrets-keeper backend URL pref (must match the Tauri backend). The bearer
 *  token is not a pref — it lives in the OS keychain (option A). */
const SECRETSKEEPER_URL_PREF = "secretskeeper.url";

export type ToastTone = "ok" | "error" | "info";
export interface Toast {
  id: string;
  tone: ToastTone;
  /** Optional bold lead line; the message reads as the body. */
  title?: string | undefined;
  message: string;
}

interface Store {
  templates: Template[];
  loaded: boolean;
  /** Favorited template names (persisted). */
  favorites: string[];
  /** Recently-used template names, most-recent first (persisted). */
  recent: string[];
  /** Studio left (catalog) sidebar visible (persisted). */
  studioLeftOpen: boolean;
  /** Studio right (AI assistant) panel visible (persisted). */
  studioRightOpen: boolean;
  /** Explore left (notebook library) sidebar visible (persisted). */
  exploreSidebarOpen: boolean;
  /** Explore right resolve/functions panel width in px (persisted). */
  exploreRightWidth: number;
  selectedName: string | null;
  // --- Derived references (computed in-store on dependency change; components
  //     read these stable refs and never recompute — no useMemo needed). ---
  /** Favorited templates resolved + sorted (deps: templates, favorites). */
  favoriteTemplates: Template[];
  /** Recently-used templates resolved + sorted for stable UI display (deps: templates, recent). */
  recentTemplates: Template[];
  /** Full catalog grouped by category (deps: templates). */
  catalogGroups: { cat: string; items: Template[] }[];
  /** Templates grouped by the pack they came from (deps: templates, packs). */
  packGroups: { pack: string; items: Template[] }[];
  /** Distinct category names incl. "All" (deps: templates). */
  categories: string[];
  /** Marketplace grid filtered by category+kind+search (deps: templates, market, search). */
  marketItems: Template[];
  /** Favorites filtered by kind+search for the marketplace (deps: templates, favorites, market, search). */
  marketFavorites: Template[];
  /** Recents filtered by kind+search for the marketplace (deps: templates, recent, market, search). */
  marketRecent: Template[];
  /** Studio sidebar: favorites filtered by search (deps: favorites, templates, search). */
  sidebarFavorites: Template[];
  /** Studio sidebar: recents filtered by search (deps: recent, templates, search). */
  sidebarRecent: Template[];
  /** Studio sidebar: category groups filtered by search (deps: templates, search). */
  sidebarGroups: { cat: string; items: Template[] }[];
  /** Studio sidebar: pack-provenance groups filtered by search (deps: templates, packs, search). */
  sidebarPackGroups: { pack: string; items: Template[] }[];
  view: View;
  /** Which connector role the connectors surface is scoped to (Sources vs Targets). */
  connectorRole: ConnectorBindRole;
  search: string;
  openCategories: Record<string, boolean>;
  run: {
    values: Record<string, string>;
    result: RunResult | null;
    running: boolean;
  };
  ai: {
    messages: { role: "user" | "assistant"; content: string }[];
    draft: string;
    busy: boolean;
  };
  market: { category: string; kind: KindFilter };
  targets: Target[];
  activeTargetId: string | null;
  cluster: {
    status: ClusterStatus | null;
    workers: WorkerStatus[];
    loading: boolean;
    error: string | null;
    /** Progressive enhancement: reveal the operator-grade admin UI (topology,
     *  control plane, membership, bootstrap). Default off in embedded mode. */
    advancedOpen: boolean;
  };
  /** Embedded in-process engine compute config (cores). `draft` is the pending
   *  control value; differs from `config.cores` until saved. */
  embedded: {
    config: EmbeddedConfig | null;
    draft: number | null;
    busy: boolean;
    loaded: boolean;
  };
  jobs: Job[];
  jobsLoading: boolean;
  jobsError: string | null;
  jobDraft: Job | null;
  jobBusy: boolean;
  // --- Cohesive packs (ADR-0027) ---
  /** Discovered packs (manifest + contributed connectors/templates/job templates). */
  packs: Pack[];
  /** connector spec name → pack name, for provenance badges (deps: packs). */
  connectorPack: Record<string, string>;
  /** task-template name → pack name, for provenance badges (deps: packs). */
  templatePack: Record<string, string>;
  // --- Pack editor (IDE-like) ---
  /** Pack id currently open in the editor, or null. */
  editingPackId: string | null;
  /** Editable text files of the open pack. */
  packFiles: PackFile[];
  /** Selected file path in the editor. */
  packFilePath: string | null;
  /** Per-file unsaved-edits flag. */
  packFileDirty: Record<string, boolean>;
  packBusy: boolean;
  packSources: PackSource[];
  packVersions: PackVersion[];
  packSyncEvents: PackSyncEvent[];
  githubPackPanelOpen: boolean;
  githubPackDraft: LinkGithubPackRequest;
  githubPackError: string | null;
  packUpdateBusy: boolean;
  packUpdateError: string | null;
  packUpdateResolution: JobPackUpdateResolution | null;
  // --- Global confirmation modal (risky actions) ---
  confirm: {
    open: boolean;
    title: string;
    message: string;
    confirmLabel: string;
    tone: "danger" | "default";
  };
  // --- Global toast/notification queue (action feedback) ---
  /** Transient toasts, newest last. Rendered by `<Toaster/>` at app root;
   *  raised by `actions.notify(...)` so any mutating action can confirm itself
   *  or surface an error instead of failing silently. */
  toasts: Toast[];
  /** UI-only: whether the open job draws its input from a file or a source
   *  connector. The two are mutually exclusive; not persisted. */
  jobSourceMode: "file" | "connector";
  // --- Connectors (ADR-0026) ---
  /** Connector specs from the TOML library (the available types). */
  connectorSpecs: ConnectorSpec[];
  /** Configured connector instances (persisted via prefs). */
  connectorInstances: ConnectorInstance[];
  /** The instance currently open in the wizard (new or being edited). */
  connectorDraft: ConnectorInstance | null;
  /** Context for a connector being configured from a job/pack setup flow. */
  connectorReturnIntent: ConnectorReturnIntent | null;
  connectorBusy: boolean;
  connectorLoading: boolean;
  connectorError: string | null;
  connectorTestResult: ConnectorTestResult | null;
  connectorActionResult: ConnectorActionResult | null;
  /** UI-only per-job values used for connector preview/actions, not persisted. */
  connectorPreviewJobParams: Record<string, string>;
  connectorCatalogMessage: string | null;
  /** Catalog filter text (name / description / category / driver). */
  connectorSearch: string;
  /** Hide connectors with no installed driver from the catalog. */
  connectorHideUnavailable: boolean;
  // --- Bindings (ADR-0032) ---
  bindings: Binding[];
  /** Repo-owned connector bindings from connectors/bindings/ (ADR 0006). */
  sharedConnectorBindings: SharedConnectorBinding[];
  contracts: ContractInfo[];
  bindingDraft: Binding | null;
  bindingRunParamsJson: string;
  bindingMappingJson: string;
  bindingBusy: boolean;
  bindingMessage: string | null;
  sharedConnectorBusy: string | null;
  sharedConnectorRunName: string | null;
  sharedConnectorRunResult: ConnectorActionResult | null;
  bindingTestResult: ConnectorTestResult | null;
  bindingDiscoverResult: ConnectorActionResult | null;
  bindingPreviewResult: ConnectorActionResult | null;
  // --- Connector Store / remote registry (ADR-0016) ---
  /** Packages fetched from the remote registry (sources AND targets). */
  registryPackages: RegistryPackage[];
  /** A registry fetch is in flight. */
  registryLoading: boolean;
  registryError: string | null;
  /** Free-text store search. */
  registrySearch: string;
  /** Type filter: all / sources / targets. */
  registryTypeFilter: RegistryTypeFilter;
  /** Category filter ("all" = no filter). */
  registryCategoryFilter: string;
  /** The package open in the full-details view, or null for the catalog. */
  selectedPackage: RegistryPackage | null;
  /** Configured registry URLs (prefs-backed). */
  registryUrls: string[];
  /** Per-package install-in-flight flags, keyed by package name. */
  installing: Record<string, boolean>;
  /** Last install result for the selected package's status/log panel. */
  installResult: {
    name: string;
    status: string;
    message: string;
    log: string;
  } | null;
  // --- Staged relations scratchpad (ADR-0033) ---
  stagedRelations: StagedRelation[];
  stagedSelectedId: string | null;
  stagedRows: Record<string, unknown>[];
  stagedBusy: boolean;
  stagedMessage: string | null;
  // Derived (recomputed in-store on dependency change — no useMemo):
  /** Source-kind instances (deps: connectorInstances). */
  sourceInstances: ConnectorInstance[];
  /** Target-kind instances (deps: connectorInstances). */
  targetInstances: ConnectorInstance[];
  /** Enabled source instances — selectable as a job source (deps: instances). */
  enabledSources: ConnectorInstance[];
  /** Enabled target instances — selectable as a job target (deps: instances). */
  enabledTargets: ConnectorInstance[];
  /** Catalog grouped kind → category (deps: connectorSpecs) for the add picker. */
  connectorCatalog: {
    kind: "source" | "target";
    category: string;
    specs: ConnectorSpec[];
  }[];
  /** Spec backing the open draft (deps: connectorDraft, connectorSpecs). */
  connectorDraftSpec: ConnectorSpec | null;
  runs: JobRun[];
  runsTotal: number;
  runsLoading: boolean;
  runsError: string | null;
  openRunId: string | null;
  runSteps: JobRunStep[];
  allRuns: JobRun[];
  allRunsLoading: boolean;
  allRunsError: string | null;
  sourceColumns: SourceColumn[];
  sourceInspecting: boolean;
  /** Output columns of each step (index i = step i's output) — feeds step i+1's
   *  column pickers. Computed by `describeColumns()`. */
  stepColumns: string[][];
  dryResult: RunResult | null;
  dryBusy: boolean;
  /** Sample rows from a "Test source" — previews the configured source (file or
   *  connector) without running the job's steps. */
  sourcePreview: RunResult | null;
  sourcePreviewBusy: boolean;
  bootstrap: { plan: ClusterPlan; blocks: CommandBlock[]; busy: boolean };
  membership: {
    members: RaftMember[];
    voters: number[];
    busy: boolean;
    error: string | null;
  };
  // ---- Explore notebook (ADR-0029) ----
  /** The open notebook document (persisted as the working draft). */
  notebook: Notebook;
  /** Saved-file path of the open notebook, or null when it has never been saved. */
  notebookPath: string | null;
  /** Open notebook has unsaved edits since the last save / open. */
  notebookDirty: boolean;
  /** Writable default directory new notebooks save to (from the backend). */
  defaultNotebookDir: string;
  /** Extra notebook directories the user has added (persisted). */
  notebookPaths: string[];
  /** Saved notebooks discovered across the default dir + extra paths. */
  notebookLibrary: NotebookFile[];
  /** Library grouped by directory for the sidebar (deps: library, paths, default). */
  notebookGroups: { dir: string; isDefault: boolean; items: NotebookFile[] }[];
  /** Pruned tree of queryable datasets under the data roots (Explore sidebar). */
  dataTree: DataTreeNode[];
  /** Data-tree scan in flight. */
  dataTreeBusy: boolean;
  /** Last data-tree scan outcome message (transient). */
  dataTreeMessage: string | null;
  /** Per-cell run state, keyed by cell id (transient, not persisted). */
  cellRuns: Record<string, { result: RunResult | null; running: boolean }>;
  /** Markdown cells currently in edit (vs rendered) mode, keyed by id. */
  cellEditing: Record<string, boolean>;
  /** Focused cell — drives the live TOML resolution panel. */
  activeCellId: string | null;
  /** Right-hand "Resolve to TOML" panel visible. */
  exploreTomlOpen: boolean;
  /** Function name under the cursor in the active Explore SQL cell, if any. */
  exploreFunctionFocus: string | null;
  /** Save / add-to-pack in flight. */
  exploreBusy: boolean;
  /** Last save/add-to-pack outcome message (transient). */
  exploreMessage: string | null;
  /** External Postgres-wire endpoint coordinates (loaded on demand). */
  externalEndpoint: ExternalEndpoint | null;
  // ---- App settings (persisted via prefs) ----
  settings: {
    serverUrl: string;
    serverUrlDraft: string;
    serverOk: boolean | null;
    serverMessage: string | null;
    /** Saved OpenAI API key (the value last persisted to prefs). */
    openaiKey: string;
    /** In-progress edit buffer for the key field. */
    openaiDraft: string;
    /** Save in flight. */
    busy: boolean;
    /** Prefs have been read at least once. */
    loaded: boolean;
    /** Transient confirmation/status line shown after a save. */
    message: string | null;
    /** secrets-keeper backend (ADR-0022 store). The bearer token lives in the OS
     *  keychain (option A) and is write-only from the UI — only `tokenSet` (is one
     *  stored) is read back. The AI key field above may hold a
     *  `secrets-keeper://mount/path#field` reference resolved from here. */
    secretsKeeper: {
      url: string;
      urlDraft: string;
      /** Whether a bearer token is stored in the keychain (never the value). */
      tokenSet: boolean;
      /** Write-only input buffer for setting/replacing the token. */
      tokenDraft: string;
      busy: boolean;
      /** Last test/save status line. */
      message: string | null;
      /** Whether the last test reported a usable (reachable, unsealed) backend. */
      ok: boolean | null;
    };
    registryTestMessage: string | null;
    registryTestOk: boolean | null;
    registryTestUrl: string | null;
    theme: "dark" | "light";
  };
  runtime: {
    engineMode: "desktop" | "server" | "preview";
    bannerVisible: boolean;
    retrying: boolean;
    bannerMessage: string | null;
    narrowLayout: boolean;
    railOpen: boolean;
  };
}

function defaultPlan(): ClusterPlan {
  return {
    coordinatorBind: "0.0.0.0:7000",
    coordinatorUrl: "http://127.0.0.1:7000",
    workers: [
      {
        bind: "0.0.0.0:7101",
        advertise: "http://127.0.0.1:7101",
        name: "",
        slots: 4,
      },
    ],
    raft: {
      enabled: false,
      dataDir: "/var/lib/quantfabric/raft",
      nodeId: 1,
      peers: [],
      init: true,
    },
    mtls: {
      enabled: false,
      caBundle: "/etc/quantfabric/ca.pem",
      required: true,
    },
  };
}

export const store = proxy<Store>({
  templates: [],
  loaded: false,
  favorites: [],
  recent: [],
  studioLeftOpen: true,
  studioRightOpen: true,
  exploreSidebarOpen: true,
  exploreRightWidth: 380,
  selectedName: null,
  favoriteTemplates: [],
  recentTemplates: [],
  catalogGroups: [],
  packGroups: [],
  categories: ["All"],
  marketItems: [],
  marketFavorites: [],
  marketRecent: [],
  sidebarFavorites: [],
  sidebarRecent: [],
  sidebarGroups: [],
  sidebarPackGroups: [],
  view: "connectors",
  connectorRole: "source",
  search: "",
  openCategories: {},
  run: { values: {}, result: null, running: false },
  ai: { messages: [], draft: "", busy: false },
  market: { category: "All", kind: "all" },
  targets: [],
  activeTargetId: null,
  cluster: {
    status: null,
    workers: [],
    loading: false,
    error: null,
    advancedOpen: false,
  },
  embedded: { config: null, draft: null, busy: false, loaded: false },
  jobs: [],
  jobsLoading: false,
  jobsError: null,
  jobDraft: null,
  jobBusy: false,
  packs: [],
  connectorPack: {},
  templatePack: {},
  editingPackId: null,
  packFiles: [],
  packFilePath: null,
  packFileDirty: {},
  packBusy: false,
  packSources: [],
  packVersions: [],
  packSyncEvents: [],
  githubPackPanelOpen: false,
  githubPackDraft: {
    repoUrl: "",
    refName: "main",
    packPath: "",
    monitor: "manual",
    authRef: "",
  },
  githubPackError: null,
  packUpdateBusy: false,
  packUpdateError: null,
  packUpdateResolution: null,
  confirm: {
    open: false,
    title: "",
    message: "",
    confirmLabel: "Confirm",
    tone: "default",
  },
  toasts: [],
  jobSourceMode: "file",
  connectorSpecs: [],
  connectorInstances: [],
  connectorDraft: null,
  connectorReturnIntent: null,
  connectorBusy: false,
  connectorLoading: false,
  connectorError: null,
  connectorTestResult: null,
  connectorActionResult: null,
  connectorPreviewJobParams: {},
  connectorCatalogMessage: null,
  connectorSearch: "",
  connectorHideUnavailable: false,
  bindings: [],
  sharedConnectorBindings: [],
  contracts: [],
  bindingDraft: null,
  bindingRunParamsJson: "",
  bindingMappingJson: "",
  bindingBusy: false,
  bindingMessage: null,
  sharedConnectorBusy: null,
  sharedConnectorRunName: null,
  sharedConnectorRunResult: null,
  bindingTestResult: null,
  bindingDiscoverResult: null,
  bindingPreviewResult: null,
  registryPackages: [],
  registryLoading: false,
  registryError: null,
  registrySearch: "",
  registryTypeFilter: "all",
  registryCategoryFilter: "all",
  selectedPackage: null,
  registryUrls: defaultRegistryUrl() ? [defaultRegistryUrl()] : [],
  installing: {},
  installResult: null,
  stagedRelations: [],
  stagedSelectedId: null,
  stagedRows: [],
  stagedBusy: false,
  stagedMessage: null,
  sourceInstances: [],
  targetInstances: [],
  enabledSources: [],
  enabledTargets: [],
  connectorCatalog: [],
  connectorDraftSpec: null,
  runs: [],
  runsTotal: 0,
  runsLoading: false,
  runsError: null,
  openRunId: null,
  runSteps: [],
  allRuns: [],
  allRunsLoading: false,
  allRunsError: null,
  sourceColumns: [],
  sourceInspecting: false,
  stepColumns: [],
  dryResult: null,
  dryBusy: false,
  sourcePreview: null,
  sourcePreviewBusy: false,
  bootstrap: { plan: defaultPlan(), blocks: [], busy: false },
  membership: {
    members: [{ node_id: 1, base_url: "http://127.0.0.1:7000" }],
    voters: [],
    busy: false,
    error: null,
  },
  notebook: newNotebook(),
  notebookPath: null,
  notebookDirty: false,
  defaultNotebookDir: "",
  notebookPaths: [],
  notebookLibrary: [],
  notebookGroups: [],
  dataTree: [],
  dataTreeBusy: false,
  dataTreeMessage: null,
  cellRuns: {},
  cellEditing: {},
  activeCellId: null,
  exploreTomlOpen: true,
  exploreFunctionFocus: null,
  exploreBusy: false,
  exploreMessage: null,
  externalEndpoint: null,
  settings: {
    serverUrl: "",
    serverUrlDraft: "",
    serverOk: null,
    serverMessage: null,
    openaiKey: "",
    openaiDraft: "",
    busy: false,
    loaded: false,
    message: null,
    secretsKeeper: {
      url: "",
      urlDraft: "",
      tokenSet: false,
      tokenDraft: "",
      busy: false,
      message: null,
      ok: null,
    },
    registryTestMessage: null,
    registryTestOk: null,
    registryTestUrl: null,
    theme: "dark",
  },
  runtime: {
    engineMode: "preview",
    bannerVisible: false,
    retrying: false,
    bannerMessage: null,
    narrowLayout: false,
    railOpen: false,
  },
});

export function activeTarget(): Target | null {
  return store.targets.find((t) => t.id === store.activeTargetId) ?? null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * The embedded cluster binds an ephemeral port (`127.0.0.1:0`) each launch, so
 * its persisted target URL goes stale across restarts (hitting it gives
 * "error sending request … connection refused"). Resolve the live coordinator
 * URL from the running cluster and reconcile the stored "embedded" target
 * (insert if missing, update if changed). Returns the live URL, or null if the
 * cluster isn't ready yet.
 */
async function reconcileEmbeddedUrl(): Promise<string | null> {
  const overview = await clusterOverview();
  const url = overview?.coordinatorUrl;
  if (!url) return null;
  const existing = store.targets.find((t) => t.id === "embedded");
  if (existing) {
    if (existing.url !== url) {
      existing.url = url;
      void saveTarget({
        id: "embedded",
        name: existing.name,
        kind: "embedded",
        url,
      });
    }
  } else {
    const embedded = {
      id: "embedded",
      name: "Embedded (in-app)",
      kind: "embedded" as const,
      url,
    };
    store.targets.unshift(embedded);
    void saveTarget(embedded);
  }
  return url;
}

/** The currently-selected template, derived from `selectedName`. */
export function selectedTemplate(): Template | null {
  return store.templates.find((t) => t.name === store.selectedName) ?? null;
}

const byName = () => new Map(store.templates.map((t) => [t.name, t]));
const templateLabel = (t: Template) => t.shortName || t.displayName || t.name;
const sortByName = (a: Template, b: Template) =>
  templateLabel(a).localeCompare(templateLabel(b));
const searchTerms = (query: string) =>
  query
    .trim()
    .toLowerCase()
    .split(/[\s,._/-]+/)
    .filter(Boolean);
const isSubsequence = (term: string, value: string) => {
  let i = 0;
  for (const ch of value) if (ch === term[i]) i += 1;
  return i === term.length;
};
const termMatches = (term: string, value: string) => {
  if (!term) return true;
  const normalized = value.toLowerCase();
  if (normalized.includes(term)) return true;
  return normalized
    .split(/[\s,._/-]+/)
    .some((part) => term.length >= 3 && isSubsequence(term, part));
};
const templateMatches = (template: Template, query: string) => {
  const terms = searchTerms(query);
  if (terms.length === 0) return true;
  const text = [
    templateLabel(template) ?? "",
    template.name ?? "",
    template.category ?? "",
    template.kind ?? "",
    template.description ?? "",
    ...(template.inputs ?? []).flatMap((input) => [
      input.name ?? "",
      input.kind ?? "",
    ]),
  ].join(" ");
  return terms.every((term) => termMatches(term, text));
};

/** Recompute catalog-derived references. Deps: templates, favorites, recent. */
function recomputeCatalog() {
  const fav = new Set(store.favorites);
  const lookup = byName();
  store.favoriteTemplates = store.templates
    .filter((t) => fav.has(t.name))
    .sort(sortByName);
  store.recentTemplates = store.recent
    .map((n) => lookup.get(n))
    .filter((t): t is Template => !!t)
    .sort(sortByName);
  const by: Record<string, Template[]> = {};
  for (const t of store.templates) (by[t.category] ??= []).push(t);
  store.catalogGroups = Object.keys(by)
    .sort()
    .map((cat) => ({ cat, items: by[cat].slice().sort(sortByName) }));
  store.categories = ["All", ...Object.keys(by).sort()];
  // Pack provenance grouping (deps: templatePack from loadPacks): templates that
  // came from a pack, grouped under the pack's name. Templates with no pack stay
  // only in the category catalog above.
  const byPack: Record<string, Template[]> = {};
  for (const t of store.templates) {
    const pk = store.templatePack[t.name];
    if (pk) (byPack[pk] ??= []).push(t);
  }
  store.packGroups = Object.keys(byPack)
    .sort()
    .map((pack) => ({ pack, items: byPack[pack].slice().sort(sortByName) }));
}

/** Recompute marketplace references. Deps: templates, favorites, recent, market, search. */
function recomputeMarket() {
  const q = store.search.trim();
  const { kind, category } = store.market;
  const match = (t: Template) =>
    (kind === "all" || t.kind === kind) && templateMatches(t, q);
  store.marketItems = store.templates.filter(
    (t) => (category === "All" || t.category === category) && match(t),
  );
  const fav = new Set(store.favorites);
  store.marketFavorites = store.templates
    .filter((t) => fav.has(t.name) && match(t))
    .sort(sortByName);
  store.marketRecent = store.recentTemplates.filter(match).slice(0, 8);
}

/** Recompute Studio sidebar references (search-filtered). Reuses the catalog-
 *  derived base refs. Deps: templates, favorites, recent, search. */
function recomputeSidebar() {
  const q = store.search.trim();
  const m = (t: Template) => templateMatches(t, q);
  store.sidebarFavorites = q
    ? store.favoriteTemplates.filter(m)
    : store.favoriteTemplates.slice();
  store.sidebarRecent = (
    q ? store.recentTemplates.filter(m) : store.recentTemplates
  ).slice(0, 10);
  store.sidebarGroups = q
    ? store.catalogGroups
        .map((g) => ({ cat: g.cat, items: g.items.filter(m) }))
        .filter((g) => g.items.length > 0)
    : store.catalogGroups.map((g) => ({ cat: g.cat, items: g.items.slice() }));
  store.sidebarPackGroups = q
    ? store.packGroups
        .map((g) => ({ pack: g.pack, items: g.items.filter(m) }))
        .filter((g) => g.items.length > 0)
    : store.packGroups.map((g) => ({ pack: g.pack, items: g.items.slice() }));
}

// Wire the derivations to their dependencies: recompute once when an input
// changes (in the store), not on every component render. Catalog runs first so
// sidebar/market reuse its fresh base refs.
subscribeKey(store, "templates", () => {
  recomputeCatalog();
  recomputeMarket();
  recomputeSidebar();
});
subscribeKey(store, "favorites", () => {
  recomputeCatalog();
  recomputeMarket();
  recomputeSidebar();
});
subscribeKey(store, "recent", () => {
  recomputeCatalog();
  recomputeMarket();
  recomputeSidebar();
});
subscribeKey(store, "search", () => {
  recomputeMarket();
  recomputeSidebar();
});
subscribeKey(store, "market", recomputeMarket);

/** Group the notebook library by directory for the Explore sidebar. The default
 *  directory always appears first (and even when empty, so it's a visible save
 *  target); extra paths follow in configured order. Deps: notebookLibrary,
 *  notebookPaths, defaultNotebookDir. */
function recomputeNotebookGroups() {
  const byDir = new Map<string, NotebookFile[]>();
  for (const nb of store.notebookLibrary) {
    const list = byDir.get(nb.dir) ?? [];
    list.push(nb);
    byDir.set(nb.dir, list);
  }
  const order: string[] = [];
  const push = (dir: string) => {
    if (dir && !order.includes(dir)) order.push(dir);
  };
  push(store.defaultNotebookDir);
  store.notebookPaths.forEach(push);
  // Surface any directory the backend returned that we didn't explicitly track.
  for (const dir of byDir.keys()) push(dir);
  store.notebookGroups = order.map((dir) => ({
    dir,
    isDefault: dir === store.defaultNotebookDir,
    items: (byDir.get(dir) ?? [])
      .slice()
      .sort((a, b) => a.name.localeCompare(b.name)),
  }));
}
subscribeKey(store, "notebookLibrary", recomputeNotebookGroups);
subscribeKey(store, "notebookPaths", recomputeNotebookGroups);
subscribeKey(store, "defaultNotebookDir", recomputeNotebookGroups);

// ---- Connectors (ADR-0026) derived state ----

const connectorLabel = (c: ConnectorInstance) => c.name || c.spec;
const sortInstances = (a: ConnectorInstance, b: ConnectorInstance) =>
  connectorLabel(a).localeCompare(connectorLabel(b));

/** Recompute instance-derived refs. Deps: connectorInstances. */
function recomputeConnectorInstances() {
  const all = store.connectorInstances.slice();
  store.sourceInstances = all
    .filter((c) => c.kind === "source")
    .sort(sortInstances);
  store.targetInstances = all
    .filter((c) => c.kind === "target")
    .sort(sortInstances);
  store.enabledSources = store.sourceInstances.filter((c) => c.enabled);
  store.enabledTargets = store.targetInstances.filter((c) => c.enabled);
}

/** Recompute catalog grouping, applying the catalog filter + availability
 *  toggle. Deps: connectorSpecs, connectorSearch, connectorHideUnavailable. */
function recomputeConnectorCatalog() {
  const q = store.connectorSearch.trim().toLowerCase();
  const matches = (s: ConnectorSpec) => {
    if (store.connectorHideUnavailable && !s.available) return false;
    if (!q) return true;
    return (
      s.name.toLowerCase().includes(q) ||
      connectorSpecLabel(s).toLowerCase().includes(q) ||
      s.description.toLowerCase().includes(q) ||
      s.category.toLowerCase().includes(q) ||
      s.driver.toLowerCase().includes(q)
    );
  };
  const groups: Record<string, ConnectorSpec[]> = {};
  for (const s of store.connectorSpecs) {
    if (!matches(s)) continue;
    const key = `${s.kind}::${s.category}`;
    (groups[key] ??= []).push(s);
  }
  store.connectorCatalog = Object.keys(groups)
    .sort()
    .map((key) => {
      const [kind, category] = key.split("::");
      return {
        kind: kind as "source" | "target",
        category,
        specs: groups[key]
          .slice()
          .sort((a, b) =>
            connectorSpecLabel(a).localeCompare(connectorSpecLabel(b)),
          ),
      };
    });
}

/** Recompute the spec backing the open draft. Deps: connectorDraft, connectorSpecs. */
function recomputeConnectorDraftSpec() {
  const d = store.connectorDraft;
  store.connectorDraftSpec = d
    ? (store.connectorSpecs.find((s) => s.name === d.spec) ?? null)
    : null;
}

subscribeKey(store, "connectorInstances", recomputeConnectorInstances);
subscribeKey(store, "connectorSpecs", () => {
  recomputeConnectorCatalog();
  recomputeConnectorDraftSpec();
});
subscribeKey(store, "connectorSearch", recomputeConnectorCatalog);
subscribeKey(store, "connectorHideUnavailable", recomputeConnectorCatalog);
subscribeKey(store, "connectorDraft", recomputeConnectorDraftSpec);

// Store/registry filters refetch the registry (server-side filtering, with a
// client-side seed fallback) when the user changes them — only while on the
// Store view so we don't fetch in the background.
function reloadRegistryIfActive() {
  if (store.view !== "store") return;
  window.clearTimeout(
    (
      reloadRegistryIfActive as typeof reloadRegistryIfActive & {
        timer?: number;
      }
    ).timer,
  );
  (
    reloadRegistryIfActive as typeof reloadRegistryIfActive & { timer?: number }
  ).timer = window.setTimeout(() => {
    void actions.loadRegistry();
  }, 250);
}
subscribeKey(store, "registrySearch", reloadRegistryIfActive);
subscribeKey(store, "registryTypeFilter", reloadRegistryIfActive);
subscribeKey(store, "registryCategoryFilter", reloadRegistryIfActive);

/** Look up a connector spec by name. */
export function connectorSpec(
  name: string | undefined | null,
): ConnectorSpec | null {
  if (!name) return null;
  return store.connectorSpecs.find((s) => s.name === name) ?? null;
}

/** Resolve a connector instance by id. */
export function connectorInstance(
  id: string | undefined | null,
): ConnectorInstance | null {
  if (!id) return null;
  return store.connectorInstances.find((c) => c.id === id) ?? null;
}

function persistConnectors() {
  void setPref(CONNECTORS_KEY, JSON.stringify(store.connectorInstances));
}

function connectorRecord(inst: ConnectorInstance): ConnectorInstanceRecord {
  return {
    id: inst.id,
    driver: inst.driver,
    name: inst.name,
    params: { ...inst.params },
    jobParamDefaults: { ...(inst.jobParamDefaults ?? {}) },
    lock: inst.lock ?? null,
  };
}

function applyConnectorRecord(
  inst: ConnectorInstance,
  record: ConnectorInstanceRecord | null,
): ConnectorInstance {
  if (!record) return inst;
  return {
    ...inst,
    params: { ...record.params },
    jobParamDefaults: { ...(record.jobParamDefaults ?? {}) },
    lock: record.lock ?? null,
  };
}

function notifyAsyncError(error: unknown, fallbackTitle: string) {
  const desc = describeUiError(error);
  actions.notify({
    tone: "error",
    title: desc.title || fallbackTitle,
    message: desc.message,
  });
  return desc.message;
}

function persistNotebook() {
  // Strip the valtio proxy before serializing.
  void setPref(
    NOTEBOOK_KEY,
    JSON.stringify(JSON.parse(JSON.stringify(store.notebook))),
  );
  // Any working-draft edit diverges the open notebook from its saved file.
  store.notebookDirty = true;
}

/** Build the SQL that reads a data-tree file directly, by detected format. The
 *  absolute path is single-quote-escaped for the string literal. */
function dataFileSql(node: DataTreeNode): string {
  const path = node.path.replace(/'/g, "''");
  const reader =
    node.format === "parquet"
      ? `read_parquet('${path}')`
      : node.format === "json"
        ? `read_json_auto('${path}')`
        : node.format === "tsv"
          ? `read_csv_auto('${path}', delim = '\\t')`
          : `read_csv_auto('${path}')`;
  return `SELECT *\nFROM ${reader}\nLIMIT 100`;
}

/** Serialize the open notebook to the JSON we persist on disk (proxy-stripped). */
function notebookJson(): string {
  return JSON.stringify(JSON.parse(JSON.stringify(store.notebook)), null, 2);
}

function persistNotebookPaths() {
  void setPref(NOTEBOOK_PATHS_KEY, JSON.stringify(store.notebookPaths));
}

function persistPanels() {
  void setPref(
    PANELS_KEY,
    JSON.stringify({
      left: store.studioLeftOpen,
      right: store.studioRightOpen,
      explore: store.exploreSidebarOpen,
      exploreRightWidth: store.exploreRightWidth,
    }),
  );
}

function applyTheme(theme: "dark" | "light") {
  if (typeof document === "undefined") return;
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

/** Reconcile the params list against the placeholders used across SQL cells:
 *  add a default param for any new `{{name}}`, keeping existing ones untouched.
 *  (Unused params are kept so the user doesn't lose configured kinds/values.) */
function syncNotebookParams() {
  const used = notebookParamNames(store.notebook);
  const have = new Set(store.notebook.params.map((p) => p.name));
  for (const name of used) {
    if (!have.has(name)) {
      store.notebook.params.push({
        name,
        kind: "Column",
        value: "",
      } as NotebookParam);
    }
  }
}

function humanize(name: string): string {
  return name
    .replace(/_export$/, "")
    .replace(/[_-]+/g, " ")
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

const connectorSpecLabel = (s: ConnectorSpec) => humanize(s.name);

/** A readable default name for a new connector instance. */
function defaultInstanceName(spec: ConnectorSpec): string {
  const verb = spec.kind === "source" ? "Import" : "Export";
  return `${humanize(spec.name)} ${verb}`;
}

/** Default per-job params for an instance, from its spec's `jobParams`. */
function defaultJobParams(
  inst: ConnectorInstance | null,
): Record<string, string> {
  if (!inst) return {};
  const spec = connectorSpec(inst.spec);
  const out: Record<string, string> = {};
  for (const p of spec?.jobParams ?? [])
    out[p.name] = inst.jobParamDefaults?.[p.name] ?? p.default ?? "";
  return out;
}

function isSecretLikeConnectorParam(
  name: string,
  kind: string,
  label?: string | null,
  help?: string | null,
): boolean {
  const haystack = `${name} ${label ?? ""} ${help ?? ""}`.toLowerCase();
  return (
    kind === "secret" ||
    /\b(password|passwd|secret|token|api[_ -]?key|access[_ -]?key|private[_ -]?key|credential)\b/.test(
      haystack,
    )
  );
}

function formatJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function parseStringRecord(
  value: string,
  label: string,
): Record<string, string> {
  let parsed: unknown;
  try {
    parsed = value.trim().length ? JSON.parse(value) : {};
  } catch (e) {
    throw new Error(
      `${label} JSON is invalid: ${e instanceof Error ? e.message : String(e)}`,
      { cause: e },
    );
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} JSON must be an object.`);
  }
  const out: Record<string, string> = {};
  for (const [key, raw] of Object.entries(parsed)) {
    if (typeof raw !== "string") {
      throw new Error(`${label} value for \`${key}\` must be a string.`);
    }
    out[key] = raw;
  }
  return out;
}

function parseOptionalStringRecord(
  value: string,
  label: string,
): Record<string, string> | null {
  if (!value.trim()) return null;
  const parsed = parseStringRecord(value, label);
  return Object.keys(parsed).length ? parsed : null;
}

function syncBindingJsonEditors() {
  store.bindingRunParamsJson = formatJson(store.bindingDraft?.runParams ?? {});
  store.bindingMappingJson = store.bindingDraft?.mapping
    ? formatJson(store.bindingDraft.mapping)
    : "";
}

function defaultConnectorJobParamDefaults(
  spec: ConnectorSpec,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of spec.jobParams) out[p.name] = p.default ?? "";
  return out;
}

function normalizeConnectorJobParamDefaults(
  spec: ConnectorSpec,
  values: Record<string, string> | undefined,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const p of spec.jobParams)
    out[p.name] = values?.[p.name] ?? p.default ?? "";
  return out;
}

function effectiveConnectorPreviewJobParams(
  draft: ConnectorInstance,
): Record<string, string> {
  const spec = connectorSpec(draft.spec);
  const out: Record<string, string> = {};
  for (const p of spec?.jobParams ?? []) {
    const value =
      store.connectorPreviewJobParams[p.name] ??
      draft.jobParamDefaults?.[p.name] ??
      "";
    if (value.trim()) out[p.name] = value;
  }
  return out;
}

function actionMatching(
  spec: ConnectorSpec | null,
  patterns: RegExp[],
): ConnectorAction | null {
  if (!spec) return null;
  return (
    spec.actions.find((action) =>
      patterns.some((pattern) =>
        pattern.test(`${action.name} ${action.label} ${action.flag}`),
      ),
    ) ?? null
  );
}

function inferSchemaFromRows(
  rows: readonly Readonly<Record<string, unknown>>[],
) {
  const names = new Set<string>();
  for (const row of rows) {
    Object.keys(row).forEach((name) => names.add(name));
  }
  return {
    columns: Array.from(names).map((name) => ({
      name,
      dataType: inferColumnType(rows.map((row) => row[name])),
    })),
    preview: rows.slice(0, 20).map((row) => ({ ...row })),
  };
}

function inferColumnType(values: unknown[]): string {
  const value = values.find(
    (candidate) => candidate !== null && candidate !== undefined,
  );
  if (value === undefined) return "unknown";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number")
    return Number.isInteger(value) ? "integer" : "number";
  if (typeof value === "string") {
    return /^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-Z]*)?$/.test(value)
      ? "date"
      : "string";
  }
  return "unknown";
}

/** Pull `[template.example]` arg bindings from TOML source to pre-fill the run form. */
export function parseExample(
  source: string | null | undefined,
): Record<string, string> {
  if (!source) return {};
  const block = source.match(
    /\[template\.example\][\s\S]*?args\s*=\s*\{([\s\S]*?)\}/,
  );
  if (!block) return {};
  const out: Record<string, string> = {};
  const re = /([A-Za-z0-9_]+)\s*=\s*(\[[^\]]*\]|"[^"]*"|[-0-9.eE]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(block[1]))) {
    let v = m[2].trim();
    if (v.startsWith("[")) v = v.slice(1, -1).replace(/"/g, "").trim();
    else if (v.startsWith('"')) v = v.slice(1, -1);
    out[m[1]] = v;
  }
  return out;
}

/** The pending confirm callback lives outside the proxy (functions aren't proxy
 *  state); the store holds only what the modal renders. */
let pendingConfirm: (() => void | Promise<void>) | null = null;

/** Monotonic counter so two toasts raised in the same millisecond stay unique. */
let toastSeq = 0;
let registryLoadSeq = 0;

function connectorCanEnable(
  connector: ConnectorInstance,
  spec: ConnectorSpec | null,
  reliabilityOverride?: boolean,
): boolean {
  if (reliabilityOverride) return true;
  if (!spec?.available) {
    actions.notify({
      tone: "error",
      title: "Driver unavailable",
      message: `Install or link the ${connector.driver} driver before enabling this connector.`,
    });
    return false;
  }
  if (connector.lastTest?.status === "pass") return true;

  const testLabel =
    connector.lastTest?.status === "example"
      ? "the last test was unverified"
      : connector.lastTest?.status === "fail"
        ? "the last test failed"
        : "it has not passed a connection test";

  actions.requestConfirm(
    {
      title: "Enable unverified connector",
      message: `“${connector.name}” can be enabled, but ${testLabel}. Run Test first for a verified connection, or enable anyway if you intentionally want to proceed.`,
      confirmLabel: "Enable anyway",
      tone: "default",
    },
    () => {
      if (store.connectorDraft?.id === connector.id) {
        store.connectorDraft.enabled = true;
        actions.saveConnector({ reliabilityOverride: true });
      } else {
        actions.toggleConnectorEnabled(connector.id, true, {
          reliabilityOverride: true,
        });
      }
    },
  );
  return false;
}

export const actions = {
  async load() {
    const t = await listTemplates();
    store.templates = t;
    const initial =
      t.find((x) => x.kind === "projection" && x.source) ?? t[0] ?? null;
    if (initial) actions.select(initial.name);
    store.loaded = true;
    // Hydrate persisted control targets (control-plane store), then ensure the
    // in-app embedded engine is registered as a default.
    store.targets = await listTargets();
    // The embedded cluster starts asynchronously; retry briefly so its live URL
    // is registered (and any stale persisted URL is corrected) before first use.
    let live = await reconcileEmbeddedUrl();
    for (let i = 0; i < 6 && !live; i++) {
      await sleep(700);
      live = await reconcileEmbeddedUrl();
    }
    if (!store.activeTargetId && store.targets[0])
      store.activeTargetId = store.targets[0].id;
    if (store.activeTargetId) void actions.refreshTopology();
    void actions.loadEmbeddedConfig();
    void actions.loadJobs();
    void actions.loadConnectors();
    void actions.loadContracts();
    void actions.loadPacks().then(() => actions.syncMonitoredPackSources());
    void actions.loadPrefs();
    void actions.loadSettings();
  },

  // ---- Favorites + recently-used (persisted) ----
  async loadPrefs() {
    const parse = (s: string | null): string[] => {
      if (!s) return [];
      try {
        const v = JSON.parse(s);
        return Array.isArray(v)
          ? v.filter((x): x is string => typeof x === "string")
          : [];
      } catch {
        return [];
      }
    };
    const [fav, rec, panels, conns, notebook, nbPaths] = await Promise.all([
      getPref(FAV_KEY),
      getPref(RECENT_KEY),
      getPref(PANELS_KEY),
      getPref(CONNECTORS_KEY),
      getPref(NOTEBOOK_KEY),
      getPref(NOTEBOOK_PATHS_KEY),
    ]);
    store.favorites = parse(fav);
    store.recent = parse(rec);
    store.notebookPaths = parse(nbPaths);
    if (notebook) {
      try {
        const nb = JSON.parse(notebook) as Notebook;
        if (nb && Array.isArray(nb.cells) && nb.cells.length > 0) {
          store.notebook = nb;
          store.activeCellId =
            nb.cells.find((c) => c.kind === "sql")?.id ?? nb.cells[0].id;
          syncNotebookParams();
        }
      } catch {
        /* keep the starter notebook */
      }
    }
    if (conns) {
      try {
        const v = JSON.parse(conns);
        if (Array.isArray(v))
          store.connectorInstances = v as ConnectorInstance[];
      } catch {
        /* keep empty */
      }
    }
    if (panels) {
      try {
        const p = JSON.parse(panels) as {
          left?: boolean;
          right?: boolean;
          explore?: boolean;
          exploreRightWidth?: number;
        };
        if (typeof p.left === "boolean") store.studioLeftOpen = p.left;
        if (typeof p.right === "boolean") store.studioRightOpen = p.right;
        if (typeof p.explore === "boolean")
          store.exploreSidebarOpen = p.explore;
        if (typeof p.exploreRightWidth === "number")
          store.exploreRightWidth = Math.max(
            300,
            Math.min(680, Math.round(p.exploreRightWidth)),
          );
      } catch {
        /* keep defaults */
      }
    }
  },

  toggleStudioLeft(open?: boolean) {
    store.studioLeftOpen = open ?? !store.studioLeftOpen;
    persistPanels();
  },

  toggleStudioRight() {
    store.studioRightOpen = !store.studioRightOpen;
    persistPanels();
  },

  toggleExploreSidebar() {
    store.exploreSidebarOpen = !store.exploreSidebarOpen;
    persistPanels();
  },

  setExploreRightWidth(width: number, persist = true) {
    store.exploreRightWidth = Math.max(300, Math.min(680, Math.round(width)));
    if (persist) persistPanels();
  },

  toggleFavorite(name: string) {
    store.favorites = store.favorites.includes(name)
      ? store.favorites.filter((n) => n !== name)
      : [...store.favorites, name];
    void setPref(FAV_KEY, JSON.stringify(store.favorites));
  },

  recordUsage(name: string) {
    if (!name) return;
    store.recent = [name, ...store.recent.filter((n) => n !== name)].slice(
      0,
      RECENT_MAX,
    );
    void setPref(RECENT_KEY, JSON.stringify(store.recent));
  },

  addTarget(name: string, url: string, token?: string) {
    void token;
    actions.notify({
      tone: "info",
      title: "Remote targets coming soon",
      message: `Target “${name || url}” was not added. This UI currently runs only against the embedded local engine.`,
    });
  },

  removeTarget(id: string) {
    store.targets = store.targets.filter((t) => t.id !== id);
    void removeTarget(id);
    if (store.activeTargetId === id)
      actions.switchTarget(store.targets[0]?.id ?? "");
  },

  switchTarget(id: string) {
    store.activeTargetId = id;
    store.cluster.status = null;
    store.cluster.workers = [];
    void actions.refreshTopology();
  },

  async refreshTopology() {
    const target = activeTarget();
    if (!target) return;
    store.cluster.loading = true;
    store.cluster.error = null;
    try {
      if (target.kind === "remote") {
        store.cluster.status = null;
        store.cluster.workers = [];
        store.cluster.error =
          "Remote targets are coming soon. Use the embedded local engine for now.";
        return;
      }
      // Embedded port is ephemeral — resolve the live URL before querying.
      let url = target.url;
      if (target.kind === "embedded") {
        const live = await reconcileEmbeddedUrl();
        if (live) url = live;
        void actions.loadEmbeddedConfig();
      }
      store.cluster.status = await clusterStatusAt(url);
      try {
        store.cluster.workers = await listWorkers();
      } catch {
        store.cluster.workers = [];
      }
    } catch (e) {
      store.cluster.error = String(e);
      store.cluster.status = null;
      store.cluster.workers = [];
    } finally {
      store.cluster.loading = false;
    }
  },

  // ---- Embedded compute (parallelise over cores) ----
  /** Reveal/hide the operator-grade advanced cluster admin UI. */
  setClusterAdvanced(open: boolean) {
    store.cluster.advancedOpen = open;
  },

  async loadEmbeddedConfig() {
    try {
      const config = await embeddedConfig();
      store.embedded.config = config;
      // Keep an in-flight draft if the user is mid-edit; otherwise sync.
      if (store.embedded.draft === null) store.embedded.draft = config.cores;
      store.embedded.loaded = true;
    } catch {
      // Embedded engine may still be starting; leave prior state.
    }
  },

  /** Update the pending core count (local draft, not yet applied). */
  setEmbeddedCoresDraft(cores: number) {
    const max = store.embedded.config?.maxCores ?? cores;
    store.embedded.draft = Math.min(Math.max(1, Math.round(cores)), max);
  },

  /** Persist the drafted core count. Applies to the engine on next launch. */
  async saveEmbeddedCores() {
    const draft = store.embedded.draft;
    if (draft == null) return;
    store.embedded.busy = true;
    try {
      store.embedded.config = await setEmbeddedCores(draft);
      store.embedded.draft = store.embedded.config.cores;
      actions.notify({
        tone: "ok",
        title: "Compute updated",
        message: `Parallelism set to ${store.embedded.config.cores} core${
          store.embedded.config.cores === 1 ? "" : "s"
        }. Restart the app to apply.`,
      });
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Could not update compute",
        message: String(e),
      });
    } finally {
      store.embedded.busy = false;
    }
  },

  // ---- Bootstrap wizard (P6) ----
  setBootstrap(
    patch: Partial<Pick<ClusterPlan, "coordinatorBind" | "coordinatorUrl">>,
  ) {
    Object.assign(store.bootstrap.plan, patch);
  },
  setBootstrapRaft(patch: Partial<ClusterPlan["raft"]>) {
    Object.assign(store.bootstrap.plan.raft, patch);
  },
  setBootstrapMtls(patch: Partial<ClusterPlan["mtls"]>) {
    Object.assign(store.bootstrap.plan.mtls, patch);
  },
  setRaftPeers(csv: string) {
    store.bootstrap.plan.raft.peers = csv
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  },
  addBootstrapWorker() {
    const n = store.bootstrap.plan.workers.length + 1;
    store.bootstrap.plan.workers.push({
      bind: `0.0.0.0:${7100 + n}`,
      advertise: `http://127.0.0.1:${7100 + n}`,
      name: "",
      slots: 4,
    });
  },
  removeBootstrapWorker(idx: number) {
    store.bootstrap.plan.workers.splice(idx, 1);
  },
  setBootstrapWorker(idx: number, patch: Partial<WorkerPlan>) {
    const w = store.bootstrap.plan.workers[idx];
    if (w) Object.assign(w, patch);
  },
  async runBootstrap() {
    store.bootstrap.busy = true;
    try {
      const plan = JSON.parse(
        JSON.stringify(store.bootstrap.plan),
      ) as ClusterPlan;
      store.bootstrap.blocks = await generateBootstrap(plan);
    } finally {
      store.bootstrap.busy = false;
    }
  },

  // ---- Cluster membership (P6 live node ops) ----
  addMember() {
    const m = store.membership.members;
    m.push({ node_id: (m.at(-1)?.node_id ?? 0) + 1, base_url: "http://" });
  },
  removeMember(idx: number) {
    store.membership.members.splice(idx, 1);
  },
  setMember(idx: number, patch: Partial<RaftMember>) {
    const m = store.membership.members[idx];
    if (m) Object.assign(m, patch);
  },
  async applyMembership() {
    const target = activeTarget();
    if (!target) return;
    store.membership.busy = true;
    store.membership.error = null;
    try {
      const members = JSON.parse(
        JSON.stringify(store.membership.members),
      ) as RaftMember[];
      const res = await setClusterMembership(target.url, members);
      store.membership.voters = res.voters;
      void actions.refreshTopology();
    } catch (e) {
      store.membership.error = String(e);
    } finally {
      store.membership.busy = false;
    }
  },

  // ---- Jobs (P3) ----
  async loadJobs() {
    store.jobsLoading = true;
    store.jobsError = null;
    try {
      store.jobs = await listJobs();
    } catch (e) {
      store.jobsError = notifyAsyncError(e, "Could not load jobs");
    } finally {
      store.jobsLoading = false;
    }
  },

  async loadAllRuns() {
    store.allRunsLoading = true;
    store.allRunsError = null;
    try {
      const result = await listRuns({ limit: 200, offset: 0 });
      store.allRuns = result.runs;
    } catch (e) {
      store.allRunsError = notifyAsyncError(e, "Could not load run telemetry");
    } finally {
      store.allRunsLoading = false;
    }
  },

  async loadObservability() {
    await Promise.all([
      actions.loadJobs(),
      actions.loadAllRuns(),
      actions.loadStagedRelations(),
      actions.loadBindings(),
    ]);
  },

  newJob() {
    store.jobDraft = {
      id: `job-${Date.now()}`,
      name: "New job",
      description: "",
      definition: { source: "", steps: [] },
      enabled: true,
    };
    store.jobSourceMode = "file";
    store.runs = [];
    store.openRunId = null;
    store.runSteps = [];
    store.sourceColumns = [];
    store.stepColumns = [];
    store.dryResult = null;
    store.sourcePreview = null;
  },

  editJob(id: string) {
    const j = store.jobs.find((x) => x.id === id);
    if (!j) return;
    store.jobDraft = JSON.parse(JSON.stringify(j)) as Job;
    store.jobSourceMode =
      j.definition.sourceConnectorId || j.definition.sourceBindingId
        ? "connector"
        : "file";
    store.openRunId = null;
    store.runSteps = [];
    store.dryResult = null;
    store.sourcePreview = null;
    store.sourceColumns = [];
    store.stepColumns = [];
    if (store.jobDraft.definition.source) {
      void actions.inspectSource(store.jobDraft.definition.source);
      void actions.describeColumns();
    }
    void actions.refreshRuns(id);
  },

  setJobField(patch: Partial<Pick<Job, "name" | "description" | "enabled">>) {
    if (store.jobDraft) Object.assign(store.jobDraft, patch);
  },

  async setDraftSource(source: string) {
    if (!store.jobDraft) return;
    store.sourcePreview = null;
    store.jobDraft.definition.source = source;
    await actions.inspectSource(source);
    void actions.describeColumns();
  },

  async inspectSource(path: string) {
    store.dryResult = null;
    if (!path.trim()) {
      store.sourceColumns = [];
      return;
    }
    store.sourceInspecting = true;
    try {
      store.sourceColumns = (await inspectSource(path)).columns.map(
        (column) => ({
          ...column,
        }),
      );
    } catch {
      store.sourceColumns = [];
    } finally {
      store.sourceInspecting = false;
    }
  },

  async browseSource() {
    const path = await pickDataFile();
    if (path) await actions.setDraftSource(path);
  },

  /** Test the configured source on its own — sample rows from the file (a
   *  `SELECT * … LIMIT` head) or the source connector's preview action — without
   *  running any of the job's steps. */
  async testSource() {
    const d = store.jobDraft;
    if (!d) return;
    store.sourcePreviewBusy = true;
    store.sourcePreview = null;
    try {
      if (store.jobSourceMode === "file") {
        const path = d.definition.source.trim();
        if (!path) {
          store.sourcePreview = {
            status: "failed",
            rows: [],
            error: "Choose a source file first.",
          };
          return;
        }
        const schema = await inspectSource(path);
        store.sourceColumns = schema.columns.map((column) => ({ ...column }));
        store.sourcePreview = {
          status: "complete",
          rows: schema.preview,
          rowCount: schema.preview.length,
        };
      } else {
        const binding = d.definition.sourceBindingId
          ? store.bindings.find((b) => b.id === d.definition.sourceBindingId)
          : null;
        const inst = connectorInstance(
          binding?.instanceId ?? d.definition.sourceConnectorId ?? null,
        );
        if (!inst) {
          store.sourcePreview = {
            status: "failed",
            rows: [],
            error: "Select a source binding or source connector first.",
          };
          return;
        }
        const spec = connectorSpec(inst.spec);
        const preview = spec?.actions.find(
          (a) => /preview/i.test(a.flag) || /preview/i.test(a.name),
        );
        if (!preview) {
          store.sourcePreview = {
            status: "failed",
            rows: [],
            error: "This connector exposes no preview action.",
          };
          return;
        }
        const res = await runConnectorActionRemote(
          inst.driver,
          preview.flag,
          { ...inst.params },
          {
            ...(binding?.runParams ?? d.definition.sourceConnectorParams ?? {}),
          },
        );
        const failed = res.status !== "complete";
        store.sourcePreview = {
          status: failed ? "failed" : "complete",
          rows: res.rows,
          rowCount: res.rows.length,
          elapsedMs: res.elapsedMs,
          // Surface the driver note on failure or when it resolved with no rows
          // (e.g. the current mock driver), so the result isn't misleading.
          error: failed || res.rows.length === 0 ? res.message : null,
        };
      }
    } catch (e) {
      store.sourcePreview = { status: "failed", rows: [], error: String(e) };
    } finally {
      store.sourcePreviewBusy = false;
    }
  },

  async dryRun() {
    const d = store.jobDraft;
    if (!d || !d.definition.source) return;
    store.dryBusy = true;
    store.dryResult = null;
    try {
      // Backend chains the steps (each output feeds the next) and returns the
      // final output — no history recorded.
      const steps = JSON.parse(JSON.stringify(d.definition.steps));
      store.dryResult = await dryRunJob(d.definition.source, steps);
      // The chain just ran, so refresh per-step output columns too.
      void actions.describeColumns();
    } finally {
      store.dryBusy = false;
    }
  },

  /** Compute each step's output columns (runs the chain embedded) so downstream
   *  step pickers reflect the real schema. Cheap to call on demand. */
  async describeColumns() {
    const d = store.jobDraft;
    if (!d || !d.definition.source) return;
    const steps = JSON.parse(JSON.stringify(d.definition.steps));
    try {
      store.stepColumns = await describeSteps(d.definition.source, steps);
    } catch {
      store.stepColumns = [];
    }
  },

  setDraftSchedule(schedule: string) {
    if (store.jobDraft) store.jobDraft.definition.schedule = schedule;
  },

  /** Opt a job in/out of scheduled execution. On → seed a sensible default cron
   *  if none is set yet; off → clear the schedule so the job is on-demand. */
  setDraftScheduled(on: boolean) {
    const d = store.jobDraft;
    if (!d) return;
    d.definition.schedule = on
      ? d.definition.schedule?.trim() || "0 2 * * *"
      : undefined;
  },

  /** Toggle a job's enabled flag from the editor and persist immediately so the
   *  jobs list (and the scheduler) reflect it. */
  async toggleDraftEnabled() {
    if (!store.jobDraft) return;
    store.jobDraft.enabled = !store.jobDraft.enabled;
    await actions.saveDraft();
  },

  setDraftSink(sink: string) {
    if (store.jobDraft) store.jobDraft.definition.sink = sink;
  },

  addStep() {
    if (!store.jobDraft) return;
    store.jobDraft.definition.steps.push({
      kind: "unit",
      template: "",
      args: {},
    });
    store.dryResult = null;
  },

  addSqlStep() {
    if (!store.jobDraft) return;
    store.jobDraft.definition.steps.push({
      kind: "sql",
      template: "",
      args: {},
      sql: "SELECT *\nFROM input\nLIMIT 20",
    });
    store.dryResult = null;
    store.stepColumns = [];
  },

  addMarkdownStep() {
    if (!store.jobDraft) return;
    store.jobDraft.definition.steps.push({
      kind: "markdown",
      template: "",
      args: {},
      markdown:
        "## Runbook note\n\nDescribe the operational context for this flow.",
    });
    store.dryResult = null;
  },

  removeStep(idx: number) {
    store.jobDraft?.definition.steps.splice(idx, 1);
  },

  setStepTemplate(idx: number, template: string) {
    const step = store.jobDraft?.definition.steps[idx];
    if (step) {
      step.kind = "unit";
      step.template = template;
      step.sql = undefined;
      step.markdown = undefined;
      step.args = {};
      actions.recordUsage(template);
      store.dryResult = null;
    }
  },

  setStepArg(idx: number, key: string, value: string) {
    const step = store.jobDraft?.definition.steps[idx];
    if (step) {
      step.args[key] = value;
      store.dryResult = null;
    }
  },

  setStepSql(idx: number, sql: string) {
    const step = store.jobDraft?.definition.steps[idx];
    if (!step) return;
    step.kind = "sql";
    step.sql = sql;
    step.template = "";
    step.args = {};
    store.dryResult = null;
    store.stepColumns = [];
  },

  setStepMarkdown(idx: number, markdown: string) {
    const step = store.jobDraft?.definition.steps[idx];
    if (!step) return;
    step.kind = "markdown";
    step.markdown = markdown;
    step.template = "";
    step.args = {};
    store.dryResult = null;
  },

  async saveDraft() {
    if (!store.jobDraft) return;
    const name = store.jobDraft.name;
    try {
      await saveJob(JSON.parse(JSON.stringify(store.jobDraft)) as Job);
      await actions.loadJobs();
      actions.notify({ tone: "ok", message: `Saved job “${name}”.` });
    } catch (e) {
      notifyAsyncError(e, "Save failed");
    }
  },

  async deleteJob(id: string) {
    try {
      await deleteJobRemote(id);
      if (store.jobDraft?.id === id) store.jobDraft = null;
      await actions.loadJobs();
    } catch (e) {
      notifyAsyncError(e, "Delete failed");
    }
  },

  async runDraft() {
    if (!store.jobDraft || !store.activeTargetId) return;
    const readiness = deriveJobReadiness({
      draft: store.jobDraft,
      jobSourceMode: store.jobSourceMode,
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
    });
    if (!readiness.canRun) return;
    const jobName = store.jobDraft.name;
    store.jobBusy = true;
    try {
      if (store.jobDraft.definition.packBinding?.updatePolicy === "track") {
        await actions.applyCompatibleJobPackUpdate(true);
      }
      await saveJob(JSON.parse(JSON.stringify(store.jobDraft)) as Job);
      await actions.loadJobs();
      await runJob(store.jobDraft.id, store.activeTargetId);
      await actions.refreshRuns(store.jobDraft.id);
      actions.notify({
        tone: "ok",
        message: isDesktop()
          ? `Run started for “${jobName}”.`
          : `Simulated run for “${jobName}” (preview build).`,
      });
    } catch (e) {
      notifyAsyncError(e, "Run failed");
    } finally {
      store.jobBusy = false;
    }
  },

  async refreshRuns(jobId: string) {
    await actions.loadMoreRuns(jobId, { reset: true });
  },

  async loadMoreRuns(jobId: string, { reset = false } = {}) {
    store.runsLoading = true;
    store.runsError = null;
    try {
      const nextOffset = reset ? 0 : store.runs.length;
      const result = await listRuns({ jobId, limit: 50, offset: nextOffset });
      store.runs = reset ? result.runs : [...store.runs, ...result.runs];
      store.runsTotal = result.total;
    } catch (e) {
      store.runsError = notifyAsyncError(e, "Could not load runs");
    } finally {
      store.runsLoading = false;
    }
  },

  async openRun(runId: string) {
    store.openRunId = runId;
    store.runsError = null;
    try {
      store.runSteps = await listRunSteps(runId);
    } catch (e) {
      store.runSteps = [];
      store.runsError = notifyAsyncError(e, "Could not load run diagnostics");
    }
  },

  setJobPackUpdatePolicy(policy: "pinned" | "track") {
    const binding = store.jobDraft?.definition.packBinding;
    if (!binding) return;
    binding.updatePolicy = policy;
    store.packUpdateError = null;
    store.packUpdateResolution = null;
  },

  async checkJobPackUpdate() {
    const binding = store.jobDraft?.definition.packBinding;
    if (!binding?.sourceId) return null;
    store.packUpdateBusy = true;
    store.packUpdateError = null;
    try {
      await syncPackSourceRemote(binding.sourceId);
      await actions.loadPacks();
      const resolution = await resolveJobPackUpdateRemote(
        binding.sourceId,
        binding.versionId ?? undefined,
      );
      store.packUpdateResolution = resolution;
      return resolution;
    } catch (e) {
      store.packUpdateError = String(e);
      return null;
    } finally {
      store.packUpdateBusy = false;
    }
  },

  async applyCompatibleJobPackUpdate(silent = false) {
    const draft = store.jobDraft;
    const binding = draft?.definition.packBinding;
    if (!draft || !binding?.sourceId) return false;
    store.packUpdateBusy = true;
    store.packUpdateError = null;
    try {
      if (binding.updatePolicy === "track") {
        await syncPackSourceRemote(binding.sourceId);
        await actions.loadPacks();
      }
      const resolution = await resolveJobPackUpdateRemote(
        binding.sourceId,
        binding.versionId ?? undefined,
      );
      store.packUpdateResolution = resolution;
      const latest = resolution?.latest;
      if (!latest || resolution.status === "current") return true;
      if (
        resolution.status !== "compatible" &&
        resolution.status !== "available"
      ) {
        if (!silent)
          store.packUpdateError =
            "Latest pack version needs review before it can be applied.";
        return false;
      }
      const latestPack = store.packs.find((p) => p.dir === latest.cacheDir);
      const latestJob = latestPack?.jobTemplates.find(
        (j) => j.name === binding.jobTemplate,
      );
      if (!latestPack || !latestJob) {
        if (!silent)
          store.packUpdateError =
            "Latest cached pack does not contain the original job template.";
        return false;
      }
      if (latestJob.inputs.length > 0) {
        if (!silent)
          store.packUpdateError =
            "Latest job template declares new inputs and needs review.";
        return false;
      }
      draft.definition.steps = latestJob.steps.map((step, idx) => {
        const existing = draft.definition.steps[idx];
        const existingArgs =
          existing?.template === step.template ? existing.args : {};
        return {
          template: step.template,
          args: { ...step.args, ...existingArgs } as Record<
            string,
            string | number
          >,
        };
      });
      binding.versionId = latest.id;
      binding.sourceId = latest.sourceId;
      binding.refName = latest.refName;
      binding.commitSha = latest.commitSha;
      binding.contentHash = latest.contentHash;
      binding.packName = latest.packName;
      binding.packId = latest.packId;
      if (!draft.definition.schedule && latestJob.schedule)
        draft.definition.schedule = latestJob.schedule;
      void actions.describeColumns();
      return true;
    } catch (e) {
      if (!silent) store.packUpdateError = String(e);
      return false;
    } finally {
      store.packUpdateBusy = false;
    }
  },

  // ---- Connectors (ADR-0026) ----
  async loadConnectors() {
    store.connectorLoading = true;
    store.connectorError = null;
    try {
      store.connectorSpecs = await listConnectors();
    } catch (e) {
      store.connectorError = notifyAsyncError(e, "Could not load connectors");
    } finally {
      store.connectorLoading = false;
    }
  },

  // ---- Connector Store / remote registry (ADR-0016) ----

  /** Switch to the Store view and load the registry. Optionally pre-filter by
   *  type (used by the Sources/Targets "Search the store" affordance). */
  openStore(type?: RegistryTypeFilter) {
    store.selectedPackage = null;
    store.installResult = null;
    if (type) store.registryTypeFilter = type;
    store.view = "store";
    void actions.loadRegistry();
  },

  /** Fetch the registry manifest (sources AND targets) into the store. Honest:
   *  on failure leaves prior packages and surfaces a toast. */
  async loadRegistry() {
    const seq = ++registryLoadSeq;
    store.registryLoading = true;
    store.registryError = null;
    try {
      const [result, urls] = await Promise.all([
        listRegistry({
          type: store.registryTypeFilter,
          q: store.registrySearch,
          category: store.registryCategoryFilter,
        }),
        store.registryUrls.length
          ? Promise.resolve(store.registryUrls)
          : getRegistryUrls(),
      ]);
      if (seq !== registryLoadSeq) return;
      store.registryPackages = result.packages;
      store.registryUrls = urls;
    } catch (e) {
      if (seq !== registryLoadSeq) return;
      store.registryError = notifyAsyncError(e, "Could not load store");
    } finally {
      if (seq === registryLoadSeq) store.registryLoading = false;
    }
  },

  setRegistrySearch(value: string) {
    store.registrySearch = value;
  },

  setRegistryTypeFilter(type: RegistryTypeFilter) {
    store.registryTypeFilter = type;
  },

  setRegistryCategoryFilter(category: string) {
    store.registryCategoryFilter = category;
  },

  /** Open the full-details view for a package, fetching complete detail. */
  async selectPackage(name: string | null) {
    if (!name) {
      store.selectedPackage = null;
      store.installResult = null;
      return;
    }
    store.installResult = null;
    // Show the list record immediately for a snappy transition…
    const known = store.registryPackages.find((p) => p.name === name) ?? null;
    store.selectedPackage = known;
    // …then enrich with the full detail record from the registry.
    try {
      const full = await getRegistryPackage(name);
      if (full && store.selectedPackage?.name === name)
        store.selectedPackage = full;
    } catch (e) {
      notifyAsyncError(e, "Could not load package");
    }
  },

  /** Install a package; honest result; on success refreshes the local catalog so
   *  the new connector appears in Sources/Targets. */
  async installPackage(name: string) {
    if (store.installing[name]) return;
    store.installing = { ...store.installing, [name]: true };
    try {
      const result = await installRegistryPackage(name);
      store.installResult = {
        name,
        status: result.status,
        message: result.message,
        log: result.log,
      };
      if (result.status === "installed") {
        // Mark installed in-place across the list + the open detail.
        store.registryPackages = store.registryPackages.map((p) =>
          p.name === name ? { ...p, installed: true } : p,
        );
        if (store.selectedPackage?.name === name) {
          store.selectedPackage = { ...store.selectedPackage, installed: true };
        }
        actions.notify({
          tone: "ok",
          title: "Installed",
          message: result.message,
        });
        // The connector should now appear in the local catalog.
        await actions.loadConnectors();
      } else {
        actions.notify({
          tone: "error",
          title: "Install failed",
          message: result.message,
        });
      }
    } catch (e) {
      const message = notifyAsyncError(e, "Install failed");
      store.installResult = { name, status: "failed", message, log: "" };
    } finally {
      const next = { ...store.installing };
      delete next[name];
      store.installing = next;
    }
  },

  async setRegistryUrls(urls: string[]) {
    store.registryUrls = urls;
    try {
      await setRegistryUrlsRemote(urls);
    } catch (e) {
      notifyAsyncError(e, "Could not save registry URLs");
    }
  },

  async loadBindings() {
    try {
      const [bindings, shared] = await Promise.all([
        listBindings(),
        listSharedConnectorBindings(),
      ]);
      store.bindings = bindings;
      store.sharedConnectorBindings = shared;
      store.bindingMessage = null;
    } catch (e) {
      store.bindingMessage = String(e);
    }
  },

  async runSharedConnectorBinding(name: string) {
    store.sharedConnectorBusy = name;
    store.sharedConnectorRunName = name;
    store.sharedConnectorRunResult = null;
    try {
      const result = await runSharedConnectorBindingRemote(name);
      store.sharedConnectorRunResult = result;
      actions.notify({
        tone: result.status === "complete" ? "ok" : "error",
        title:
          result.status === "complete"
            ? "Connector landed"
            : "Connector run failed",
        message: result.message,
      });
      await actions.loadBindings();
    } catch (e) {
      const result = {
        status: "failed",
        message: String(e),
        rows: [],
      } as ConnectorActionResult;
      store.sharedConnectorRunResult = result;
      actions.notify({
        tone: "error",
        title: "Connector run failed",
        message: result.message,
      });
    } finally {
      store.sharedConnectorBusy = null;
    }
  },

  async loadContracts() {
    try {
      store.contracts = await listContracts();
    } catch {
      // Contracts are an optional overlay; a load failure leaves the picker empty.
    }
  },

  async loadStagedRelations() {
    store.stagedBusy = true;
    try {
      store.stagedRelations = await listStagedRelations();
      store.stagedMessage = null;
      if (
        store.stagedSelectedId &&
        !store.stagedRelations.some(
          (relation) => relation.relationId === store.stagedSelectedId,
        )
      ) {
        store.stagedSelectedId = null;
        store.stagedRows = [];
      }
    } catch (e) {
      store.stagedMessage = String(e);
    } finally {
      store.stagedBusy = false;
    }
  },

  async readStagedRelation(id: string) {
    store.stagedBusy = true;
    try {
      store.stagedSelectedId = id;
      store.stagedRows = await readStagedRelationRemote(id);
      store.stagedMessage = null;
    } catch (e) {
      store.stagedRows = [];
      store.stagedMessage = String(e);
    } finally {
      store.stagedBusy = false;
    }
  },

  async pinStagedRelation(id: string) {
    store.stagedBusy = true;
    try {
      await pinStagedRelationRemote(id);
      actions.notify({ tone: "ok", message: "Pinned staged relation." });
      await actions.loadStagedRelations();
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Pin failed",
        message: String(e),
      });
      store.stagedMessage = String(e);
    } finally {
      store.stagedBusy = false;
    }
  },

  async forgetStagedRelation(id: string) {
    store.stagedBusy = true;
    try {
      await forgetStagedRelationRemote(id);
      if (store.stagedSelectedId === id) {
        store.stagedSelectedId = null;
        store.stagedRows = [];
      }
      actions.notify({ tone: "ok", message: "Forgot staged relation." });
      await actions.loadStagedRelations();
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Forget failed",
        message: String(e),
      });
      store.stagedMessage = String(e);
    } finally {
      store.stagedBusy = false;
    }
  },

  async gcEphemeralStagedRelations() {
    store.stagedBusy = true;
    try {
      const removed = await gcStagedRelations();
      actions.notify({
        tone: "ok",
        message: `Cleared ${removed} ephemeral relation${removed === 1 ? "" : "s"}.`,
      });
      await actions.loadStagedRelations();
    } catch (e) {
      actions.notify({ tone: "error", title: "GC failed", message: String(e) });
      store.stagedMessage = String(e);
    } finally {
      store.stagedBusy = false;
    }
  },

  newBinding(instanceId?: string) {
    const inst =
      connectorInstance(instanceId ?? null) ??
      store.connectorInstances.find((c) => c.kind === "source") ??
      store.connectorInstances[0] ??
      null;
    store.bindingDraft = {
      id: `binding-${Date.now()}`,
      name: inst ? `${inst.name} binding` : "New binding",
      instanceId: inst?.id ?? "",
      runParams: inst ? defaultJobParams(inst) : {},
      contractRef: null,
      mapping: null,
      schema: null,
    };
    syncBindingJsonEditors();
    store.bindingMessage = null;
    store.bindingTestResult = null;
    store.bindingDiscoverResult = null;
    store.bindingPreviewResult = null;
  },

  editBinding(id: string) {
    const binding = store.bindings.find((b) => b.id === id);
    store.bindingDraft = binding
      ? (JSON.parse(JSON.stringify(binding)) as Binding)
      : null;
    syncBindingJsonEditors();
    store.bindingMessage = null;
    store.bindingTestResult = null;
    store.bindingDiscoverResult = null;
    store.bindingPreviewResult = null;
  },

  openBinding(id: string) {
    actions.editBinding(id);
    if (store.bindingDraft) store.view = "bindings";
  },

  closeBindingDraft() {
    store.bindingDraft = null;
    store.bindingRunParamsJson = "";
    store.bindingMappingJson = "";
    store.bindingMessage = null;
    store.bindingTestResult = null;
    store.bindingDiscoverResult = null;
    store.bindingPreviewResult = null;
  },

  setBindingField(patch: Partial<Pick<Binding, "name" | "instanceId">>) {
    if (!store.bindingDraft) return;
    const previousInstanceId = store.bindingDraft.instanceId;
    Object.assign(store.bindingDraft, patch);
    if (patch.instanceId && patch.instanceId !== previousInstanceId) {
      store.bindingDraft.runParams = defaultJobParams(
        connectorInstance(patch.instanceId),
      );
      store.bindingRunParamsJson = formatJson(store.bindingDraft.runParams);
    }
    store.bindingMessage = null;
    store.bindingDiscoverResult = null;
    store.bindingPreviewResult = null;
  },

  setBindingRunParam(key: string, value: string) {
    if (!store.bindingDraft) return;
    store.bindingDraft.runParams = {
      ...(store.bindingDraft.runParams ?? {}),
      [key]: value,
    };
    store.bindingRunParamsJson = formatJson(store.bindingDraft.runParams);
    store.bindingMessage = null;
    store.bindingPreviewResult = null;
  },

  setBindingContractRef(value: string) {
    if (!store.bindingDraft) return;
    const trimmed = value.trim();
    store.bindingDraft.contractRef = trimmed.length ? trimmed : null;
    store.bindingMessage = null;
  },

  setBindingRunParamsJson(value: string) {
    if (!store.bindingDraft) return;
    store.bindingRunParamsJson = value;
    try {
      store.bindingDraft.runParams = parseStringRecord(value, "Run parameters");
      store.bindingMessage = null;
      store.bindingPreviewResult = null;
    } catch (e) {
      store.bindingMessage = String(e);
    }
  },

  setBindingMappingJson(value: string) {
    if (!store.bindingDraft) return;
    store.bindingMappingJson = value;
    try {
      store.bindingDraft.mapping = parseOptionalStringRecord(value, "Mapping");
      store.bindingMessage = null;
    } catch (e) {
      store.bindingMessage = String(e);
    }
  },

  async testBindingConnectivity() {
    const draft = store.bindingDraft;
    const inst = connectorInstance(draft?.instanceId);
    if (!draft || !inst) return;
    store.bindingBusy = true;
    store.bindingTestResult = null;
    try {
      store.bindingTestResult = await testConnectorRemote(
        inst.kind,
        inst.driver,
        {
          ...inst.params,
        },
      );
    } catch (e) {
      store.bindingTestResult = { status: "fail", message: String(e) };
    } finally {
      store.bindingBusy = false;
    }
  },

  async discoverBinding() {
    const draft = store.bindingDraft;
    const inst = connectorInstance(draft?.instanceId);
    const spec = connectorSpec(inst?.spec);
    const action = actionMatching(spec, [/discover/i, /explain/i, /schema/i]);
    if (!draft || !inst || !action) return;
    store.bindingBusy = true;
    store.bindingDiscoverResult = null;
    try {
      store.bindingDiscoverResult = await runConnectorActionRemote(
        inst.driver,
        action.flag,
        { ...inst.params },
        { ...(draft.runParams ?? {}) },
      );
      draft.schema = inferSchemaFromRows(store.bindingDiscoverResult.rows);
      actions.notify({
        tone: "ok",
        message: `Cached ${draft.schema.columns.length} discovered column${draft.schema.columns.length === 1 ? "" : "s"} for “${draft.name}”.`,
      });
    } catch (e) {
      store.bindingDiscoverResult = {
        status: "failed",
        message: String(e),
        rows: [],
      };
      actions.notify({
        tone: "error",
        title: "Discovery failed",
        message: String(e),
      });
    } finally {
      store.bindingBusy = false;
    }
  },

  async previewBinding() {
    const draft = store.bindingDraft;
    const inst = connectorInstance(draft?.instanceId);
    const spec = connectorSpec(inst?.spec);
    const action = actionMatching(spec, [/preview/i, /sample/i, /fetch/i]);
    if (!draft || !inst || !action) return;
    store.bindingBusy = true;
    store.bindingPreviewResult = null;
    try {
      store.bindingPreviewResult = await runConnectorActionRemote(
        inst.driver,
        action.flag,
        { ...inst.params },
        { ...(draft.runParams ?? {}) },
      );
      draft.schema = inferSchemaFromRows(store.bindingPreviewResult.rows);
      actions.notify({
        tone: "ok",
        message: `Cached ${draft.schema.columns.length} preview column${draft.schema.columns.length === 1 ? "" : "s"} for “${draft.name}”.`,
      });
    } catch (e) {
      store.bindingPreviewResult = {
        status: "failed",
        message: String(e),
        rows: [],
      };
      actions.notify({
        tone: "error",
        title: "Preview failed",
        message: String(e),
      });
    } finally {
      store.bindingBusy = false;
    }
  },

  async saveBinding() {
    const draft = store.bindingDraft;
    if (!draft) return;
    const inst = connectorInstance(draft.instanceId);
    if (!draft.name.trim() || !inst) {
      store.bindingMessage =
        "Choose a connector instance and name the binding.";
      return;
    }
    try {
      draft.runParams = parseStringRecord(
        store.bindingRunParamsJson,
        "Run parameters",
      );
      draft.mapping = parseOptionalStringRecord(
        store.bindingMappingJson,
        "Mapping",
      );
    } catch (e) {
      store.bindingMessage = String(e);
      return;
    }
    store.bindingBusy = true;
    try {
      const savedRecord = await saveConnectorInstanceRecord(
        connectorRecord(inst),
      );
      if (savedRecord) {
        const next = applyConnectorRecord(inst, savedRecord);
        store.connectorInstances = store.connectorInstances.map((c) =>
          c.id === inst.id ? next : c,
        );
      }
      await saveBindingRemote(JSON.parse(JSON.stringify(draft)) as Binding);
      await actions.loadBindings();
      store.bindingDraft = null;
    } catch (e) {
      store.bindingMessage = String(e);
    } finally {
      store.bindingBusy = false;
    }
  },

  async deleteBinding(id: string) {
    store.bindingBusy = true;
    try {
      await deleteBindingRemote(id);
      if (store.bindingDraft?.id === id) store.bindingDraft = null;
      await actions.loadBindings();
    } catch (e) {
      store.bindingMessage = String(e);
    } finally {
      store.bindingBusy = false;
    }
  },

  setConnectorSearch(q: string) {
    store.connectorSearch = q;
    store.connectorCatalogMessage = null;
  },

  toggleConnectorHideUnavailable(on?: boolean) {
    store.connectorHideUnavailable = on ?? !store.connectorHideUnavailable;
    store.connectorCatalogMessage = null;
  },

  linkGithubConnectorCatalog() {
    store.connectorCatalogMessage =
      "GitHub connector linking is not implemented yet.";
  },

  // ---- Cohesive packs (ADR-0027) ----
  async loadPacks() {
    store.packs = await listPacks();
    store.packSources = await listPackSources();
    store.packVersions = await listPackVersions();
    store.packSyncEvents = await listPackSyncEvents();
    const cp: Record<string, string> = {};
    const tp: Record<string, string> = {};
    for (const p of store.packs) {
      for (const c of p.connectors) cp[c.name] = p.name;
      for (const t of p.templates) tp[t.name] = p.name;
    }
    store.connectorPack = cp;
    store.templatePack = tp;
    // Provenance just changed — refresh the pack-grouped catalog + sidebar views.
    recomputeCatalog();
    recomputeSidebar();
  },

  async linkGithubPack(request: LinkGithubPackRequest) {
    store.packBusy = true;
    try {
      const source = await linkGithubPack(request);
      await actions.loadPacks();
      return source;
    } finally {
      store.packBusy = false;
    }
  },

  async syncPackSource(sourceId: string) {
    store.packBusy = true;
    try {
      const version = await syncPackSourceRemote(sourceId);
      await actions.loadPacks();
      void actions.loadConnectors();
      listTemplates().then((templates) => {
        store.templates = templates;
      });
      return version;
    } finally {
      store.packBusy = false;
    }
  },

  async setPackSourceMonitor(sourceId: string, monitor: PackSourceMonitor) {
    store.packBusy = true;
    try {
      await updatePackSourceMonitorRemote(sourceId, monitor);
      if (monitor === "startup") {
        try {
          await syncPackSourceRemote(sourceId);
        } catch {
          // The source row will show the recorded failure after reload.
        }
      }
      await actions.loadPacks();
      if (monitor === "startup") {
        void actions.loadConnectors();
        listTemplates().then((templates) => {
          store.templates = templates;
        });
      }
    } finally {
      store.packBusy = false;
    }
  },

  async syncMonitoredPackSources() {
    const sources = store.packSources.filter(
      (source) => source.monitor === "startup",
    );
    if (sources.length === 0) return;
    store.packBusy = true;
    try {
      for (const source of sources) {
        try {
          await syncPackSourceRemote(source.id);
        } catch {
          // sync_pack_source records the failed sync event and source message.
        }
      }
      await actions.loadPacks();
      void actions.loadConnectors();
      listTemplates().then((templates) => {
        store.templates = templates;
      });
    } finally {
      store.packBusy = false;
    }
  },

  toggleGithubPackPanel(open?: boolean) {
    store.githubPackPanelOpen = open ?? !store.githubPackPanelOpen;
    store.githubPackError = null;
  },

  setGithubPackDraft(patch: Partial<LinkGithubPackRequest>) {
    store.githubPackDraft = { ...store.githubPackDraft, ...patch };
    store.githubPackError = null;
  },

  async linkAndSyncGithubPack() {
    const draft = store.githubPackDraft;
    if (!draft.repoUrl?.trim()) {
      store.githubPackError = "Repository URL is required.";
      return;
    }
    store.packBusy = true;
    store.githubPackError = null;
    try {
      const source = await linkGithubPack({
        repoUrl: draft.repoUrl.trim(),
        refName: draft.refName?.trim() || "main",
        packPath: draft.packPath?.trim() || "",
        monitor: draft.monitor || "manual",
        authRef: draft.authRef?.trim() || null,
      });
      if (source) await syncPackSourceRemote(source.id);
      store.githubPackDraft = {
        repoUrl: "",
        refName: "main",
        packPath: "",
        monitor: "manual",
        authRef: "",
      };
      store.githubPackPanelOpen = false;
      await actions.loadPacks();
      void actions.loadConnectors();
      listTemplates().then((templates) => {
        store.templates = templates;
      });
    } catch (e) {
      store.githubPackError = String(e);
      await actions.loadPacks();
    } finally {
      store.packBusy = false;
    }
  },

  /** Scaffold a runnable job draft from a pack's job template and open it in the
   *  job editor: source connector resolved to an enabled instance, steps/args
   *  and any default schedule pre-wired. The user finishes in the editor. */
  instantiateJobTemplate(packId: string, jobName: string) {
    const pack = store.packs.find((p) => p.id === packId);
    const jt = pack?.jobTemplates.find((j) => j.name === jobName);
    if (!pack || !jt) return;

    let mode: "file" | "connector" = "file";
    let source = "";
    let sourceConnectorId: string | undefined;
    let sourceConnectorParams: Record<string, string> | undefined;
    if (jt.source.connector) {
      mode = "connector";
      const inst = store.connectorInstances.find(
        (c) =>
          c.spec === jt.source.connector && c.kind === "source" && c.enabled,
      );
      if (inst) {
        sourceConnectorId = inst.id;
        sourceConnectorParams = defaultJobParams(inst);
      }
    } else if (jt.source.file) {
      source = jt.source.file;
    }

    const steps = jt.steps.map((s) => ({
      template: s.template,
      args: { ...s.args } as Record<string, string | number>,
    }));
    const version =
      pack.origin === "github"
        ? store.packVersions.find(
            (v) => v.cacheDir === pack.dir || v.packId === pack.id,
          )
        : null;
    store.jobDraft = {
      id: `job-${Date.now()}`,
      name: humanize(jt.name),
      description: jt.description,
      definition: {
        source,
        steps,
        schedule: jt.schedule ?? undefined,
        packBinding: {
          origin: pack.origin,
          packId: version?.packId ?? pack.id,
          packName: pack.name,
          jobTemplate: jt.name,
          sourceId: version?.sourceId ?? null,
          versionId: version?.id ?? null,
          refName: version?.refName ?? null,
          commitSha: version?.commitSha ?? null,
          contentHash: version?.contentHash ?? null,
          // New jobs are pinned to the pack version they were created from so a
          // run never silently changes what it executes (reproducible by
          // default, matching ADR-0033). Switching a job to "track" — to follow
          // remote pack updates — is an explicit user choice via
          // `setJobPackUpdatePolicy`, not a creation-time default.
          updatePolicy: "pinned",
        },
        sourceConnectorId,
        sourceConnectorParams,
      },
      enabled: true,
    };
    store.jobSourceMode = mode;
    store.runs = [];
    store.openRunId = null;
    store.runSteps = [];
    store.sourceColumns = [];
    store.stepColumns = [];
    store.dryResult = null;
    store.sourcePreview = null;
    for (const s of steps) if (s.template) actions.recordUsage(s.template);
    if (mode === "connector" && jt.source.connector && !sourceConnectorId) {
      const spec = connectorSpec(jt.source.connector);
      if (spec) {
        actions.newConnectorForReturn(spec.name, {
          from: "pack-job-template",
          returnView: "jobs",
          specName: spec.name,
          kind: "source",
          pack: { packId: pack.id, jobTemplateName: jt.name },
          bind: { jobDraftId: store.jobDraft.id, role: "source" },
        });
        return;
      }
    }
    store.view = "jobs";
  },

  // ---- Pack editor (IDE-like) + import / export ----
  async openPackEditor(id: string) {
    store.editingPackId = id;
    store.packFileDirty = {};
    store.packFiles = await readPackFiles(id);
    store.packFilePath = store.packFiles[0]?.path ?? null;
  },
  closePackEditor() {
    store.editingPackId = null;
    store.packFiles = [];
    store.packFilePath = null;
    store.packFileDirty = {};
  },
  selectPackFile(path: string) {
    store.packFilePath = path;
  },
  setPackFileContents(path: string, contents: string) {
    const f = store.packFiles.find((x) => x.path === path);
    if (!f) return;
    f.contents = contents;
    store.packFileDirty = { ...store.packFileDirty, [path]: true };
  },
  async savePackFile(path: string) {
    const id = store.editingPackId;
    const f = store.packFiles.find((x) => x.path === path);
    if (!id || !f) return;
    store.packBusy = true;
    try {
      await writePackFile(id, path, f.contents);
      const rest = { ...store.packFileDirty };
      delete rest[path];
      store.packFileDirty = rest;
      // Manifest / artifact edits can change the catalog — refresh derived state.
      await actions.loadPacks();
      void actions.loadConnectors();
    } finally {
      store.packBusy = false;
    }
  },
  /** Copy the pack to `<id>-copy` and open the copy for editing. */
  async duplicatePack(id: string) {
    const pack = store.packs.find((p) => p.id === id);
    if (!pack) return;
    store.packBusy = true;
    try {
      const copy = await duplicatePackRemote(
        id,
        `${id}-copy`,
        `${pack.name} (copy)`,
      );
      await actions.loadPacks();
      if (copy) await actions.openPackEditor(copy.id);
      actions.notify({ tone: "ok", message: `Duplicated “${pack.name}”.` });
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Duplicate failed",
        message: String(e),
      });
    } finally {
      store.packBusy = false;
    }
  },
  async exportPack(id: string) {
    const dest = await pickSavePath(`${id}.qfpack`);
    if (!dest) return;
    store.packBusy = true;
    try {
      await exportPackRemote(id, dest);
      actions.notify({ tone: "ok", message: `Exported pack to ${dest}.` });
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Export failed",
        message: String(e),
      });
    } finally {
      store.packBusy = false;
    }
  },
  async importPack() {
    const src = await pickPackBundle();
    if (!src) return;
    store.packBusy = true;
    try {
      const pack = await importPackRemote(src);
      await actions.loadPacks();
      void actions.loadConnectors();
      if (pack) {
        await actions.openPackEditor(pack.id);
        actions.notify({
          tone: "ok",
          message: `Imported pack “${pack.name}”.`,
        });
      } else {
        actions.notify({
          tone: "info",
          message: "Pack import is only available in the desktop app.",
        });
      }
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Import failed",
        message: String(e),
      });
    } finally {
      store.packBusy = false;
    }
  },
  async deletePack(id: string) {
    const name = store.packs.find((p) => p.id === id)?.name ?? id;
    store.packBusy = true;
    try {
      await deletePackRemote(id);
      if (store.editingPackId === id) actions.closePackEditor();
      await actions.loadPacks();
      void actions.loadConnectors();
      actions.notify({ tone: "ok", message: `Deleted pack “${name}”.` });
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Delete failed",
        message: String(e),
      });
    } finally {
      store.packBusy = false;
    }
  },

  // ---- Global confirmation modal (risky actions) ----
  /** Open the confirm modal; `onConfirm` runs only if the user confirms. */
  requestConfirm(
    opts: {
      title: string;
      message: string;
      confirmLabel?: string;
      tone?: "danger" | "default";
    },
    onConfirm: () => void | Promise<void>,
  ) {
    pendingConfirm = onConfirm;
    store.confirm = {
      open: true,
      title: opts.title,
      message: opts.message,
      confirmLabel: opts.confirmLabel ?? "Confirm",
      tone: opts.tone ?? "danger",
    };
  },
  async resolveConfirm(ok: boolean) {
    const fn = pendingConfirm;
    pendingConfirm = null;
    store.confirm = { ...store.confirm, open: false };
    if (ok && fn) await fn();
  },

  /** Raise a transient toast. `error` toasts persist until dismissed; `ok`/`info`
   *  auto-dismiss. Returns the toast id so a caller can dismiss it early. */
  notify(input: {
    tone?: ToastTone | undefined;
    title?: string | undefined;
    message: string;
  }): string {
    const id = `toast-${Date.now()}-${++toastSeq}`;
    const tone = input.tone ?? "info";
    store.toasts = [
      ...store.toasts.slice(-3),
      { id, tone, title: input.title, message: input.message },
    ];
    if (tone !== "error") {
      setTimeout(() => actions.dismissToast(id), tone === "ok" ? 3500 : 5000);
    }
    return id;
  },
  dismissToast(id: string) {
    store.toasts = store.toasts.filter((t) => t.id !== id);
  },

  /** Open the wizard for a NEW instance of `specName`, prefilled from the
   *  spec's example/defaults. */
  newConnector(specName: string) {
    store.connectorReturnIntent = null;
    const spec = connectorSpec(specName);
    if (!spec) return;
    const params: Record<string, string> = {};
    for (const p of spec.params)
      params[p.name] = spec.example[p.name] ?? p.default ?? "";
    store.connectorDraft = {
      id: `conn-${Date.now()}`,
      spec: spec.name,
      kind: spec.kind,
      driver: spec.driver,
      name: defaultInstanceName(spec),
      params,
      jobParamDefaults: defaultConnectorJobParamDefaults(spec),
      // Saving a connector and enabling it are separate reliability gates.
      // The power toggle enables it after a passing test, or after an explicit
      // user override for unverified connectors.
      enabled: false,
      lastTest: null,
    };
    store.connectorTestResult = null;
    store.connectorActionResult = null;
    store.connectorPreviewJobParams = {};
  },

  newConnectorForReturn(
    specName: string,
    intent: Omit<ConnectorReturnIntent, "id">,
  ) {
    actions.newConnector(specName);
    if (!store.connectorDraft) return;
    store.connectorReturnIntent = {
      ...intent,
      id: `connector-return-${Date.now()}`,
      specName,
    };
    store.view = "connectors";
  },

  editConnector(id: string) {
    store.connectorReturnIntent = null;
    const inst = connectorInstance(id);
    if (!inst) return;
    const draft = JSON.parse(JSON.stringify(inst)) as ConnectorInstance;
    const spec = connectorSpec(draft.spec);
    if (spec) {
      draft.jobParamDefaults = normalizeConnectorJobParamDefaults(
        spec,
        draft.jobParamDefaults,
      );
    }
    store.connectorDraft = draft;
    store.connectorTestResult = inst.lastTest
      ? { status: inst.lastTest.status, message: inst.lastTest.message }
      : null;
    store.connectorActionResult = null;
    store.connectorPreviewJobParams = {};
  },

  openConnector(id: string) {
    actions.editConnector(id);
    if (store.connectorDraft) store.view = "connectors";
  },

  closeConnectorDraft() {
    actions.clearConnectorDraftSecrets();
    store.connectorDraft = null;
    store.connectorReturnIntent = null;
    store.connectorTestResult = null;
    store.connectorActionResult = null;
    store.connectorPreviewJobParams = {};
  },

  cancelConnectorReturn() {
    actions.clearConnectorDraftSecrets();
    const intent = store.connectorReturnIntent;
    store.connectorReturnIntent = null;
    if (intent) store.view = intent.returnView;
  },

  setConnectorName(name: string) {
    if (store.connectorDraft) store.connectorDraft.name = name;
  },

  setConnectorParam(key: string, value: string) {
    if (store.connectorDraft) store.connectorDraft.params[key] = value;
  },
  setConnectorJobParamDefault(key: string, value: string) {
    if (!store.connectorDraft) return;
    store.connectorDraft.jobParamDefaults = {
      ...(store.connectorDraft.jobParamDefaults ?? {}),
      [key]: value,
    };
  },
  setConnectorPreviewJobParam(key: string, value: string) {
    store.connectorPreviewJobParams = {
      ...store.connectorPreviewJobParams,
      [key]: value,
    };
  },

  clearConnectorDraftSecrets() {
    const draft = store.connectorDraft;
    if (!draft) return;
    const spec = connectorSpec(draft.spec);
    if (!spec) return;
    for (const param of [...spec.params, ...spec.jobParams]) {
      if (
        !isSecretLikeConnectorParam(
          param.name,
          param.kind,
          param.label,
          param.help,
        )
      )
        continue;
      const current =
        param.name in draft.params
          ? draft.params[param.name]
          : draft.jobParamDefaults?.[param.name];
      if (
        !current ||
        current.startsWith("secrets-keeper://") ||
        current.startsWith("secret-session://")
      )
        continue;
      if (param.name in draft.params) draft.params[param.name] = "";
      if (draft.jobParamDefaults && param.name in draft.jobParamDefaults) {
        draft.jobParamDefaults[param.name] = "";
      }
      if (param.name in store.connectorPreviewJobParams) {
        const next = { ...store.connectorPreviewJobParams };
        delete next[param.name];
        store.connectorPreviewJobParams = next;
      }
    }
  },

  async testConnectorDraft() {
    const d = store.connectorDraft;
    if (!d) return;
    store.connectorBusy = true;
    store.connectorActionResult = null;
    try {
      const res = await testConnectorRemote(d.kind, d.driver, { ...d.params });
      store.connectorTestResult = res;
      d.lastTest = {
        status: res.status,
        message: res.message,
        at: new Date().toISOString(),
      };
    } catch (e) {
      const message = notifyAsyncError(e, "Connector test failed");
      store.connectorTestResult = { status: "fail", message };
    } finally {
      store.connectorBusy = false;
    }
  },

  async runConnectorActionDraft(action: ConnectorAction) {
    const d = store.connectorDraft;
    if (!d) return;
    store.connectorBusy = true;
    store.connectorActionResult = null;
    try {
      store.connectorActionResult = await runConnectorActionRemote(
        d.driver,
        action.flag,
        { ...d.params },
        effectiveConnectorPreviewJobParams(d),
      );
    } catch (e) {
      const message = notifyAsyncError(e, "Connector action failed");
      store.connectorActionResult = {
        status: "failed",
        message,
        rows: [],
      };
    } finally {
      store.connectorBusy = false;
    }
  },

  saveConnector(opts?: {
    completeReturn?: boolean;
    reliabilityOverride?: boolean;
  }) {
    const d = store.connectorDraft;
    if (!d) return;
    const spec = connectorSpec(d.spec);
    if (d.enabled && !connectorCanEnable(d, spec, opts?.reliabilityOverride))
      return;
    if (spec) {
      d.jobParamDefaults = normalizeConnectorJobParamDefaults(
        spec,
        d.jobParamDefaults,
      );
    }
    const next = store.connectorInstances.filter((c) => c.id !== d.id);
    const saved = JSON.parse(JSON.stringify(d)) as ConnectorInstance;
    next.push(saved);
    store.connectorInstances = next;
    persistConnectors();
    void saveConnectorInstanceRecord(connectorRecord(saved))
      .then((record) => {
        if (!record) return;
        const updated = applyConnectorRecord(saved, record);
        store.connectorInstances = store.connectorInstances.map((c) =>
          c.id === saved.id ? updated : c,
        );
        if (store.connectorDraft?.id === saved.id) {
          store.connectorDraft = JSON.parse(
            JSON.stringify(updated),
          ) as ConnectorInstance;
        }
        persistConnectors();
      })
      .catch((e) => {
        notifyAsyncError(e, "Could not persist connector");
      });
    if (opts?.completeReturn) actions.completeConnectorReturn(saved);
  },

  completeConnectorReturn(saved: ConnectorInstance) {
    const intent = store.connectorReturnIntent;
    if (!intent) return;
    if (intent.specName !== saved.spec || intent.kind !== saved.kind) return;
    if (intent.bind) {
      const draft = store.jobDraft;
      if (!draft || draft.id !== intent.bind.jobDraftId) return;
      if (intent.bind.role === "source") {
        store.jobSourceMode = "connector";
        draft.definition.source = "";
        draft.definition.sourceConnectorId = saved.id;
        draft.definition.sourceConnectorParams = defaultJobParams(saved);
        store.sourceColumns = [];
        store.stepColumns = [];
        store.sourcePreview = null;
        store.dryResult = null;
      } else {
        draft.definition.targetConnectorId = saved.id;
        draft.definition.targetConnectorParams = defaultJobParams(saved);
      }
    }
    store.connectorReturnIntent = null;
    store.connectorDraft = null;
    store.connectorTestResult = null;
    store.connectorActionResult = null;
    store.connectorPreviewJobParams = {};
    store.view = intent.returnView;
  },

  deleteConnector(id: string) {
    store.connectorInstances = store.connectorInstances.filter(
      (c) => c.id !== id,
    );
    if (store.connectorDraft?.id === id) actions.closeConnectorDraft();
    persistConnectors();
    void deleteConnectorInstanceRecord(id).catch((e) => {
      notifyAsyncError(e, "Delete failed");
    });
  },

  /** Enable/disable a saved instance (and keep an open draft in sync). */
  toggleConnectorEnabled(
    id: string,
    enabled?: boolean,
    opts?: { reliabilityOverride?: boolean },
  ) {
    const current =
      store.connectorDraft?.id === id
        ? store.connectorDraft
        : store.connectorInstances.find((c) => c.id === id);
    if (!current) return;
    const nextEnabled = enabled ?? !current.enabled;
    if (
      nextEnabled &&
      !connectorCanEnable(
        current,
        connectorSpec(current.spec),
        opts?.reliabilityOverride,
      )
    ) {
      return;
    }
    store.connectorInstances = store.connectorInstances.map((c) =>
      c.id === id ? { ...c, enabled: nextEnabled } : c,
    );
    if (store.connectorDraft?.id === id) {
      store.connectorDraft.enabled = nextEnabled;
    }
    persistConnectors();
  },

  // ---- Job ↔ connector binding ----
  /** Switch a job's input between a file and a source connector. Clears the
   *  unused side so the two stay mutually exclusive. */
  setJobSourceMode(mode: "file" | "connector") {
    store.jobSourceMode = mode;
    store.sourcePreview = null;
    const d = store.jobDraft;
    if (!d) return;
    if (mode === "file") {
      d.definition.sourceConnectorId = undefined;
      d.definition.sourceBindingId = undefined;
      d.definition.sourceConnectorParams = undefined;
    } else {
      d.definition.source = "";
      store.sourceColumns = [];
      store.stepColumns = [];
    }
  },

  setJobSourceConnector(id: string) {
    const d = store.jobDraft;
    if (!d) return;
    d.definition.sourceConnectorId = id || undefined;
    d.definition.sourceBindingId = undefined;
    d.definition.sourceConnectorParams = id
      ? defaultJobParams(connectorInstance(id))
      : undefined;
    store.sourcePreview = null;
    store.dryResult = null;
  },
  setJobSourceBinding(id: string) {
    const d = store.jobDraft;
    if (!d) return;
    const binding = store.bindings.find((b) => b.id === id);
    d.definition.sourceBindingId = binding?.id || undefined;
    d.definition.sourceConnectorId = undefined;
    d.definition.sourceConnectorParams = undefined;
    if (binding?.schema) {
      store.sourceColumns = binding.schema.columns.map((column) => ({
        ...column,
      }));
      store.sourcePreview = binding.schema.preview.length
        ? {
            status: "complete",
            rows: binding.schema.preview,
            rowCount: binding.schema.preview.length,
          }
        : null;
    } else {
      store.sourceColumns = [];
      store.sourcePreview = null;
    }
    store.dryResult = null;
  },
  setJobSourceParam(key: string, value: string) {
    const d = store.jobDraft;
    if (!d) return;
    d.definition.sourceConnectorParams = {
      ...(d.definition.sourceConnectorParams ?? {}),
      [key]: value,
    };
    store.sourcePreview = null;
    store.dryResult = null;
  },
  setJobTargetConnector(id: string) {
    const d = store.jobDraft;
    if (!d) return;
    d.definition.targetConnectorId = id || undefined;
    d.definition.targetConnectorParams = id
      ? defaultJobParams(connectorInstance(id))
      : undefined;
    store.dryResult = null;
  },
  setJobTargetParam(key: string, value: string) {
    const d = store.jobDraft;
    if (!d) return;
    d.definition.targetConnectorParams = {
      ...(d.definition.targetConnectorParams ?? {}),
      [key]: value,
    };
    store.dryResult = null;
  },

  configureJobConnector(role: ConnectorBindRole, specName?: string) {
    const draft = store.jobDraft;
    if (!draft) return;
    const kind: ConnectorKind = role === "source" ? "source" : "target";
    const spec =
      specName && connectorSpec(specName)?.kind === kind
        ? connectorSpec(specName)
        : (store.connectorSpecs.find((s) => s.kind === kind && s.available) ??
          store.connectorSpecs.find((s) => s.kind === kind));
    if (!spec) {
      store.connectorSearch = kind;
      store.view = "connectors";
      return;
    }
    actions.newConnectorForReturn(spec.name, {
      from: "job-empty-state",
      returnView: "jobs",
      specName: spec.name,
      kind,
      bind: { jobDraftId: draft.id, role },
    });
  },

  setView(v: View) {
    store.view = v;
    if (v === "connectors" && store.connectorSpecs.length === 0)
      void actions.loadConnectors();
    if (v === "store" && store.registryPackages.length === 0)
      void actions.loadRegistry();
    if (v === "bindings") {
      void actions.loadConnectors();
      void actions.loadBindings();
    }
    if (v === "staged") void actions.loadStagedRelations();
    if (v === "observability") void actions.loadObservability();
    if (v === "jobs") void actions.loadBindings();
    if (v === "explore") {
      void actions.loadExplore();
      void actions.loadDataTree();
    }
    if (v === "settings") void actions.loadSettings();
  },

  /** Open the connectors surface scoped to a role (Sources or Targets). */
  setConnectorRole(role: ConnectorBindRole) {
    store.connectorRole = role;
    actions.setView("connectors");
  },

  // ---- App settings (persisted via prefs) ----
  /** Read persisted settings into the store (idempotent; safe to re-call). */
  async loadSettings() {
    // The bearer token lives in the OS keychain (option A) and is write-only from
    // the UI — we only learn *whether* one is set, never its value.
    const [key, skUrl, tokenSet, serverUrl, theme] = await Promise.all([
      getPref(OPENAI_KEY_PREF),
      getPref(SECRETSKEEPER_URL_PREF),
      hasSecretsKeeperToken(),
      getPref(SERVER_URL_PREF),
      getPref(THEME_PREF),
    ]);
    const sk = store.settings.secretsKeeper;
    const normalizedServerUrl = (serverUrl?.trim() || getServerBase()).replace(
      /\/+$/,
      "",
    );
    const nextTheme = theme === "light" ? "light" : "dark";
    setServerBase(normalizedServerUrl || defaultServerBase());
    applyTheme(nextTheme);
    store.settings.serverUrl = normalizedServerUrl;
    store.settings.openaiKey = key ?? "";
    sk.url = skUrl ?? "";
    sk.tokenSet = tokenSet;
    store.settings.theme = nextTheme;
    // Don't clobber an in-flight edit if the user is already typing.
    if (!store.settings.loaded) {
      store.settings.serverUrlDraft = normalizedServerUrl;
      store.settings.openaiDraft = key ?? "";
      sk.urlDraft = skUrl ?? "";
    }
    store.settings.loaded = true;
  },

  setServerUrlDraft(value: string) {
    store.settings.serverUrlDraft = value;
    store.settings.serverOk = null;
    store.settings.serverMessage = null;
  },

  async testServerConnection() {
    const draft =
      store.settings.serverUrlDraft.trim().replace(/\/+$/, "") ||
      defaultServerBase();
    const previous = getServerBase();
    store.settings.busy = true;
    store.settings.serverOk = null;
    try {
      setServerBase(draft);
      const health = await serverHealth();
      store.settings.serverOk = Boolean(health?.ok);
      store.settings.serverMessage = health?.ok
        ? `Connected to ${draft}.`
        : `Could not reach ${draft}.`;
    } catch (e) {
      store.settings.serverOk = false;
      store.settings.serverMessage = `Test failed: ${e}`;
    } finally {
      setServerBase(previous);
      store.settings.busy = false;
    }
  },

  setOpenaiDraft(value: string) {
    store.settings.openaiDraft = value;
    store.settings.message = null;
  },

  setSecretsKeeperUrlDraft(value: string) {
    store.settings.secretsKeeper.urlDraft = value;
    store.settings.secretsKeeper.message = null;
  },
  setSecretsKeeperTokenDraft(value: string) {
    store.settings.secretsKeeper.tokenDraft = value;
    store.settings.secretsKeeper.message = null;
  },

  /** Persist the secrets-keeper backend URL (pref store) and, when a new token
   *  was entered, write it to the OS keychain (option A). An empty token field
   *  leaves the stored token unchanged; clearing is an explicit action. */
  async saveSecretsKeeper() {
    const sk = store.settings.secretsKeeper;
    if (sk.busy) return;
    sk.busy = true;
    try {
      const url = sk.urlDraft.trim().replace(/\/+$/, "");
      await setPref(SECRETSKEEPER_URL_PREF, url);
      sk.url = url;
      sk.urlDraft = url;
      const token = sk.tokenDraft.trim();
      if (token) {
        await setSecretsKeeperToken(token);
        sk.tokenSet = true;
        sk.tokenDraft = "";
      }
      actions.notify({ tone: "ok", message: "Secrets backend saved." });
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Save failed",
        message: String(e),
      });
    } finally {
      sk.busy = false;
    }
  },

  /** Remove the stored bearer token from the OS keychain. */
  async clearSecretsKeeperToken() {
    const sk = store.settings.secretsKeeper;
    if (sk.busy) return;
    sk.busy = true;
    try {
      await setSecretsKeeperToken("");
      sk.tokenSet = false;
      sk.tokenDraft = "";
      actions.notify({ tone: "ok", message: "Secrets backend token cleared." });
    } catch (e) {
      actions.notify({
        tone: "error",
        title: "Could not clear token",
        message: String(e),
      });
    } finally {
      sk.busy = false;
    }
  },

  /** Probe the secrets backend (reachability + seal status + token validity). */
  async testSecretsKeeper() {
    const sk = store.settings.secretsKeeper;
    if (sk.busy) return;
    sk.busy = true;
    sk.message = null;
    try {
      const status = await secretsKeeperTest(
        sk.urlDraft.trim(),
        sk.tokenDraft.trim(),
      );
      sk.ok = status.ok;
      sk.message = status.message;
    } catch (e) {
      sk.ok = false;
      sk.message = String(e);
    } finally {
      sk.busy = false;
    }
  },

  /** Persist the OpenAI API key (trimmed). An empty value clears it. */
  async saveSettings() {
    if (store.settings.busy) return;
    store.settings.busy = true;
    try {
      const url =
        store.settings.serverUrlDraft.trim().replace(/\/+$/, "") ||
        defaultServerBase();
      await setPref(SERVER_URL_PREF, url);
      setServerBase(url);
      store.settings.serverUrl = url;
      store.settings.serverUrlDraft = url;
      store.settings.serverMessage = `Saved ${url}.`;
      const key = store.settings.openaiDraft.trim();
      await setPref(OPENAI_KEY_PREF, key);
      store.settings.openaiKey = key;
      store.settings.openaiDraft = key;
      store.settings.message = key ? "Settings saved." : "Settings saved.";
    } catch (e) {
      store.settings.message = `Save failed: ${e}`;
    } finally {
      store.settings.busy = false;
    }
  },

  async testRegistryUrl(url: string) {
    store.settings.registryTestMessage = null;
    store.settings.registryTestOk = null;
    store.settings.registryTestUrl = url;
    try {
      const result = await listRegistry({ url });
      store.settings.registryTestOk = true;
      store.settings.registryTestMessage = `${result.packages.length} package${result.packages.length === 1 ? "" : "s"} found.`;
    } catch (e) {
      store.settings.registryTestOk = false;
      store.settings.registryTestMessage = String(e);
    }
  },

  addRegistryUrl(url: string) {
    const normalized = url.trim().replace(/\/+$/, "");
    if (!normalized || store.registryUrls.includes(normalized)) return;
    void actions.setRegistryUrls([...store.registryUrls, normalized]);
  },

  removeRegistryUrl(url: string) {
    void actions.setRegistryUrls(
      store.registryUrls.filter((item) => item !== url),
    );
  },

  async saveTheme(theme: "dark" | "light") {
    store.settings.theme = theme;
    applyTheme(theme);
    try {
      await setPref(THEME_PREF, theme);
    } catch {
      /* local fallback handled by setPref */
    }
  },

  setEngineMode(mode: "desktop" | "server" | "preview") {
    store.runtime.engineMode = mode;
  },

  setConnectionBanner(input: {
    visible: boolean;
    message?: string | null;
    retrying?: boolean;
  }) {
    store.runtime.bannerVisible = input.visible;
    store.runtime.bannerMessage = input.message ?? null;
    if (typeof input.retrying === "boolean")
      store.runtime.retrying = input.retrying;
  },

  setNarrowLayout(narrow: boolean) {
    store.runtime.narrowLayout = narrow;
    if (!narrow) store.runtime.railOpen = false;
  },

  toggleRail(force?: boolean) {
    store.runtime.railOpen = force ?? !store.runtime.railOpen;
  },

  setSearch(s: string) {
    store.search = s;
  },

  toggleCategory(cat: string, open?: boolean) {
    store.openCategories[cat] = open ?? !store.openCategories[cat];
  },

  select(name: string) {
    store.selectedName = name;
    store.view = "studio";
    const tpl = store.templates.find((t) => t.name === name);
    store.run.values = parseExample(tpl?.source);
    store.run.result = null;
    store.run.running = false;
    if (store.loaded) actions.recordUsage(name); // skip the startup default
  },

  open(name: string) {
    actions.select(name);
  },

  setRunValue(field: string, value: string) {
    store.run.values[field] = value;
  },

  async run(template: Template) {
    actions.recordUsage(template.name);
    store.run.running = true;
    store.run.result = null;
    try {
      const args: Record<string, string> = {};
      for (const inp of template.inputs)
        args[inp.name] = store.run.values[inp.name] ?? "";
      store.run.result = await runTemplate(template.name, args);
    } catch (e) {
      store.run.result = { status: "failed", rows: [], error: String(e) };
    } finally {
      store.run.running = false;
    }
  },

  async browse(field: string) {
    const p = await pickDataFile();
    if (p) store.run.values[field] = p;
  },

  setDraft(s: string) {
    store.ai.draft = s;
  },

  async sendAi(text: string, template: Template | null) {
    const content = text.trim();
    if (!content || store.ai.busy) return;
    store.ai.messages.push({ role: "user", content });
    store.ai.draft = "";
    store.ai.busy = true;
    const context = template
      ? `Active template: ${template.name} (${template.kind}, ${template.category}).\n\n${template.source ?? ""}`
      : "No template selected.";
    try {
      const reply = await aiChat([...store.ai.messages], context);
      store.ai.messages.push({ role: "assistant", content: reply });
    } finally {
      store.ai.busy = false;
      requestAnimationFrame(() => {
        const el = document.getElementById("ai-scroll");
        el?.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
      });
    }
  },

  setMarketCategory(c: string) {
    store.market.category = c;
  },
  setMarketKind(k: KindFilter) {
    store.market.kind = k;
  },

  // ---- Explore notebook (ADR-0029) ----
  async loadExplore() {
    if (!store.activeCellId) {
      store.activeCellId =
        store.notebook.cells.find((c) => c.kind === "sql")?.id ??
        store.notebook.cells[0]?.id ??
        null;
    }
    syncNotebookParams();
    if (!store.externalEndpoint) {
      store.externalEndpoint = await externalEndpointInfo();
    }
    if (!store.defaultNotebookDir) {
      store.defaultNotebookDir = await defaultNotebookDirRemote();
    }
    await actions.refreshNotebookLibrary();
  },

  /** Re-scan the default dir + extra paths for saved notebooks. */
  async refreshNotebookLibrary() {
    store.notebookLibrary = await listNotebooks([...store.notebookPaths]);
  },

  /** Scan the data roots for queryable datasets (Explore sidebar tree). */
  async loadDataTree() {
    store.dataTreeBusy = true;
    try {
      store.dataTree = await listDataTree();
      store.dataTreeMessage =
        store.dataTree.length === 0
          ? "No datasets found under the data roots."
          : null;
    } catch (e) {
      store.dataTree = [];
      store.dataTreeMessage = String(e);
    } finally {
      store.dataTreeBusy = false;
    }
  },

  /** Append a SQL cell that reads the clicked data file directly, and focus it.
   *  The cell reads the file's absolute path via the engine's `read_*` table
   *  functions, so it is self-contained (independent of the notebook source). */
  addCellForDataFile(node: DataTreeNode) {
    if (node.isDir || !node.queryable) return;
    const cell = newCell("sql", dataFileSql(node));
    store.notebook.cells.push(cell);
    store.activeCellId = cell.id;
    persistNotebook();
  },

  newExploration() {
    store.notebook = newNotebook();
    store.cellRuns = {};
    store.cellEditing = {};
    store.activeCellId =
      store.notebook.cells.find((c) => c.kind === "sql")?.id ?? null;
    store.notebookPath = null;
    store.exploreMessage = null;
    persistNotebook();
    store.notebookDirty = false;
  },

  /** Open a saved notebook file into the editor. */
  async openNotebookFile(path: string) {
    try {
      const raw = await readNotebook(path);
      const nb = JSON.parse(raw) as Notebook;
      if (!nb || !Array.isArray(nb.cells)) throw new Error("not a notebook");
      store.notebook = nb;
      store.cellRuns = {};
      store.cellEditing = {};
      store.activeCellId =
        nb.cells.find((c) => c.kind === "sql")?.id ?? nb.cells[0]?.id ?? null;
      store.notebookPath = path;
      syncNotebookParams();
      // syncNotebookParams may push params (→ persistNotebook → dirty); a freshly
      // opened file is clean. persistNotebook is not called here, so just clear.
      store.notebookDirty = false;
      store.exploreMessage = null;
    } catch (e) {
      store.exploreMessage = `Open failed: ${e}`;
    }
  },

  /** Save the open notebook back to its file (or to the default dir if new). */
  async saveCurrentNotebook() {
    const dir =
      (store.notebookPath
        ? store.notebookPath.slice(0, store.notebookPath.lastIndexOf("/"))
        : "") || store.defaultNotebookDir;
    await actions.saveNotebookToDir(dir);
  },

  /** Save the open notebook into a specific directory (Save as / first save). */
  async saveNotebookToDir(dir: string) {
    if (!dir) {
      store.exploreMessage = "No notebook directory configured.";
      return;
    }
    store.exploreBusy = true;
    store.exploreMessage = null;
    try {
      const slug = slugify(store.notebook.name || "untitled");
      const path = await saveNotebookRemote(dir, slug, notebookJson());
      store.notebookPath = path;
      store.notebookDirty = false;
      store.exploreMessage = `Saved notebook to ${path}.`;
      await actions.refreshNotebookLibrary();
    } catch (e) {
      store.exploreMessage = `Save failed: ${e}`;
    } finally {
      store.exploreBusy = false;
    }
  },

  /** Delete a saved notebook file from the library. */
  async deleteNotebookFile(path: string) {
    try {
      await deleteNotebookRemote(path);
      if (store.notebookPath === path) store.notebookPath = null;
      await actions.refreshNotebookLibrary();
    } catch (e) {
      store.exploreMessage = `Delete failed: ${e}`;
    }
  },

  /** Add a directory to the library via the native folder picker. */
  async addNotebookPath() {
    const dir = await pickDirectory();
    if (!dir) return;
    if (dir === store.defaultNotebookDir || store.notebookPaths.includes(dir))
      return;
    store.notebookPaths = [...store.notebookPaths, dir];
    persistNotebookPaths();
    await actions.refreshNotebookLibrary();
  },

  /** Remove an extra directory (the default dir can't be removed). */
  async removeNotebookPath(dir: string) {
    if (dir === store.defaultNotebookDir) return;
    store.notebookPaths = store.notebookPaths.filter((d) => d !== dir);
    persistNotebookPaths();
    await actions.refreshNotebookLibrary();
  },

  setNotebookName(name: string) {
    store.notebook.name = name;
    persistNotebook();
  },

  setNotebookSource(source: string) {
    store.notebook.source = source;
    persistNotebook();
  },

  async browseNotebookSource() {
    const p = await pickDataFile();
    if (p) actions.setNotebookSource(p);
  },

  addCell(kind: CellKind, afterId?: string) {
    const cell = newCell(
      kind,
      kind === "sql" ? "SELECT *\nFROM input\nLIMIT 20" : "## Notes\n",
    );
    const idx = afterId
      ? store.notebook.cells.findIndex((c) => c.id === afterId)
      : -1;
    if (idx >= 0) store.notebook.cells.splice(idx + 1, 0, cell);
    else store.notebook.cells.push(cell);
    store.activeCellId = cell.id;
    if (kind === "markdown") store.cellEditing[cell.id] = true;
    persistNotebook();
  },

  removeCell(id: string) {
    const idx = store.notebook.cells.findIndex((c) => c.id === id);
    if (idx < 0) return;
    store.notebook.cells.splice(idx, 1);
    delete store.cellRuns[id];
    delete store.cellEditing[id];
    if (store.activeCellId === id) {
      store.activeCellId =
        store.notebook.cells[Math.max(0, idx - 1)]?.id ?? null;
    }
    persistNotebook();
  },

  moveCell(id: string, dir: -1 | 1) {
    const cells = store.notebook.cells;
    const idx = cells.findIndex((c) => c.id === id);
    const next = idx + dir;
    if (idx < 0 || next < 0 || next >= cells.length) return;
    const [cell] = cells.splice(idx, 1);
    cells.splice(next, 0, cell);
    persistNotebook();
  },

  setCellSource(id: string, source: string) {
    const cell = store.notebook.cells.find((c) => c.id === id);
    if (!cell) return;
    cell.source = source;
    store.activeCellId = id;
    if (cell.kind === "sql") syncNotebookParams();
    persistNotebook();
  },

  setActiveCell(id: string) {
    store.activeCellId = id;
    store.exploreFunctionFocus = null;
  },

  setExploreFunctionFocus(name: string | null) {
    store.exploreFunctionFocus = name;
  },

  setCellEditing(id: string, editing: boolean) {
    store.cellEditing[id] = editing;
  },

  async runCell(id: string) {
    const cell = store.notebook.cells.find((c) => c.id === id);
    if (!cell || cell.kind !== "sql") return;
    store.activeCellId = id;
    const sql = substitute(cell.source, store.notebook.params);
    store.cellRuns[id] = { result: null, running: true };
    try {
      const result = await runQuery(sql, store.notebook.source || null);
      store.cellRuns[id] = { result, running: false };
    } catch (e) {
      store.cellRuns[id] = {
        result: { status: "failed", rows: [], error: String(e) },
        running: false,
      };
    }
  },

  async runAllCells() {
    for (const cell of store.notebook.cells) {
      if (cell.kind === "sql") await actions.runCell(cell.id);
    }
  },

  addParam() {
    const base = "param";
    let n = store.notebook.params.length + 1;
    let name = `${base}${n}`;
    const taken = new Set(store.notebook.params.map((p) => p.name));
    while (taken.has(name)) name = `${base}${++n}`;
    store.notebook.params.push({ name, kind: "Column", value: "" });
    persistNotebook();
  },

  setParam(index: number, patch: Partial<NotebookParam>) {
    const p = store.notebook.params[index];
    if (!p) return;
    Object.assign(p, patch);
    persistNotebook();
  },

  removeParam(index: number) {
    store.notebook.params.splice(index, 1);
    persistNotebook();
  },

  toggleExploreToml(open?: boolean) {
    store.exploreTomlOpen = open ?? !store.exploreTomlOpen;
  },

  /** The resolved TOML for the focused SQL cell (live preview). */
  resolvedToml(): string | null {
    const id = store.activeCellId;
    const cell = store.notebook.cells.find((c) => c.id === id);
    if (!cell || cell.kind !== "sql") return null;
    const name =
      `${slugify(store.notebook.name)}_${slugify(cell.source.slice(0, 24)) || "query"}`.slice(
        0,
        64,
      );
    return toTemplateToml(cell, store.notebook, {
      name,
      description: store.notebook.name,
    });
  },

  async saveCellAsTemplate(
    cellId: string,
    name: string,
    category: string,
    description: string,
  ) {
    const cell = store.notebook.cells.find((c) => c.id === cellId);
    if (!cell || cell.kind !== "sql") return;
    const slug = slugify(name);
    const toml = toTemplateToml(cell, store.notebook, {
      name: slug,
      description,
    });
    store.exploreBusy = true;
    store.exploreMessage = null;
    try {
      await saveTemplate(category || "Explore", slug, toml);
      store.templates = await listTemplates();
      store.exploreMessage = `Saved template “${slug}” to ${category || "Explore"}.`;
    } catch (e) {
      store.exploreMessage = `Save failed: ${e}`;
    } finally {
      store.exploreBusy = false;
    }
  },

  /** Promote the whole notebook to a NEW user pack (ADR-0030 P5): every SQL
   *  cell becomes a template, chained into a job template, so the notebook → pack
   *  → job lineage is one click. Navigates to the new pack on success. */
  async promoteNotebookToPack() {
    const bundle = toPackBundle(store.notebook);
    store.exploreBusy = true;
    store.exploreMessage = null;
    try {
      await createPackRemote(
        bundle.id,
        bundle.name,
        bundle.description,
        bundle.files,
      );
      await actions.loadPacks();
      store.templates = await listTemplates();
      store.exploreMessage = `Promoted to pack “${bundle.name}”.`;
      store.view = "packs";
    } catch (e) {
      store.exploreMessage = `Promote to pack failed: ${e}`;
    } finally {
      store.exploreBusy = false;
    }
  },

  /** Create a one-off job draft from the current Explore notebook without
   *  publishing a pack. Generated SQL templates are saved under Explore so the
   *  existing job runner can execute the chain immediately. */
  async createJobFromCurrentNotebook() {
    const draft = toNotebookJobDraft(store.notebook);
    store.exploreBusy = true;
    store.exploreMessage = null;
    try {
      if (draft.sqlCellCount === 0) {
        store.exploreMessage =
          "Add at least one SQL cell before creating a job.";
        return;
      }
      if (!draft.source.trim()) {
        store.exploreMessage =
          "Choose a source file in Explore before creating a job. The runner will not substitute sample data.";
        actions.notify({
          tone: "error",
          title: "Source required",
          message:
            "Choose a CSV or Parquet source in Explore, then create the job again.",
        });
        return;
      }
      await Promise.all(
        draft.templates.map((template) =>
          saveTemplate("Explore", template.name, template.source),
        ),
      );
      store.templates = await listTemplates();
      recomputeCatalog();
      recomputeSidebar();
      store.jobDraft = {
        id: `job-${Date.now()}`,
        name: `${store.notebook.name || "Untitled exploration"} job`,
        description: draft.description,
        definition: {
          source: draft.source,
          steps: draft.steps,
        },
        enabled: true,
      };
      store.jobSourceMode = "file";
      store.runs = [];
      store.openRunId = null;
      store.runSteps = [];
      store.sourceColumns = [];
      store.stepColumns = [];
      store.dryResult = null;
      store.sourcePreview = null;
      store.exploreMessage = `Created job draft from "${store.notebook.name}".`;
      if (draft.source.trim()) {
        void actions.inspectSource(draft.source);
        void actions.describeColumns();
      }
      store.view = "jobs";
    } catch (e) {
      store.exploreMessage = `Create job failed: ${e}`;
    } finally {
      store.exploreBusy = false;
    }
  },

  openPackJobTemplateInExplore(packId: string, jobTemplateName: string) {
    const pack = store.packs.find((p) => p.id === packId);
    const jobTemplate = pack?.jobTemplates.find(
      (jt) => jt.name === jobTemplateName,
    );
    if (!pack || !jobTemplate) return;
    store.notebook = notebookFromPackJobTemplate(
      pack,
      jobTemplate,
      store.templates,
    );
    store.notebookPath = null;
    store.notebookDirty = true;
    store.cellRuns = {};
    store.cellEditing = {};
    store.activeCellId =
      store.notebook.cells.find((cell) => cell.kind === "sql")?.id ??
      store.notebook.cells[0]?.id ??
      null;
    store.exploreMessage = `Opened packaged flow "${jobTemplateName}" from ${pack.name}.`;
    store.view = "explore";
    store.exploreTomlOpen = true;
  },

  async addCellToPack(
    cellId: string,
    packId: string,
    name: string,
    description: string,
  ) {
    const cell = store.notebook.cells.find((c) => c.id === cellId);
    const pack = store.packs.find((p) => p.id === packId);
    if (!cell || cell.kind !== "sql" || !pack) return;
    const slug = slugify(name);
    const toml = toTemplateToml(cell, store.notebook, {
      name: slug,
      description,
    });
    store.exploreBusy = true;
    store.exploreMessage = null;
    try {
      await writePackFile(packId, `templates/${slug}.toml`, toml);
      await actions.loadPacks();
      store.templates = await listTemplates();
      store.exploreMessage = `Added “${slug}” to pack ${pack.name}.`;
    } catch (e) {
      store.exploreMessage = `Add to pack failed: ${e}`;
    } finally {
      store.exploreBusy = false;
    }
  },
};

// Load the catalog once at startup (singleton store — no mount effect needed).
void actions.load();
