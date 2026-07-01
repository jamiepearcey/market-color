import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { ConnectorInstance, ConnectorSpec } from "@/lib/types";

vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: Record<string, unknown>) => (
      <div {...props}>{children as React.ReactNode}</div>
    ),
  },
}));

vi.mock("@/lib/store", async () => {
  const { proxy } = await import("valtio");

  const actions = new Proxy(
    {
      newConnector: vi.fn(),
      clearConnectorDraftSecrets: vi.fn(),
    } as Record<string, ReturnType<typeof vi.fn>>,
    {
      get(target, prop: string) {
        if (!(prop in target)) target[prop] = vi.fn();
        return target[prop];
      },
    },
  );

  const store = proxy({
    connectorRole: "source" as "source" | "target",
    connectorSpecs: [] as ConnectorSpec[],
    connectorInstances: [] as ConnectorInstance[],
    sourceInstances: [] as ConnectorInstance[],
    targetInstances: [] as ConnectorInstance[],
    enabledSources: [] as ConnectorInstance[],
    enabledTargets: [] as ConnectorInstance[],
    connectorDraft: null as ConnectorInstance | null,
    connectorDraftSpec: null as ConnectorSpec | null,
    connectorReturnIntent: null,
    connectorBusy: false,
    connectorLoading: false,
    connectorError: null as string | null,
    connectorTestResult: null,
    connectorActionResult: null,
    connectorPreviewJobParams: {} as Record<string, string>,
    connectorCatalogMessage: null as string | null,
    connectorSearch: "",
    connectorHideUnavailable: false,
    runtime: {
      engineMode: "preview" as const,
    },
    connectorCatalog: [] as {
      kind: "source" | "target";
      category: string;
      specs: ConnectorSpec[];
    }[],
  });

  return { actions, store };
});

vi.mock("@/lib/api", () => ({
  pickDataFile: vi.fn().mockResolvedValue(null),
  pickDirectory: vi.fn().mockResolvedValue(null),
}));

import { ConnectorsView } from "./ConnectorsView";
import { actions, store } from "@/lib/store";

function makeSpec(overrides: Partial<ConnectorSpec>): ConnectorSpec {
  return {
    name: "filesystem-source",
    description: "Read CSV files from disk.",
    kind: "source",
    driver: "filesystem",
    category: "Files",
    params: [],
    jobParams: [],
    actions: [],
    example: {},
    exampleUnresolved: false,
    available: true,
    ...overrides,
  };
}

function makeInstance(
  spec: ConnectorSpec,
  overrides: Partial<ConnectorInstance> = {},
): ConnectorInstance {
  return {
    id: `${spec.name}-instance`,
    spec: spec.name,
    kind: spec.kind,
    driver: spec.driver,
    name: `${spec.name} instance`,
    params: {},
    enabled: true,
    ...overrides,
  };
}

describe("ConnectorsView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    store.connectorRole = "source";
    store.connectorSpecs = [];
    store.connectorInstances = [];
    store.sourceInstances = [];
    store.targetInstances = [];
    store.enabledSources = [];
    store.enabledTargets = [];
    store.connectorDraft = null;
    store.connectorDraftSpec = null;
    store.connectorReturnIntent = null;
    store.connectorBusy = false;
    store.connectorLoading = false;
    store.connectorError = null;
    store.connectorTestResult = null;
    store.connectorActionResult = null;
    store.connectorPreviewJobParams = {};
    store.connectorCatalogMessage = null;
    store.connectorSearch = "";
    store.connectorHideUnavailable = false;
    store.runtime.engineMode = "preview";
    store.connectorCatalog = [];
  });

  it("scopes the catalog to source connectors and opens the selected spec", () => {
    const sourceSpec = makeSpec({
      name: "filesystem-source",
      description: "Read CSV files from disk.",
      kind: "source",
    });
    const targetSpec = makeSpec({
      name: "filesystem-target",
      description: "Write Parquet files to disk.",
      kind: "target",
      driver: "filesystem-target",
    });

    store.connectorRole = "source";
    store.connectorSpecs = [sourceSpec, targetSpec];
    store.connectorCatalog = [
      { kind: "source", category: "Files", specs: [sourceSpec] },
      { kind: "target", category: "Files", specs: [targetSpec] },
    ];

    render(<ConnectorsView />);

    expect(screen.getByText("Import connectors")).toBeInTheDocument();
    expect(screen.getByText("Read CSV files from disk.")).toBeInTheDocument();
    expect(
      screen.queryByText("Write Parquet files to disk."),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /filesystem source/i }));
    expect(actions.newConnector).toHaveBeenCalledWith("filesystem-source");
  });

  it("renders the target connector configuration form for the active draft", () => {
    const targetSpec = makeSpec({
      name: "filesystem-target",
      description: "Write Parquet files to disk.",
      kind: "target",
      driver: "filesystem-target",
      params: [
        {
          name: "output_path",
          label: "Output path",
          kind: "path",
          input: "file",
          required: true,
          options: [],
        },
      ],
      jobParams: [
        {
          name: "compression",
          label: "Compression",
          kind: "enum",
          required: false,
          options: ["snappy", "zstd"],
        },
      ],
      actions: [
        {
          name: "preview",
          label: "Preview",
          flag: "--preview",
        },
      ],
    });
    const targetInstance = makeInstance(targetSpec, {
      name: "Nightly target",
      params: { output_path: "/tmp/out.parquet" },
      jobParamDefaults: { compression: "snappy" },
    });

    store.connectorRole = "target";
    store.connectorSpecs = [targetSpec];
    store.connectorInstances = [targetInstance];
    store.targetInstances = [targetInstance];
    store.enabledTargets = [targetInstance];
    store.connectorDraft = targetInstance;
    store.connectorDraftSpec = targetSpec;

    render(<ConnectorsView />);

    expect(screen.getByDisplayValue("Nightly target")).toBeInTheDocument();
    expect(screen.getByText("Connection")).toBeInTheDocument();
    expect(screen.getByText("Output path")).toBeInTheDocument();
    expect(screen.getByText("Per-job defaults")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /preview/i })).toBeInTheDocument();
  });

  it("renders the preview notice when running against offline bundled data", () => {
    const sourceSpec = makeSpec({
      name: "filesystem-source",
      description: "Read CSV files from disk.",
      kind: "source",
    });

    store.connectorRole = "source";
    store.connectorSpecs = [sourceSpec];
    store.connectorCatalog = [
      { kind: "source", category: "Files", specs: [sourceSpec] },
    ];
    store.runtime.engineMode = "preview";

    render(<ConnectorsView />);

    expect(
      screen.getByText(
        "Preview mode: connector catalog and driver actions may use representative sample data instead of a live engine.",
      ),
    ).toBeInTheDocument();
  });
});
