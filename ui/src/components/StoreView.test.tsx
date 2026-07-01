import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { RegistryPackage } from "@/lib/types";

vi.mock("@/lib/store", async () => {
  const { proxy } = await import("valtio");

  const actions = new Proxy(
    {
      loadRegistry: vi.fn(),
      setRegistrySearch: vi.fn(),
      setRegistryTypeFilter: vi.fn(),
      setRegistryCategoryFilter: vi.fn(),
      selectPackage: vi.fn(),
      installPackage: vi.fn(),
    } as Record<string, ReturnType<typeof vi.fn>>,
    {
      get(target, prop: string) {
        if (!(prop in target)) target[prop] = vi.fn();
        return target[prop];
      },
    },
  );

  const store = proxy({
    registryPackages: [] as RegistryPackage[],
    registryLoading: false,
    registryError: null as string | null,
    registrySearch: "",
    registryTypeFilter: "all" as "all" | "source" | "target",
    registryCategoryFilter: "all",
    selectedPackage: null as RegistryPackage | null,
    installing: {} as Record<string, boolean>,
    installResult: null as
      | { name: string; status: "installed" | "failed"; message: string; log: string }
      | null,
    runtime: {
      engineMode: "preview" as const,
    },
  });

  return { actions, store };
});

import { PackageDetailView } from "./PackageDetailView";
import { StoreView } from "./StoreView";
import { actions, store } from "@/lib/store";

function makePackage(overrides: Partial<RegistryPackage> = {}): RegistryPackage {
  return {
    name: "filesystem-csv-source",
    title: "Filesystem CSV Source",
    summary: "Read CSV files from disk.",
    description: "Reads CSV files from disk and emits rows into the pipeline.",
    type: "source",
    version: "1.2.3",
    author: { name: "Celeritas", email: "team@example.com" },
    repository: "https://github.com/celeritas/fs-csv-source",
    license: "MIT",
    artifact: {
      publisher: "npm",
      package: "@celeritas/fs-csv-source",
      install_command: "npm install -g @celeritas/fs-csv-source",
    },
    manifest: {
      capabilities: ["discover", "test"],
      settings: [
        {
          name: "root",
          kind: "string",
          required: true,
          default: "/data",
          description: "Directory to scan.",
          options: null,
          scope: "connection",
        },
      ],
    },
    tags: ["csv", "filesystem"],
    category: "Files",
    installed: false,
    installCommand: "npm install -g @celeritas/fs-csv-source",
    packagedBy: {
      name: "Ops Team",
      email: "ops@example.com",
      url: "https://example.com/ops",
    },
    packagedAt: "2026-06-15T10:30:00Z",
    ...overrides,
  };
}

describe("StoreView and PackageDetailView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    store.registryPackages = [];
    store.registryLoading = false;
    store.registryError = null;
    store.registrySearch = "";
    store.registryTypeFilter = "all";
    store.registryCategoryFilter = "all";
    store.selectedPackage = null;
    store.installing = {};
    store.installResult = null;
    store.runtime.engineMode = "preview";
  });

  it("wires search, type, and category filters through the store actions", async () => {
    const sourcePkg = makePackage();
    const targetPkg = makePackage({
      name: "filesystem-parquet-target",
      title: "Filesystem Parquet Target",
      summary: "Write Parquet files to disk.",
      description: "Writes Parquet files to disk.",
      type: "target",
      category: "Warehouse",
      repository: "https://github.com/celeritas/fs-parquet-target",
      artifact: {
        publisher: "npm",
        package: "@celeritas/fs-parquet-target",
        install_command: "npm install -g @celeritas/fs-parquet-target",
      },
      installCommand: "npm install -g @celeritas/fs-parquet-target",
    });

    store.registryPackages = [sourcePkg, targetPkg];

    render(<StoreView />);

    fireEvent.change(screen.getByLabelText("Search the store"), {
      target: { value: "csv" },
    });
    await waitFor(() => {
      expect(actions.setRegistrySearch).toHaveBeenCalledWith("csv");
    });

    fireEvent.click(screen.getByRole("button", { name: /targets/i }));
    expect(actions.setRegistryTypeFilter).toHaveBeenCalledWith("target");

    fireEvent.click(screen.getByRole("button", { name: "Warehouse" }));
    expect(actions.setRegistryCategoryFilter).toHaveBeenCalledWith("Warehouse");
  });

  it("renders package detail fields and install progress to installed state", () => {
    const pkg = makePackage();

    const { rerender } = render(<PackageDetailView pkg={pkg} />);

    expect(screen.getByText("Package / install command")).toBeInTheDocument();
    expect(
      screen.getByText("npm install -g @celeritas/fs-csv-source"),
    ).toBeInTheDocument();
    expect(screen.getByText("Packaged by")).toBeInTheDocument();
    expect(screen.getByText("Ops Team")).toBeInTheDocument();
    expect(screen.getByText("Packaged at")).toBeInTheDocument();
    expect(screen.getByText(/Jun/)).toBeInTheDocument();
    expect(screen.getByText("Version")).toBeInTheDocument();
    expect(screen.getByText("1.2.3")).toBeInTheDocument();
    expect(screen.getByText("Repository")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /celeritas\/fs-csv-source/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Reads CSV files from disk and emits rows into the pipeline.",
      ),
    ).toBeInTheDocument();

    store.installing = { [pkg.name]: true };
    rerender(<PackageDetailView pkg={pkg} />);
    expect(
      screen.getByRole("button", { name: /installing/i }),
    ).toBeInTheDocument();

    store.installing = {};
    store.installResult = {
      name: pkg.name,
      status: "installed",
      message: "Installed successfully.",
      log: "npm install complete",
    };
    rerender(<PackageDetailView pkg={{ ...pkg, installed: true }} />);

    expect(screen.getByText("Installed successfully.")).toBeInTheDocument();
    expect(screen.getByText("npm install complete")).toBeInTheDocument();
    expect(screen.getAllByText("Installed").length).toBeGreaterThan(0);
  });

  it("renders the preview notice when the sidecar is offline", () => {
    store.registryPackages = [makePackage()];
    store.runtime.engineMode = "preview";

    render(<StoreView />);

    expect(
      screen.getByText(
        "Preview mode: store results come from the bundled offline registry seed, not a live sidecar registry fetch.",
      ),
    ).toBeInTheDocument();
  });
});
