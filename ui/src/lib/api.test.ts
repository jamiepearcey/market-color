import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const {
  mockServerReady,
  mockSrv,
  mockSrvGet,
  mockSrvPost,
  mockSrvDelete,
  mockInvoke,
} = vi.hoisted(() => ({
  mockServerReady: vi.fn<() => Promise<boolean>>(),
  mockSrv: vi.fn(),
  mockSrvGet: vi.fn(),
  mockSrvPost: vi.fn(),
  mockSrvDelete: vi.fn(),
  mockInvoke: vi.fn(),
}));

vi.mock("./server", () => ({
  serverReady: mockServerReady,
  srv: mockSrv,
  srvGet: mockSrvGet,
  srvPost: mockSrvPost,
  srvDelete: mockSrvDelete,
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: mockInvoke,
}));

import { listConnectors, listRegistry, runJob } from "./api";

function setTauriAvailable(value: boolean) {
  const tauriWindow = window as Window & { __TAURI_INTERNALS__?: unknown };
  if (value) {
    tauriWindow.__TAURI_INTERNALS__ = {};
    return;
  }
  delete tauriWindow.__TAURI_INTERNALS__;
}

describe("api mode fallback", () => {
  beforeEach(() => {
    mockServerReady.mockReset();
    mockSrv.mockReset();
    mockSrvGet.mockReset();
    mockSrvPost.mockReset();
    mockSrvDelete.mockReset();
    mockInvoke.mockReset();
    setTauriAvailable(false);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("uses Tauri for listConnectors when desktop internals are present", async () => {
    setTauriAvailable(true);
    const desktopConnectors = [
      {
        name: "desktop-source",
        description: "Desktop source",
        kind: "source",
        driver: "desktop-driver",
        category: "Files",
        params: [],
        jobParams: [],
        actions: [],
        example: {},
        exampleUnresolved: false,
        available: true,
      },
    ];
    mockInvoke.mockResolvedValueOnce(desktopConnectors);

    await expect(listConnectors()).resolves.toEqual(desktopConnectors);
    expect(mockInvoke).toHaveBeenCalledWith("list_connectors", undefined);
    expect(mockServerReady).not.toHaveBeenCalled();
    expect(mockSrvGet).not.toHaveBeenCalled();
  });

  it("falls back from Tauri to server for listConnectors after desktop failure", async () => {
    setTauriAvailable(true);
    const serverConnectors = [
      {
        name: "server-source",
        description: "Server source",
        kind: "source",
        driver: "server-driver",
        category: "Files",
        params: [],
        jobParams: [],
        actions: [],
        example: {},
        exampleUnresolved: false,
        available: true,
      },
    ];
    mockInvoke.mockRejectedValueOnce(new Error("tauri failed"));
    mockServerReady.mockResolvedValueOnce(true);
    mockSrvGet.mockResolvedValueOnce(serverConnectors);

    await expect(listConnectors()).resolves.toEqual(serverConnectors);
    expect(mockServerReady).toHaveBeenCalledTimes(1);
    expect(mockSrvGet).toHaveBeenCalledWith("/connectors");
  });

  it("falls back to preview data for listConnectors when desktop and server are unavailable", async () => {
    mockServerReady.mockResolvedValueOnce(false);

    const connectors = await listConnectors();

    expect(mockInvoke).not.toHaveBeenCalled();
    expect(mockSrvGet).not.toHaveBeenCalled();
    expect(connectors.length).toBeGreaterThan(0);
    expect(connectors.some((connector) => connector.kind === "source")).toBe(
      true,
    );
  });

  it("uses Tauri for runJob when desktop internals are present", async () => {
    setTauriAvailable(true);
    mockInvoke.mockResolvedValueOnce("run-desktop-1");

    await expect(runJob("job-1", "embedded")).resolves.toBe("run-desktop-1");
    expect(mockInvoke).toHaveBeenCalledWith("run_job", {
      jobId: "job-1",
      targetId: "embedded",
    });
    expect(mockSrvPost).not.toHaveBeenCalled();
  });

  it("falls back from Tauri to server for runJob after desktop failure", async () => {
    setTauriAvailable(true);
    mockInvoke.mockRejectedValueOnce(new Error("tauri failed"));
    mockServerReady.mockResolvedValueOnce(true);
    mockSrvPost.mockResolvedValueOnce({
      run: { id: "run-server-1" },
      steps: [],
    });

    await expect(runJob("job-2", "embedded")).resolves.toBe("run-server-1");
    expect(mockSrvPost).toHaveBeenCalledWith("/jobs/job-2/run", {});
  });

  it("falls back to preview mode for runJob when desktop and server are unavailable", async () => {
    vi.useFakeTimers();
    mockServerReady.mockResolvedValueOnce(false);

    const pending = runJob("job-3", "embedded");
    await vi.runAllTimersAsync();

    await expect(pending).resolves.toMatch(/^run-preview-/);
    expect(mockSrvPost).not.toHaveBeenCalled();
  });

  it("uses Tauri for listRegistry when desktop internals are present", async () => {
    setTauriAvailable(true);
    mockInvoke.mockResolvedValueOnce({
      registry: { name: "Desktop registry" },
      packages: [],
    });

    await expect(listRegistry({ type: "source", q: "csv" })).resolves.toEqual({
      registry: { name: "Desktop registry" },
      packages: [],
    });
    expect(mockInvoke).toHaveBeenCalledWith("list_registry", {
      url: null,
      type: "source",
      q: "csv",
      category: null,
    });
    expect(mockSrvGet).not.toHaveBeenCalled();
  });

  it("uses the sidecar for listRegistry when the server is ready", async () => {
    mockServerReady.mockResolvedValueOnce(true);
    mockSrvGet.mockResolvedValueOnce({
      registry: { name: "Test registry" },
      packages: [],
    });

    await expect(
      listRegistry({ q: "csv", type: "source", category: "Files" }),
    ).resolves.toEqual({
      registry: { name: "Test registry" },
      packages: [],
    });
    expect(mockSrvGet).toHaveBeenCalledWith(
      "/registry?type=source&q=csv&category=Files",
    );
  });

  it("falls back to the bundled seed for listRegistry when the server is unavailable", async () => {
    mockServerReady.mockResolvedValueOnce(false);

    const result = await listRegistry();

    expect(mockSrvGet).not.toHaveBeenCalled();
    expect(Array.isArray(result.packages)).toBe(true);
    expect(result.registry).toBeDefined();
  });
});
