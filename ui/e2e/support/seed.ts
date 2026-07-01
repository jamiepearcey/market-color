import { expect, type APIRequestContext } from "@playwright/test";

export const apiBase = "http://127.0.0.1:38787/api";
const connectorPrefsKey = "connectors:instances";

export const sourceInstance = {
  id: "ci-source-smoke",
  spec: "source-filesystem-csv-ipc",
  kind: "source" as const,
  driver: "source-filesystem-csv-ipc",
  name: "Smoke source",
  params: {},
  jobParamDefaults: {},
  enabled: true,
};

export const targetInstance = {
  id: "ci-target-smoke",
  spec: "sink-redis-ipc",
  kind: "target" as const,
  driver: "sink-redis-ipc",
  name: "Smoke target",
  params: {},
  jobParamDefaults: {},
  enabled: true,
};

export async function seedSmokeState(request: APIRequestContext) {
  const existingJobs = await request.get(`${apiBase}/jobs`);
  expect(existingJobs.ok()).toBeTruthy();
  for (const job of (await existingJobs.json()) as Array<{ id: string }>) {
    const deleteResponse = await request.delete(
      `${apiBase}/jobs/${encodeURIComponent(job.id)}`,
    );
    expect(deleteResponse.ok()).toBeTruthy();
  }

  const existingInstances = await request.get(`${apiBase}/connector-instances`);
  expect(existingInstances.ok()).toBeTruthy();
  for (const instance of (await existingInstances.json()) as Array<{ id: string }>) {
    const deleteResponse = await request.delete(
      `${apiBase}/connector-instances/${encodeURIComponent(instance.id)}`,
    );
    expect(deleteResponse.ok()).toBeTruthy();
  }

  const sourceResponse = await request.post(`${apiBase}/connector-instances`, {
    data: {
      id: sourceInstance.id,
      driver: sourceInstance.driver,
      name: sourceInstance.name,
      params: sourceInstance.params,
      jobParamDefaults: sourceInstance.jobParamDefaults,
    },
  });
  expect(sourceResponse.ok()).toBeTruthy();

  const targetResponse = await request.post(`${apiBase}/connector-instances`, {
    data: {
      id: targetInstance.id,
      driver: targetInstance.driver,
      name: targetInstance.name,
      params: targetInstance.params,
      jobParamDefaults: targetInstance.jobParamDefaults,
    },
  });
  expect(targetResponse.ok()).toBeTruthy();

  const registryUrlsResponse = await request.put(`${apiBase}/registry/urls`, {
    data: {
      urls: ["http://127.0.0.1:38788/registry.manifest.json"],
    },
  });
  expect(registryUrlsResponse.ok()).toBeTruthy();

  const connectorPrefsResponse = await request.put(
    `${apiBase}/prefs/${encodeURIComponent(connectorPrefsKey)}`,
    {
      data: {
        value: JSON.stringify([sourceInstance, targetInstance]),
      },
    },
  );
  expect(connectorPrefsResponse.ok()).toBeTruthy();

  const jobResponse = await request.post(`${apiBase}/jobs`, {
    data: {
      id: "job-smoke",
      name: "Smoke job",
      definition: {
        source: "",
        sourceConnectorId: sourceInstance.id,
        targetConnectorId: targetInstance.id,
        steps: [{ template: targetInstance.driver, args: {} }],
      },
    },
  });
  expect(jobResponse.ok()).toBeTruthy();

  return {
    source: await sourceResponse.json(),
    target: await targetResponse.json(),
    job: await jobResponse.json(),
  };
}
