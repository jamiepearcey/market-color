import { expect, test } from "@playwright/test";

import { seedSmokeState, sourceInstance, targetInstance } from "./support/seed";

test("smoke flow covers sources, targets, store, and job run", async ({
  page,
  request,
}) => {
  const seeded = await seedSmokeState(request);
  await page.addInitScript(() => {
    window.__CELERITAS_E2E__ = true;
  });

  await page.goto("/");
  await expect(page.getByText("engine: server", { exact: true })).toBeVisible();
  await page.evaluate(async ({ instances, job }) => {
    await window.__CELERITAS_TEST_HOOK__?.actions.loadPrefs();
    await window.__CELERITAS_TEST_HOOK__?.actions.loadJobs();
    if (!window.__CELERITAS_TEST_HOOK__) return;
    const { actions, store } = window.__CELERITAS_TEST_HOOK__;
    const template =
      store.templates.find((entry) =>
        entry.inputs.every(
          (input) =>
            input.kind === "Table" || input.kind === "TypedRelation",
        ),
      ) ?? store.templates[0];
    const sourceColumns = [{ name: "users", dataType: "integer" }];
    const args = Object.fromEntries(
      (template?.inputs ?? [])
        .filter(
          (input) =>
            input.kind !== "Table" && input.kind !== "TypedRelation",
        )
        .map((input) => [
          input.name,
          input.kind === "Number"
            ? "1"
            : input.kind === "Date"
              ? "2024-01-01"
              : "users",
        ]),
    );
    const runnableJob = {
      ...job,
      definition: {
        ...job.definition,
        steps: template
          ? [{ template: template.name, args }]
          : job.definition.steps,
      },
    };
    store.connectorInstances = instances;
    store.jobs = [runnableJob];
    store.jobsLoading = false;
    store.jobDraft = runnableJob;
    store.jobSourceMode = "connector";
    store.activeTargetId = "embedded";
    store.sourceColumns = sourceColumns;
    store.sourcePreview = {
      status: "complete",
      rows: [{ users: 1 }],
      rowCount: 1,
      elapsedMs: 10,
    };
  }, {
    instances: [sourceInstance, targetInstance],
    job: seeded.job,
  });
  const primaryNav = page.getByRole("navigation", { name: "Primary" });
  await expect(page.getByText("Import connectors", { exact: true })).toBeVisible();
  await expect(page.getByText("Sources (import)", { exact: true })).toBeVisible();

  await primaryNav.getByRole("button", { name: /Targets/ }).click();
  await expect(page.getByText("Export connectors", { exact: true })).toBeVisible();
  await expect(page.getByText("Targets (export)", { exact: true })).toBeVisible();

  await primaryNav.getByRole("button", { name: "Store", exact: true }).click();
  const packageSearch = page.getByPlaceholder(
    "Search the store by name, tag, or description",
  );
  await expect(packageSearch).toBeVisible();
  await packageSearch.fill("Filesystem CSV Source");
  await page
    .getByRole("button", {
      name:
        "Filesystem CSV Source source files v0.1.0 Installed Filesystem CSV Source — reads from local files into Celeritas over native Arrow IPC.",
      exact: true,
    })
    .click();
  await expect(
    page.getByText(
      "Filesystem CSV Source — reads from local files into Celeritas over native Arrow IPC.",
    ),
  ).toBeVisible();

  await primaryNav
    .getByRole("button", { name: /Scheduled Jobs/ })
    .click();
  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByRole("button", { name: "Run now" })).toBeVisible();
  await page.getByRole("button", { name: "Run now" }).click();
  await page.getByRole("tab", { name: "Run history" }).click();
  await expect(page.locator('[data-job-section="runs"] button').first()).toBeVisible();
  await page.locator('[data-job-section="runs"] button').first().click();
  await expect(page.getByText("Run diagnostics").first()).toBeVisible();
  await expect(
    page.getByText("Job is missing a source or sink connector.").first(),
  ).toBeVisible();
  await expect(page.getByText(/average_fill_price/).first()).toBeVisible();
});
