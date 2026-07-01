import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { seedSmokeState } from "./support/seed";

const seriousOrCritical = new Set(["serious", "critical"]);

test("primary views expose no serious or critical axe violations", async ({
  page,
  request,
}) => {
  await seedSmokeState(request);
  await page.goto("/");
  await expect(page.getByText("engine: server", { exact: true })).toBeVisible();

  const primaryNav = page.getByRole("navigation", { name: "Primary" });
  const views = [
    {
      name: /Sources/,
      ready: () => page.getByText("Import connectors", { exact: true }),
    },
    {
      name: /Targets/,
      ready: () => page.getByText("Export connectors", { exact: true }),
    },
    {
      name: "Store",
      ready: () =>
        page.getByPlaceholder("Search the store by name, tag, or description"),
    },
    {
      name: /Scheduled Jobs/,
      ready: () => page.getByText("Jobs · bound flows", { exact: true }),
    },
    {
      name: "Settings",
      ready: () => page.getByText("Sidecar connection", { exact: true }),
    },
  ] as const;

  for (const view of views) {
    await primaryNav.getByRole("button", { name: view.name }).click();
    await expect(view.ready()).toBeVisible();

    const results = await new AxeBuilder({ page }).analyze();
    const violations = results.violations.filter((violation) =>
      seriousOrCritical.has(violation.impact ?? ""),
    );
    expect(
      violations,
      `${String(view.name)} serious/critical violations: ${violations
        .map((violation) => `${violation.id} (${violation.impact})`)
        .join(", ")}`,
    ).toEqual([]);
  }
});
