import { defineConfig, devices } from "@playwright/test";

const uiPort = 4173;
const apiPort = 38787;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: process.env.CI ? [["html"], ["list"]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${uiPort}`,
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: `node e2e/support/run-sidecar-with-stub.mjs --port ${apiPort}`,
      url: `http://127.0.0.1:${apiPort}/api/health`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command: `npm run preview -- --host 127.0.0.1 --port ${uiPort}`,
      url: `http://127.0.0.1:${uiPort}`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
  ],
});
