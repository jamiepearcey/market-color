import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The UI is a pure static client. In dev, Vite serves it on :1420 and proxies
// /api + /chat to the standalone API server (default :8787, run via `npm run api`).
// In a static/Tauri build the client points at the API via the configurable base
// URL (src/lib/api-base.ts), so no backend lives in Vite.
const API = process.env.VITE_API_PROXY || "http://localhost:8787";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
    dedupe: ["react", "react-dom"],
  },
  server: {
    port: 1420,
    strictPort: true,
    proxy: {
      "/api": { target: API, changeOrigin: true },
      "/chat": { target: API, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    target: "es2022",
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
