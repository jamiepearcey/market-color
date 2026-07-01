# Celeritas — ETL Control UI

A focused management UI for **Celeritas**, an open-source Rust ETL engine.

The app is a React 18 + Vite 6 + TypeScript + Tailwind v3 single-page app that
can run two ways:

- **Browser preview (mock data)** — the default for development. The data layer
  in `src/lib/api.ts` detects the absence of Tauri (`isPreview()`) and serves
  mocked connectors, jobs, and runs, so the UI builds and runs without any
  backend.
- **Desktop (Tauri)** — bundled as a native app that talks to a running
  Celeritas engine.

## Primary navigation

The left rail promotes the three things an operator manages, plus settings:

- **Sources** — configure and test the connectors that import data.
- **Targets** — configure and test the connectors that export data.
- **Scheduled Jobs** — define, schedule (cron), and run ETL jobs; inspect runs.
- **Settings** — engine/target endpoints, secrets backend, API keys.

Sources and Targets are two role-scoped views over the same connectors surface
(`ConnectorsView`): the sidebar, catalog, and empty states adapt to the active
role (`store.connectorRole`).

## Develop

```bash
npm install
npm run dev      # browser preview at the Vite dev URL (mock data, no Tauri)
npm run build    # tsc --noEmit && vite build — the web build needs no Tauri
```

## Production env

Copy `ui/.env.production.example` when producing a browser build that talks to a
deployed sidecar:

- `VITE_CELERITAS_API`: default sidecar base URL for the browser build.
- `VITE_CELERITAS_REGISTRY_URL`: initial registry URL shown in Settings before
  the sidecar or desktop layer overrides it.
- `VITE_CELERITAS_BUILD_VERSION`: build/version string surfaced in the UI
  footer.
- `VITE_CELERITAS_BUILD_SHA`: git SHA surfaced in the UI footer.
- `VITE_CELERITAS_ERROR_REPORT_URL`: optional webhook for UI boundary/global
  error reports.

The sidecar uses the matching runtime env vars `CELERITAS_BUILD_VERSION` and
`CELERITAS_BUILD_SHA`, so `/api/health` and the UI footer can report the same
build identity.

## Desktop (Tauri)

The `src-tauri/` crate wraps the web build as a desktop app. Building the
desktop bundle requires the Rust toolchain and the Tauri prerequisites; the web
build above does **not**.

Common desktop bundle commands:

- `npm run build:desktop:macos`
- `npm run build:desktop:macos:unsigned`
- `npm run build:desktop:windows`
- `npm run build:desktop:windows:unsigned`
- `npm run build:desktop:linux`

Release signing and notarization inputs are documented in
[`src-tauri/SIGNING.md`](./src-tauri/SIGNING.md). The repo does not store any
certificate or notarization secret.

Desktop secret custody:

- Connector secrets and the secrets-backend bearer token are stored in the OS
  keychain.
- The JSON state under the app data directory persists only
  `secret-session://tauri-keychain/...` references, not plaintext secret values.
