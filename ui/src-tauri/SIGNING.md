# Desktop Signing And Release Notes

This app's desktop release path is intentionally env-driven. No signing
certificate, private key, or notarization credential is committed in the repo.

## Build targets

- macOS: `npm run build:desktop:macos`
- macOS unsigned local smoke build: `npm run build:desktop:macos:unsigned`
- Windows: `npm run build:desktop:windows`
- Windows unsigned local smoke build: `npm run build:desktop:windows:unsigned`
- Linux AppImage + deb: `npm run build:desktop:linux`

The macOS script requests `app` and `dmg` bundles. The Linux script requests
`appimage` and `deb` bundles. The Windows script uses Tauri's default Windows
bundles for the host toolchain.

## macOS signing and notarization

Tauri v2 reads macOS signing and notarization credentials from environment
variables during `tauri build`.

For local signing with a certificate already installed in the login keychain:

- `APPLE_SIGNING_IDENTITY`

For CI signing with an exported `.p12` certificate:

- `APPLE_CERTIFICATE`
- `APPLE_CERTIFICATE_PASSWORD`

For notarization, provide one of these auth sets:

- Apple ID flow:
  - `APPLE_ID`
  - `APPLE_PASSWORD`
  - `APPLE_TEAM_ID`
- App Store Connect API key flow:
  - `APPLE_API_KEY`
  - `APPLE_API_ISSUER`
  - `APPLE_API_KEY_PATH` or `API_PRIVATE_KEYS_DIR`
  - `APPLE_TEAM_ID`

Example:

```bash
export APPLE_CERTIFICATE="$(cat /secure/path/celeritas-cert-base64.txt)"
export APPLE_CERTIFICATE_PASSWORD="..."
export APPLE_ID="release@example.com"
export APPLE_PASSWORD="app-specific-password"
export APPLE_TEAM_ID="TEAMID1234"
npm run build:desktop:macos
```

If you only need a local bundle validation pass without signing credentials,
use `npm run build:desktop:macos:unsigned`.

## Windows signing

The committed scaffold assumes native Tauri Windows signing rather than a
checked-in custom `signCommand`. Supply the certificate material through the
Windows machine's certificate store and let Tauri drive `signtool.exe`.

Useful environment variables:

- `TAURI_WINDOWS_SIGNTOOL_PATH`

If the chosen signing provider requires a non-default `signtool.exe` location,
set `TAURI_WINDOWS_SIGNTOOL_PATH` before running the build. For cross-platform
or Azure-backed signing, Tauri also supports `bundle.windows.signCommand`, but
that command is intentionally not committed here because it is provider-specific
and should live in CI/release configuration instead of the shared app config.

Example PowerShell:

```powershell
$env:TAURI_WINDOWS_SIGNTOOL_PATH="C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe"
npm run build:desktop:windows
```

If you only need an installer smoke build without signing, use
`npm run build:desktop:windows:unsigned`.

## Linux targets

The Linux release script builds:

- `AppImage`
- `deb`

Current scope is packaging only; no Linux signing material is wired in this
tranche. If RPM signing is added later, Tauri supports
`TAURI_SIGNING_RPM_KEY` and `TAURI_SIGNING_RPM_KEY_PASSPHRASE`, but RPM is not
part of the current Celeritas desktop target set.

## Notes

- Tauri's `--no-sign` flag is the expected escape hatch for local debug and CI
  validation builds when signing secrets are intentionally absent.
- `.env` files are not a safe source of truth for release credentials here;
  provide these variables from the shell, CI secret store, or release runner.
