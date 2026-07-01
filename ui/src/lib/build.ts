export const BUILD_VERSION =
  (import.meta.env.VITE_CELERITAS_BUILD_VERSION as string | undefined) ??
  "0.1.0-dev";

export const BUILD_SHA =
  (import.meta.env.VITE_CELERITAS_BUILD_SHA as string | undefined) ?? "dev";

export function shortBuildSha(): string {
  return BUILD_SHA.slice(0, 12);
}
