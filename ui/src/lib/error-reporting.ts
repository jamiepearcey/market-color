const ERROR_REPORT_URL =
  (import.meta.env.VITE_CELERITAS_ERROR_REPORT_URL as string | undefined)?.trim() ??
  "";

function redactSecrets(text: string): string {
  return text
    .replace(/Bearer\s+[A-Za-z0-9._~+/-]+=*/gi, "Bearer [redacted]")
    .replace(
      /\b(password|token|secret|api[_ -]?key|access[_ -]?key)\b\s*[:=]\s*([^\s,;]+)/gi,
      "$1=[redacted]",
    )
    .replace(/secrets-keeper:\/\/[^\s"'`]+/g, "secrets-keeper://[redacted]")
    .replace(/secret-session:\/\/[^\s"'`]+/g, "secret-session://[redacted]");
}

function serializeError(error: unknown): {
  name?: string;
  message: string;
  stack?: string;
} {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: redactSecrets(error.message || "Unknown UI error"),
      ...(error.stack ? { stack: redactSecrets(error.stack) } : {}),
    };
  }
  return { message: redactSecrets(String(error)) };
}

export function errorReportingConfigured(): boolean {
  return Boolean(ERROR_REPORT_URL);
}

export async function reportUiError(
  kind: string,
  error: unknown,
  context: Record<string, unknown> = {},
): Promise<void> {
  if (!ERROR_REPORT_URL) return;
  const payload = {
    kind,
    ts: new Date().toISOString(),
    url: typeof window === "undefined" ? null : window.location.href,
    userAgent: typeof navigator === "undefined" ? null : navigator.userAgent,
    error: serializeError(error),
    context: Object.fromEntries(
      Object.entries(context).map(([key, value]) => [
        key,
        typeof value === "string" ? redactSecrets(value) : value,
      ]),
    ),
  };
  try {
    await fetch(ERROR_REPORT_URL, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    /* no-op */
  }
}

export function installGlobalErrorReporting() {
  if (!ERROR_REPORT_URL || typeof window === "undefined") return;
  window.addEventListener("error", (event) => {
    void reportUiError("window.error", event.error ?? event.message, {
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    });
  });
  window.addEventListener("unhandledrejection", (event) => {
    void reportUiError("window.unhandledrejection", event.reason);
  });
}
