import type { ReactNode } from "react";

export interface UiErrorLike {
  name?: string | undefined;
  message: string;
  status?: number | undefined;
  code: string | undefined;
  field: string | null | undefined;
  details: unknown;
}

export class ServerApiError extends Error implements UiErrorLike {
  status: number;
  code: string | undefined;
  field: string | null | undefined;
  details: unknown;

  constructor(input: {
    message: string;
    status: number;
    code?: string | undefined;
    field?: string | null | undefined;
    details?: unknown;
  }) {
    super(input.message);
    this.name = "ServerApiError";
    this.status = input.status;
    this.code = input.code;
    this.field = input.field ?? null;
    this.details = input.details;
  }
}

export function asUiError(error: unknown): UiErrorLike {
  if (error instanceof ServerApiError) return error;
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      status: undefined,
      code: undefined,
      field: undefined,
      details: undefined,
    };
  }
  return {
    message: String(error),
    status: undefined,
    code: undefined,
    field: undefined,
    details: undefined,
  };
}

export function describeUiError(error: unknown): {
  title: string;
  message: string;
  field?: string | null | undefined;
} {
  const err = asUiError(error);
  const fallback = err.message || "Unexpected error.";
  switch (err.code) {
    case "NEEDS_INSTALL":
      return {
        title: "Connector install required",
        message:
          "Install the connector first, or retry with auto-install enabled.",
        field: err.field,
      };
    case "ENGINE_MISSING":
      return {
        title: "Engine not found",
        message:
          "Start the Celeritas sidecar with a valid engine binary, then retry.",
      };
    case "TIMEOUT":
      return {
        title: "Operation timed out",
        message:
          "The engine took too long to respond. Check connectivity or retry.",
      };
    case "AUTH_REQUIRED":
    case "AUTH_FAILED":
      return {
        title: "Authentication failed",
        message: "Check the configured token or credentials, then retry.",
        field: err.field,
      };
    case "CONNECTIVITY":
      return {
        title: "Connectivity error",
        message: fallback,
        field: err.field,
      };
    case "CONFIG_INVALID":
    case "VALIDATION":
      return {
        title: "Validation failed",
        message: fallback,
        field: err.field,
      };
    case "REGISTRY_FETCH_FAILED":
    case "REGISTRY_SCHEMA":
    case "REGISTRY_UNTRUSTED":
    case "REGISTRY_PACKAGE_NOT_FOUND":
      return {
        title: "Registry error",
        message: fallback,
      };
    case "ORIGIN_FORBIDDEN":
      return {
        title: "Origin blocked",
        message:
          "This browser origin is not allowed by the sidecar CORS policy.",
      };
    case "SHUTTING_DOWN":
      return {
        title: "Server shutting down",
        message: "The sidecar is draining in-flight work. Retry in a moment.",
      };
    default:
      return {
        title:
          err.status && err.status >= 500 ? "Server error" : "Request failed",
        message: fallback,
        field: err.field,
      };
  }
}

export function formatInlineFieldError(
  error: unknown,
  fieldName: string,
): string | null {
  const desc = describeUiError(error);
  return desc.field === fieldName ? desc.message : null;
}

export function errorPanelMessage(error: unknown): ReactNode {
  const desc = describeUiError(error);
  return desc.message;
}
