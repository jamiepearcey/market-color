export const ERROR_CODES = {
  AUTH_REQUIRED: "AUTH_REQUIRED",
  AUTH_FAILED: "AUTH_FAILED",
  CONFIG_INVALID: "CONFIG_INVALID",
  CONNECTIVITY: "CONNECTIVITY",
  CONNECTOR_NOT_FOUND: "CONNECTOR_NOT_FOUND",
  ENGINE_FAILED: "ENGINE_FAILED",
  ENGINE_MISSING: "ENGINE_MISSING",
  INTERNAL: "INTERNAL",
  NEEDS_INSTALL: "NEEDS_INSTALL",
  NOT_FOUND: "NOT_FOUND",
  ORIGIN_FORBIDDEN: "ORIGIN_FORBIDDEN",
  RATE_LIMITED: "RATE_LIMITED",
  REGISTRY_FETCH_FAILED: "REGISTRY_FETCH_FAILED",
  REGISTRY_PACKAGE_NOT_FOUND: "REGISTRY_PACKAGE_NOT_FOUND",
  REGISTRY_SCHEMA: "REGISTRY_SCHEMA",
  REGISTRY_UNTRUSTED: "REGISTRY_UNTRUSTED",
  SHUTTING_DOWN: "SHUTTING_DOWN",
  TIMEOUT: "TIMEOUT",
  VALIDATION: "VALIDATION",
};

export function appError(code, statusCode, message, details) {
  const err = new Error(message);
  err.code = code;
  err.statusCode = statusCode;
  if (details !== undefined) {
    err.details = details;
  }
  return err;
}

export function errorBody(err) {
  return {
    code: err?.code || ERROR_CODES.INTERNAL,
    message: err?.message || "internal error",
    ...(err?.details !== undefined ? { details: err.details } : {}),
    error: err?.message || "internal error",
  };
}
