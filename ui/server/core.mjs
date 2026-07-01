import { ERROR_CODES, appError } from "./errors.mjs";

export const DEFAULT_CORS_ORIGINS = [
  "http://127.0.0.1:4500",
  "http://127.0.0.1:5173",
  "http://localhost:5173",
];
export const SAFE_IDENTIFIER_RE = /^[a-z0-9][a-z0-9._/-]*$/;

export function isPlainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function isLoopbackHost(host) {
  const normalized = String(host || "")
    .trim()
    .replace(/^\[(.*)\]$/, "$1")
    .toLowerCase();
  return normalized === "127.0.0.1" || normalized === "localhost" || normalized === "::1";
}

export function resolveCorsOrigins(envValue) {
  if (typeof envValue === "string" && envValue.trim()) {
    return envValue
      .split(",")
      .map((origin) => origin.trim())
      .filter(Boolean);
  }
  return [...DEFAULT_CORS_ORIGINS];
}

export function isOriginAllowed(origin, allowlist) {
  if (!origin) return true;
  return allowlist.includes(origin);
}

export function isMutatingMethod(method) {
  return method === "POST" || method === "PUT" || method === "DELETE" || method === "PATCH";
}

function validationError(field, message) {
  const err = appError(ERROR_CODES.VALIDATION, 400, message);
  err.field = field;
  return err;
}

export function assertSafeIdentifier(field, value) {
  if (typeof value !== "string" || !SAFE_IDENTIFIER_RE.test(value)) {
    throw validationError(field, `${field} must match ${SAFE_IDENTIFIER_RE}`);
  }
  return value;
}

function expectObject(routeName, body) {
  if (!isPlainObject(body)) {
    throw validationError(routeName, `${routeName} body must be an object`);
  }
}

function rejectUnknownFields(body, allowedFields) {
  for (const field of Object.keys(body)) {
    if (!allowedFields.includes(field)) {
      throw validationError(field, `unexpected field '${field}'`);
    }
  }
}

function validateOptionalString(body, field) {
  if (field in body && typeof body[field] !== "string") {
    throw validationError(field, `${field} must be a string`);
  }
}

function validateOptionalBoolean(body, field) {
  if (field in body && typeof body[field] !== "boolean") {
    throw validationError(field, `${field} must be a boolean`);
  }
}

function validateOptionalObject(body, field) {
  if (field in body && !isPlainObject(body[field])) {
    throw validationError(field, `${field} must be an object`);
  }
}

function requireNonEmptyString(body, field) {
  if (typeof body[field] !== "string" || !body[field].trim()) {
    throw validationError(field, `${field} must be a non-empty string`);
  }
}

function validateConnectorBody(routeName, body, { allowAction = false } = {}) {
  const allowedFields = allowAction
    ? ["action", "autoInstall", "driver", "jobParams", "kind", "params", "selection", "spec"]
    : ["autoInstall", "driver", "jobParams", "kind", "params", "selection", "spec"];
  expectObject(routeName, body);
  rejectUnknownFields(body, allowedFields);
  validateOptionalBoolean(body, "autoInstall");
  validateOptionalString(body, "driver");
  validateOptionalString(body, "spec");
  validateOptionalString(body, "kind");
  validateOptionalObject(body, "params");
  validateOptionalObject(body, "jobParams");
  if ("selection" in body && !Array.isArray(body.selection)) {
    throw validationError("selection", "selection must be an array");
  }
  if (allowAction) validateOptionalString(body, "action");
  const driver = typeof body.driver === "string" && body.driver.trim()
    ? body.driver.trim()
    : typeof body.spec === "string" && body.spec.trim()
      ? body.spec.trim()
      : null;
  if (!driver) {
    throw validationError("driver", "driver or spec is required");
  }
  assertSafeIdentifier("driver", driver);
  return body;
}

export function validateBody(routeName, body) {
  switch (routeName) {
    case "connectorTest":
      return validateConnectorBody(routeName, body);
    case "connectorInspect":
      return validateConnectorBody(routeName, body);
    case "connectorAction":
      return validateConnectorBody(routeName, body, { allowAction: true });
    case "jobSave":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["definition", "description", "enabled", "id", "name"]);
      requireNonEmptyString(body, "name");
      validateOptionalString(body, "id");
      validateOptionalString(body, "description");
      validateOptionalBoolean(body, "enabled");
      validateOptionalObject(body, "definition");
      return body;
    case "jobDryRun":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["autoInstall", "fullRefresh", "job", "source", "steps"]);
      validateOptionalBoolean(body, "autoInstall");
      validateOptionalBoolean(body, "fullRefresh");
      validateOptionalObject(body, "job");
      validateOptionalString(body, "source");
      if ("source" in body) requireNonEmptyString(body, "source");
      if ("steps" in body && !Array.isArray(body.steps)) {
        throw validationError("steps", "steps must be an array");
      }
      if (!body.job && !body.source) {
        throw validationError("source", "source or job is required");
      }
      return body;
    case "targetSave":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["id", "kind", "name", "url"]);
      requireNonEmptyString(body, "name");
      validateOptionalString(body, "id");
      validateOptionalString(body, "kind");
      validateOptionalString(body, "url");
      return body;
    case "prefPut":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["value"]);
      if (!Object.hasOwn(body, "value")) {
        throw validationError("value", "value is required");
      }
      return body;
    case "registryInstall":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["installId", "name", "url"]);
      requireNonEmptyString(body, "name");
      assertSafeIdentifier("name", body.name);
      validateOptionalString(body, "installId");
      if (typeof body.installId === "string" && body.installId.trim()) {
        assertSafeIdentifier("installId", body.installId);
      }
      validateOptionalString(body, "url");
      return body;
    case "registryUrlsPut":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["urls"]);
      if (!Array.isArray(body.urls)) {
        throw validationError("urls", "urls must be an array");
      }
      for (const url of body.urls) {
        if (typeof url !== "string" || !url.trim()) {
          throw validationError("urls", "urls must contain only non-empty strings");
        }
      }
      return body;
    case "activeEnvironmentPut":
      expectObject(routeName, body);
      rejectUnknownFields(body, ["name"]);
      if (body.name !== null && body.name !== undefined && typeof body.name !== "string") {
        throw validationError("name", "name must be a string or null");
      }
      if (typeof body.name === "string" && body.name.trim()) {
        assertSafeIdentifier("name", body.name.trim());
      }
      return body;
    default:
      return body;
  }
}

export class KeyedAsyncQueue {
  constructor() {
    this.states = new Map();
  }

  async run(key, task) {
    const state = this.states.get(key) ?? { pending: 0, tail: Promise.resolve() };
    state.pending += 1;
    const waitFor = state.tail.catch(() => {});
    let release;
    state.tail = new Promise((resolve) => {
      release = resolve;
    });
    this.states.set(key, state);

    await waitFor;
    try {
      return await task();
    } finally {
      state.pending -= 1;
      release();
      if (state.pending === 0) {
        this.states.delete(key);
      }
    }
  }
}
